from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from vix_spread.products.base import Product
from vix_spread.utils.errors import ForwardSelectionError

if TYPE_CHECKING:
    from vix_spread.data.snapshot import OptionsMarketSnapshot


ForwardSource = Literal[
    'settlement_date_match',
    'put_call_parity',
    'interpolated',
    'spot_vix',
]


@dataclass(frozen=True)
class Forward:
    value: float
    selection_method: Literal['settlement_date_match',
                              'put_call_parity',
                              'interpolated']
    model_risk_flag: bool  # True iff selection_method == 'interpolated'
    settlement_date: datetime


class ForwardSelector:
    """Selects the Black-76 forward input. Hierarchy:
       1. Exact same-settlement-date VX future (preferred).
       2. Put-call parity implied forward from same-expiry options.
       3. Term-structure interpolation (FALLBACK ONLY, sets model_risk_flag).
    Selecting from spot VIX is FORBIDDEN and raises ForwardSelectionError.

    Phase-2 scope:
      * settlement_date_match: WIRED. Looks up the VX future whose
        settlement_date matches `product.settlement_event.date()` from
        `market.vx_curve` and returns it as the Forward.
      * put_call_parity / interpolated: NotImplementedError until the
        option-chain snapshot lands. These are NOT-WIRED, not FORBIDDEN —
        spot_vix is the only permanent refusal."""

    def select(
        self,
        product: Product,
        market: 'OptionsMarketSnapshot',
        as_of: datetime,
        source: ForwardSource = 'settlement_date_match',
    ) -> Forward:
        """Returns Forward with selection_method tag for audit.

        For VXFutureOption, only branch (1) is valid — the deliverable VX
        is the forward; PCP and interpolation are unreachable.
        """
        if source == 'spot_vix':
            raise ForwardSelectionError(
                "Spot VIX is FORBIDDEN as a Black-76 forward input. "
                "Use the same-settlement-date VX future, a PCP-implied "
                "forward, or (with model_risk_flag) a term-structure "
                "interpolation."
            )
        if source == 'put_call_parity':
            raise NotImplementedError(
                "PCP-implied forward not wired in Phase 2; requires the "
                "option-chain snapshot. Use 'settlement_date_match' until "
                "the chain ingestion lands."
            )
        if source == 'interpolated':
            raise NotImplementedError(
                "Term-structure interpolation not wired in Phase 2. When "
                "enabled, it must set model_risk_flag=True on the returned "
                "Forward."
            )
        if source == 'settlement_date_match':
            return self._settlement_date_match(product, market)
        raise ValueError(f"Unknown ForwardSource: {source!r}.")

    @staticmethod
    def _settlement_date_match(
        product: Product,
        market: 'OptionsMarketSnapshot',
    ) -> Forward:
        """Look up VX future price by `product.settlement_event.date()`.

        Raises ForwardSelectionError if the market lacks a `vx_curve`,
        if no contract matches the settlement date, or if the price is
        non-positive (would blow up `log(F/K)` in the pricer)."""
        if not hasattr(market, 'vx_curve'):
            raise ForwardSelectionError(
                "market lacks 'vx_curve' attribute required for "
                "settlement_date_match selection."
            )
        target = product.settlement_event.date()
        if target not in market.vx_curve:
            raise ForwardSelectionError(
                f"No VX future with settlement_date={target} in market.vx_curve."
            )
        price = float(market.vx_curve[target])
        if price <= 0:
            raise ForwardSelectionError(
                f"VX future price for settlement_date={target} is "
                f"non-positive: {price}."
            )
        return Forward(
            value=price,
            selection_method='settlement_date_match',
            model_risk_flag=False,
            settlement_date=product.settlement_event,
        )
