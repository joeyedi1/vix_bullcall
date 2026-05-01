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
    Selecting from spot VIX is FORBIDDEN and raises ForwardSelectionError."""

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

        Spike-phase enforcement:
          * source='spot_vix' is a permanent refusal — validation memo
            prohibits spot VIX as a Black-76 input under any circumstance.
          * source='interpolated' raises until the fallback path is wired
            (later phase). When wired, it must set model_risk_flag=True.
        """
        if source == 'spot_vix':
            raise ForwardSelectionError(
                "Spot VIX is FORBIDDEN as a Black-76 forward input. "
                "Use the same-settlement-date VX future, a PCP-implied "
                "forward, or (with model_risk_flag) a term-structure "
                "interpolation."
            )
        if source == 'interpolated':
            raise ForwardSelectionError(
                "Interpolated fallback is not wired in the spike phase. "
                "When enabled, it must set model_risk_flag=True on the "
                "returned Forward."
            )
        # source in {'settlement_date_match', 'put_call_parity'}:
        # selection logic deferred to Phase 2.
        ...
