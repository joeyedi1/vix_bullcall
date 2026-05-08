"""SpreadEvaluator — end-to-end spread evaluation.

The composer that wires Phase-2/3/4 layers into one method:

    BullCallSpread + VIXSnapshot + as_of  →  SpreadEvaluation

Inside one `evaluate(...)` call:

  1. `ForwardSelector` resolves the Black-76 forward from `vx_curve`
     (settlement-date match per ARCHITECTURE §4.1).
  2. `ChainIVProvider` resolves a per-leg `LegIV` (vendor IVOL_LAST or
     B76-inverted from chain midpoint) per §4.3.
  3. `Black76Pricer.price_spread` produces the diagnostic
     `TheoreticalSpreadPrice` (FlatVolError-guarded against VVIX
     substitution per validation memo §4 / §12.4).
  4. `FillEngine.attempt_fill` produces an `ExecutedFill` or typed
     `RejectedOrder` per §5 — the executable answer.

Returns a `SpreadEvaluation` bundling forward, both LegIVs (with source
audit), the theoretical, and the fill. Computing edge_bleed
(`theoretical - executed_debit`) is a property on the result; this is
the quantity the §8.3 edge-bleed report rolls up across the backtest.

This is the unit a Strategy will call once per signal — Phase-5's
SpreadSelector chooses the spread, sizes it, and feeds it here.

Out of scope for this first pass
--------------------------------
  - Settlement-event integration (this composer is for entry/intra-life
    evaluation; payoff at SOQ goes through ExitEngine + SettlementMarket).
  - PCP / interpolated forward sources — `ForwardSelector` currently
    raises NotImplementedError on those.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from vix_spread.data.vix_index_options import vix_option_active_contract_id
from vix_spread.execution.fill_engine import (
    ExecutedFill,
    FillEngine,
    RejectedOrder,
)
from vix_spread.execution.fill_modes import FillMode
from vix_spread.execution.liquidity_gates import LiquidityGates
from vix_spread.products.base import Product
from vix_spread.products.spread import BullCallSpread

from .black76 import Black76Pricer
from .forward_selector import Forward, ForwardSelector
from .leg_iv import ChainIVProvider, LegIV
from .theoretical import TheoreticalSpreadPrice

if TYPE_CHECKING:
    from vix_spread.data.snapshot import VIXSnapshot


@dataclass(frozen=True)
class SpreadEvaluation:
    """Complete result of evaluating a spread at one moment.

    Both `theoretical` (vendor-or-inverted IVs into Black-76) and `fill`
    (executable price under chosen mode + gates) are produced. Comparing
    them gives `edge_bleed`, the quantity §8.3 reports roll up across
    the backtest.

    `iv_long` / `iv_short` carry the `LegIVSource` audit tag so the
    backtest reporting layer (§8.2) can bin model-risk by source.
    """
    spread: BullCallSpread
    as_of: datetime
    forward: Forward
    iv_long: LegIV
    iv_short: LegIV
    theoretical: TheoreticalSpreadPrice
    fill: "ExecutedFill | RejectedOrder"

    @property
    def is_filled(self) -> bool:
        return isinstance(self.fill, ExecutedFill)

    @property
    def edge_bleed(self) -> float | None:
        """`theoretical.value - fill.debit_per_spread` if filled, else None.

        Sign convention: positive = paid LESS than fair value (favourable);
        negative = paid MORE than fair value (the typical case for a
        debit spread crossed against a non-zero NBBO width).
        """
        if isinstance(self.fill, ExecutedFill):
            return float(self.theoretical.value - self.fill.debit_per_spread)
        return None


class SpreadEvaluator:
    """Composes ForwardSelector + ChainIVProvider + Black76Pricer + FillEngine.

    Constructor injection — every collaborator is explicit so tests can
    swap any of them and a Strategy can keep one Evaluator per backtest
    run. `gates` are also constructor-bound: a backtest run uses one
    gate configuration; per-call gate overrides go on `evaluate(gates=...)`.

    `contract_id_for` defaults to `vix_option_active_contract_id` (the
    Tuesday-last-trade form matching the per-minute NBBO panel). Inject
    a different mapper for VX-OOF or other products when those wire in.
    """

    def __init__(
        self,
        chain_iv_provider: ChainIVProvider,
        forward_selector: ForwardSelector,
        pricer: Black76Pricer,
        fill_engine: FillEngine,
        gates: LiquidityGates,
        contract_id_for: Callable[[Product], str] | None = None,
    ) -> None:
        self.chain_iv_provider = chain_iv_provider
        self.forward_selector = forward_selector
        self.pricer = pricer
        self.fill_engine = fill_engine
        self.gates = gates
        self.contract_id_for = contract_id_for or vix_option_active_contract_id

    def evaluate(
        self,
        spread: BullCallSpread,
        snapshot: "VIXSnapshot",
        order_size: int = 1,
        fill_mode: FillMode = FillMode.SYNTHETIC_BIDASK,
        *,
        accept_midpoint_optimism: bool = False,
        slippage_ticks_per_leg: int = 1,
        slippage_apply_to_short_leg_only: bool = False,
        tick_value: float | None = None,
    ) -> SpreadEvaluation:
        """Run the full evaluation pipeline. Never raises on a missing
        quote — that branch becomes a `RejectedOrder` with
        `reason='gate_fail'`, `sub_reason='no_quote'` so backtest reports
        bin no-quote misses alongside other gate failures.
        """
        as_of = snapshot.timestamp

        # 1. Forward (settlement-date match against vx_curve).
        forward = self.forward_selector.select(
            spread.long_leg, snapshot, as_of, source="settlement_date_match",
        )

        # 2. Per-leg IVs.
        iv_long = self.chain_iv_provider.get(
            product=spread.long_leg, as_of=as_of,
            forward=forward, risk_free_rate=snapshot.risk_free_rate,
        )
        iv_short = self.chain_iv_provider.get(
            product=spread.short_leg, as_of=as_of,
            forward=forward, risk_free_rate=snapshot.risk_free_rate,
        )

        # 3. Theoretical spread price (FlatVolError-guarded inside).
        theoretical = self.pricer.price_spread(
            spread, forward, iv_long.value, iv_short.value,
            as_of, snapshot.risk_free_rate,
        )

        # 4. Quote lookup → 5. Fill (or no-quote rejection).
        long_id = self.contract_id_for(spread.long_leg)
        short_id = self.contract_id_for(spread.short_leg)
        long_q = snapshot.options_quotes.get(long_id)
        short_q = snapshot.options_quotes.get(short_id)

        if long_q is None or short_q is None:
            missing = [
                name for name, q in (("long", long_q), ("short", short_q))
                if q is None
            ]
            fill: ExecutedFill | RejectedOrder = RejectedOrder(
                timestamp=as_of, spread=spread, reason="gate_fail",
                detail={
                    "sub_reason": "no_quote",
                    "missing_legs": missing,
                    "long_id": long_id,
                    "short_id": short_id,
                },
            )
        else:
            fill = self.fill_engine.attempt_fill(
                spread=spread, long_q=long_q, short_q=short_q,
                order_size=order_size, mode=fill_mode, gates=self.gates,
                decision_timestamp=as_of,
                accept_midpoint_optimism=accept_midpoint_optimism,
                slippage_ticks_per_leg=slippage_ticks_per_leg,
                slippage_apply_to_short_leg_only=slippage_apply_to_short_leg_only,
                tick_value=tick_value,
            )

        return SpreadEvaluation(
            spread=spread, as_of=as_of, forward=forward,
            iv_long=iv_long, iv_short=iv_short,
            theoretical=theoretical, fill=fill,
        )
