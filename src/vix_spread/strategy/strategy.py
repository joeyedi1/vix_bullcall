"""VIXBullCallSpreadStrategy — per-minute decision composer.

ARCHITECTURE §7.1. The thin layer between regime/curve signals and the
executable spread. Composes:

  - `StrategyHypothesis` — entry-regime + entry-curve filters
  - `SpreadSelector`     — picks (long_strike, short_strike, expiry)
  - `SpreadEvaluator`    — forward + IVs + theoretical + fill in one call
  - `FixedRiskSizer`     — integer position size from max-loss budget
  - `ExitPolicy`         — declared exit mode (audit-only here; ExitEngine
                            consumes during the loop)

Returns a `StrategyDecision` describing what the strategy WANTS to do
at `as_of`. Per ARCH §7.2 timing rule, **the decision is for action
AFTER `as_of`**: the backtest loop uses the next eligible quote minute
to actually book the trade. The FillEngine's same-minute-fill rejection
enforces this structurally — this method does NOT shift `as_of` itself.

Decision flow
-------------
  1. Lookahead guard: signal.as_of must be <= as_of.
  2. `entry_regime_filter` — skip if False.
  3. `entry_curve_filter`  — skip if False.
  4. `SpreadSelector.select(...)` — skip if None (no fillable strikes).
  5. `SpreadEvaluator.evaluate(...)` — produces theoretical + preview fill.
  6. If preview fill is `RejectedOrder` — skip (gate failure).
  7. `FixedRiskSizer.size(...)` against the preview debit — skip if 0.
  8. Return `Enter(spread, evaluation, size)` — backtest re-evaluates at
     the next eligible quote minute to lock the actual fill.

`evaluation` and `size` are computed at `as_of` against `market[as_of]`
quotes — they are PREVIEWS the backtest engine should re-confirm at
`as_of + 1min`. Same as Phase-2's TheoreticalPrice / executed-fill
separation, applied at the strategy level: the decision is the spread
+ a forecast of what the fill will look like; the actual fill belongs
to the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from vix_spread.execution.exit_policy import ExitPolicy
from vix_spread.execution.fill_engine import ExecutedFill, RejectedOrder
from vix_spread.execution.fill_modes import FillMode
from vix_spread.products.spread import BullCallSpread
from vix_spread.utils.errors import LookaheadError

from .hypothesis import StrategyHypothesis
from .sizing import FixedRiskSizer
from .spread_selector import SpreadSelector

if TYPE_CHECKING:
    from vix_spread.data.snapshot import VIXSnapshot
    from vix_spread.pricing.evaluator import SpreadEvaluation, SpreadEvaluator
    from vix_spread.regime.base import RegimeSignal


SkipReason = Literal[
    "regime_filter",
    "curve_filter",
    "no_spread",
    "fill_rejected",
    "size_zero",
]


@dataclass(frozen=True)
class StrategyDecision:
    """Outcome of one `VIXBullCallSpreadStrategy.evaluate(...)` call.

    `as_of` is the decision timestamp (T). Any actual execution happens
    at T+1 in the backtest loop — the FillEngine's same-minute-fill
    rejection enforces this. `evaluation` and `size`, if populated, are
    PREVIEWS at T to be re-confirmed against T+1's market.

    Attributes
    ----------
    as_of
        Decision timestamp (tz-aware UTC).
    action
        `'enter'` if all filters + selection + sizing succeeded;
        `'skip'` otherwise. (`'hold'` and `'exit'` are returned by the
        ExitEngine for open positions; the entry-strategy returns only
        these two.)
    reason
        For `'enter'`: `"entered"`. For `'skip'`: a `SkipReason` tag.
    spread, evaluation, size
        Populated on `'enter'`; `None` on `'skip'`. `evaluation` may
        carry a `RejectedOrder` even on skip when `reason='fill_rejected'`
        — that's how the audit log records the rejection cause.
    hypothesis_name
        Tag from the bound `StrategyHypothesis` for audit; ARCH §3.5.
    """
    as_of: datetime
    action: Literal["enter", "skip"]
    reason: str
    hypothesis_name: str
    spread: BullCallSpread | None = None
    evaluation: "SpreadEvaluation | None" = None
    size: int | None = None


class VIXBullCallSpreadStrategy:
    """Per-minute strategy composer (ARCH §7.1)."""

    def __init__(
        self,
        hypothesis: StrategyHypothesis,
        spread_selector: SpreadSelector,
        evaluator: "SpreadEvaluator",
        sizer: FixedRiskSizer,
        exit_policy: ExitPolicy,
    ) -> None:
        self.hypothesis = hypothesis
        self.spread_selector = spread_selector
        self.evaluator = evaluator
        self.sizer = sizer
        self.exit_policy = exit_policy

    def evaluate(
        self,
        market: "VIXSnapshot",
        signal: "RegimeSignal",
        as_of: datetime,
        *,
        equity: float,
        fill_mode: FillMode = FillMode.SYNTHETIC_BIDASK,
        accept_midpoint_optimism: bool = False,
        slippage_ticks_per_leg: int = 1,
    ) -> StrategyDecision:
        """Decide at T (= `as_of`); execution happens at T+1 in the loop.

        Raises `LookaheadError` if `signal.as_of > as_of` — a structural
        guard against feeding the strategy a signal computed from data
        the decision time shouldn't know about.
        """
        if signal.as_of > as_of:
            raise LookaheadError(
                f"Strategy.evaluate received signal.as_of={signal.as_of} > "
                f"as_of={as_of}. Signals must be functions of data with "
                f"timestamp <= decision time."
            )

        skip = lambda reason: StrategyDecision(
            as_of=as_of, action="skip", reason=reason,
            hypothesis_name=self.hypothesis.name,
        )

        # 1. Hypothesis filters.
        if not self.hypothesis.entry_regime_filter(signal):
            return skip("regime_filter")
        if not self.hypothesis.entry_curve_filter(signal.curve_features):
            return skip("curve_filter")

        # 2. Spread selection (pre-filtered for no-bid-short).
        spread = self.spread_selector.select(market, signal, as_of)
        if spread is None:
            return skip("no_spread")

        # 3. Preview evaluation (theoretical + simulated fill at as_of).
        evaluation = self.evaluator.evaluate(
            spread=spread, snapshot=market,
            order_size=1, fill_mode=fill_mode,
            accept_midpoint_optimism=accept_midpoint_optimism,
            slippage_ticks_per_leg=slippage_ticks_per_leg,
        )
        if isinstance(evaluation.fill, RejectedOrder):
            return StrategyDecision(
                as_of=as_of, action="skip", reason="fill_rejected",
                hypothesis_name=self.hypothesis.name,
                spread=spread, evaluation=evaluation,
            )

        # 4. Sizing against the preview debit. The backtest loop should
        #    re-size against the actual T+1 fill if it diverges materially.
        assert isinstance(evaluation.fill, ExecutedFill)
        size = self.sizer.size(
            spread=spread,
            debit_per_spread=evaluation.fill.debit_per_spread,
            equity=equity,
        )
        if size <= 0:
            return StrategyDecision(
                as_of=as_of, action="skip", reason="size_zero",
                hypothesis_name=self.hypothesis.name,
                spread=spread, evaluation=evaluation,
            )

        return StrategyDecision(
            as_of=as_of, action="enter", reason="entered",
            hypothesis_name=self.hypothesis.name,
            spread=spread, evaluation=evaluation, size=size,
        )
