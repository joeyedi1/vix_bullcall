"""WalkForwardBacktest — the only backtest mode in the codebase (ARCH §8.1).

Drives `VIXBullCallSpreadStrategy` across a minute grid with **strict
T → T+1 execution**: a decision generated at minute T is filled against
`market[T+1]` (or the next eligible quote AFTER T), never `market[T]`.
Same-minute fills are structurally impossible — the loop's fill phase
only attempts pending entries with `T > queued_at`. The FillEngine's
documented same-minute rejection is the fail-safe; this loop is the
primary enforcer.

Per-minute order of operations
------------------------------
  1. **Fill pending entries** queued from PRIOR minutes.
     Strict T+1+ rule: only attempt if `T > pending.queued_at`.
     Successful fill → add to open positions; equity decreases.
     Rejected fill → log (no retry — single-attempt for first pass).

  2. **Evaluate exits** for each open position via `exit_decider(pos, T)`:
     - Returns `None`: position holds.
     - Returns `ExitPolicy`: call `ExitEngine.execute_exit(...)`.
        * `ExecutedFill` (Tuesday close)  → completed_trade; equity ↑
        * `SettlementOutcome`             → completed_trade; equity ↑
        * `FailedExit`                    → log; position remains open

  3. **Generate new decisions** via `strategy.evaluate(market, signal, as_of=T)`.
     If `action='enter'`, queue for execution at next minute. The decision
     log records every evaluation (enter or skip) for audit.

  4. **Snapshot equity** at end of T.

Exits run BEFORE new decisions per ARCH §7.2. Pending fills run BEFORE
exits because they're executions of PRIOR decisions, not new decisions
for this minute.

Out of scope for this first pass (Turn 1)
-----------------------------------------
  - Three-execution-scenario reporting (Turn 2).
  - Walk-forward refit cadence (regime classifier reused as-is per call
    via the injected `signal_at` callable).
  - Persistent retry of pending fills past T+1.
  - Mark-to-market sizing (sizer uses cash equity = starting + closed P&L
    − open entry outflows, computed implicitly from realized fills).
  - Concurrent-spread caps (loop accepts whatever the strategy returns).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Iterable

import pandas as pd

from vix_spread.execution.exit_engine import (
    ExitEngine,
    FailedExit,
    OpenPosition,
    SettlementOutcome,
)
from vix_spread.execution.exit_policy import ExitPolicy
from vix_spread.execution.fill_engine import (
    ExecutedFill,
    FillEngine,
    RejectedOrder,
)
from vix_spread.execution.fill_modes import FillMode
from vix_spread.execution.liquidity_gates import LiquidityGates
from vix_spread.products.base import Product

from .results import BacktestResults, CompletedTrade

if TYPE_CHECKING:
    from vix_spread.data.snapshot import VIXSnapshot
    from vix_spread.regime.base import RegimeSignal
    from vix_spread.strategy.strategy import (
        StrategyDecision,
        VIXBullCallSpreadStrategy,
    )


@dataclass(frozen=True)
class _PendingEntry:
    """An enter decision queued for T+1 (or later) execution."""
    decision: "StrategyDecision"
    queued_at: datetime


class WalkForwardBacktest:
    """Per-minute orchestrator for the entry / exit / settlement lifecycle.

    Components are constructor-injected so the backtest engine itself is
    pure orchestration logic (testable with stubs; same wiring used in
    real-data runs).
    """

    def __init__(
        self,
        strategy: "VIXBullCallSpreadStrategy",
        fill_engine: FillEngine,
        exit_engine: ExitEngine,
        gates: LiquidityGates,
        starting_equity: float,
        market_at: Callable[[datetime], "VIXSnapshot | None"],
        signal_at: Callable[[datetime], "RegimeSignal | None"],
        exit_decider: Callable[[OpenPosition, datetime], ExitPolicy | None],
        contract_id_for: Callable[[Product], str] | None = None,
        fill_mode: FillMode = FillMode.SYNTHETIC_BIDASK,
        accept_midpoint_optimism: bool = False,
        slippage_ticks_per_leg: int = 1,
    ) -> None:
        from vix_spread.data.vix_index_options import vix_option_active_contract_id
        if starting_equity <= 0:
            raise ValueError(
                f"starting_equity must be positive; got {starting_equity}."
            )
        self.strategy = strategy
        self.fill_engine = fill_engine
        self.exit_engine = exit_engine
        self.gates = gates
        self.starting_equity = float(starting_equity)
        self.market_at = market_at
        self.signal_at = signal_at
        self.exit_decider = exit_decider
        self.contract_id_for = contract_id_for or vix_option_active_contract_id
        self.fill_mode = fill_mode
        self.accept_midpoint_optimism = accept_midpoint_optimism
        self.slippage_ticks_per_leg = slippage_ticks_per_leg

    def run(self, minute_grid: Iterable[datetime]) -> BacktestResults:
        equity = self.starting_equity
        open_positions: list[OpenPosition] = []
        pending_entries: list[_PendingEntry] = []
        completed_trades: list[CompletedTrade] = []
        rejection_log: list[RejectedOrder | FailedExit] = []
        decisions_log: list[StrategyDecision] = []
        eq_snapshots: list[tuple[datetime, float]] = []

        for T in minute_grid:
            market = self.market_at(T)

            # ---- Step 1: fill pending entries from prior minutes. ----
            # Strict T → T+1+ rule: only attempt if T > queued_at.
            if market is not None and pending_entries:
                still_pending: list[_PendingEntry] = []
                for pe in pending_entries:
                    if T <= pe.queued_at:
                        still_pending.append(pe)
                        continue
                    fill = self._attempt_pending_fill(pe.decision, market)
                    if isinstance(fill, ExecutedFill):
                        equity -= self._fill_cash_outflow(fill)
                        open_positions.append(
                            OpenPosition(
                                spread=fill.spread, size=fill.size,
                                entry_fill=fill,
                            )
                        )
                    else:
                        rejection_log.append(fill)
                pending_entries = still_pending

            # ---- Step 2: evaluate exits BEFORE new decisions. ----
            still_open: list[OpenPosition] = []
            for pos in open_positions:
                policy = self.exit_decider(pos, T)
                if policy is None:
                    still_open.append(pos)
                    continue
                # Force-Tuesday needs market quotes; settlement does not.
                if (
                    policy is ExitPolicy.FORCED_TUESDAY_LIQUIDATION
                    and market is None
                ):
                    still_open.append(pos)
                    continue

                outcome = self.exit_engine.execute_exit(pos, market, policy)
                if isinstance(outcome, ExecutedFill):
                    equity -= self._fill_cash_outflow(outcome)
                    completed_trades.append(
                        CompletedTrade(
                            spread=pos.spread, size=pos.size,
                            entry_fill=pos.entry_fill,  # type: ignore[arg-type]
                            exit_outcome=outcome,
                            pnl=self._tuesday_pnl(pos, outcome),
                        )
                    )
                elif isinstance(outcome, SettlementOutcome):
                    equity += outcome.net_payoff_per_spread * outcome.size
                    completed_trades.append(
                        CompletedTrade(
                            spread=pos.spread, size=pos.size,
                            entry_fill=pos.entry_fill,  # type: ignore[arg-type]
                            exit_outcome=outcome,
                            pnl=self._settlement_pnl(pos, outcome),
                        )
                    )
                elif isinstance(outcome, FailedExit):
                    rejection_log.append(outcome)
                    still_open.append(pos)
                else:
                    raise RuntimeError(
                        f"unknown ExitEngine outcome type: {type(outcome).__name__}"
                    )
            open_positions = still_open

            # ---- Step 3: generate new decision (only with signal + market). ----
            signal = self.signal_at(T)
            if signal is not None and market is not None:
                decision = self.strategy.evaluate(
                    market=market, signal=signal, as_of=T,
                    equity=equity, fill_mode=self.fill_mode,
                    accept_midpoint_optimism=self.accept_midpoint_optimism,
                    slippage_ticks_per_leg=self.slippage_ticks_per_leg,
                )
                decisions_log.append(decision)
                if decision.action == "enter":
                    pending_entries.append(
                        _PendingEntry(decision=decision, queued_at=T)
                    )

            # ---- Step 4: snapshot equity. ----
            eq_snapshots.append((T, equity))

        eq_curve = pd.Series(
            [v for _, v in eq_snapshots],
            index=pd.DatetimeIndex(
                [t for t, _ in eq_snapshots], name="timestamp",
            ),
            name="equity",
        )
        return BacktestResults(
            completed_trades=completed_trades,
            rejection_log=rejection_log,
            open_positions=open_positions,
            decisions_log=decisions_log,
            equity_curve=eq_curve,
            starting_equity=self.starting_equity,
            final_equity=equity,
        )

    # ---------------------------------------------------------------- #
    # Helpers                                                          #
    # ---------------------------------------------------------------- #

    def _attempt_pending_fill(
        self, decision: "StrategyDecision", market: "VIXSnapshot",
    ) -> ExecutedFill | RejectedOrder:
        """Re-fill the queued decision against `market`. Quote lookup is
        done HERE (not on the strategy's preview) so the fill price is
        the actual T+1 NBBO."""
        spread = decision.spread
        assert spread is not None, "enter decisions must carry a spread"
        long_id = self.contract_id_for(spread.long_leg)
        short_id = self.contract_id_for(spread.short_leg)
        long_q = market.options_quotes.get(long_id)
        short_q = market.options_quotes.get(short_id)

        if long_q is None or short_q is None:
            missing = [
                name for name, q in (("long", long_q), ("short", short_q))
                if q is None
            ]
            return RejectedOrder(
                timestamp=market.timestamp, spread=spread,
                reason="gate_fail",
                detail={
                    "sub_reason": "no_quote",
                    "long_id": long_id, "short_id": short_id,
                    "missing_legs": missing,
                },
            )

        return self.fill_engine.attempt_fill(
            spread=spread, long_q=long_q, short_q=short_q,
            order_size=decision.size or 1,
            mode=self.fill_mode, gates=self.gates,
            decision_timestamp=decision.as_of,
            accept_midpoint_optimism=self.accept_midpoint_optimism,
            slippage_ticks_per_leg=self.slippage_ticks_per_leg,
        )

    @staticmethod
    def _fill_cash_outflow(fill: ExecutedFill) -> float:
        """Dollar cash outflow from a fill: `debit × multiplier × size`.

        Positive on entries (debit > 0); negative on close-credit exits
        (debit < 0) — i.e. the close is a cash inflow.
        """
        return (
            fill.debit_per_spread
            * fill.size
            * fill.spread.long_leg.option_multiplier()
        )

    @staticmethod
    def _tuesday_pnl(pos: OpenPosition, exit_fill: ExecutedFill) -> float:
        """P&L for a Tuesday-close trade: entry outflow + exit outflow,
        negated. Exit `debit_per_spread` is negative on the typical
        credit close (per ExitEngine sign convention)."""
        assert pos.entry_fill is not None
        multiplier = pos.spread.long_leg.option_multiplier()
        entry_out = pos.entry_fill.debit_per_spread * pos.size * multiplier
        exit_out = exit_fill.debit_per_spread * exit_fill.size * multiplier
        return -(entry_out + exit_out)

    @staticmethod
    def _settlement_pnl(
        pos: OpenPosition, settlement: SettlementOutcome,
    ) -> float:
        """P&L for a settlement: payoff (already multiplier-scaled) minus
        entry outflow."""
        assert pos.entry_fill is not None
        multiplier = pos.spread.long_leg.option_multiplier()
        entry_out = pos.entry_fill.debit_per_spread * pos.size * multiplier
        return settlement.net_payoff_per_spread * settlement.size - entry_out
