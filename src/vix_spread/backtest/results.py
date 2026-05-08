"""BacktestResults — Phase-5 first-pass results bundle.

ARCHITECTURE §8.2 specifies a richer three-execution-scenario bundle
(`base_case`, `optimistic`, `stressed` plus rejection log + regime
audit + forward-selection audit + held-to-settlement sensitivity).
This first-pass scaffold carries the ledger fields the loop produces;
Turn 2 will wire the three-scenario report on top.

Fields here are minimal and observable from one walk-forward run:

  - `completed_trades`  — every closed spread with realized P&L
  - `rejection_log`     — RejectedOrder + FailedExit entries
  - `open_positions`    — positions still open at the end of the run
  - `decisions_log`     — every StrategyDecision (audit)
  - `equity_curve`      — pd.Series of equity per minute
  - `starting_equity`, `final_equity` — bookends
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from vix_spread.execution.exit_engine import (
        FailedExit,
        OpenPosition,
        SettlementOutcome,
    )
    from vix_spread.execution.fill_engine import ExecutedFill, RejectedOrder
    from vix_spread.products.spread import BullCallSpread
    from vix_spread.strategy.strategy import StrategyDecision


@dataclass(frozen=True)
class CompletedTrade:
    """One closed spread trade — entry + exit pairing with realized P&L.

    P&L convention (dollars):
      - Entry outflow = `entry_fill.debit_per_spread × multiplier × size`
      - For Tuesday close exit: cash inflow = `-exit.debit_per_spread × multiplier × size`
        (exit.debit is NEGATIVE on credit close — see ExitEngine docstring)
        → `pnl = -(entry_fill.debit + exit.debit) × multiplier × size`
      - For settlement: `pnl = exit.net_payoff_per_spread × size − entry_outflow`
        (`net_payoff_per_spread` is already multiplier-scaled by Product.settlement_value)
    """
    spread: "BullCallSpread"
    size: int
    entry_fill: "ExecutedFill"
    exit_outcome: "ExecutedFill | SettlementOutcome"
    pnl: float


@dataclass(frozen=True)
class BacktestResults:
    """First-pass results bundle (Phase-5)."""
    completed_trades: list[CompletedTrade]
    rejection_log: list["RejectedOrder | FailedExit"]
    open_positions: list["OpenPosition"]
    decisions_log: list["StrategyDecision"]
    equity_curve: pd.Series
    starting_equity: float
    final_equity: float
