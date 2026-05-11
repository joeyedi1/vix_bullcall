"""Three-execution-scenario reporting (ARCHITECTURE §8.2).

Every backtest run produces results under THREE scenarios — base /
optimistic / stressed — to bracket execution-model risk:

  - **base_case** (`SYNTHETIC_BIDASK`): conservative cross-the-spread
    fills. Headline number; the report MUST lead with this.
  - **optimistic** (`MIDPOINT`): mid-of-NBBO fills. Inflates performance;
    used for sensitivity analysis only. Requires `accept_midpoint_optimism=True`.
  - **stressed** (`SYNTHETIC_PLUS_SLIPPAGE`): synthetic + 1-tick adverse
    slippage per leg. Bounds the realistic downside of microstructure drag.

Cost: ~3× the single-scenario backtest compute. ARCH §12.4 calls this
"the cheapest defense against narrative cherry-picking".

Run-construction pattern
------------------------
The three engines share market_at / signal_at / exit_decider closures
so the heavy data prep (per-minute quote dict, regime broadcast, UXK26
forward) happens ONCE. Only the backtest engine itself is rebuilt per
scenario, with the appropriate fill_mode + opt-in flags.

`run_three_scenarios` accepts a `factory: (FillMode, accept_midpoint,
slippage_ticks) -> WalkForwardBacktest` callable so the runner controls
how the engines are constructed (sharing strategy, fill_engine,
exit_engine etc. across scenarios is the caller's responsibility).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
import pandas as pd

from vix_spread.execution.exit_engine import FailedExit, SettlementOutcome
from vix_spread.execution.fill_engine import ExecutedFill, RejectedOrder
from vix_spread.execution.fill_modes import FillMode

from .results import BacktestResults, CompletedTrade

if TYPE_CHECKING:
    from .walk_forward import WalkForwardBacktest


# --------------------------------------------------------------------------- #
# Metrics + dataframe shaping                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExecutionScenarioMetrics:
    """Headline metrics for one execution scenario.

    Computed from the scenario's equity_curve + completed_trades. The
    metric set here is intentionally thin (Phase-5 first-pass); ARCH §8.3
    diagnostics — Sharpe / Sortino / regime-overlap — wire on top.
    """
    total_pnl: float
    n_trades: int
    n_open_at_end: int
    mean_pnl_per_trade: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    hit_rate: float


@dataclass(frozen=True)
class ExecutionScenarioResult:
    """One execution-scenario slice of a backtest run.

    `trade_log` is a wide DataFrame derived from `BacktestResults.completed_trades`
    so the caller can inspect / serialize without re-walking the dataclasses.
    """
    fill_mode: FillMode
    starting_equity: float
    final_equity: float
    equity_curve: pd.Series
    trade_log: pd.DataFrame
    completed_trades: list[CompletedTrade]
    rejection_log: list[RejectedOrder | FailedExit]
    metrics: ExecutionScenarioMetrics


@dataclass(frozen=True)
class ThreeScenarioResults:
    """The mandated three-scenario bundle (ARCH §8.2)."""
    base_case: ExecutionScenarioResult
    optimistic: ExecutionScenarioResult
    stressed: ExecutionScenarioResult

    def __iter__(self):
        yield self.base_case
        yield self.optimistic
        yield self.stressed


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def trades_to_dataframe(trades: list[CompletedTrade]) -> pd.DataFrame:
    """Convert `CompletedTrade` list → wide trade-log DataFrame.

    Columns: entry_time, exit_time, exit_kind ('tuesday_close' | 'settlement'),
    long_strike, short_strike, expiry, size, entry_debit_per_spread,
    exit_debit_or_payoff, pnl.
    """
    cols = [
        "entry_time", "exit_time", "exit_kind",
        "long_strike", "short_strike", "expiry",
        "size",
        "entry_debit_per_spread",
        "exit_debit_or_payoff",
        "pnl",
    ]
    if not trades:
        return pd.DataFrame(columns=cols)
    rows = []
    for t in trades:
        if isinstance(t.exit_outcome, ExecutedFill):
            kind = "tuesday_close"
            exit_time = t.exit_outcome.timestamp
            exit_payment = t.exit_outcome.debit_per_spread
        elif isinstance(t.exit_outcome, SettlementOutcome):
            kind = "settlement"
            exit_time = t.exit_outcome.timestamp
            exit_payment = t.exit_outcome.net_payoff_per_spread
        else:  # pragma: no cover — should be unreachable
            kind = "unknown"
            exit_time = None
            exit_payment = float("nan")
        rows.append({
            "entry_time": t.entry_fill.timestamp,
            "exit_time": exit_time,
            "exit_kind": kind,
            "long_strike": float(t.spread.long_leg.strike),
            "short_strike": float(t.spread.short_leg.strike),
            "expiry": t.spread.long_leg.expiry,
            "size": int(t.size),
            "entry_debit_per_spread": float(t.entry_fill.debit_per_spread),
            "exit_debit_or_payoff": float(exit_payment),
            "pnl": float(t.pnl),
        })
    return pd.DataFrame(rows, columns=cols)


def _max_drawdown(equity_curve: pd.Series) -> tuple[float, float]:
    """Returns `(max_dd_dollars, max_dd_pct)` from a peak.

    `max_dd_pct` is the worst peak-to-trough drop expressed as a fraction
    of the running peak. Returns (0.0, 0.0) on an empty / all-zero curve.
    """
    if equity_curve.empty:
        return 0.0, 0.0
    peak = equity_curve.cummax()
    dd_dollars = (peak - equity_curve).fillna(0.0)
    max_dd_d = float(dd_dollars.max())
    if max_dd_d <= 0.0:
        return 0.0, 0.0
    # Avoid div-by-zero when peak<=0; mask those rows out.
    safe = peak.where(peak > 0)
    pct_dd = (dd_dollars / safe).fillna(0.0)
    return max_dd_d, float(pct_dd.max())


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[CompletedTrade],
    open_at_end: int,
) -> ExecutionScenarioMetrics:
    pnls = [t.pnl for t in trades]
    total = float(sum(pnls))
    n = len(trades)
    mean = (total / n) if n > 0 else 0.0
    if n > 0:
        wins = sum(1 for p in pnls if p > 0)
        hit = wins / n
    else:
        hit = float("nan")
    dd_d, dd_pct = _max_drawdown(equity_curve)
    return ExecutionScenarioMetrics(
        total_pnl=total,
        n_trades=n,
        n_open_at_end=int(open_at_end),
        mean_pnl_per_trade=mean,
        max_drawdown_dollars=dd_d,
        max_drawdown_pct=dd_pct,
        hit_rate=hit,
    )


def from_backtest_results(
    fill_mode: FillMode, results: BacktestResults,
) -> ExecutionScenarioResult:
    """Wrap a single-mode `BacktestResults` as an `ExecutionScenarioResult`."""
    metrics = compute_metrics(
        results.equity_curve, results.completed_trades,
        open_at_end=len(results.open_positions),
    )
    return ExecutionScenarioResult(
        fill_mode=fill_mode,
        starting_equity=results.starting_equity,
        final_equity=results.final_equity,
        equity_curve=results.equity_curve,
        trade_log=trades_to_dataframe(results.completed_trades),
        completed_trades=results.completed_trades,
        rejection_log=results.rejection_log,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Three-scenario orchestrator                                                  #
# --------------------------------------------------------------------------- #


# Scenario tags + their canonical (fill_mode, accept_midpoint, slippage_ticks)
# tuples. Settled at module level so callers (and tests) can introspect.
SCENARIOS: tuple[tuple[str, FillMode, bool, int], ...] = (
    ("base_case",  FillMode.SYNTHETIC_BIDASK,        False, 1),
    ("optimistic", FillMode.MIDPOINT,                True,  1),
    ("stressed",   FillMode.SYNTHETIC_PLUS_SLIPPAGE, False, 1),
)


def run_three_scenarios(
    factory: Callable[[FillMode, bool, int], "WalkForwardBacktest"],
    minute_grid: Iterable[datetime],
) -> ThreeScenarioResults:
    """Run the backtest three times (once per fill_mode) and bundle.

    `factory(fill_mode, accept_midpoint_optimism, slippage_ticks_per_leg)`
    returns a fresh `WalkForwardBacktest` configured for that scenario.
    The caller is responsible for sharing strategy / fill_engine /
    exit_engine / market_at / signal_at across scenarios so the heavy
    data prep happens once.

    Each scenario runs the FULL minute_grid; total cost ≈ 3× single-scenario.
    """
    scenarios: dict[str, ExecutionScenarioResult] = {}
    grid_list = list(minute_grid)
    for tag, mode, accept_mid, slip in SCENARIOS:
        engine = factory(mode, accept_mid, slip)
        results = engine.run(grid_list)
        scenarios[tag] = from_backtest_results(mode, results)
    return ThreeScenarioResults(
        base_case=scenarios["base_case"],
        optimistic=scenarios["optimistic"],
        stressed=scenarios["stressed"],
    )


# --------------------------------------------------------------------------- #
# Pretty-print helper                                                          #
# --------------------------------------------------------------------------- #


def format_three_scenario_summary(
    bundle: ThreeScenarioResults, currency_fmt: str = "{:>14,.2f}",
) -> str:
    """Side-by-side comparison of base / optimistic / stressed metrics."""
    headers = ("base_case", "optimistic", "stressed")
    cols = [bundle.base_case, bundle.optimistic, bundle.stressed]

    def fmt_money(v: float) -> str:
        return currency_fmt.format(v)

    def fmt_pct(v: float) -> str:
        return f"{v * 100:>13.2f}%"

    def fmt_int(v: int) -> str:
        return f"{v:>14,}"

    def fmt_float(v: float, prec: int = 4) -> str:
        if np.isnan(v):
            return f"{'  n/a':>14}"
        return f"{v:>14.{prec}f}"

    rows = [
        ("starting_equity",           [fmt_money(c.starting_equity) for c in cols]),
        ("final_equity",              [fmt_money(c.final_equity) for c in cols]),
        ("total_pnl",                 [fmt_money(c.metrics.total_pnl) for c in cols]),
        ("n_trades",                  [fmt_int(c.metrics.n_trades) for c in cols]),
        ("n_open_at_end",             [fmt_int(c.metrics.n_open_at_end) for c in cols]),
        ("mean_pnl_per_trade",        [fmt_money(c.metrics.mean_pnl_per_trade) for c in cols]),
        ("max_drawdown_dollars",      [fmt_money(c.metrics.max_drawdown_dollars) for c in cols]),
        ("max_drawdown_pct",          [fmt_pct(c.metrics.max_drawdown_pct) for c in cols]),
        ("hit_rate",                  [fmt_pct(c.metrics.hit_rate) if not np.isnan(c.metrics.hit_rate) else fmt_float(c.metrics.hit_rate) for c in cols]),
        ("rejections_logged",         [fmt_int(len(c.rejection_log)) for c in cols]),
    ]

    lines = []
    lines.append("=" * 78)
    lines.append("THREE-SCENARIO BACKTEST RESULTS (ARCH §8.2)")
    lines.append("=" * 78)
    lines.append(f"{'metric':<26s}" + "".join(f"{h:>16s}" for h in headers))
    lines.append("-" * 78)
    for label, vals in rows:
        lines.append(f"{label:<26s}" + "".join(f"{v:>16s}" for v in vals))
    lines.append("-" * 78)
    lines.append("Headline = base_case (SYNTHETIC_BIDASK). Optimistic and stressed")
    lines.append("are sensitivity-analysis only; never report midpoint as the")
    lines.append("base figure.")
    return "\n".join(lines)
