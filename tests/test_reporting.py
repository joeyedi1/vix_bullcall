"""Three-scenario reporting tests.

Headline checks:
  - `compute_metrics` produces correct totals + drawdown + hit rate
  - `trades_to_dataframe` renders the wide schema correctly
  - `run_three_scenarios` produces three results with three different
    fill_modes, and (with a deterministic synthetic backtest) the
    optimistic midpoint scenario produces a less-negative debit than
    the synthetic-bidask base case.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from vix_spread.backtest.reporting import (
    SCENARIOS,
    ExecutionScenarioMetrics,
    ExecutionScenarioResult,
    ThreeScenarioResults,
    compute_metrics,
    format_three_scenario_summary,
    from_backtest_results,
    run_three_scenarios,
    trades_to_dataframe,
)
from vix_spread.backtest.results import BacktestResults, CompletedTrade
from vix_spread.backtest.walk_forward import WalkForwardBacktest
from vix_spread.data.snapshot import VIXSnapshot
from vix_spread.execution.exit_engine import (
    ExitEngine,
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
from vix_spread.execution.quote import OptionQuote
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.strategy.strategy import StrategyDecision


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


SOQ_DT = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)


def _spread(long_strike: float = 20.0, short_strike: float = 22.0) -> BullCallSpread:
    return BullCallSpread(
        long_leg=VIXIndexOption(
            contract_root="VIX", expiry=SOQ_DT, settlement_event=SOQ_DT,
            strike=long_strike, right="call",
        ),
        short_leg=VIXIndexOption(
            contract_root="VIX", expiry=SOQ_DT, settlement_event=SOQ_DT,
            strike=short_strike, right="call",
        ),
    )


def _entry_fill(
    timestamp: datetime, debit: float = 1.25, size: int = 1,
) -> ExecutedFill:
    return ExecutedFill(
        timestamp=timestamp, spread=_spread(), debit_per_spread=debit,
        size=size, fill_mode=FillMode.SYNTHETIC_BIDASK,
        long_leg_fill=2.20, short_leg_fill=0.95,
        tick_rounded=False, fees_per_spread=0.0,
    )


def _exit_fill(
    timestamp: datetime, exit_debit: float = -1.30, size: int = 1,
) -> ExecutedFill:
    return ExecutedFill(
        timestamp=timestamp, spread=_spread(), debit_per_spread=exit_debit,
        size=size, fill_mode=FillMode.SYNTHETIC_BIDASK,
        long_leg_fill=2.50, short_leg_fill=1.20,
        tick_rounded=False, fees_per_spread=0.0,
    )


def _settlement(
    payoff_per_spread: float, vro: float = 23.0, size: int = 1,
) -> SettlementOutcome:
    return SettlementOutcome(
        timestamp=SOQ_DT, spread=_spread(), size=size,
        long_leg_payoff=300.0, short_leg_payoff=100.0,
        net_payoff_per_spread=payoff_per_spread, vro_print=vro,
    )


def _completed_trade(
    pnl: float, exit_kind: str = "tuesday_close",
) -> CompletedTrade:
    entry = _entry_fill(datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc))
    if exit_kind == "tuesday_close":
        exit_o = _exit_fill(datetime(2026, 4, 14, 19, 0, tzinfo=timezone.utc))
    else:
        exit_o = _settlement(payoff_per_spread=200.0)
    return CompletedTrade(
        spread=entry.spread, size=entry.size, entry_fill=entry,
        exit_outcome=exit_o, pnl=pnl,
    )


# --------------------------------------------------------------------------- #
# compute_metrics                                                             #
# --------------------------------------------------------------------------- #


def test_compute_metrics_basic():
    trades = [
        _completed_trade(pnl=100.0),
        _completed_trade(pnl=-30.0),
        _completed_trade(pnl=50.0),
        _completed_trade(pnl=-20.0),
    ]
    eq = pd.Series(
        [100_000.0, 100_100.0, 100_070.0, 100_120.0, 100_100.0],
        index=pd.date_range("2026-04-01", periods=5, freq="D", tz="UTC"),
    )
    m = compute_metrics(eq, trades, open_at_end=0)
    assert m.total_pnl == pytest.approx(100.0)
    assert m.n_trades == 4
    assert m.mean_pnl_per_trade == pytest.approx(25.0)
    # 2 winners (100, 50) of 4 trades → 0.5
    assert m.hit_rate == pytest.approx(0.5)
    # Peak-to-trough: peak=100,100 at T1 → trough=100,070 at T2 → dd=30.
    # The later (100,120 → 100,100) drop is only 20 — smaller, so not the max.
    assert m.max_drawdown_dollars == pytest.approx(30.0)
    assert m.max_drawdown_pct == pytest.approx(30.0 / 100_100.0)


def test_compute_metrics_empty_trades():
    eq = pd.Series([100_000.0], index=pd.DatetimeIndex(["2026-04-01"], tz="UTC"))
    m = compute_metrics(eq, trades=[], open_at_end=0)
    assert m.total_pnl == 0.0
    assert m.n_trades == 0
    assert m.mean_pnl_per_trade == 0.0
    assert np.isnan(m.hit_rate)
    assert m.max_drawdown_dollars == 0.0
    assert m.max_drawdown_pct == 0.0


def test_compute_metrics_drawdown_on_falling_curve():
    """Steady decline: drawdown is the total drop from start to end."""
    eq = pd.Series(
        [100_000.0, 95_000.0, 90_000.0, 85_000.0, 80_000.0],
        index=pd.date_range("2026-04-01", periods=5, freq="D", tz="UTC"),
    )
    m = compute_metrics(eq, trades=[_completed_trade(pnl=-20_000.0)], open_at_end=0)
    assert m.max_drawdown_dollars == pytest.approx(20_000.0)
    assert m.max_drawdown_pct == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
# trades_to_dataframe                                                         #
# --------------------------------------------------------------------------- #


def test_trades_to_dataframe_schema():
    trades = [
        _completed_trade(pnl=100.0, exit_kind="tuesday_close"),
        _completed_trade(pnl=200.0, exit_kind="settlement"),
    ]
    df = trades_to_dataframe(trades)
    expected_cols = {
        "entry_time", "exit_time", "exit_kind",
        "long_strike", "short_strike", "expiry",
        "size",
        "entry_debit_per_spread", "exit_debit_or_payoff", "pnl",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == 2
    assert df.iloc[0]["exit_kind"] == "tuesday_close"
    assert df.iloc[1]["exit_kind"] == "settlement"
    assert df.iloc[0]["long_strike"] == 20.0
    assert df.iloc[0]["short_strike"] == 22.0
    assert df.iloc[0]["pnl"] == pytest.approx(100.0)


def test_trades_to_dataframe_empty():
    df = trades_to_dataframe([])
    assert len(df) == 0
    # Schema still present for downstream concatenation.
    assert "pnl" in df.columns
    assert "entry_time" in df.columns


def test_trades_to_dataframe_settlement_uses_net_payoff_for_exit():
    trade = _completed_trade(pnl=200.0, exit_kind="settlement")
    df = trades_to_dataframe([trade])
    # SettlementOutcome.net_payoff_per_spread = 200.0 in the fixture
    assert df.iloc[0]["exit_debit_or_payoff"] == pytest.approx(200.0)


# --------------------------------------------------------------------------- #
# from_backtest_results                                                        #
# --------------------------------------------------------------------------- #


def test_from_backtest_results_propagates_fields():
    trades = [_completed_trade(pnl=50.0)]
    eq = pd.Series([100_000.0, 100_050.0], index=pd.date_range(
        "2026-04-01", periods=2, freq="D", tz="UTC",
    ))
    rejection = RejectedOrder(
        timestamp=datetime(2026, 4, 1, tzinfo=timezone.utc),
        spread=_spread(), reason="stale_quote", detail={},
    )
    backtest = BacktestResults(
        completed_trades=trades, rejection_log=[rejection],
        open_positions=[], decisions_log=[], equity_curve=eq,
        starting_equity=100_000.0, final_equity=100_050.0,
    )
    scenario = from_backtest_results(FillMode.SYNTHETIC_BIDASK, backtest)
    assert scenario.fill_mode is FillMode.SYNTHETIC_BIDASK
    assert scenario.starting_equity == 100_000.0
    assert scenario.final_equity == 100_050.0
    assert len(scenario.completed_trades) == 1
    assert len(scenario.trade_log) == 1
    assert scenario.rejection_log == [rejection]
    assert scenario.metrics.total_pnl == 50.0


# --------------------------------------------------------------------------- #
# run_three_scenarios                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class _CannedStrategy:
    """Stub strategy that always emits an enter at the first minute and
    skip thereafter — keeps the three-scenario test deterministic."""
    canned: dict[datetime, StrategyDecision] = field(default_factory=dict)

    def evaluate(self, *, market, signal, as_of, equity, fill_mode=None, **kwargs):
        return self.canned.get(
            as_of,
            StrategyDecision(
                as_of=as_of, action="skip", reason="stub",
                hypothesis_name="test",
            ),
        )


def _make_engine_factory(
    *,
    grid: list[datetime],
    quotes_per_minute: dict[datetime, dict[str, OptionQuote]],
    canned_decisions: dict[datetime, StrategyDecision],
    gates: LiquidityGates,
    starting_equity: float = 100_000.0,
) -> Callable[[FillMode, bool, int], WalkForwardBacktest]:
    """Returns a factory that builds an engine per scenario, sharing the
    quote / signal / exit_decider closures."""
    def market_at(T):
        return (
            VIXSnapshot(
                timestamp=T, vx_curve={}, options_quotes=quotes_per_minute[T],
                risk_free_rate=0.04,
            )
            if T in quotes_per_minute else None
        )

    def signal_at(T):
        return object()  # any non-None placeholder; stub strategy ignores

    def exit_decider(pos, T):
        return None

    def factory(mode, accept_mid, slip):
        return WalkForwardBacktest(
            strategy=_CannedStrategy(canned=canned_decisions),  # type: ignore[arg-type]
            fill_engine=FillEngine(),
            exit_engine=ExitEngine(gates=gates),
            gates=gates,
            starting_equity=starting_equity,
            market_at=market_at,
            signal_at=signal_at,
            exit_decider=exit_decider,
            fill_mode=mode,
            accept_midpoint_optimism=accept_mid,
            slippage_ticks_per_leg=slip,
        )

    return factory


def test_run_three_scenarios_produces_three_results_with_distinct_fill_modes():
    """Smoke: run_three_scenarios returns base/optimistic/stressed each
    with the right fill_mode tag. The actual debit math differs across
    scenarios (verified in the next test)."""
    base = datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc)
    grid = [base + timedelta(minutes=i) for i in range(3)]
    T1, T2 = grid[1], grid[2]

    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"

    def _q(ts, cid, bid, ask, age=2.0):
        return OptionQuote(
            timestamp=ts, contract_id=cid, bid=bid, ask=ask,
            bid_size=100, ask_size=100,
            last_trade=None, last_trade_age_seconds=None,
            is_locked=False, is_crossed=False,
            quote_age_seconds=age,
        )

    # June 17 SOQ → ticker date June 16 (Tuesday)
    soq = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    spread = BullCallSpread(
        long_leg=VIXIndexOption(
            contract_root="VIX", expiry=soq, settlement_event=soq,
            strike=20.0, right="call",
        ),
        short_leg=VIXIndexOption(
            contract_root="VIX", expiry=soq, settlement_event=soq,
            strike=22.0, right="call",
        ),
    )
    quotes = {
        T1: {
            long_id: _q(T1, long_id, bid=2.10, ask=2.20),
            short_id: _q(T1, short_id, bid=0.95, ask=1.05),
        },
        T2: {
            long_id: _q(T2, long_id, bid=2.10, ask=2.20),
            short_id: _q(T2, short_id, bid=0.95, ask=1.05),
        },
    }
    canned = {
        T1: StrategyDecision(
            as_of=T1, action="enter", reason="entered",
            hypothesis_name="test", spread=spread, size=1,
        )
    }
    gates = LiquidityGates(
        max_leg_spread_pct=0.5, min_displayed_size=1,
        max_quote_age_seconds=30.0,
    )
    factory = _make_engine_factory(
        grid=grid, quotes_per_minute=quotes,
        canned_decisions=canned, gates=gates,
    )
    bundle = run_three_scenarios(factory, grid)

    assert isinstance(bundle, ThreeScenarioResults)
    assert bundle.base_case.fill_mode is FillMode.SYNTHETIC_BIDASK
    assert bundle.optimistic.fill_mode is FillMode.MIDPOINT
    assert bundle.stressed.fill_mode is FillMode.SYNTHETIC_PLUS_SLIPPAGE


def test_three_scenario_debits_increase_from_optimistic_to_stressed():
    """For the same trade with the same NBBO:
       MIDPOINT debit       = 1.15  (mid 2.15 - mid 1.00)
       SYNTHETIC_BIDASK     = 1.25  (long.ask 2.20 - short.bid 0.95)
       SYNTHETIC_PLUS_SLIP  = 1.35  (synthetic + 2 * 0.05 slippage)

    Pre-condition: the entry fills successfully under all three modes.
    """
    base = datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc)
    grid = [base + timedelta(minutes=i) for i in range(3)]
    T1, T2 = grid[1], grid[2]
    long_id = "VIX US 06/16/26 C20 Index"
    short_id = "VIX US 06/16/26 C22 Index"
    soq = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    spread = BullCallSpread(
        long_leg=VIXIndexOption(
            contract_root="VIX", expiry=soq, settlement_event=soq,
            strike=20.0, right="call",
        ),
        short_leg=VIXIndexOption(
            contract_root="VIX", expiry=soq, settlement_event=soq,
            strike=22.0, right="call",
        ),
    )

    def _q(ts, cid, bid, ask):
        return OptionQuote(
            timestamp=ts, contract_id=cid, bid=bid, ask=ask,
            bid_size=100, ask_size=100,
            last_trade=None, last_trade_age_seconds=None,
            is_locked=False, is_crossed=False, quote_age_seconds=2.0,
        )

    quotes = {
        T1: {
            long_id: _q(T1, long_id, 2.10, 2.20),
            short_id: _q(T1, short_id, 0.95, 1.05),
        },
        T2: {
            long_id: _q(T2, long_id, 2.10, 2.20),
            short_id: _q(T2, short_id, 0.95, 1.05),
        },
    }
    canned = {
        T1: StrategyDecision(
            as_of=T1, action="enter", reason="entered",
            hypothesis_name="test", spread=spread, size=1,
        )
    }
    gates = LiquidityGates(
        max_leg_spread_pct=0.5, min_displayed_size=1,
        max_quote_age_seconds=30.0,
    )
    factory = _make_engine_factory(
        grid=grid, quotes_per_minute=quotes,
        canned_decisions=canned, gates=gates,
    )
    bundle = run_three_scenarios(factory, grid)

    # Each scenario opens one position at T2; we read the entry debit
    # from the open_positions's entry_fill.
    base_pos = bundle.base_case.completed_trades or []
    base_entry = (
        bundle.base_case.completed_trades[0].entry_fill
        if bundle.base_case.completed_trades
        else None
    )
    # No exit -> position open at end. Read open positions in the
    # underlying BacktestResults via metrics.n_open_at_end.
    assert bundle.base_case.metrics.n_open_at_end == 1
    assert bundle.optimistic.metrics.n_open_at_end == 1
    assert bundle.stressed.metrics.n_open_at_end == 1
    # Equity reflects the entry debit × multiplier × size for each scenario.
    base_eq = bundle.base_case.final_equity
    opt_eq = bundle.optimistic.final_equity
    stressed_eq = bundle.stressed.final_equity
    # Base debit = 1.25 × 100 = 125 outflow → equity 99,875
    # Optimistic = 1.15 × 100 = 115 outflow → equity 99,885
    # Stressed = 1.35 × 100 = 135 outflow → equity 99,865
    assert base_eq == pytest.approx(99_875.0)
    assert opt_eq == pytest.approx(99_885.0)
    assert stressed_eq == pytest.approx(99_865.0)
    # Ordering: optimistic > base > stressed
    assert opt_eq > base_eq > stressed_eq


def test_three_scenarios_iterable():
    """ThreeScenarioResults is iterable in (base, optimistic, stressed)
    order — useful for tabular reporting."""
    eq = pd.Series([100_000.0], index=pd.DatetimeIndex(["2026-04-01"], tz="UTC"))
    backtest = BacktestResults(
        completed_trades=[], rejection_log=[], open_positions=[],
        decisions_log=[], equity_curve=eq,
        starting_equity=100_000.0, final_equity=100_000.0,
    )
    bundle = ThreeScenarioResults(
        base_case=from_backtest_results(FillMode.SYNTHETIC_BIDASK, backtest),
        optimistic=from_backtest_results(FillMode.MIDPOINT, backtest),
        stressed=from_backtest_results(FillMode.SYNTHETIC_PLUS_SLIPPAGE, backtest),
    )
    out = list(bundle)
    assert [s.fill_mode for s in out] == [
        FillMode.SYNTHETIC_BIDASK,
        FillMode.MIDPOINT,
        FillMode.SYNTHETIC_PLUS_SLIPPAGE,
    ]


# --------------------------------------------------------------------------- #
# SCENARIOS table + summary formatter                                          #
# --------------------------------------------------------------------------- #


def test_scenarios_table_specifies_canonical_modes():
    """The module-level SCENARIOS table is the single source of truth for
    (tag, fill_mode, accept_midpoint_optimism, slippage_ticks)."""
    tags = [s[0] for s in SCENARIOS]
    modes = [s[1] for s in SCENARIOS]
    accept_mid = [s[2] for s in SCENARIOS]
    assert tags == ["base_case", "optimistic", "stressed"]
    assert modes == [
        FillMode.SYNTHETIC_BIDASK,
        FillMode.MIDPOINT,
        FillMode.SYNTHETIC_PLUS_SLIPPAGE,
    ]
    # Optimistic MUST have accept_midpoint_optimism=True (the explicit opt-in).
    assert accept_mid == [False, True, False]


def test_format_three_scenario_summary_renders():
    """Smoke: the text formatter produces a string with the headlines
    and one line per metric. Doesn't pin exact layout (would be brittle)."""
    eq = pd.Series(
        [100_000.0, 99_900.0],
        index=pd.date_range("2026-04-01", periods=2, freq="D", tz="UTC"),
    )
    backtest = BacktestResults(
        completed_trades=[_completed_trade(pnl=-100.0)],
        rejection_log=[], open_positions=[], decisions_log=[],
        equity_curve=eq, starting_equity=100_000.0, final_equity=99_900.0,
    )
    bundle = ThreeScenarioResults(
        base_case=from_backtest_results(FillMode.SYNTHETIC_BIDASK, backtest),
        optimistic=from_backtest_results(FillMode.MIDPOINT, backtest),
        stressed=from_backtest_results(FillMode.SYNTHETIC_PLUS_SLIPPAGE, backtest),
    )
    text = format_three_scenario_summary(bundle)
    assert "THREE-SCENARIO BACKTEST RESULTS" in text
    assert "base_case" in text
    assert "optimistic" in text
    assert "stressed" in text
    assert "total_pnl" in text
    assert "max_drawdown" in text
    assert "hit_rate" in text
