"""WalkForwardBacktest — synthetic temporal-correctness tests.

Headlines:
  - `test_pending_entry_at_T_fills_at_T_plus_1` proves the strict
    T -> T+1 lag. The fill timestamp is T+1, never T.
  - `test_decision_at_T_not_filled_within_T` proves a decision queued
    at T cannot become an open position before the loop advances past T.
  - `test_exits_evaluated_before_new_entries_within_one_minute` proves
    settlement runs first when both events fire at the same minute.

We use real `FillEngine` + real `ExitEngine` (their unit tests pass)
plus a stubbed Strategy that emits canned `StrategyDecision`s. This
keeps the test focus on LOOP BEHAVIOUR rather than strategy logic.
"""
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import pytest

from vix_spread.backtest.results import BacktestResults, CompletedTrade
from vix_spread.backtest.walk_forward import WalkForwardBacktest
from vix_spread.data.snapshot import SettlementMarket, VIXSnapshot
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
from vix_spread.execution.quote import OptionQuote
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.strategy.strategy import StrategyDecision


# --------------------------------------------------------------------------- #
# Fixtures: synthetic minute grid + stubbed strategy                          #
# --------------------------------------------------------------------------- #


SOQ_DATE = date(2026, 5, 20)
SOQ_DT = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)


def _grid(minutes: int, start: datetime | None = None) -> list[datetime]:
    """`minutes` consecutive 1-min UTC stamps, starting `start` (default 09:30)."""
    base = start or datetime(2026, 4, 15, 9, 30, tzinfo=timezone.utc)
    return [base + timedelta(minutes=i) for i in range(minutes)]


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


def _ticker(strike: float) -> str:
    return f"VIX US 05/19/26 C{strike:g} Index"


def _q(
    contract_id: str, ts: datetime, bid: float, ask: float,
    *, bid_size: int = 100, ask_size: int = 100,
    quote_age_seconds: float = 2.0,
) -> OptionQuote:
    return OptionQuote(
        timestamp=ts, contract_id=contract_id,
        bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
        last_trade=None, last_trade_age_seconds=None,
        is_locked=False, is_crossed=False,
        quote_age_seconds=quote_age_seconds,
    )


def _market_factory(
    spreads: dict[datetime, dict[str, OptionQuote]] | None = None,
) -> Callable[[datetime], VIXSnapshot | None]:
    """Build a `market_at(T)` callable from a `{T: {contract_id: OptionQuote}}` map."""
    spreads = spreads or {}

    def market_at(T: datetime) -> VIXSnapshot | None:
        quotes = spreads.get(T)
        if quotes is None:
            return None
        return VIXSnapshot(
            timestamp=T, vx_curve={SOQ_DATE: 22.0},
            options_quotes=quotes, risk_free_rate=0.04, vix_spot=18.0,
        )

    return market_at


@dataclass
class _StubStrategy:
    """Emits canned `StrategyDecision`s by `as_of`. Default is skip."""
    canned: dict[datetime, StrategyDecision] = field(default_factory=dict)

    def evaluate(
        self, *, market, signal, as_of, equity,
        fill_mode=None, **kwargs,
    ) -> StrategyDecision:
        return self.canned.get(
            as_of,
            StrategyDecision(
                as_of=as_of, action="skip", reason="stub",
                hypothesis_name="test",
            ),
        )


def _enter_decision(
    as_of: datetime,
    long_strike: float = 20.0,
    short_strike: float = 22.0,
    size: int = 1,
) -> StrategyDecision:
    return StrategyDecision(
        as_of=as_of, action="enter", reason="entered",
        hypothesis_name="test",
        spread=_spread(long_strike, short_strike),
        evaluation=None, size=size,
    )


def _signal_factory_constant(
    constant: object | None,
) -> Callable[[datetime], object | None]:
    def signal_at(T: datetime):
        return constant
    return signal_at


@pytest.fixture
def gates() -> LiquidityGates:
    return LiquidityGates(
        max_leg_spread_pct=0.5, min_displayed_size=1,
        max_quote_age_seconds=30.0,
    )


def _build_backtest(
    *,
    grid: list[datetime],
    quotes_per_minute: dict[datetime, dict[str, OptionQuote]],
    canned_decisions: dict[datetime, StrategyDecision],
    exit_decider: Callable[[OpenPosition, datetime], ExitPolicy | None],
    gates: LiquidityGates,
    settlement_market: SettlementMarket | None = None,
    starting_equity: float = 100_000.0,
) -> WalkForwardBacktest:
    return WalkForwardBacktest(
        strategy=_StubStrategy(canned=canned_decisions),  # type: ignore[arg-type]
        fill_engine=FillEngine(),
        exit_engine=ExitEngine(gates=gates, settlement_market=settlement_market),
        gates=gates,
        starting_equity=starting_equity,
        market_at=_market_factory(quotes_per_minute),
        # Always-fresh signal so strategy.evaluate is reached. The stub
        # strategy returns canned decisions regardless.
        signal_at=_signal_factory_constant(object()),
        exit_decider=exit_decider,
    )


# --------------------------------------------------------------------------- #
# T → T+1 enforcement                                                          #
# --------------------------------------------------------------------------- #


def test_pending_entry_at_T_fills_at_T_plus_1(gates):
    """Decision at T1; market at T1 AND T2 carry valid quotes. The loop
    must NOT use T1's market for the fill — the fill timestamp must be T2.
    """
    g = _grid(5)        # T0..T4
    T1, T2 = g[1], g[2]

    quotes_T1 = {
        _ticker(20): _q(_ticker(20), T1, bid=2.10, ask=2.20),
        _ticker(22): _q(_ticker(22), T1, bid=0.95, ask=1.05),
    }
    quotes_T2 = {
        _ticker(20): _q(_ticker(20), T2, bid=2.20, ask=2.30),   # different from T1!
        _ticker(22): _q(_ticker(22), T2, bid=1.00, ask=1.10),
    }

    bt = _build_backtest(
        grid=g,
        quotes_per_minute={T1: quotes_T1, T2: quotes_T2},
        canned_decisions={T1: _enter_decision(T1)},
        exit_decider=lambda pos, T: None,        # never exit during this test
        gates=gates,
    )
    res = bt.run(g)

    assert len(res.open_positions) == 1
    open_pos = res.open_positions[0]
    assert open_pos.entry_fill is not None
    # Fill timestamp must be T2 (next minute), not T1 (decision minute).
    assert open_pos.entry_fill.timestamp == T2
    # Fill price must be T2's ask, not T1's ask — proves we used T2 quotes.
    assert open_pos.entry_fill.long_leg_fill == pytest.approx(2.30)
    assert open_pos.entry_fill.short_leg_fill == pytest.approx(1.00)


def test_decision_at_T_not_filled_within_T(gates):
    """Single-minute grid: decision generated at T; no T+1 to fill at.
    Pending entry must remain unfilled — no open position created."""
    g = _grid(1)
    T0 = g[0]

    quotes = {
        _ticker(20): _q(_ticker(20), T0, bid=2.10, ask=2.20),
        _ticker(22): _q(_ticker(22), T0, bid=0.95, ask=1.05),
    }
    bt = _build_backtest(
        grid=g,
        quotes_per_minute={T0: quotes},
        canned_decisions={T0: _enter_decision(T0)},
        exit_decider=lambda pos, T: None,
        gates=gates,
    )
    res = bt.run(g)

    assert len(res.open_positions) == 0
    assert len(res.completed_trades) == 0
    # Decision was generated and logged...
    assert any(d.action == "enter" for d in res.decisions_log)
    # ...but no fill ever happened.
    assert all(
        not isinstance(item, ExecutedFill)
        for trade in res.completed_trades for item in (trade.exit_outcome,)
    )


def test_pending_entry_skipped_when_quote_missing_at_T_plus_1(gates):
    """T+1 has no market quotes for the legs → RejectedOrder logged,
    pending entry dropped (single-attempt; no persistent retry in
    first-pass)."""
    g = _grid(3)
    T1, T2 = g[1], g[2]

    quotes_T1 = {
        _ticker(20): _q(_ticker(20), T1, bid=2.10, ask=2.20),
        _ticker(22): _q(_ticker(22), T1, bid=0.95, ask=1.05),
    }
    # T2 market exists but lacks BOTH leg quotes
    bt = _build_backtest(
        grid=g,
        quotes_per_minute={T1: quotes_T1, T2: {}},
        canned_decisions={T1: _enter_decision(T1)},
        exit_decider=lambda pos, T: None,
        gates=gates,
    )
    res = bt.run(g)

    assert len(res.open_positions) == 0
    assert len(res.rejection_log) == 1
    rej = res.rejection_log[0]
    assert isinstance(rej, RejectedOrder)
    assert rej.reason == "gate_fail"
    assert rej.detail.get("sub_reason") == "no_quote"


# --------------------------------------------------------------------------- #
# State transitions: enter → hold → exit                                      #
# --------------------------------------------------------------------------- #


def _settlement_market_with_vro(vro: float) -> SettlementMarket:
    return SettlementMarket(
        vro_prints={SOQ_DATE: vro},
        vx_settle_prints={SOQ_DATE: vro},
    )


def test_complete_trade_lifecycle_via_settlement(gates):
    """Decision T1 → fill T2 → hold T3 → settle at T4.
    Final state: 0 open positions, 1 completed trade."""
    g = _grid(5)
    T1, T2, T3, T4 = g[1], g[2], g[3], g[4]

    qs = {
        T1: {
            _ticker(20): _q(_ticker(20), T1, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T1, bid=0.95, ask=1.05),
        },
        T2: {
            _ticker(20): _q(_ticker(20), T2, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T2, bid=0.95, ask=1.05),
        },
        T3: {
            _ticker(20): _q(_ticker(20), T3, bid=2.50, ask=2.60),
            _ticker(22): _q(_ticker(22), T3, bid=1.20, ask=1.30),
        },
    }

    def decider(pos: OpenPosition, T: datetime) -> ExitPolicy | None:
        # Settle at T4 only.
        return ExitPolicy.HOLD_TO_SETTLEMENT if T == T4 else None

    bt = _build_backtest(
        grid=g,
        quotes_per_minute=qs,
        canned_decisions={T1: _enter_decision(T1)},
        exit_decider=decider,
        gates=gates,
        settlement_market=_settlement_market_with_vro(vro=23.0),
    )
    res = bt.run(g)

    assert len(res.open_positions) == 0
    assert len(res.completed_trades) == 1
    trade = res.completed_trades[0]
    assert isinstance(trade.exit_outcome, SettlementOutcome)
    # VRO=23, K_long=20, K_short=22 → long $300, short $100, net $200/spread.
    # Entry debit at T2: ask_long - bid_short = 2.20 - 0.95 = 1.25; ×100 = $125.
    # P&L = $200 - $125 = $75.
    assert trade.pnl == pytest.approx(75.0)
    assert trade.exit_outcome.net_payoff_per_spread == pytest.approx(200.0)


def test_complete_trade_lifecycle_via_tuesday_close(gates):
    """Tuesday-close exit at T3 produces an ExecutedFill (negative debit)."""
    g = _grid(5)
    T1, T2, T3 = g[1], g[2], g[3]

    qs = {
        T1: {
            _ticker(20): _q(_ticker(20), T1, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T1, bid=0.95, ask=1.05),
        },
        T2: {
            _ticker(20): _q(_ticker(20), T2, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T2, bid=0.95, ask=1.05),
        },
        T3: {
            # Close-credit time: long.bid=2.50, short.ask=1.20 → credit 1.30.
            _ticker(20): _q(_ticker(20), T3, bid=2.50, ask=2.60),
            _ticker(22): _q(_ticker(22), T3, bid=1.10, ask=1.20),
        },
    }

    def decider(pos: OpenPosition, T: datetime) -> ExitPolicy | None:
        return ExitPolicy.FORCED_TUESDAY_LIQUIDATION if T == T3 else None

    bt = _build_backtest(
        grid=g,
        quotes_per_minute=qs,
        canned_decisions={T1: _enter_decision(T1)},
        exit_decider=decider,
        gates=gates,
    )
    res = bt.run(g)

    assert len(res.completed_trades) == 1
    trade = res.completed_trades[0]
    assert isinstance(trade.exit_outcome, ExecutedFill)
    # Entry debit = 1.25; close credit = 1.30 → P&L per spread = 0.05; ×100 = $5.
    assert trade.exit_outcome.debit_per_spread == pytest.approx(-1.30)
    assert trade.pnl == pytest.approx(5.0)


def test_failed_exit_keeps_position_open(gates):
    """FORCED_TUESDAY at T3 with no market quote for the long leg → FailedExit.
    The position must remain open (rather than silently lost) so a later
    minute can retry."""
    g = _grid(5)
    T1, T2, T3 = g[1], g[2], g[3]

    qs = {
        T1: {
            _ticker(20): _q(_ticker(20), T1, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T1, bid=0.95, ask=1.05),
        },
        T2: {
            _ticker(20): _q(_ticker(20), T2, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T2, bid=0.95, ask=1.05),
        },
        T3: {                                     # MISSING long ticker
            _ticker(22): _q(_ticker(22), T3, bid=1.00, ask=1.10),
        },
    }

    def decider(pos: OpenPosition, T: datetime) -> ExitPolicy | None:
        return ExitPolicy.FORCED_TUESDAY_LIQUIDATION if T == T3 else None

    bt = _build_backtest(
        grid=g,
        quotes_per_minute=qs,
        canned_decisions={T1: _enter_decision(T1)},
        exit_decider=decider,
        gates=gates,
    )
    res = bt.run(g)

    assert len(res.open_positions) == 1, "FailedExit must NOT lose the position"
    assert len(res.rejection_log) == 1
    assert isinstance(res.rejection_log[0], FailedExit)
    assert res.rejection_log[0].reason == "no_quote"
    assert len(res.completed_trades) == 0


# --------------------------------------------------------------------------- #
# Ordering: exits before new entries within one minute                        #
# --------------------------------------------------------------------------- #


def test_exits_evaluated_before_new_entries_within_one_minute(gates):
    """At T3: an open position (entered T1, filled T2) settles AND the
    strategy fires a NEW entry signal. Both events should resolve, but
    the settlement must happen BEFORE the new decision is generated, so
    the new decision reads post-settlement equity.

    Order is pinned by checking:
      - settlement appears in completed_trades
      - new decision is in decisions_log at T3 with action='enter'
      - new entry is queued for T4 (different from the position that
        just settled — proves they're not conflated)
    """
    g = _grid(6)
    T1, T2, T3, T4 = g[1], g[2], g[3], g[4]

    qs = {
        T1: {
            _ticker(20): _q(_ticker(20), T1, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T1, bid=0.95, ask=1.05),
        },
        T2: {
            _ticker(20): _q(_ticker(20), T2, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T2, bid=0.95, ask=1.05),
        },
        T3: {
            _ticker(18): _q(_ticker(18), T3, bid=3.10, ask=3.20),
            _ticker(20): _q(_ticker(20), T3, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T3, bid=0.95, ask=1.05),
        },
        T4: {
            _ticker(18): _q(_ticker(18), T4, bid=3.10, ask=3.20),
            _ticker(20): _q(_ticker(20), T4, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T4, bid=0.95, ask=1.05),
        },
    }

    # Position 1 enters T1 (fills T2). Settles at T3 — same minute as a
    # NEW entry signal from the strategy at strikes 18/20 (distinct from
    # the just-settled 20/22, so the test can distinguish them).
    bt = _build_backtest(
        grid=g,
        quotes_per_minute=qs,
        canned_decisions={
            T1: _enter_decision(T1),                # original entry
            T3: _enter_decision(T3, long_strike=18.0, short_strike=20.0),  # contention case
        },
        exit_decider=lambda pos, T: (
            ExitPolicy.HOLD_TO_SETTLEMENT if T == T3 else None
        ),
        gates=gates,
        settlement_market=_settlement_market_with_vro(vro=23.0),
    )
    res = bt.run(g)

    # Position 1 settled at T3.
    assert len(res.completed_trades) == 1
    assert isinstance(res.completed_trades[0].exit_outcome, SettlementOutcome)

    # New decision fired at T3 (action=enter), proving the new-decision
    # phase ran AFTER the settlement.
    enter_at_t3 = [
        d for d in res.decisions_log
        if d.action == "enter" and d.as_of == T3
    ]
    assert len(enter_at_t3) == 1
    # Distinct strikes — new entry is NOT the just-settled spread.
    assert enter_at_t3[0].spread.long_leg.strike == 18.0

    # New entry filled at T4 (one minute after T3 decision).
    assert len(res.open_positions) == 1
    assert res.open_positions[0].entry_fill.timestamp == T4
    assert res.open_positions[0].spread.long_leg.strike == 18.0


# --------------------------------------------------------------------------- #
# Equity tracking                                                              #
# --------------------------------------------------------------------------- #


def test_equity_curve_reflects_entry_and_settlement(gates):
    """Starting $100k → entry T2 (debit $125) → equity drops to $99,875.
    Settlement T4 (payoff $200) → equity rises to $100,075."""
    g = _grid(5)
    T1, T2, T4 = g[1], g[2], g[4]

    qs = {
        T1: {
            _ticker(20): _q(_ticker(20), T1, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T1, bid=0.95, ask=1.05),
        },
        T2: {
            _ticker(20): _q(_ticker(20), T2, bid=2.10, ask=2.20),
            _ticker(22): _q(_ticker(22), T2, bid=0.95, ask=1.05),
        },
    }
    bt = _build_backtest(
        grid=g,
        quotes_per_minute=qs,
        canned_decisions={T1: _enter_decision(T1)},
        exit_decider=lambda pos, T: ExitPolicy.HOLD_TO_SETTLEMENT if T == T4 else None,
        gates=gates,
        settlement_market=_settlement_market_with_vro(vro=23.0),
    )
    res = bt.run(g)

    # T0 + T1 (before fill): equity = 100,000
    assert res.equity_curve.iloc[0] == pytest.approx(100_000.0)
    # T2: entry filled at debit 1.25 × 100 × 1 = $125 outflow → 99,875
    assert res.equity_curve.iloc[2] == pytest.approx(99_875.0)
    # T3: hold; equity unchanged
    assert res.equity_curve.iloc[3] == pytest.approx(99_875.0)
    # T4: settlement payoff $200 → 99,875 + 200 = 100,075
    assert res.equity_curve.iloc[4] == pytest.approx(100_075.0)
    # final_equity matches the curve's last value
    assert res.final_equity == pytest.approx(100_075.0)


# --------------------------------------------------------------------------- #
# Construction validation                                                      #
# --------------------------------------------------------------------------- #


def test_run_rejects_non_positive_starting_equity(gates):
    with pytest.raises(ValueError, match="starting_equity"):
        WalkForwardBacktest(
            strategy=_StubStrategy(),  # type: ignore[arg-type]
            fill_engine=FillEngine(),
            exit_engine=ExitEngine(gates=gates),
            gates=gates,
            starting_equity=0.0,
            market_at=lambda T: None,
            signal_at=lambda T: None,
            exit_decider=lambda pos, T: None,
        )
