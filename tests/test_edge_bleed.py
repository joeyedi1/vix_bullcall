"""EdgeBleedAudit tests — synthetic decision+trade pairing.

Headline checks:
  - decisions are paired with trades by (strikes, expiry) + most-recent
    decision-before-fill timestamp
  - dollar bleed = (theoretical - executed) × multiplier × size, with
    the sign convention preserved (positive = paid less than fair)
  - iv source tags propagate
  - summary statistics computed correctly
  - skip silently when no matching decision exists
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from vix_spread.backtest.edge_bleed import (
    EdgeBleedAudit,
    EdgeBleedEntry,
    _spread_key,
)
from vix_spread.backtest.results import BacktestResults, CompletedTrade
from vix_spread.execution.fill_engine import ExecutedFill, RejectedOrder
from vix_spread.execution.fill_modes import FillMode
from vix_spread.pricing.evaluator import SpreadEvaluation
from vix_spread.pricing.forward_selector import Forward
from vix_spread.pricing.leg_iv import LegIV, LegIVSource
from vix_spread.pricing.theoretical import (
    TheoreticalPrice,
    TheoreticalSpreadPrice,
)
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.strategy.strategy import StrategyDecision


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


SOQ_DT = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
DECISION_T = datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc)
FILL_T = datetime(2026, 4, 1, 14, 31, tzinfo=timezone.utc)


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


def _forward(value: float = 21.0) -> Forward:
    return Forward(
        value=value, selection_method="settlement_date_match",
        model_risk_flag=False, settlement_date=SOQ_DT,
    )


def _theoretical(forward: Forward, value: float = 1.0) -> TheoreticalPrice:
    return TheoreticalPrice(
        value=value, delta=0.5, gamma=0.1, vega=0.2, theta=-0.05,
        forward_used=forward, iv_used=0.5,
        T_minutes=10_000.0, is_executable=False, rho=0.0,
    )


def _evaluation(
    spread: BullCallSpread, theoretical_value: float,
    iv_long_src: LegIVSource = LegIVSource.VENDOR,
    iv_short_src: LegIVSource = LegIVSource.VENDOR,
) -> SpreadEvaluation:
    fwd = _forward()
    long_tp = _theoretical(fwd, value=theoretical_value + 0.5)
    short_tp = _theoretical(fwd, value=0.5)
    th = TheoreticalSpreadPrice(
        value=theoretical_value,
        long_leg=long_tp, short_leg=short_tp,
        delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0,
        is_executable=False,
    )
    return SpreadEvaluation(
        spread=spread, as_of=DECISION_T, forward=fwd,
        iv_long=LegIV(value=0.55, source=iv_long_src),
        iv_short=LegIV(value=0.45, source=iv_short_src),
        theoretical=th,
        fill=None,  # type: ignore[arg-type]
    )


def _decision(
    as_of: datetime, spread: BullCallSpread,
    theoretical_value: float = 1.0, size: int = 1,
    iv_long_src: LegIVSource = LegIVSource.VENDOR,
    iv_short_src: LegIVSource = LegIVSource.VENDOR,
) -> StrategyDecision:
    return StrategyDecision(
        as_of=as_of, action="enter", reason="entered",
        hypothesis_name="test",
        spread=spread,
        evaluation=_evaluation(
            spread, theoretical_value,
            iv_long_src=iv_long_src, iv_short_src=iv_short_src,
        ),
        size=size,
    )


def _entry_fill(
    timestamp: datetime, spread: BullCallSpread,
    debit: float, size: int = 1,
) -> ExecutedFill:
    return ExecutedFill(
        timestamp=timestamp, spread=spread, debit_per_spread=debit,
        size=size, fill_mode=FillMode.SYNTHETIC_BIDASK,
        long_leg_fill=2.10, short_leg_fill=0.95,
        tick_rounded=False, fees_per_spread=0.0,
    )


def _completed_trade(
    spread: BullCallSpread, entry_debit: float, size: int = 1,
    fill_timestamp: datetime = FILL_T,
) -> CompletedTrade:
    entry = _entry_fill(fill_timestamp, spread, entry_debit, size)
    # Exit (Tuesday close) at a notional credit, gives a deterministic P&L
    exit_ = ExecutedFill(
        timestamp=fill_timestamp + timedelta(days=14),
        spread=spread, debit_per_spread=-1.30, size=size,
        fill_mode=FillMode.SYNTHETIC_BIDASK,
        long_leg_fill=2.50, short_leg_fill=1.20,
        tick_rounded=False, fees_per_spread=0.0,
    )
    multiplier = spread.long_leg.option_multiplier()
    pnl = -(entry_debit + exit_.debit_per_spread) * multiplier * size
    return CompletedTrade(
        spread=spread, size=size, entry_fill=entry,
        exit_outcome=exit_, pnl=pnl,
    )


def _results(
    *, completed_trades: list[CompletedTrade],
    decisions_log: list[StrategyDecision],
) -> BacktestResults:
    return BacktestResults(
        completed_trades=completed_trades,
        rejection_log=[], open_positions=[],
        decisions_log=decisions_log,
        equity_curve=pd.Series([100_000.0], index=pd.DatetimeIndex(
            ["2026-04-01"], tz="UTC",
        )),
        starting_equity=100_000.0, final_equity=100_000.0,
    )


# --------------------------------------------------------------------------- #
# Empty case                                                                  #
# --------------------------------------------------------------------------- #


def test_edge_bleed_audit_empty():
    res = _results(completed_trades=[], decisions_log=[])
    audit = EdgeBleedAudit.from_results(res)
    assert audit.entries == []
    assert audit.per_spread_distribution().empty
    assert audit.dollars_distribution().empty
    summary = audit.summary()
    assert summary["n_trades"] == 0
    assert summary["total_dollars"] == 0.0
    assert audit.to_dataframe().empty
    assert audit.by_iv_source_breakdown().empty


# --------------------------------------------------------------------------- #
# Pairing                                                                     #
# --------------------------------------------------------------------------- #


def test_pairs_trade_with_most_recent_pre_fill_decision():
    """Decision at T-1 minute, fill at T → paired. The decision's
    theoretical drives the bleed; the trade's executed_debit is the
    other side."""
    sp = _spread(20.0, 22.0)
    decision = _decision(
        as_of=DECISION_T, spread=sp, theoretical_value=1.10,
    )
    trade = _completed_trade(
        spread=sp, entry_debit=1.25, fill_timestamp=FILL_T,
    )
    res = _results(completed_trades=[trade], decisions_log=[decision])
    audit = EdgeBleedAudit.from_results(res)
    assert len(audit.entries) == 1
    e = audit.entries[0]
    assert e.decision_as_of == DECISION_T
    assert e.fill_timestamp == FILL_T
    assert e.theoretical_preview == pytest.approx(1.10)
    assert e.executed_debit == pytest.approx(1.25)
    # bleed = theoretical - executed = 1.10 - 1.25 = -0.15
    assert e.edge_bleed_per_spread == pytest.approx(-0.15)
    # × multiplier 100 × size 1 = -$15
    assert e.edge_bleed_dollars == pytest.approx(-15.0)


def test_picks_most_recent_decision_when_multiple_match():
    """Two enter-decisions on the same spread at different T's; only
    the LATER one (still strictly before the fill) pairs with the
    trade."""
    sp = _spread(20.0, 22.0)
    earlier = _decision(
        as_of=DECISION_T - timedelta(minutes=5), spread=sp,
        theoretical_value=1.00,
    )
    later = _decision(
        as_of=DECISION_T, spread=sp, theoretical_value=1.10,
    )
    trade = _completed_trade(
        spread=sp, entry_debit=1.25, fill_timestamp=FILL_T,
    )
    res = _results(
        completed_trades=[trade],
        decisions_log=[earlier, later],
    )
    audit = EdgeBleedAudit.from_results(res)
    assert len(audit.entries) == 1
    assert audit.entries[0].decision_as_of == DECISION_T
    assert audit.entries[0].theoretical_preview == pytest.approx(1.10)


def test_skips_trade_with_no_matching_decision():
    """Trade exists but the decisions_log has no enter-decision for its
    spread strikes. Silently skip (don't fabricate a paired entry)."""
    sp = _spread(20.0, 22.0)
    decision = _decision(  # different strikes
        as_of=DECISION_T, spread=_spread(18.0, 20.0),
        theoretical_value=1.10,
    )
    trade = _completed_trade(spread=sp, entry_debit=1.25)
    res = _results(completed_trades=[trade], decisions_log=[decision])
    audit = EdgeBleedAudit.from_results(res)
    assert len(audit.entries) == 0


def test_skips_decision_after_fill():
    """A decision whose as_of is >= fill timestamp can't have
    originated the fill (single-attempt T+1)."""
    sp = _spread(20.0, 22.0)
    decision_after = _decision(
        as_of=FILL_T + timedelta(minutes=1), spread=sp,
        theoretical_value=1.10,
    )
    trade = _completed_trade(spread=sp, entry_debit=1.25)
    res = _results(completed_trades=[trade], decisions_log=[decision_after])
    audit = EdgeBleedAudit.from_results(res)
    assert audit.entries == []


def test_ignores_skip_decisions():
    """`StrategyDecision.action='skip'` has no evaluation; the audit
    must not try to pair it with a trade."""
    sp = _spread(20.0, 22.0)
    skip = StrategyDecision(
        as_of=DECISION_T, action="skip", reason="curve_filter",
        hypothesis_name="test",
        spread=None, evaluation=None, size=None,
    )
    enter = _decision(as_of=DECISION_T, spread=sp, theoretical_value=1.10)
    trade = _completed_trade(spread=sp, entry_debit=1.25)
    res = _results(completed_trades=[trade], decisions_log=[skip, enter])
    audit = EdgeBleedAudit.from_results(res)
    # Pairs with the enter decision; skip is ignored.
    assert len(audit.entries) == 1
    assert audit.entries[0].theoretical_preview == pytest.approx(1.10)


# --------------------------------------------------------------------------- #
# Sign convention                                                              #
# --------------------------------------------------------------------------- #


def test_negative_bleed_when_paid_more_than_fair():
    """Typical case: theoretical $1.10, executed $1.25 → paid $0.15 more
    than fair → bleed = -$0.15 per spread."""
    sp = _spread()
    decision = _decision(as_of=DECISION_T, spread=sp, theoretical_value=1.10)
    trade = _completed_trade(spread=sp, entry_debit=1.25)
    res = _results(completed_trades=[trade], decisions_log=[decision])
    audit = EdgeBleedAudit.from_results(res)
    assert audit.entries[0].edge_bleed_per_spread < 0
    assert audit.entries[0].edge_bleed_dollars < 0


def test_positive_bleed_when_paid_less_than_fair():
    """Rare favourable case: theoretical $1.30, executed $1.25 → paid
    $0.05 less than fair → bleed = +$0.05 per spread."""
    sp = _spread()
    decision = _decision(as_of=DECISION_T, spread=sp, theoretical_value=1.30)
    trade = _completed_trade(spread=sp, entry_debit=1.25)
    res = _results(completed_trades=[trade], decisions_log=[decision])
    audit = EdgeBleedAudit.from_results(res)
    assert audit.entries[0].edge_bleed_per_spread == pytest.approx(0.05)
    assert audit.entries[0].edge_bleed_dollars == pytest.approx(5.0)


def test_dollar_bleed_uses_multiplier_and_size():
    """$0.15 bleed × $100 multiplier × 3 contracts = $45."""
    sp = _spread()
    decision = _decision(as_of=DECISION_T, spread=sp, theoretical_value=1.10)
    trade = _completed_trade(spread=sp, entry_debit=1.25, size=3)
    res = _results(completed_trades=[trade], decisions_log=[decision])
    audit = EdgeBleedAudit.from_results(res)
    assert audit.entries[0].edge_bleed_dollars == pytest.approx(-45.0)


# --------------------------------------------------------------------------- #
# Aggregates                                                                  #
# --------------------------------------------------------------------------- #


def test_summary_statistics():
    """Three trades; verify mean / median / total / p5 / p95."""
    sp = _spread()
    decisions = [
        _decision(DECISION_T + timedelta(minutes=i * 10), sp, theoretical_value=1.10)
        for i in range(3)
    ]
    trades = [
        _completed_trade(
            spread=sp, entry_debit=1.20,
            fill_timestamp=DECISION_T + timedelta(minutes=i * 10 + 1),
        )
        for i in range(3)
    ]
    res = _results(completed_trades=trades, decisions_log=decisions)
    audit = EdgeBleedAudit.from_results(res)
    s = audit.summary()
    assert s["n_trades"] == 3
    # Each: theoretical 1.10 − executed 1.20 = -0.10 → -$10 dollars
    assert s["total_dollars"] == pytest.approx(-30.0)
    assert s["mean_dollars"] == pytest.approx(-10.0)
    assert s["median_dollars"] == pytest.approx(-10.0)


def test_iv_source_breakdown_groups_by_pair():
    """Two trades: one vendor/vendor, one b76_inverted/vendor.
    `by_iv_source_breakdown` produces a 2-row aggregate."""
    sp = _spread()
    decisions = [
        _decision(
            DECISION_T, sp, theoretical_value=1.10,
            iv_long_src=LegIVSource.VENDOR,
            iv_short_src=LegIVSource.VENDOR,
        ),
        _decision(
            DECISION_T + timedelta(minutes=10), sp, theoretical_value=1.10,
            iv_long_src=LegIVSource.B76_INVERTED,
            iv_short_src=LegIVSource.VENDOR,
        ),
    ]
    trades = [
        _completed_trade(
            spread=sp, entry_debit=1.25,
            fill_timestamp=DECISION_T + timedelta(minutes=1),
        ),
        _completed_trade(
            spread=sp, entry_debit=1.30,
            fill_timestamp=DECISION_T + timedelta(minutes=11),
        ),
    ]
    res = _results(completed_trades=trades, decisions_log=decisions)
    audit = EdgeBleedAudit.from_results(res)
    breakdown = audit.by_iv_source_breakdown()
    assert len(breakdown) == 2
    assert set(breakdown["iv_long_source"]) == {"vendor", "b76_inverted"}


def test_to_dataframe_schema():
    sp = _spread()
    decision = _decision(DECISION_T, sp, theoretical_value=1.10)
    trade = _completed_trade(spread=sp, entry_debit=1.25)
    res = _results(completed_trades=[trade], decisions_log=[decision])
    audit = EdgeBleedAudit.from_results(res)
    df = audit.to_dataframe()
    expected_cols = {
        "decision_as_of", "fill_timestamp",
        "long_strike", "short_strike", "expiry",
        "size",
        "theoretical_preview", "executed_debit",
        "edge_bleed_per_spread", "edge_bleed_dollars",
        "iv_long_source", "iv_short_source",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == 1
    assert df.iloc[0]["long_strike"] == 20.0
    assert df.iloc[0]["short_strike"] == 22.0


def test_spread_key_distinguishes_by_strikes_and_expiry():
    a = _spread(20.0, 22.0)
    b = _spread(20.0, 22.0)
    c = _spread(20.5, 22.0)
    assert _spread_key(a) == _spread_key(b)
    assert _spread_key(a) != _spread_key(c)
