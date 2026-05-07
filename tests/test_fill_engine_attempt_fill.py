"""FillEngine.attempt_fill — happy-path execution + each rejection class.

Companion to the existing structural guard tests
(`test_fill_engine_no_midpoint_default.py`, `test_fill_engine_rejects_theoretical.py`).
This file exercises the body's correctness contract:

  - SYNTHETIC_BIDASK debit = long.ask - short.bid
  - MIDPOINT debit       = mid(long) - mid(short)  (with explicit opt-in flag)
  - Each LiquidityGate violation produces a typed RejectedOrder with the
    correct reason and detail.
  - Rejection priority is observed.
"""
from datetime import datetime, timezone

import pytest

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


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_spread(long_strike: float = 20.0, short_strike: float = 22.0) -> BullCallSpread:
    expiry = datetime(2026, 6, 17, tzinfo=timezone.utc)
    settle = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    long_leg = VIXIndexOption(
        contract_root="VIX", expiry=expiry, settlement_event=settle,
        strike=long_strike, right="call",
    )
    short_leg = VIXIndexOption(
        contract_root="VIX", expiry=expiry, settlement_event=settle,
        strike=short_strike, right="call",
    )
    return BullCallSpread(long_leg=long_leg, short_leg=short_leg)


def _q(
    contract_id: str = "VIX_X",
    bid: float = 1.40,
    ask: float = 1.50,
    bid_size: int = 100,
    ask_size: int = 100,
    *,
    is_locked: bool = False,
    is_crossed: bool = False,
    quote_age_seconds: float = 2.0,
) -> OptionQuote:
    return OptionQuote(
        timestamp=datetime(2026, 5, 1, 15, 31, tzinfo=timezone.utc),
        contract_id=contract_id,
        bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
        last_trade=None, last_trade_age_seconds=None,
        is_locked=is_locked, is_crossed=is_crossed,
        quote_age_seconds=quote_age_seconds,
    )


@pytest.fixture
def engine() -> FillEngine:
    return FillEngine()


@pytest.fixture
def spread() -> BullCallSpread:
    return _make_spread()


@pytest.fixture
def gates() -> LiquidityGates:
    """Default-permissive gates for happy-path tests; individual tests
    override what they want to violate."""
    return LiquidityGates(
        max_leg_spread_pct=0.5,
        min_displayed_size=1,
        max_quote_age_seconds=30.0,
        reject_locked_or_crossed=True,
        reject_no_bid_short_leg=True,
        max_order_size_pct_of_displayed=0.5,
    )


# --------------------------------------------------------------------------- #
# Happy path — debit math                                                     #
# --------------------------------------------------------------------------- #


def test_synthetic_bidask_debit_is_long_ask_minus_short_bid(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, ExecutedFill)
    # long.ask - short.bid = 2.20 - 0.95 = 1.25
    assert out.debit_per_spread == pytest.approx(1.25)
    assert out.long_leg_fill == pytest.approx(2.20)
    assert out.short_leg_fill == pytest.approx(0.95)
    assert out.fill_mode is FillMode.SYNTHETIC_BIDASK
    assert out.tick_rounded is False
    assert out.fees_per_spread == 0.0
    assert out.size == 1


def test_midpoint_debit_is_mid_minus_mid(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)   # mid 2.15
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)   # mid 1.00
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.MIDPOINT, gates=gates,
        accept_midpoint_optimism=True,
    )
    assert isinstance(out, ExecutedFill)
    assert out.debit_per_spread == pytest.approx(1.15)  # 2.15 - 1.00
    assert out.long_leg_fill == pytest.approx(2.15)
    assert out.short_leg_fill == pytest.approx(1.00)
    assert out.fill_mode is FillMode.MIDPOINT


def test_synthetic_plus_slippage_not_yet_wired(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    with pytest.raises(NotImplementedError, match="SYNTHETIC_PLUS_SLIPPAGE"):
        engine.attempt_fill(
            spread=spread, long_q=long_q, short_q=short_q,
            order_size=1, mode=FillMode.SYNTHETIC_PLUS_SLIPPAGE, gates=gates,
        )


def test_executed_fill_size_propagates(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20, ask_size=20)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05, bid_size=20)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=5, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, ExecutedFill)
    assert out.size == 5


# --------------------------------------------------------------------------- #
# Rejection contract                                                          #
# --------------------------------------------------------------------------- #


def test_rejects_no_bid_short_leg(engine, spread, gates):
    """short.bid <= 0 → RejectedOrder reason='no_bid_short' (highest priority)."""
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=0.0, ask=1.05)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "no_bid_short"
    assert out.detail == {"short_bid": 0.0}


def test_rejects_locked_quote(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.20, ask=2.20, is_locked=True)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "locked"


def test_rejects_crossed_quote(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=1.10, ask=1.05, is_crossed=True)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "crossed"


def test_rejects_stale_quote_long_leg(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20, quote_age_seconds=120.0)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05, quote_age_seconds=2.0)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "stale_quote"
    assert out.detail["leg"] == "long"
    assert out.detail["age_s"] == 120.0


def test_rejects_stale_quote_short_leg(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20, quote_age_seconds=2.0)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05, quote_age_seconds=120.0)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "stale_quote"
    assert out.detail["leg"] == "short"


def test_rejects_min_displayed_size(engine, spread):
    """min_displayed_size violation -> gate_fail with sub_reason."""
    gates_thin = LiquidityGates(
        max_leg_spread_pct=0.5, min_displayed_size=20,
        max_quote_age_seconds=30.0,
    )
    long_q = _q("VIX_C20", bid=2.10, ask=2.20, bid_size=5, ask_size=5)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05, bid_size=50, ask_size=50)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates_thin,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "gate_fail"
    assert out.detail["sub_reason"] == "min_displayed_size"


def test_rejects_max_leg_spread_pct(engine, spread):
    """leg with bid 1.00 / ask 2.00: spread_pct = 1.0/1.5 ≈ 0.67 > 0.15."""
    gates_tight = LiquidityGates(
        max_leg_spread_pct=0.15, min_displayed_size=1,
        max_quote_age_seconds=30.0,
    )
    long_q = _q("VIX_C20", bid=1.00, ask=2.00)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates_tight,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "gate_fail"
    assert out.detail["sub_reason"] == "leg_spread_pct"
    assert out.detail["leg"] == "long"


def test_rejects_order_size_pct_of_displayed(engine, spread):
    """order 60 vs ask_size 100 = 60% > max 50%."""
    gates_size = LiquidityGates(
        max_leg_spread_pct=0.5, min_displayed_size=1,
        max_quote_age_seconds=30.0,
        max_order_size_pct_of_displayed=0.5,
    )
    long_q = _q("VIX_C20", bid=2.10, ask=2.20, ask_size=100)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05, bid_size=100)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=60, mode=FillMode.SYNTHETIC_BIDASK, gates=gates_size,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "gate_fail"
    assert out.detail["sub_reason"] == "order_size_pct_of_displayed"


# --------------------------------------------------------------------------- #
# Rejection priority — earliest violation wins                                #
# --------------------------------------------------------------------------- #


def test_rejection_priority_no_bid_short_beats_locked(engine, spread, gates):
    """Both no_bid_short AND locked apply; no_bid_short wins (priority 1)."""
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=0.0, ask=0.0, is_locked=True)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "no_bid_short"


def test_rejection_priority_locked_beats_stale(engine, spread, gates):
    """Both locked AND stale apply; locked wins (priority 2 vs 3)."""
    long_q = _q("VIX_C20", bid=2.20, ask=2.20, is_locked=True,
                quote_age_seconds=200.0)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert isinstance(out, RejectedOrder)
    assert out.reason == "locked"


# --------------------------------------------------------------------------- #
# Other contract checks                                                       #
# --------------------------------------------------------------------------- #


def test_requires_explicit_gates(engine, spread):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    with pytest.raises(ValueError, match="gates"):
        engine.attempt_fill(
            spread=spread, long_q=long_q, short_q=short_q,
            order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=None,
        )


def test_rejects_zero_or_negative_order_size(engine, spread, gates):
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    for bad_size in (0, -1):
        with pytest.raises(ValueError, match="order_size"):
            engine.attempt_fill(
                spread=spread, long_q=long_q, short_q=short_q,
                order_size=bad_size, mode=FillMode.SYNTHETIC_BIDASK,
                gates=gates,
            )


def test_returned_timestamp_is_long_quote_timestamp(engine, spread, gates):
    """Caller-supplied matched-timestamp legs; engine surfaces long_q.timestamp
    on the result as the canonical fill timestamp."""
    long_q = _q("VIX_C20", bid=2.10, ask=2.20)
    short_q = _q("VIX_C22", bid=0.95, ask=1.05)
    out = engine.attempt_fill(
        spread=spread, long_q=long_q, short_q=short_q,
        order_size=1, mode=FillMode.SYNTHETIC_BIDASK, gates=gates,
    )
    assert out.timestamp == long_q.timestamp
