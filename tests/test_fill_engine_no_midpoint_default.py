import inspect
from datetime import datetime, timezone

import pytest

from vix_spread.execution.fill_engine import FillEngine
from vix_spread.execution.fill_modes import FillMode
from vix_spread.execution.quote import OptionQuote
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption


def _make_spread() -> BullCallSpread:
    expiry = datetime(2026, 6, 17, tzinfo=timezone.utc)
    settlement = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    long_leg = VIXIndexOption(
        contract_root="VIX",
        expiry=expiry,
        settlement_event=settlement,
        strike=20.0,
        right="call",
    )
    short_leg = VIXIndexOption(
        contract_root="VIX",
        expiry=expiry,
        settlement_event=settlement,
        strike=22.0,
        right="call",
    )
    return BullCallSpread(long_leg=long_leg, short_leg=short_leg)


def _make_quote(contract_id: str, bid: float, ask: float) -> OptionQuote:
    return OptionQuote(
        timestamp=datetime(2026, 5, 15, 15, 31, tzinfo=timezone.utc),
        contract_id=contract_id,
        bid=bid,
        ask=ask,
        bid_size=10,
        ask_size=10,
        last_trade=None,
        last_trade_age_seconds=None,
        is_locked=False,
        is_crossed=False,
        quote_age_seconds=2.0,
    )


def test_synthetic_bidask_is_the_default_mode():
    """Validation-memo constraint: SYNTHETIC_BIDASK is the headline base case.
    The signature default for `mode` MUST be SYNTHETIC_BIDASK so that callers
    who omit the kwarg cannot accidentally land on the optimistic midpoint."""
    sig = inspect.signature(FillEngine.attempt_fill)
    assert sig.parameters['mode'].default is FillMode.SYNTHETIC_BIDASK


def test_midpoint_without_explicit_flag_raises():
    """MIDPOINT is the optimistic sensitivity scenario, not the base case.
    Selecting it without `accept_midpoint_optimism=True` must raise — a
    structural opt-in, not a documentation note."""
    engine = FillEngine()
    spread = _make_spread()
    long_q = _make_quote("VIX_C20", bid=2.10, ask=2.20)
    short_q = _make_quote("VIX_C22", bid=0.95, ask=1.05)

    with pytest.raises(ValueError):
        engine.attempt_fill(
            spread=spread,
            long_q=long_q,
            short_q=short_q,
            order_size=1,
            mode=FillMode.MIDPOINT,
            decision_timestamp=datetime(2026, 5, 15, 15, 30, tzinfo=timezone.utc),
        )
