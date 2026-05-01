from datetime import datetime, timezone

import pytest

from vix_spread.execution.fill_engine import FillEngine
from vix_spread.execution.fill_modes import FillMode
from vix_spread.pricing.forward_selector import Forward
from vix_spread.pricing.theoretical import TheoreticalPrice
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


def _make_theoretical() -> TheoreticalPrice:
    settlement = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    forward = Forward(
        value=21.5,
        selection_method='settlement_date_match',
        model_risk_flag=False,
        settlement_date=settlement,
    )
    return TheoreticalPrice(
        value=1.50,
        delta=0.42,
        gamma=0.05,
        vega=0.18,
        theta=-0.02,
        forward_used=forward,
        iv_used=0.65,
        T_minutes=30000.0,
    )


def test_attempt_fill_rejects_theoretical_price():
    """Validation-memo constraint: TheoreticalPrice (is_executable=False) is
    NEVER a fill. Passing it where an OptionQuote is expected must raise
    TypeError at the entry of attempt_fill — at the type/structural layer,
    not by code-review convention."""
    engine = FillEngine()
    spread = _make_spread()
    theo = _make_theoretical()

    with pytest.raises(TypeError):
        engine.attempt_fill(
            spread=spread,
            long_q=theo,            # forbidden — TheoreticalPrice, not OptionQuote
            short_q=theo,
            order_size=1,
            mode=FillMode.SYNTHETIC_BIDASK,
            decision_timestamp=datetime(2026, 5, 15, 15, 30, tzinfo=timezone.utc),
        )
