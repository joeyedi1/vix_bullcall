from datetime import datetime, timezone

import pytest

from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.products.vx_future_option import VXFutureOption


def test_mixing_product_types_raises_type_error():
    """Validation-memo constraint: VIXIndexOption and VXFutureOption MUST NOT
    be combined into a single spread. The check fires at __post_init__ —
    the type system, not a code-review convention, blocks the mistake."""
    expiry = datetime(2026, 6, 17, tzinfo=timezone.utc)
    settlement = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)

    vix_idx_leg = VIXIndexOption(
        contract_root="VIX",
        expiry=expiry,
        settlement_event=settlement,
        strike=20.0,
        right="call",
    )
    vx_fut_leg = VXFutureOption(
        contract_root="VX",
        expiry=expiry,
        settlement_event=settlement,
        strike=22.0,
        right="call",
        deliverable_vx=None,
    )

    with pytest.raises(TypeError):
        BullCallSpread(long_leg=vix_idx_leg, short_leg=vx_fut_leg)
