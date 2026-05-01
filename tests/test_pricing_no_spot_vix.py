from datetime import datetime, timezone

import pytest

from vix_spread.pricing.forward_selector import ForwardSelector
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.utils.errors import ForwardSelectionError


def test_vix_index_option_cannot_select_spot_vix_forward():
    """Validation-memo constraint: spot VIX is FORBIDDEN as a Black-76
    forward input. ForwardSelector must refuse — refusal at the structural
    layer, not by code-review convention."""
    selector = ForwardSelector()
    leg = VIXIndexOption(
        contract_root="VIX",
        expiry=datetime(2026, 6, 17, tzinfo=timezone.utc),
        settlement_event=datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc),
        strike=20.0,
        right="call",
    )
    as_of = datetime(2026, 5, 1, 15, 30, tzinfo=timezone.utc)

    with pytest.raises(ForwardSelectionError):
        selector.select(
            product=leg,
            market=None,
            as_of=as_of,
            source='spot_vix',
        )
