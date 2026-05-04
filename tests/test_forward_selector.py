"""Tests for ForwardSelector (ARCHITECTURE §4.1).

Three contracts:
  1. settlement_date_match: returns Forward with the VX-curve price for
     the option's settlement-event date and model_risk_flag=False.
  2. PCP and interpolated branches raise NotImplementedError (NOT
     ForwardSelectionError) — they are not-yet-wired, not forbidden.
     spot_vix is the only permanent refusal (kept tested in
     test_pricing_no_spot_vix.py).
  3. Missing/non-positive VX prices raise ForwardSelectionError loudly
     rather than producing a NaN forward that silently breaks pricing.
"""
from datetime import date, datetime, timezone

import pytest

from vix_spread.data.snapshot import OptionsMarketSnapshot
from vix_spread.pricing.forward_selector import Forward, ForwardSelector
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.utils.errors import ForwardSelectionError


_SETTLEMENT = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
_AS_OF = datetime(2026, 5, 1, 15, 30, tzinfo=timezone.utc)


def _option(strike: float = 20.0) -> VIXIndexOption:
    return VIXIndexOption(
        contract_root="VIX",
        expiry=_SETTLEMENT,
        settlement_event=_SETTLEMENT,
        strike=strike,
        right='call',
    )


def _market(price: float = 21.5) -> OptionsMarketSnapshot:
    return OptionsMarketSnapshot(
        timestamp=_AS_OF,
        vx_curve={_SETTLEMENT.date(): price},
    )


# --------------------------------------------------------------------- #
# settlement_date_match (the wired branch)                              #
# --------------------------------------------------------------------- #


def test_settlement_date_match_returns_curve_price():
    selector = ForwardSelector()
    fwd = selector.select(
        product=_option(), market=_market(price=21.5), as_of=_AS_OF,
    )
    assert isinstance(fwd, Forward)
    assert fwd.value == pytest.approx(21.5)
    assert fwd.selection_method == 'settlement_date_match'
    assert fwd.model_risk_flag is False
    assert fwd.settlement_date == _SETTLEMENT


def test_settlement_date_match_default_source():
    """Default `source` argument is settlement_date_match — the preferred
    branch in the §4.1 hierarchy."""
    selector = ForwardSelector()
    # Same call without explicit source kwarg.
    fwd = selector.select(product=_option(), market=_market(), as_of=_AS_OF)
    assert fwd.selection_method == 'settlement_date_match'


def test_settlement_date_match_missing_curve_entry_raises():
    selector = ForwardSelector()
    market = OptionsMarketSnapshot(timestamp=_AS_OF, vx_curve={})
    with pytest.raises(ForwardSelectionError, match='No VX future'):
        selector.select(product=_option(), market=market, as_of=_AS_OF)


def test_settlement_date_match_non_positive_price_raises():
    selector = ForwardSelector()
    market = _market(price=0.0)
    with pytest.raises(ForwardSelectionError, match='non-positive'):
        selector.select(product=_option(), market=market, as_of=_AS_OF)


def test_settlement_date_match_market_without_vx_curve_raises():
    """A market object lacking the vx_curve attribute fails fast rather
    than producing AttributeError deep in the selector body."""
    class BadMarket:
        timestamp = _AS_OF
    selector = ForwardSelector()
    with pytest.raises(ForwardSelectionError, match='vx_curve'):
        selector.select(product=_option(), market=BadMarket(), as_of=_AS_OF)


# --------------------------------------------------------------------- #
# Not-yet-wired branches                                                #
# --------------------------------------------------------------------- #


def test_put_call_parity_not_implemented():
    selector = ForwardSelector()
    with pytest.raises(NotImplementedError, match='PCP'):
        selector.select(
            product=_option(), market=_market(), as_of=_AS_OF,
            source='put_call_parity',
        )


def test_interpolated_not_implemented():
    selector = ForwardSelector()
    with pytest.raises(NotImplementedError, match='interpolation'):
        selector.select(
            product=_option(), market=_market(), as_of=_AS_OF,
            source='interpolated',
        )
