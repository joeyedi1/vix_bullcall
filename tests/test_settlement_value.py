"""Tests for Product.settlement_value (ARCHITECTURE §2.1 / §12.7).

Two products, two payoff conventions:
  * VIXIndexOption: cash payoff = max(VRO − K, 0) · 100  (calls)
                                  max(K − VRO, 0) · 100  (puts).
    VRO is the actual SOQ print — never spot VIX, never theoretical.
  * VXFutureOption: cash equivalent of the resulting VX position =
    max(VX_settle − K, 0) · 1000   (calls)
    max(K − VX_settle, 0) · 1000   (puts).
"""
from datetime import date, datetime, timezone

import pytest

from vix_spread.data.snapshot import SettlementMarket
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.products.vx_future import VXFuture
from vix_spread.products.vx_future_option import VXFutureOption


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


_EXPIRY = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
_SETTLE_DATE = date(2026, 6, 17)


def _market(vro: float = 22.5, vx_settle: float = 21.0) -> SettlementMarket:
    return SettlementMarket(
        vro_prints={_SETTLE_DATE: vro},
        vx_settle_prints={_SETTLE_DATE: vx_settle},
    )


def _vix_call(strike: float) -> VIXIndexOption:
    return VIXIndexOption(
        contract_root="VIX",
        expiry=_EXPIRY,
        settlement_event=_EXPIRY,
        strike=strike,
        right='call',
    )


def _vix_put(strike: float) -> VIXIndexOption:
    return VIXIndexOption(
        contract_root="VIX",
        expiry=_EXPIRY,
        settlement_event=_EXPIRY,
        strike=strike,
        right='put',
    )


def _vx_call(strike: float) -> VXFutureOption:
    return VXFutureOption(
        contract_root="VX",
        expiry=_EXPIRY,
        settlement_event=_EXPIRY,
        strike=strike,
        right='call',
        deliverable_vx=VXFuture(contract_root="VX", settlement_date=_SETTLE_DATE),
    )


def _vx_put(strike: float) -> VXFutureOption:
    return VXFutureOption(
        contract_root="VX",
        expiry=_EXPIRY,
        settlement_event=_EXPIRY,
        strike=strike,
        right='put',
        deliverable_vx=VXFuture(contract_root="VX", settlement_date=_SETTLE_DATE),
    )


# --------------------------------------------------------------------- #
# VIXIndexOption                                                        #
# --------------------------------------------------------------------- #


def test_vix_index_call_itm():
    """VRO=22.5, K=20.0 → payoff = (22.5 − 20.0) · 100 = 250."""
    market = _market(vro=22.5)
    assert _vix_call(20.0).settlement_value(market) == pytest.approx(250.0)


def test_vix_index_call_otm_pays_zero():
    market = _market(vro=18.0)
    assert _vix_call(20.0).settlement_value(market) == 0.0


def test_vix_index_put_itm():
    """VRO=18.0, K=20.0 → payoff = (20.0 − 18.0) · 100 = 200."""
    market = _market(vro=18.0)
    assert _vix_put(20.0).settlement_value(market) == pytest.approx(200.0)


def test_vix_index_put_otm_pays_zero():
    market = _market(vro=22.0)
    assert _vix_put(20.0).settlement_value(market) == 0.0


def test_vix_index_missing_vro_raises():
    market = SettlementMarket(vro_prints={}, vx_settle_prints={})
    with pytest.raises(KeyError):
        _vix_call(20.0).settlement_value(market)


# --------------------------------------------------------------------- #
# VXFutureOption                                                        #
# --------------------------------------------------------------------- #


def test_vx_future_call_itm():
    """VX_settle=21.0, K=20.0 → payoff = (21.0 − 20.0) · 1000 = 1000."""
    market = _market(vx_settle=21.0)
    assert _vx_call(20.0).settlement_value(market) == pytest.approx(1000.0)


def test_vx_future_call_otm_pays_zero():
    market = _market(vx_settle=19.5)
    assert _vx_call(20.0).settlement_value(market) == 0.0


def test_vx_future_put_itm():
    """VX_settle=18.0, K=20.0 → payoff = (20.0 − 18.0) · 1000 = 2000."""
    market = _market(vx_settle=18.0)
    assert _vx_put(20.0).settlement_value(market) == pytest.approx(2000.0)


def test_vx_future_put_otm_pays_zero():
    market = _market(vx_settle=22.0)
    assert _vx_put(20.0).settlement_value(market) == 0.0


def test_vx_future_uses_deliverable_settlement_date_not_option_expiry():
    """The deliverable VX's settlement_date — not the option's expiry —
    is the lookup key. Demonstrate by giving the deliverable a DIFFERENT
    settlement_date and verifying the lookup follows it."""
    other_date = date(2026, 7, 22)
    deliverable = VXFuture(contract_root="VX", settlement_date=other_date)
    opt = VXFutureOption(
        contract_root="VX",
        expiry=_EXPIRY,
        settlement_event=_EXPIRY,
        strike=20.0,
        right='call',
        deliverable_vx=deliverable,
    )
    market = SettlementMarket(
        vro_prints={},
        vx_settle_prints={other_date: 25.0, _SETTLE_DATE: 999.0},
    )
    # Should use other_date's price (25.0), not _SETTLE_DATE's (999.0).
    assert opt.settlement_value(market) == pytest.approx(5_000.0)


# --------------------------------------------------------------------- #
# VXFutureOption construction guard                                     #
# --------------------------------------------------------------------- #


def test_vx_future_option_rejects_non_vxfuture_deliverable():
    """deliverable_vx is REQUIRED to be a VXFuture instance — passing
    anything else (including None) raises at construction time. This is
    the structural guard against the validation-memo product-conflation
    defect."""
    with pytest.raises(TypeError, match="VXFuture"):
        VXFutureOption(
            contract_root="VX",
            expiry=_EXPIRY,
            settlement_event=_EXPIRY,
            strike=20.0,
            right='call',
            deliverable_vx=None,  # type: ignore[arg-type]
        )
