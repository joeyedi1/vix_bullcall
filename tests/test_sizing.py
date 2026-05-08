"""FixedRiskSizer — fixed-percent-of-equity max-loss sizing for bull call spreads."""
from datetime import datetime, timezone

import pytest

from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.strategy.sizing import FixedRiskSizer


def _spread(long_strike: float = 20.0, short_strike: float = 22.0) -> BullCallSpread:
    expiry = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    return BullCallSpread(
        long_leg=VIXIndexOption(
            contract_root="VIX", expiry=expiry, settlement_event=expiry,
            strike=long_strike, right="call",
        ),
        short_leg=VIXIndexOption(
            contract_root="VIX", expiry=expiry, settlement_event=expiry,
            strike=short_strike, right="call",
        ),
    )


# --------------------------------------------------------------------------- #
# Constructor validation                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_pct", [0.0, 1.0, -0.01, 1.5, -1.0])
def test_constructor_rejects_pct_outside_zero_one(bad_pct):
    with pytest.raises(ValueError):
        FixedRiskSizer(risk_per_trade_pct=bad_pct)


# --------------------------------------------------------------------------- #
# Happy path — risk-budget math                                               #
# --------------------------------------------------------------------------- #


def test_sizes_to_floor_of_risk_dollars_over_max_loss():
    """equity 100k, risk 0.5%, debit 1.25 -> risk=$500, max_loss/spread=$125,
    size = floor(500/125) = 4 contracts."""
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    n = sizer.size(spread=_spread(), debit_per_spread=1.25, equity=100_000.0)
    assert n == 4


def test_sizes_floor_correctly_when_division_is_inexact():
    """equity 100k, risk 0.5%, debit 1.30 -> risk=$500, max_loss/spread=$130,
    size = floor(500/130) = 3."""
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    n = sizer.size(spread=_spread(), debit_per_spread=1.30, equity=100_000.0)
    assert n == 3


def test_size_scales_linearly_with_equity():
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    # 4 contracts at 100k -> 8 at 200k -> 12 at 300k (debit unchanged)
    assert sizer.size(spread=_spread(), debit_per_spread=1.25, equity=100_000.0) == 4
    assert sizer.size(spread=_spread(), debit_per_spread=1.25, equity=200_000.0) == 8
    assert sizer.size(spread=_spread(), debit_per_spread=1.25, equity=300_000.0) == 12


def test_size_uses_product_option_multiplier():
    """Multiplier comes from the spread's long leg — VIXIndexOption -> 100,
    so a $0.50 debit costs $50/spread."""
    sizer = FixedRiskSizer(risk_per_trade_pct=0.01)
    # equity 100k, risk 1% = $1000; max_loss/spread = $0.50 * 100 = $50.
    # size = floor(1000 / 50) = 20.
    n = sizer.size(spread=_spread(), debit_per_spread=0.50, equity=100_000.0)
    assert n == 20


# --------------------------------------------------------------------------- #
# Edge cases — return 0                                                        #
# --------------------------------------------------------------------------- #


def test_size_zero_when_debit_exceeds_risk_budget():
    """equity 1000, risk 0.5%, debit 1.00 -> risk=$5, max_loss=$100,
    size = floor(5/100) = 0 (the strategy can't afford even one contract)."""
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    n = sizer.size(spread=_spread(), debit_per_spread=1.00, equity=1_000.0)
    assert n == 0


def test_size_zero_on_non_positive_equity():
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    assert sizer.size(spread=_spread(), debit_per_spread=1.25, equity=0.0) == 0
    assert sizer.size(spread=_spread(), debit_per_spread=1.25, equity=-100.0) == 0


def test_size_zero_on_non_positive_debit():
    """Zero or negative debit (a credit spread) makes max-loss undefined for
    this rule; the sizer refuses rather than producing infinite size."""
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    assert sizer.size(spread=_spread(), debit_per_spread=0.0, equity=100_000.0) == 0
    assert sizer.size(spread=_spread(), debit_per_spread=-1.0, equity=100_000.0) == 0


def test_size_returns_int():
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    n = sizer.size(spread=_spread(), debit_per_spread=1.25, equity=100_000.0)
    assert isinstance(n, int)
    assert not isinstance(n, bool)
