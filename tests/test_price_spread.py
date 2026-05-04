"""Tests for Black76Pricer.price_spread (ARCHITECTURE §4.4 / validation memo).

Three contracts:
  1. Net debit and aggregated Greeks equal long − short.
  2. FlatVolError fires when iv_long == iv_short — the validation-memo
     defect signature of substituting VVIX for strike-specific IVs.
  3. Returned object is TheoreticalSpreadPrice with is_executable=False.
"""
from datetime import datetime, timedelta, timezone

import pytest

from vix_spread.pricing.black76 import Black76Pricer
from vix_spread.pricing.forward_selector import Forward
from vix_spread.pricing.theoretical import TheoreticalSpreadPrice
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.utils.errors import FlatVolError


_SETTLEMENT = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
_AS_OF = _SETTLEMENT - timedelta(days=30)


def _leg(strike: float, right: str = 'call') -> VIXIndexOption:
    return VIXIndexOption(
        contract_root="VIX",
        expiry=_SETTLEMENT,
        settlement_event=_SETTLEMENT,
        strike=strike,
        right=right,
    )


def _spread(long_K: float = 20.0, short_K: float = 22.0) -> BullCallSpread:
    return BullCallSpread(long_leg=_leg(long_K), short_leg=_leg(short_K))


def _forward(F: float = 21.0) -> Forward:
    return Forward(
        value=F,
        selection_method='settlement_date_match',
        model_risk_flag=False,
        settlement_date=_SETTLEMENT,
    )


# --------------------------------------------------------------------- #
# 1. Spread arithmetic = long − short                                    #
# --------------------------------------------------------------------- #


def test_spread_value_equals_long_minus_short():
    pricer = Black76Pricer()
    spread = _spread(long_K=20.0, short_K=22.0)
    fwd = _forward(F=21.0)

    sp = pricer.price_spread(
        spread, fwd, iv_long=0.65, iv_short=0.60,
        as_of=_AS_OF, risk_free_rate=0.05,
    )

    long_p = pricer.price(spread.long_leg, fwd, leg_iv=0.65,
                          as_of=_AS_OF, risk_free_rate=0.05)
    short_p = pricer.price(spread.short_leg, fwd, leg_iv=0.60,
                           as_of=_AS_OF, risk_free_rate=0.05)

    assert sp.value == pytest.approx(long_p.value - short_p.value, rel=1e-12)
    assert sp.delta == pytest.approx(long_p.delta - short_p.delta, rel=1e-12)
    assert sp.gamma == pytest.approx(long_p.gamma - short_p.gamma, rel=1e-12)
    assert sp.vega == pytest.approx(long_p.vega - short_p.vega, rel=1e-12)
    assert sp.theta == pytest.approx(long_p.theta - short_p.theta, rel=1e-12)
    assert sp.rho == pytest.approx(long_p.rho - short_p.rho, rel=1e-12)


def test_spread_value_is_positive_for_bull_call_in_expectation():
    """For a bull call spread (buy lower K, sell higher K) with reasonable
    inputs, the long leg value > short leg value, so net debit > 0."""
    pricer = Black76Pricer()
    sp = pricer.price_spread(
        _spread(long_K=20.0, short_K=22.0),
        _forward(F=21.0),
        iv_long=0.65, iv_short=0.60,
        as_of=_AS_OF, risk_free_rate=0.05,
    )
    assert sp.value > 0


def test_spread_carries_per_leg_breakdown():
    """The spread object retains both leg theoreticals for audit /
    edge-bleed analysis."""
    pricer = Black76Pricer()
    sp = pricer.price_spread(
        _spread(long_K=20.0, short_K=22.0),
        _forward(F=21.0),
        iv_long=0.65, iv_short=0.60,
        as_of=_AS_OF, risk_free_rate=0.05,
    )
    assert sp.long_leg.iv_used == pytest.approx(0.65)
    assert sp.short_leg.iv_used == pytest.approx(0.60)
    assert sp.long_leg.is_executable is False
    assert sp.short_leg.is_executable is False


# --------------------------------------------------------------------- #
# 2. FlatVolError                                                        #
# --------------------------------------------------------------------- #


def test_flat_vol_raises_flatvolerror():
    """The validation-memo defect: passing the same IV to both legs of a
    non-zero-width spread (the signature of substituting VVIX for
    strike-specific IVs)."""
    pricer = Black76Pricer()
    with pytest.raises(FlatVolError, match='iv_long'):
        pricer.price_spread(
            _spread(long_K=20.0, short_K=22.0),
            _forward(F=21.0),
            iv_long=0.65, iv_short=0.65,    # ← copy-paste defect
            as_of=_AS_OF, risk_free_rate=0.05,
        )


def test_close_but_distinct_vols_pass():
    """The flat-vol check is strict equality. Two close-but-different
    vols from a real chain must NOT raise — only bit-identical inputs
    fire the guard."""
    pricer = Black76Pricer()
    sp = pricer.price_spread(
        _spread(long_K=20.0, short_K=22.0),
        _forward(F=21.0),
        iv_long=0.6500001, iv_short=0.65,
        as_of=_AS_OF, risk_free_rate=0.05,
    )
    assert sp.value > 0  # priced cleanly


# --------------------------------------------------------------------- #
# 3. Sentinel: TheoreticalSpreadPrice / is_executable=False              #
# --------------------------------------------------------------------- #


def test_returns_theoretical_spread_price_with_is_executable_false():
    pricer = Black76Pricer()
    out = pricer.price_spread(
        _spread(long_K=20.0, short_K=22.0),
        _forward(F=21.0),
        iv_long=0.65, iv_short=0.60,
        as_of=_AS_OF, risk_free_rate=0.05,
    )
    assert isinstance(out, TheoreticalSpreadPrice)
    assert out.is_executable is False
