"""Tests for Black76Pricer (ARCHITECTURE §4.4).

Five contracts:
  1. ATM zero-rate hand-computed reference matches to 1e-4. F=K=100,
     σ=0.20, T=1, r=0 — the textbook case where N(0.1)=0.539828 from
     standard normal tables fully determines the answer.
  2. General call/put parametrized cases match an independent inline
     Black-76 reference (scipy.stats.norm) to 1e-10. Catches any
     formula drift in the production pricer.
  3. Put-call parity holds: C − P = e^(-rT)(F − K), exact to 1e-10.
  4. Returned object is a TheoreticalPrice with is_executable=False
     (the structural anti-leak gate the FillEngine relies on).
  5. Bad inputs (F<=0, σ<=0, unknown 'right') raise ValueError loudly
     rather than producing NaN Greeks.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest
from scipy.stats import norm

from vix_spread.pricing.black76 import Black76Pricer
from vix_spread.pricing.forward_selector import Forward
from vix_spread.pricing.theoretical import TheoreticalPrice
from vix_spread.products.vix_index_option import VIXIndexOption


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


def _option(strike: float, right: str) -> VIXIndexOption:
    settlement = datetime(2027, 5, 1, 16, 0, tzinfo=timezone.utc)
    return VIXIndexOption(
        contract_root="VIX",
        expiry=settlement,
        settlement_event=settlement,
        strike=strike,
        right=right,
    )


def _forward(F: float) -> Forward:
    settlement = datetime(2027, 5, 1, 16, 0, tzinfo=timezone.utc)
    return Forward(
        value=F,
        selection_method='settlement_date_match',
        model_risk_flag=False,
        settlement_date=settlement,
    )


def _ref_black76(F, K, T, sigma, r, right):
    """Independent Black-76 reference using scipy.stats.norm directly."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df = math.exp(-r * T)
    pdf_d1 = float(norm.pdf(d1))
    if right == 'call':
        price = df * (F * float(norm.cdf(d1)) - K * float(norm.cdf(d2)))
        delta = df * float(norm.cdf(d1))
    else:
        price = df * (K * float(norm.cdf(-d2)) - F * float(norm.cdf(-d1)))
        delta = -df * float(norm.cdf(-d1))
    gamma = df * pdf_d1 / (F * sigma * sqrt_T)
    vega = df * F * pdf_d1 * sqrt_T
    theta = r * price - df * F * pdf_d1 * sigma / (2.0 * sqrt_T)
    rho = -T * price
    return price, delta, gamma, vega, theta, rho


# --------------------------------------------------------------------- #
# 1. ATM zero-rate hand-computed reference                              #
# --------------------------------------------------------------------- #


def test_atm_zero_rate_hand_computed_reference():
    """F = K = 100, σ = 0.20, T = 1.0 (calendar year), r = 0.

    From the closed-form:
      d1 =  0.5 · σ · √T = 0.10
      d2 = -0.10
      N( 0.10) = 0.539828   (standard normal table)
      N(-0.10) = 0.460172
      φ( 0.10) = 0.39694793

      C = 100·(0.539828 − 0.460172) = 7.9656
      P = C  (since F = K and r = 0 ⇒ put-call parity zero)

      Δ_call = 0.539828
      Δ_put  = -0.460172
      Γ      = φ(0.10) / (F·σ) = 0.39694793 / 20 = 0.01984740
      ν      = F · φ(0.10) = 39.694793
      Θ      = -F · φ(0.10) · σ / 2 = -3.9694793   (r·price = 0)
      ρ      = -T · price = -7.9656
    """
    pricer = Black76Pricer()
    settlement = datetime(2027, 5, 1, 16, 0, tzinfo=timezone.utc)
    as_of = settlement - timedelta(days=365)  # exactly T = 1.0
    fwd = _forward(100.0)

    cp = pricer.price(_option(100.0, 'call'), fwd,
                      leg_iv=0.20, as_of=as_of, risk_free_rate=0.0)
    pp = pricer.price(_option(100.0, 'put'), fwd,
                      leg_iv=0.20, as_of=as_of, risk_free_rate=0.0)

    assert cp.value == pytest.approx(7.96557, abs=1e-4)
    assert pp.value == pytest.approx(7.96557, abs=1e-4)

    assert cp.delta == pytest.approx(0.539828, abs=1e-4)
    assert pp.delta == pytest.approx(-0.460172, abs=1e-4)

    assert cp.gamma == pytest.approx(0.01984740, abs=1e-6)
    assert pp.gamma == pytest.approx(0.01984740, abs=1e-6)

    assert cp.vega == pytest.approx(39.694793, abs=1e-3)
    assert pp.vega == pytest.approx(39.694793, abs=1e-3)

    assert cp.theta == pytest.approx(-3.9694793, abs=1e-4)
    assert pp.theta == pytest.approx(-3.9694793, abs=1e-4)

    assert cp.rho == pytest.approx(-7.96557, abs=1e-4)
    assert pp.rho == pytest.approx(-7.96557, abs=1e-4)


# --------------------------------------------------------------------- #
# 2. General parametrized cases match the independent reference         #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("F,K,sigma,dte_days,r,right", [
    (20.0, 22.0, 0.65, 30, 0.05, 'call'),    # OTM call, 30 DTE — typical VIX
    (22.0, 20.0, 0.65, 30, 0.05, 'put'),     # OTM put
    (15.0, 15.0, 0.30, 60, 0.04, 'call'),    # ATM, low vol, 60 DTE
    (15.0, 15.0, 0.30, 60, 0.04, 'put'),
    (25.0, 30.0, 0.80, 7,  0.05, 'put'),     # 7 DTE high-vol OTM put
    (50.0, 50.0, 1.20, 14, 0.06, 'call'),    # Crisis-grade vol
    (100.0, 50.0, 0.20, 365, 0.05, 'call'),  # Deep ITM call (positive theta region)
])
def test_pricer_matches_independent_reference(F, K, sigma, dte_days, r, right):
    pricer = Black76Pricer()
    opt = _option(strike=K, right=right)
    as_of = opt.settlement_event - timedelta(days=dte_days)
    fwd = _forward(F)

    out = pricer.price(product=opt, forward=fwd, leg_iv=sigma,
                       as_of=as_of, risk_free_rate=r)

    T = dte_days * 24 * 60 / 525_600.0
    p_ref, d_ref, g_ref, v_ref, t_ref, rho_ref = _ref_black76(
        F, K, T, sigma, r, right,
    )
    assert out.value == pytest.approx(p_ref, rel=1e-10)
    assert out.delta == pytest.approx(d_ref, rel=1e-10)
    assert out.gamma == pytest.approx(g_ref, rel=1e-10)
    assert out.vega == pytest.approx(v_ref, rel=1e-10)
    assert out.theta == pytest.approx(t_ref, rel=1e-10)
    assert out.rho == pytest.approx(rho_ref, rel=1e-10)


# --------------------------------------------------------------------- #
# 3. Put-call parity                                                    #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("F,K,T_days,sigma,r", [
    (20.0, 22.0, 30, 0.65, 0.05),
    (50.0, 45.0, 60, 0.40, 0.04),
    (15.0, 18.0, 14, 0.90, 0.06),
    (100.0, 100.0, 365, 0.20, 0.0),  # the hand-computed case
])
def test_put_call_parity(F, K, T_days, sigma, r):
    pricer = Black76Pricer()
    opt_call = _option(K, 'call')
    opt_put = _option(K, 'put')
    as_of = opt_call.settlement_event - timedelta(days=T_days)
    fwd = _forward(F)

    cp = pricer.price(opt_call, fwd, leg_iv=sigma, as_of=as_of, risk_free_rate=r)
    pp = pricer.price(opt_put, fwd, leg_iv=sigma, as_of=as_of, risk_free_rate=r)

    T = T_days * 24 * 60 / 525_600.0
    parity = math.exp(-r * T) * (F - K)
    assert (cp.value - pp.value) == pytest.approx(parity, abs=1e-10)


# --------------------------------------------------------------------- #
# 4. Sentinel: TheoreticalPrice / is_executable=False                   #
# --------------------------------------------------------------------- #


def test_returns_theoretical_price_with_is_executable_false():
    """The structural anti-leak gate: the FillEngine refuses any object
    with `is_executable=False` (in fact any non-OptionQuote). The pricer
    must produce TheoreticalPrice with that sentinel set, audit-tagged
    with the Forward used and the IV used."""
    pricer = Black76Pricer()
    opt = _option(20.0, 'call')
    as_of = opt.settlement_event - timedelta(days=30)
    fwd = _forward(20.0)
    out = pricer.price(opt, fwd, leg_iv=0.65, as_of=as_of, risk_free_rate=0.05)

    assert isinstance(out, TheoreticalPrice)
    assert out.is_executable is False
    assert out.iv_used == pytest.approx(0.65)
    assert out.forward_used is fwd
    # T_minutes audit field equals the actual minute count.
    expected_minutes = 30 * 24 * 60
    assert out.T_minutes == pytest.approx(expected_minutes, abs=1e-6)


# --------------------------------------------------------------------- #
# 5. Bad-input refusals                                                 #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("F,K", [(0.0, 20.0), (-1.0, 20.0), (20.0, 0.0), (20.0, -1.0)])
def test_non_positive_forward_or_strike_raises(F, K):
    pricer = Black76Pricer()
    settlement = datetime(2027, 5, 1, 16, 0, tzinfo=timezone.utc)
    opt = VIXIndexOption(
        contract_root="VIX",
        expiry=settlement,
        settlement_event=settlement,
        strike=K,
        right='call',
    )
    fwd = Forward(
        value=F,
        selection_method='settlement_date_match',
        model_risk_flag=False,
        settlement_date=settlement,
    )
    as_of = settlement - timedelta(days=30)
    with pytest.raises(ValueError):
        pricer.price(opt, fwd, leg_iv=0.5, as_of=as_of, risk_free_rate=0.05)


def test_non_positive_iv_raises():
    pricer = Black76Pricer()
    opt = _option(20.0, 'call')
    as_of = opt.settlement_event - timedelta(days=30)
    fwd = _forward(20.0)
    with pytest.raises(ValueError, match='Implied vol'):
        pricer.price(opt, fwd, leg_iv=0.0, as_of=as_of, risk_free_rate=0.05)


def test_unknown_right_raises():
    """A Product whose `right` is neither 'call' nor 'put' must raise.
    Constructed via object.__setattr__ to bypass dataclass validation."""
    pricer = Black76Pricer()
    opt = _option(20.0, 'call')
    object.__setattr__(opt, 'right', 'straddle')  # forced bad value
    as_of = opt.settlement_event - timedelta(days=30)
    fwd = _forward(20.0)
    with pytest.raises(ValueError, match='right'):
        pricer.price(opt, fwd, leg_iv=0.5, as_of=as_of, risk_free_rate=0.05)
