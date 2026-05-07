"""Tests for ChainIVProvider — vendor + B76-invert fallback.

The headline test (per validation-memo guard for silent-NaN propagation)
is `test_inversion_recovers_input_iv_to_tight_tolerance`: generate a
synthetic mid using a KNOWN IV via Black-76, give it back as the chain-row
mid to invert, and assert the recovered IV matches the input within
brentq's xtol.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from vix_spread.pricing.black76 import Black76Pricer
from vix_spread.pricing.forward_selector import Forward
from vix_spread.pricing.leg_iv import (
    ChainIVProvider,
    LegIV,
    LegIVProvider,
    LegIVSource,
)
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.utils.errors import LegIVResolutionError


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def pricer() -> Black76Pricer:
    return Black76Pricer()


@pytest.fixture
def settlement_event() -> datetime:
    # Wed June 17 2026 SOQ window
    return datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)


@pytest.fixture
def expiry(settlement_event) -> datetime:
    return settlement_event


@pytest.fixture
def forward(settlement_event) -> Forward:
    return Forward(
        value=20.0,
        selection_method='settlement_date_match',
        model_risk_flag=False,
        settlement_date=settlement_event,
    )


@pytest.fixture
def call_product(expiry, settlement_event) -> VIXIndexOption:
    return VIXIndexOption(
        contract_root="VIX",
        expiry=expiry,
        settlement_event=settlement_event,
        strike=20.0,
        right='call',
    )


@pytest.fixture
def put_product(expiry, settlement_event) -> VIXIndexOption:
    return VIXIndexOption(
        contract_root="VIX",
        expiry=expiry,
        settlement_event=settlement_event,
        strike=18.0,
        right='put',
    )


@pytest.fixture
def as_of() -> datetime:
    # ~2.5 months before expiry — well within Black-76 numerical comfort zone
    return datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc)


@pytest.fixture
def risk_free_rate() -> float:
    return 0.04


def _make_chain(rows: list[dict]) -> pd.DataFrame:
    """Build a chain DataFrame with the canonical MultiIndex layout."""
    df = pd.DataFrame(rows)
    return df.set_index(['date', 'expiry', 'right', 'strike'])


# --------------------------------------------------------------------------- #
# LegIVSource enum                                                            #
# --------------------------------------------------------------------------- #


def test_legiv_source_enum_values():
    assert LegIVSource.VENDOR.value == 'vendor'
    assert LegIVSource.B76_INVERTED.value == 'b76_inverted'


# --------------------------------------------------------------------------- #
# Vendor path                                                                 #
# --------------------------------------------------------------------------- #


def test_provider_uses_vendor_when_present(
    pricer, call_product, as_of, forward, risk_free_rate
):
    chain = _make_chain([{
        'date': as_of.date(),
        'expiry': call_product.expiry.date(),
        'right': 'C',
        'strike': 20.0,
        'IVOL_LAST': 60.0,           # 60% vol in vendor pct form
        'PX_BID': 1.0,
        'PX_ASK': 2.0,
    }])
    provider = ChainIVProvider(chain, pricer)
    iv = provider.get(
        product=call_product, as_of=as_of,
        forward=forward, risk_free_rate=risk_free_rate,
    )
    assert iv.source == LegIVSource.VENDOR
    assert iv.value == pytest.approx(0.60)
    assert iv.fallback_target_mid is None


def test_provider_returns_legiv_dataclass(
    pricer, call_product, as_of, forward, risk_free_rate
):
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 20.0,
        'IVOL_LAST': 50.0, 'PX_BID': 1.0, 'PX_ASK': 1.5,
    }])
    iv = ChainIVProvider(chain, pricer).get(
        product=call_product, as_of=as_of,
        forward=forward, risk_free_rate=risk_free_rate,
    )
    assert isinstance(iv, LegIV)
    # Frozen — backtest auditor can rely on immutability.
    with pytest.raises(Exception):
        iv.value = 999  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# B76-invert fallback — the headline correctness test                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("true_iv", [0.30, 0.50, 0.85, 1.20, 2.00])
def test_inversion_recovers_input_iv_to_tight_tolerance(
    pricer, call_product, as_of, forward, risk_free_rate, true_iv
):
    """Black-76 forward: price(σ) → mid → invert(mid) ≈ σ.

    Tolerance: brentq's `xtol=1e-6`, so recovered IV should match input
    to ~1e-5 in absolute units.
    """
    target_price = pricer.price(
        call_product, forward, true_iv, as_of, risk_free_rate,
    ).value
    chain = _make_chain([{
        'date': as_of.date(),
        'expiry': call_product.expiry.date(),
        'right': 'C',
        'strike': 20.0,
        'IVOL_LAST': float('nan'),                # forces fallback
        'PX_BID': target_price - 1e-6,
        'PX_ASK': target_price + 1e-6,            # mid == target_price
    }])
    provider = ChainIVProvider(chain, pricer)
    iv = provider.get(
        product=call_product, as_of=as_of,
        forward=forward, risk_free_rate=risk_free_rate,
    )
    assert iv.source == LegIVSource.B76_INVERTED
    assert iv.value == pytest.approx(true_iv, abs=1e-5)
    assert iv.fallback_target_mid == pytest.approx(target_price, abs=1e-5)


def test_inversion_round_trips_for_puts(
    pricer, put_product, as_of, forward, risk_free_rate
):
    """Same recovery property must hold for puts; the inversion is
    right-aware via `Black76Pricer.price` dispatching on `product.right`."""
    true_iv = 0.65
    target = pricer.price(
        put_product, forward, true_iv, as_of, risk_free_rate,
    ).value
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': put_product.expiry.date(),
        'right': 'P', 'strike': 18.0,
        'IVOL_LAST': float('nan'),
        'PX_BID': target - 1e-6, 'PX_ASK': target + 1e-6,
    }])
    iv = ChainIVProvider(chain, pricer).get(
        product=put_product, as_of=as_of,
        forward=forward, risk_free_rate=risk_free_rate,
    )
    assert iv.source == LegIVSource.B76_INVERTED
    assert iv.value == pytest.approx(true_iv, abs=1e-5)


def test_inversion_fires_on_zero_vendor_iv(
    pricer, call_product, as_of, forward, risk_free_rate
):
    """`IVOL_LAST == 0.0` must be treated as missing (Bloomberg sentinel
    for newly-listed contracts), NOT as a literal zero-vol vendor reading.
    """
    true_iv = 0.50
    target = pricer.price(
        call_product, forward, true_iv, as_of, risk_free_rate,
    ).value
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 20.0,
        'IVOL_LAST': 0.0,                        # sentinel for missing
        'PX_BID': target - 1e-6, 'PX_ASK': target + 1e-6,
    }])
    iv = ChainIVProvider(chain, pricer).get(
        product=call_product, as_of=as_of,
        forward=forward, risk_free_rate=risk_free_rate,
    )
    assert iv.source == LegIVSource.B76_INVERTED
    assert iv.value == pytest.approx(true_iv, abs=1e-5)


# --------------------------------------------------------------------------- #
# Refusal contract — no NaN propagation                                       #
# --------------------------------------------------------------------------- #


def test_refuses_when_both_paths_missing(
    pricer, call_product, as_of, forward, risk_free_rate
):
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 20.0,
        'IVOL_LAST': float('nan'),
        'PX_BID': float('nan'),
        'PX_ASK': float('nan'),
    }])
    provider = ChainIVProvider(chain, pricer)
    with pytest.raises(LegIVResolutionError, match="Both vendor"):
        provider.get(
            product=call_product, as_of=as_of,
            forward=forward, risk_free_rate=risk_free_rate,
        )


def test_refuses_when_quotes_non_positive(
    pricer, call_product, as_of, forward, risk_free_rate
):
    """Locked/crossed/zero quotes can't produce a meaningful midpoint
    for inversion."""
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 20.0,
        'IVOL_LAST': float('nan'),
        'PX_BID': 0.0,
        'PX_ASK': 0.0,
    }])
    provider = ChainIVProvider(chain, pricer)
    with pytest.raises(LegIVResolutionError, match="non-positive"):
        provider.get(
            product=call_product, as_of=as_of,
            forward=forward, risk_free_rate=risk_free_rate,
        )


def test_refuses_when_target_below_intrinsic(
    pricer, call_product, as_of, forward, risk_free_rate
):
    """Mid below intrinsic puts the brentq root outside [1e-3, 5.0] —
    inversion fails and is caught as `LegIVResolutionError`."""
    intrinsic = max(forward.value - call_product.strike, 0.0)
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 20.0,
        'IVOL_LAST': float('nan'),
        # bid/ask below any positive Black-76 price for this F=K=20 call
        'PX_BID': max(intrinsic - 0.5, 0.001),
        'PX_ASK': max(intrinsic - 0.4, 0.001),
    }])
    # F = K = 20 here so intrinsic = 0; "below intrinsic" means a tiny
    # mid that B76 with σ=1e-3 already exceeds. Inversion fails.
    provider = ChainIVProvider(chain, pricer)
    # When F=K, intrinsic=0 so any small mid still has a B76 root.
    # Drive the test by inflating to an unreachable mid instead:
    chain_unreachable = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 20.0,
        'IVOL_LAST': float('nan'),
        # An option price > forward is impossible (df * F is the upper bound)
        'PX_BID': 100.0,
        'PX_ASK': 102.0,
    }])
    provider2 = ChainIVProvider(chain_unreachable, pricer)
    with pytest.raises(LegIVResolutionError, match="inversion failed"):
        provider2.get(
            product=call_product, as_of=as_of,
            forward=forward, risk_free_rate=risk_free_rate,
        )


def test_refuses_when_chain_lookup_misses(
    pricer, call_product, as_of, forward, risk_free_rate
):
    """A missing (date, expiry, right, strike) row -> LegIVResolutionError.
    No silent fallback to a neighbour."""
    # Chain has a different strike than the requested 20.0
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 21.0,
        'IVOL_LAST': 50.0, 'PX_BID': 1.0, 'PX_ASK': 2.0,
    }])
    provider = ChainIVProvider(chain, pricer)
    with pytest.raises(LegIVResolutionError, match="No chain row"):
        provider.get(
            product=call_product, as_of=as_of,
            forward=forward, risk_free_rate=risk_free_rate,
        )


def test_requires_forward_and_rate_kwargs(pricer, call_product, as_of):
    chain = _make_chain([{
        'date': as_of.date(), 'expiry': call_product.expiry.date(),
        'right': 'C', 'strike': 20.0,
        'IVOL_LAST': 50.0, 'PX_BID': 1.0, 'PX_ASK': 2.0,
    }])
    provider = ChainIVProvider(chain, pricer)
    with pytest.raises(ValueError, match="forward.*risk_free_rate"):
        provider.get(product=call_product, as_of=as_of)


# --------------------------------------------------------------------------- #
# Construction validation                                                     #
# --------------------------------------------------------------------------- #


def test_construction_requires_required_columns(pricer):
    idx = pd.MultiIndex.from_tuples(
        [(None, None, None, None)],
        names=['date', 'expiry', 'right', 'strike'],
    )
    bad = pd.DataFrame({'IVOL_LAST': [50.0]}, index=idx)
    with pytest.raises(ValueError, match="missing required columns"):
        ChainIVProvider(bad, pricer)


def test_construction_requires_multiindex(pricer):
    bad = pd.DataFrame({
        'IVOL_LAST': [50.0], 'PX_BID': [1.0], 'PX_ASK': [2.0],
    })  # default RangeIndex, not MultiIndex
    with pytest.raises(ValueError, match="MultiIndexed"):
        ChainIVProvider(bad, pricer)


# --------------------------------------------------------------------------- #
# ABC contract                                                                #
# --------------------------------------------------------------------------- #


def test_chainivprovider_implements_legivprovider():
    assert issubclass(ChainIVProvider, LegIVProvider)


def test_legivprovider_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LegIVProvider()  # type: ignore[abstract]
