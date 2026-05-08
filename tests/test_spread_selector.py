"""SpreadSelector — strike-and-expiry picker with pre-selection validity filter.

The headline test is `test_skips_no_bid_short_strikes_pre_selection`:
the user-explicit guard against producing a spread the FillEngine would
reject with `reason='no_bid_short'`.
"""
from datetime import date, datetime, time, timezone

import numpy as np
import pytest

from vix_spread.data.snapshot import VIXSnapshot
from vix_spread.execution.quote import OptionQuote
from vix_spread.products.spread import BullCallSpread
from vix_spread.regime.base import RegimeSignal
from vix_spread.strategy.spread_selector import SpreadSelector


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


# May 2026 monthly VIX option: SOQ Wed = 5/20, last-trade Tue = 5/19.
SOQ_DATE = date(2026, 5, 20)
LAST_TRADE_TUE = date(2026, 5, 19)
AS_OF = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)
FORWARD = 22.0


def _q(
    contract_id: str, bid: float, ask: float,
    *, bid_size: int = 100, ask_size: int = 100,
) -> OptionQuote:
    return OptionQuote(
        timestamp=AS_OF, contract_id=contract_id,
        bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
        last_trade=None, last_trade_age_seconds=None,
        is_locked=False, is_crossed=False,
        quote_age_seconds=2.0,
    )


def _ticker(strike: float, right: str = "C") -> str:
    """Build the active-form contract_id for the May 2026 expiry."""
    return f"VIX US 05/19/26 {right}{strike:g} Index"


def _snapshot(quotes: dict[str, OptionQuote] | None = None) -> VIXSnapshot:
    return VIXSnapshot(
        timestamp=AS_OF,
        vx_curve={SOQ_DATE: FORWARD},
        options_quotes=quotes or {},
        risk_free_rate=0.04,
        vix_spot=18.0,
    )


def _signal() -> RegimeSignal:
    return RegimeSignal(
        as_of=AS_OF,
        filtered_probs=np.array([0.7, 0.3]),
        state_label=0,
        curve_features={"slope_30_182": -0.04},
        hypothesis_tag="contrarian_tail",
        as_of_inputs={"log_vix": AS_OF},
    )


@pytest.fixture
def selector() -> SpreadSelector:
    # forward 22; long_offset 2 -> long_target 24; short_offset 4 -> short_target 28
    return SpreadSelector(
        long_offset=2.0, short_offset=4.0,
        dte_min=7, dte_max=60,
    )


# --------------------------------------------------------------------------- #
# Constructor validation                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(long_offset=-1.0, short_offset=4.0, dte_min=7, dte_max=60),
        dict(long_offset=2.0, short_offset=0.0, dte_min=7, dte_max=60),
        dict(long_offset=2.0, short_offset=-1.0, dte_min=7, dte_max=60),
        dict(long_offset=2.0, short_offset=4.0, dte_min=-1, dte_max=60),
        dict(long_offset=2.0, short_offset=4.0, dte_min=60, dte_max=7),
    ],
)
def test_constructor_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        SpreadSelector(**kwargs)


# --------------------------------------------------------------------------- #
# Happy path — closest-valid strike selection                                 #
# --------------------------------------------------------------------------- #


def test_selects_closest_strikes_to_targets(selector):
    # Strikes 18, 20, 22, 24, 26, 28, 30 with normal NBBO on all.
    quotes = {
        _ticker(s): _q(_ticker(s), bid=2.0, ask=2.1)
        for s in (18, 20, 22, 24, 26, 28, 30)
    }
    spread = selector.select(_snapshot(quotes), _signal(), AS_OF)
    assert isinstance(spread, BullCallSpread)
    # forward=22, long_offset=2 -> target 24; closest valid = 24.
    assert spread.long_leg.strike == 24.0
    # long_strike=24, short_offset=4 -> target 28; closest valid > 24 = 28.
    assert spread.short_leg.strike == 28.0
    # Settlement event = SOQ Wed (normalized from Tue ticker date).
    assert spread.long_leg.settlement_event.date() == SOQ_DATE
    assert spread.short_leg.settlement_event.date() == SOQ_DATE


def test_resolves_target_to_nearest_grid_strike(selector):
    """If the target falls between strikes, the closer one wins."""
    # Strikes 23 and 25 — long_target=24 is equidistant; min() picks 23
    # (the first encountered with min |delta| in sorted order).
    quotes = {
        _ticker(s): _q(_ticker(s), bid=1.5, ask=1.6)
        for s in (23, 25, 27, 30)
    }
    spread = selector.select(_snapshot(quotes), _signal(), AS_OF)
    assert spread is not None
    # 23 and 25 are equidistant from 24; tie-breaking via min() returns 23.
    assert spread.long_leg.strike == 23.0
    # short_target = 23 + 4 = 27 -> closest in {25, 27, 30} where strike > 23
    # is 25 (delta 2) vs 27 (delta 0) vs 30 (delta 3). 27 wins.
    assert spread.short_leg.strike == 27.0


# --------------------------------------------------------------------------- #
# Pre-selection validity filter — the user's explicit requirement             #
# --------------------------------------------------------------------------- #


def test_skips_no_bid_short_strikes_pre_selection(selector):
    """The validation-memo critical guard: a strike with bid <= 0 must NOT
    be selected as the short leg, even if it's the closest to short_target.
    The selector falls back to the next-closest strike with bid > 0.
    """
    # forward 22 -> long_target 24; long_strike 24.
    # short_target 28; short candidates above 24:
    #   - 26 (bid 0.0 — INELIGIBLE as short)
    #   - 28 (bid 0.0 — INELIGIBLE as short)
    #   - 30 (bid 0.5  — eligible)
    # Without the pre-filter, selector would pick 28 (closest to target 28).
    # With it, selector falls through to 30.
    quotes = {
        _ticker(24): _q(_ticker(24), bid=1.5, ask=1.6),
        _ticker(26): _q(_ticker(26), bid=0.0, ask=0.4),    # no-bid
        _ticker(28): _q(_ticker(28), bid=0.0, ask=0.2),    # no-bid
        _ticker(30): _q(_ticker(30), bid=0.5, ask=0.6),
    }
    spread = selector.select(_snapshot(quotes), _signal(), AS_OF)
    assert spread is not None
    assert spread.long_leg.strike == 24.0
    # 26 and 28 skipped (no-bid); selector picks 30.
    assert spread.short_leg.strike == 30.0


def test_skips_no_ask_long_strikes_pre_selection(selector):
    """Mirror guard: a strike with ask <= 0 must NOT be selected as the
    long leg, since we'd have nothing to buy."""
    # forward 22 -> long_target 24
    # 24 has ask=0 (ineligible long); 22 ask>0; 26 ask>0
    # 22 (delta 2) and 26 (delta 2) are equidistant; min() picks 22.
    quotes = {
        _ticker(22): _q(_ticker(22), bid=2.0, ask=2.1),
        _ticker(24): _q(_ticker(24), bid=1.5, ask=0.0),    # no-ask, INELIGIBLE
        _ticker(26): _q(_ticker(26), bid=1.0, ask=1.1),
        _ticker(30): _q(_ticker(30), bid=0.5, ask=0.6),
    }
    spread = selector.select(_snapshot(quotes), _signal(), AS_OF)
    assert spread is not None
    assert spread.long_leg.strike == 22.0


# --------------------------------------------------------------------------- #
# Refusal branches — return None                                              #
# --------------------------------------------------------------------------- #


def test_returns_none_when_no_expiry_in_dte_window():
    """DTE window 7-30 from 4/15; the only listed expiry is 5/20 (DTE=35),
    just outside the window."""
    selector = SpreadSelector(
        long_offset=2.0, short_offset=4.0, dte_min=7, dte_max=30,
    )
    quotes = {_ticker(s): _q(_ticker(s), bid=1.0, ask=1.1) for s in (20, 24, 28)}
    spread = selector.select(_snapshot(quotes), _signal(), AS_OF)
    assert spread is None


def test_returns_none_when_no_short_eligible_above_long(selector):
    """If long_strike picks the highest valid strike, no short candidates
    remain (none > long_strike)."""
    quotes = {
        _ticker(20): _q(_ticker(20), bid=2.0, ask=2.1),
        _ticker(24): _q(_ticker(24), bid=1.5, ask=1.6),
    }
    # long_target 24 -> long_strike = 24 (highest available). No strike > 24.
    spread = selector.select(_snapshot(quotes), _signal(), AS_OF)
    assert spread is None


def test_returns_none_when_no_quotes(selector):
    spread = selector.select(_snapshot(quotes={}), _signal(), AS_OF)
    assert spread is None


def test_returns_none_when_forward_missing_for_expiry(selector):
    """Quotes exist for an expiry not present in vx_curve."""
    quotes = {_ticker(s): _q(_ticker(s), bid=1.5, ask=1.6) for s in (20, 24, 28)}
    snap = VIXSnapshot(
        timestamp=AS_OF,
        vx_curve={},                # forward missing
        options_quotes=quotes,
        risk_free_rate=0.04,
    )
    spread = selector.select(snap, _signal(), AS_OF)
    assert spread is None


def test_skips_unparseable_tickers(selector):
    """Non-VIX-monthly tickers (weeklies in different format, garbage,
    etc.) are silently skipped — same convention as filter_chain."""
    quotes = {
        "GARBAGE": _q("GARBAGE", bid=1.0, ask=1.1),
        _ticker(24): _q(_ticker(24), bid=1.5, ask=1.6),
        _ticker(28): _q(_ticker(28), bid=1.0, ask=1.1),
    }
    spread = selector.select(_snapshot(quotes), _signal(), AS_OF)
    assert spread is not None
    assert spread.long_leg.strike == 24.0


# --------------------------------------------------------------------------- #
# Multi-expiry — picks nearest in window                                      #
# --------------------------------------------------------------------------- #


def test_picks_nearest_eligible_expiry_within_dte_window(selector):
    """May 2026 (DTE=35) and June 2026 (DTE=63 → outside dte_max=60)."""
    may_quotes = {
        f"VIX US 05/19/26 C{s} Index": _q(
            f"VIX US 05/19/26 C{s} Index", bid=1.5, ask=1.6,
        )
        for s in (22, 24, 28)
    }
    june_quotes = {
        f"VIX US 06/16/26 C{s} Index": _q(
            f"VIX US 06/16/26 C{s} Index", bid=1.5, ask=1.6,
        )
        for s in (22, 24, 28)
    }
    snap = VIXSnapshot(
        timestamp=AS_OF,
        vx_curve={
            SOQ_DATE: FORWARD,                   # May 20
            date(2026, 6, 17): FORWARD,          # June 17 SOQ
        },
        options_quotes={**may_quotes, **june_quotes},
        risk_free_rate=0.04,
    )
    spread = selector.select(snap, _signal(), AS_OF)
    assert spread is not None
    # May 2026 has DTE 35, within [7, 60]; June 2026 has DTE 63, outside.
    assert spread.long_leg.settlement_event.date() == SOQ_DATE


def test_signal_argument_accepted_but_unused(selector):
    """First-pass selector ignores `signal` — accept any RegimeSignal-shaped
    object including None."""
    quotes = {
        _ticker(s): _q(_ticker(s), bid=1.5, ask=1.6)
        for s in (24, 28)
    }
    spread = selector.select(_snapshot(quotes), None, AS_OF)
    assert spread is not None
    assert spread.long_leg.strike == 24.0
    assert spread.short_leg.strike == 28.0
