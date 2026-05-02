"""Tests for minutes_to_settlement (ARCHITECTURE §4.2)."""
from datetime import datetime, timedelta, timezone

import pytest

from vix_spread.pricing.time_to_expiry import minutes_to_settlement
from vix_spread.utils.errors import ExpiryError, TimezoneError


def test_naive_as_of_raises():
    naive = datetime(2026, 5, 1)
    aware = datetime(2026, 6, 1, tzinfo=timezone.utc)
    with pytest.raises(TimezoneError):
        minutes_to_settlement(naive, aware)


def test_naive_settlement_raises():
    aware = datetime(2026, 5, 1, tzinfo=timezone.utc)
    naive = datetime(2026, 6, 1)
    with pytest.raises(TimezoneError):
        minutes_to_settlement(aware, naive)


def test_as_of_after_settlement_raises():
    later = datetime(2026, 6, 1, tzinfo=timezone.utc)
    earlier = datetime(2026, 5, 1, tzinfo=timezone.utc)
    with pytest.raises(ExpiryError):
        minutes_to_settlement(later, earlier)


def test_as_of_equal_to_settlement_raises():
    """T == 0 is not pricable; ExpiryError fires on equality, not just >."""
    ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
    with pytest.raises(ExpiryError):
        minutes_to_settlement(ts, ts)


def test_thirty_day_duration_returns_30_over_365():
    a = datetime(2026, 5, 1, tzinfo=timezone.utc)
    b = a + timedelta(days=30)
    T = minutes_to_settlement(a, b)
    assert T == pytest.approx(30.0 / 365.0, abs=1e-12)


def test_one_minute_duration_returns_1_over_525600():
    a = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    b = a + timedelta(minutes=1)
    T = minutes_to_settlement(a, b)
    assert T == pytest.approx(1.0 / 525_600.0, abs=1e-15)


def test_one_year_duration_returns_one():
    """365 calendar days exactly → T = 1.0."""
    a = datetime(2026, 5, 1, tzinfo=timezone.utc)
    b = a + timedelta(days=365)
    T = minutes_to_settlement(a, b)
    assert T == pytest.approx(1.0, abs=1e-12)
