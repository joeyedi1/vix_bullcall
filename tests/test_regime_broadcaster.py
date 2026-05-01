"""Tests for the daily-EOD → 1-minute strict-causality broadcaster.

Three contracts (ARCHITECTURE §7.2):
  1. The bar whose timestamp EQUALS a daily signal's as_of carries the
     PRIOR signal — never the same-timestamp one.
  2. The bar one minute LATER carries the new signal.
  3. Bars before the first daily signal carry NaN (no fabrication).
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from vix_spread.regime.broadcaster import broadcast_daily_to_minute


def _grid(start: datetime, n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [start + timedelta(minutes=i) for i in range(n)], tz='UTC',
    )


def test_same_minute_carries_prior_signal_strictly():
    """A signal stamped at 21:00 UTC must NOT appear on the 21:00 bar.
    The 21:00 bar must carry the prior day's signal (or NaN if first)."""
    day1_close = datetime(2026, 1, 14, 21, 0, tzinfo=timezone.utc)
    day2_close = datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc)
    daily = pd.DataFrame(
        {'state_label': [0, 1]},
        index=pd.DatetimeIndex([day1_close, day2_close], tz='UTC'),
    )
    grid = _grid(datetime(2026, 1, 15, 20, 59, tzinfo=timezone.utc), 4)
    out = broadcast_daily_to_minute(daily, grid)

    # 20:59 bar: only day1 signal known -> state_label = 0
    assert out.loc[grid[0], 'state_label'] == 0
    # 21:00 bar: day2 close just printed, but strict shift means we still
    # see day1's signal here.
    assert out.loc[grid[1], 'state_label'] == 0
    assert out.loc[grid[1], 'as_of_effective'] == pd.Timestamp(day1_close)
    # 21:01 bar: first bar STRICTLY after day2 close -> state_label = 1
    assert out.loc[grid[2], 'state_label'] == 1
    assert out.loc[grid[2], 'as_of_effective'] == pd.Timestamp(day2_close)


def test_bars_before_first_signal_are_nan():
    day1_close = datetime(2026, 1, 14, 21, 0, tzinfo=timezone.utc)
    daily = pd.DataFrame(
        {'state_label': [1]},
        index=pd.DatetimeIndex([day1_close], tz='UTC'),
    )
    grid = _grid(datetime(2026, 1, 14, 20, 58, tzinfo=timezone.utc), 4)
    out = broadcast_daily_to_minute(daily, grid)
    assert pd.isna(out.loc[grid[0], 'state_label'])
    assert pd.isna(out.loc[grid[1], 'state_label'])
    assert pd.isna(out.loc[grid[2], 'state_label'])  # 21:00 bar — strict
    assert out.loc[grid[3], 'state_label'] == 1     # 21:01 bar — first valid


def test_naive_grid_rejected():
    daily = pd.DataFrame(
        {'x': [0]},
        index=pd.DatetimeIndex([datetime(2026, 1, 1)], tz='UTC'),
    )
    grid = pd.DatetimeIndex([datetime(2026, 1, 2)])  # naive
    with pytest.raises(TypeError):
        broadcast_daily_to_minute(daily, grid)


def test_naive_daily_index_rejected():
    daily = pd.DataFrame(
        {'x': [0]}, index=pd.DatetimeIndex([datetime(2026, 1, 1)]),
    )
    grid = _grid(datetime(2026, 1, 1), 2)
    with pytest.raises(TypeError):
        broadcast_daily_to_minute(daily, grid)


def test_reserved_column_rejected():
    """`as_of_effective` is the audit column the broadcaster writes; an
    incoming signal frame that already carries it must raise rather than
    silently overwrite."""
    daily = pd.DataFrame(
        {'state_label': [0], 'as_of_effective': [pd.Timestamp('2026-01-01', tz='UTC')]},
        index=pd.DatetimeIndex([datetime(2026, 1, 1, tzinfo=timezone.utc)]),
    )
    grid = _grid(datetime(2026, 1, 2, tzinfo=timezone.utc), 2)
    with pytest.raises(ValueError, match="as_of_effective"):
        broadcast_daily_to_minute(daily, grid)
