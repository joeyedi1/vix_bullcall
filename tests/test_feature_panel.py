"""Tests for FeaturePanel.

Two contracts:
  1. Construction validates: tz-aware dates, features.index == dates,
     as_of_map covers every column with matching index.
  2. slice_causal returns rows whose as_of_effective <= as_of and drops
     rows with later release timestamps (release-boundary rule, §6.4).
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from vix_spread.data.feature_panel import FeaturePanel


def _panel(n: int = 5):
    base = datetime(2026, 1, 1, 21, 0, tzinfo=timezone.utc)
    dates = pd.DatetimeIndex(
        [base + timedelta(days=i) for i in range(n)],
        name='date',
    )
    features = pd.DataFrame({'log_vix': [2.5 + 0.01 * i for i in range(n)]},
                            index=dates)
    as_of_map = {'log_vix': pd.Series(dates, index=dates)}
    return dates, features, as_of_map


def test_construct_valid_panel():
    dates, features, as_of_map = _panel()
    panel = FeaturePanel(dates=dates, features=features, as_of_map=as_of_map)
    assert panel.features.shape == (5, 1)


def test_naive_dates_rejected():
    dates = pd.DatetimeIndex([datetime(2026, 1, 1)])
    features = pd.DataFrame({'x': [1.0]}, index=dates)
    as_of_map = {'x': pd.Series(dates, index=dates)}
    with pytest.raises(TypeError):
        FeaturePanel(dates=dates, features=features, as_of_map=as_of_map)


def test_missing_as_of_for_column_rejected():
    dates, features, _ = _panel()
    with pytest.raises(ValueError):
        FeaturePanel(dates=dates, features=features, as_of_map={})


def test_slice_causal_drops_future_rows():
    """A row with as_of_effective > as_of must be invisible to slice_causal."""
    dates, features, as_of_map = _panel(n=5)
    # Make row[3] released "tomorrow" relative to its date.
    eff = as_of_map['log_vix'].copy()
    eff.iloc[3] = eff.iloc[3] + pd.Timedelta(days=2)
    panel = FeaturePanel(dates=dates, features=features,
                         as_of_map={'log_vix': eff})

    as_of = dates[3]  # <- row[3]'s date, but eff[3] is 2 days later
    out = panel.slice_causal('log_vix', as_of)
    # Rows 0..3 by date; row 3 is excluded by release-boundary rule.
    assert list(out.index) == list(dates[[0, 1, 2]])
