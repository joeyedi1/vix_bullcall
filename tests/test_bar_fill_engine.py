"""Tests for MinuteBarFillEngine.

Four contracts:
  1. Bid/ask crossing — BUY at last_ask, SELL at last_bid; never mid/close.
  2. Per-side staleness — only the side BEING CROSSED matters; above
     max_quote_age (or NaN) rejects.
  3. No-bid / no-ask — NaN or non-positive on the cross side rejects.
  4. Augmented frame — original columns and index preserved; three new
     columns added: fill_price, fill_side, rejection_reason.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from vix_spread.execution.bar_fill_engine import MinuteBarFillEngine


def _grid(n: int) -> pd.DatetimeIndex:
    base = datetime(2025, 10, 15, 20, 0, tzinfo=timezone.utc)
    return pd.DatetimeIndex(
        [base + timedelta(minutes=i) for i in range(n)], tz='UTC',
    )


def _panel(actions, **overrides) -> pd.DataFrame:
    n = len(actions)
    df = pd.DataFrame({
        'last_bid':         [13.5] * n,
        'last_ask':         [13.7] * n,
        'bid_age_seconds':  [10.0] * n,
        'ask_age_seconds':  [10.0] * n,
        'action':           list(actions),
    }, index=_grid(n))
    for col, vals in overrides.items():
        df[col] = vals
    return df


# --------------------------------------------------------------------- #
# 1. Bid / ask crossing                                                 #
# --------------------------------------------------------------------- #


def test_buy_fills_at_last_ask():
    engine = MinuteBarFillEngine()
    panel = _panel(['BUY', None, 'BUY'])
    out = engine.execute(panel)
    assert out['fill_price'].iloc[0] == pytest.approx(13.7)
    assert out['fill_side'].iloc[0] == 'BUY'
    assert pd.isna(out['fill_price'].iloc[1])
    assert pd.isna(out['fill_side'].iloc[1])
    assert out['fill_price'].iloc[2] == pytest.approx(13.7)


def test_sell_fills_at_last_bid():
    engine = MinuteBarFillEngine()
    panel = _panel(['SELL'])
    out = engine.execute(panel)
    assert out['fill_price'].iloc[0] == pytest.approx(13.5)
    assert out['fill_side'].iloc[0] == 'SELL'


def test_no_action_minutes_have_no_fill_and_no_rejection():
    engine = MinuteBarFillEngine()
    panel = _panel([None, None, None])
    out = engine.execute(panel)
    assert out['fill_price'].isna().all()
    assert out['fill_side'].isna().all()
    assert out['rejection_reason'].isna().all()


# --------------------------------------------------------------------- #
# 2. Per-side staleness                                                 #
# --------------------------------------------------------------------- #


def test_stale_ask_rejects_buy_but_not_sell():
    """Per-side: BUY rejects on stale ask; a SELL on the same minute
    with a fresh bid still fills."""
    engine = MinuteBarFillEngine(max_quote_age_seconds=300.0)
    panel = _panel(
        ['BUY', 'SELL'],
        ask_age_seconds=[600.0, 600.0],
        bid_age_seconds=[10.0, 10.0],
    )
    out = engine.execute(panel)
    assert pd.isna(out['fill_price'].iloc[0])
    assert out['rejection_reason'].iloc[0] == 'stale_quote'
    assert out['fill_price'].iloc[1] == pytest.approx(13.5)
    assert pd.isna(out['rejection_reason'].iloc[1])


def test_stale_bid_rejects_sell_but_not_buy():
    engine = MinuteBarFillEngine(max_quote_age_seconds=300.0)
    panel = _panel(
        ['SELL', 'BUY'],
        bid_age_seconds=[600.0, 600.0],
        ask_age_seconds=[10.0, 10.0],
    )
    out = engine.execute(panel)
    assert out['rejection_reason'].iloc[0] == 'stale_quote'
    assert out['fill_price'].iloc[1] == pytest.approx(13.7)


def test_at_threshold_quote_fills():
    """Boundary: age == threshold is NOT stale (strict > to reject)."""
    engine = MinuteBarFillEngine(max_quote_age_seconds=300.0)
    panel = _panel(['BUY'], ask_age_seconds=[300.0])
    out = engine.execute(panel)
    assert out['fill_price'].iloc[0] == pytest.approx(13.7)


def test_nan_age_treated_as_stale():
    """A NaN age clock with a non-NaN price is treated as effectively
    infinite staleness rather than a fillable quote."""
    engine = MinuteBarFillEngine()
    panel = _panel(['BUY'], ask_age_seconds=[np.nan])
    out = engine.execute(panel)
    assert out['rejection_reason'].iloc[0] == 'stale_quote'


# --------------------------------------------------------------------- #
# 3. No-bid / no-ask                                                    #
# --------------------------------------------------------------------- #


def test_no_ask_rejects_buy():
    engine = MinuteBarFillEngine()
    panel = _panel(['BUY', 'BUY'], last_ask=[np.nan, 0.0])
    out = engine.execute(panel)
    assert out['rejection_reason'].iloc[0] == 'no_ask'
    assert out['rejection_reason'].iloc[1] == 'no_ask'
    assert out['fill_price'].isna().all()


def test_no_bid_rejects_sell():
    engine = MinuteBarFillEngine()
    panel = _panel(['SELL', 'SELL'], last_bid=[np.nan, -1.0])
    out = engine.execute(panel)
    assert out['rejection_reason'].iloc[0] == 'no_bid'
    assert out['rejection_reason'].iloc[1] == 'no_bid'
    assert out['fill_price'].isna().all()


def test_no_bid_takes_precedence_over_staleness():
    """When both no_bid and stale_bid are true on a SELL, the more
    specific reason (no_bid) wins."""
    engine = MinuteBarFillEngine()
    panel = _panel(['SELL'], last_bid=[np.nan], bid_age_seconds=[600.0])
    out = engine.execute(panel)
    assert out['rejection_reason'].iloc[0] == 'no_bid'


# --------------------------------------------------------------------- #
# 4. Augmented frame & end-to-end merge contract                        #
# --------------------------------------------------------------------- #


def test_invalid_action_rejects():
    engine = MinuteBarFillEngine()
    panel = _panel(['HOLD'])  # invalid sentinel — must reject loudly
    out = engine.execute(panel)
    assert out['rejection_reason'].iloc[0] == 'invalid_action'
    assert pd.isna(out['fill_price'].iloc[0])


def test_missing_required_column_raises():
    engine = MinuteBarFillEngine()
    panel = _panel(['BUY']).drop(columns=['ask_age_seconds'])
    with pytest.raises(KeyError, match='ask_age_seconds'):
        engine.execute(panel)


def test_passes_through_other_columns_and_preserves_index():
    """End-to-end merge contract: the augmented frame retains the original
    OHLCV / regime / broadcaster columns and the original DatetimeIndex;
    only the three new fill columns are added."""
    engine = MinuteBarFillEngine()
    panel = _panel([None, 'BUY', 'SELL'])
    panel['close'] = [10.1, 10.2, 10.3]                # OHLCV passenger
    panel['state_label'] = [0, 0, 1]                    # regime passenger
    panel['as_of_effective'] = pd.NaT                   # broadcaster passenger

    out = engine.execute(panel)

    assert out.index.equals(panel.index)
    np.testing.assert_array_equal(
        out['close'].to_numpy(), panel['close'].to_numpy(),
    )
    np.testing.assert_array_equal(
        out['state_label'].to_numpy(), panel['state_label'].to_numpy(),
    )
    assert {'fill_price', 'fill_side', 'rejection_reason'}.issubset(out.columns)
    # No-action row: all three new fields NaN.
    assert pd.isna(out['fill_price'].iloc[0])
    assert pd.isna(out['fill_side'].iloc[0])
    assert pd.isna(out['rejection_reason'].iloc[0])
    # BUY row → fill at last_ask.
    assert out['fill_price'].iloc[1] == pytest.approx(13.7)
    assert out['fill_side'].iloc[1] == 'BUY'
    # SELL row → fill at last_bid.
    assert out['fill_price'].iloc[2] == pytest.approx(13.5)
    assert out['fill_side'].iloc[2] == 'SELL'


def test_input_panel_not_mutated():
    """The engine returns a copy; the caller's input frame must be
    untouched (no surprise side-effects on a passenger DataFrame)."""
    engine = MinuteBarFillEngine()
    panel = _panel(['BUY', 'SELL'])
    snapshot = panel.copy()
    _ = engine.execute(panel)
    pd.testing.assert_frame_equal(panel, snapshot)
