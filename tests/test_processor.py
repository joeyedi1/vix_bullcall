"""Tests for DataProcessor.

Three behavioural contracts under test:
  1. OHLCV is reindexed but NEVER forward-filled.
  2. The `trade_age_seconds` staleness clock is correct (0 on observation,
     +60s per missing minute, NaN before the very first observation).
  3. Missing minutes are handled cleanly: NaN OHLC, volume=0, numEvents=0.

All tests use synthetic in-memory frames — no parquet, no Bloomberg.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from vix_spread.data.processor import DataProcessor, ProcessedContract


@pytest.fixture
def proc() -> DataProcessor:
    return DataProcessor(raw_root="data/raw")  # path unused in these tests


@pytest.fixture
def grid_6min(proc: DataProcessor) -> pd.DatetimeIndex:
    """Grid covering minutes 0..5 of a fixed UTC anchor."""
    return proc.build_master_grid(
        datetime(2025, 10, 16, 22, 0),
        datetime(2025, 10, 16, 22, 5),
    )


# --------------------------------------------------------------------------- #
# (1) OHLCV reindex contract: NO forward fill.                                #
# --------------------------------------------------------------------------- #


def test_align_bars_does_not_forward_fill_ohlc(proc, grid_6min):
    """Sparse bars at minutes 0, 2, 4 must yield NaN OHLC at 1, 3, 5 — never
    a repeat of the prior close."""
    bars = pd.DataFrame({
        "time": pd.to_datetime(
            ["2025-10-16 22:00", "2025-10-16 22:02", "2025-10-16 22:04"], utc=True,
        ),
        "open":  [10.0, 11.0, 12.0],
        "high":  [10.5, 11.5, 12.5],
        "low":   [ 9.5, 10.5, 11.5],
        "close": [10.2, 11.2, 12.2],
        "volume":   [100, 200, 300],
        "numEvents": [5, 10, 15],
    })

    out = proc.align_bars(bars, grid_6min)

    # Observation minutes carry the original OHLC.
    assert out.loc["2025-10-16 22:00:00+00:00", "close"] == pytest.approx(10.2)
    assert out.loc["2025-10-16 22:02:00+00:00", "close"] == pytest.approx(11.2)
    assert out.loc["2025-10-16 22:04:00+00:00", "close"] == pytest.approx(12.2)

    # Missing minutes are NaN — NOT ffilled from the prior close.
    for missing_minute in ("22:01", "22:03", "22:05"):
        ts = f"2025-10-16 {missing_minute}:00+00:00"
        for col in ("open", "high", "low", "close"):
            assert pd.isna(out.loc[ts, col]), f"{col} at {missing_minute} should be NaN, got {out.loc[ts, col]}"


def test_align_bars_missing_minutes_have_zero_volume(proc, grid_6min):
    """volume and numEvents are zero (not NaN) on minutes with no bar."""
    bars = pd.DataFrame({
        "time": pd.to_datetime(["2025-10-16 22:00", "2025-10-16 22:04"], utc=True),
        "open": [10.0, 12.0], "high": [10.5, 12.5],
        "low":  [ 9.5, 11.5], "close":[10.2, 12.2],
        "volume": [100, 300], "numEvents": [5, 15],
    })
    out = proc.align_bars(bars, grid_6min)

    for ts in [f"2025-10-16 22:0{m}:00+00:00" for m in (1, 2, 3, 5)]:
        assert out.loc[ts, "volume"] == 0
        assert out.loc[ts, "numEvents"] == 0
    assert out["volume"].dtype == np.int64
    assert out["numEvents"].dtype == np.int64


def test_align_bars_empty_frame_returns_grid_sized_nan_frame(proc, grid_6min):
    """Empty input -> output has the grid index, all OHLC NaN, all integer counts zero-able."""
    out = proc.align_bars(pd.DataFrame(), grid_6min)
    assert len(out) == len(grid_6min)
    assert out.index.equals(grid_6min)
    assert set(out.columns) == {"open", "high", "low", "close", "volume", "numEvents"}
    # All OHLC NaN since there's no input.
    assert out[["open", "high", "low", "close"]].isna().all().all()


# --------------------------------------------------------------------------- #
# (2) trade_age_seconds staleness clock.                                       #
# --------------------------------------------------------------------------- #


def test_trade_age_zero_on_observation_minute(proc, grid_6min):
    ticks = pd.DataFrame({
        "time":  pd.to_datetime(["2025-10-16 22:02:30"], utc=True),
        "type":  ["TRADE"],
        "value": [99.5],
    })
    out = proc.align_quotes(ticks, grid_6min)
    # The minute that contained the trade.
    assert out.loc["2025-10-16 22:02:00+00:00", "trade_age_seconds"] == 0.0
    assert out.loc["2025-10-16 22:02:00+00:00", "last_trade"] == pytest.approx(99.5)


def test_trade_age_increments_60s_per_missing_minute(proc, grid_6min):
    """One trade at minute 2 → age 0,60,120,180 across minutes 2,3,4,5."""
    ticks = pd.DataFrame({
        "time": pd.to_datetime(["2025-10-16 22:02:00"], utc=True),
        "type":  ["TRADE"],
        "value": [42.0],
    })
    out = proc.align_quotes(ticks, grid_6min)
    expected = {
        "2025-10-16 22:02:00+00:00": 0.0,
        "2025-10-16 22:03:00+00:00": 60.0,
        "2025-10-16 22:04:00+00:00": 120.0,
        "2025-10-16 22:05:00+00:00": 180.0,
    }
    for ts, age in expected.items():
        assert out.loc[ts, "trade_age_seconds"] == age, f"age mismatch at {ts}"


def test_trade_age_resets_to_zero_on_new_observation(proc, grid_6min):
    """Trades at minute 1 and minute 4: age must be 0 at both, with the
    in-between rising 0,60,120,0,...."""
    ticks = pd.DataFrame({
        "time": pd.to_datetime(
            ["2025-10-16 22:01:15", "2025-10-16 22:04:45"], utc=True,
        ),
        "type":  ["TRADE", "TRADE"],
        "value": [10.0, 20.0],
    })
    out = proc.align_quotes(ticks, grid_6min)
    seq = out["trade_age_seconds"].tolist()
    # Pre-first-trade (minute 0): NaN. Then 0, 60, 120, 0, 60.
    assert pd.isna(seq[0])
    assert seq[1:] == [0.0, 60.0, 120.0, 0.0, 60.0]


def test_trade_age_nan_before_first_observation(proc, grid_6min):
    """Until the first trade arrives, trade_age_seconds is NaN — there is
    no 'last known' value to be stale relative to."""
    ticks = pd.DataFrame({
        "time": pd.to_datetime(["2025-10-16 22:03:00"], utc=True),
        "type":  ["TRADE"],
        "value": [50.0],
    })
    out = proc.align_quotes(ticks, grid_6min)
    for ts in ("2025-10-16 22:00:00+00:00", "2025-10-16 22:01:00+00:00",
               "2025-10-16 22:02:00+00:00"):
        assert pd.isna(out.loc[ts, "trade_age_seconds"])
        assert pd.isna(out.loc[ts, "last_trade"])
    # First observation onward: age=0,60,120.
    assert out.loc["2025-10-16 22:03:00+00:00", "trade_age_seconds"] == 0.0
    assert out.loc["2025-10-16 22:05:00+00:00", "trade_age_seconds"] == 120.0


def test_bid_ask_ages_track_independently(proc, grid_6min):
    """Each event type has its own staleness clock."""
    ticks = pd.DataFrame({
        "time": pd.to_datetime(
            ["2025-10-16 22:01:00",  # bid only
             "2025-10-16 22:03:00"], # ask only
            utc=True,
        ),
        "type":  ["BID", "ASK"],
        "value": [9.5, 10.5],
    })
    out = proc.align_quotes(ticks, grid_6min)
    # Bid clock: NaN at 22:00, then 0/60/120/180/240 at 22:01..22:05.
    assert pd.isna(out.loc["2025-10-16 22:00:00+00:00", "bid_age_seconds"])
    assert out.loc["2025-10-16 22:01:00+00:00", "bid_age_seconds"] == 0.0
    assert out.loc["2025-10-16 22:05:00+00:00", "bid_age_seconds"] == 240.0
    # Ask clock: NaN at 22:00..22:02, then 0/60/120 at 22:03..22:05.
    for ts in ("2025-10-16 22:00:00+00:00", "2025-10-16 22:01:00+00:00",
               "2025-10-16 22:02:00+00:00"):
        assert pd.isna(out.loc[ts, "ask_age_seconds"])
    assert out.loc["2025-10-16 22:03:00+00:00", "ask_age_seconds"] == 0.0
    assert out.loc["2025-10-16 22:05:00+00:00", "ask_age_seconds"] == 120.0


# --------------------------------------------------------------------------- #
# Supporting tests — quote forward-fill, settlement parser, grid tz.          #
# --------------------------------------------------------------------------- #


def test_align_quotes_forward_fills_last_known_value(proc, grid_6min):
    """last_bid / last_ask are forward-filled (unlike OHLC). A bid at 22:01
    must persist as last_bid through 22:05 even with no new bid events."""
    ticks = pd.DataFrame({
        "time": pd.to_datetime(["2025-10-16 22:01:00"], utc=True),
        "type":  ["BID"],
        "value": [13.5],
    })
    out = proc.align_quotes(ticks, grid_6min)
    for ts in [f"2025-10-16 22:0{m}:00+00:00" for m in (1, 2, 3, 4, 5)]:
        assert out.loc[ts, "last_bid"] == pytest.approx(13.5), f"ffill failed at {ts}"


def test_align_quotes_empty_frame_returns_grid_sized_nan_frame(proc, grid_6min):
    out = proc.align_quotes(pd.DataFrame(), grid_6min)
    assert len(out) == len(grid_6min)
    assert out.index.equals(grid_6min)
    assert out.isna().all().all()


def test_build_master_grid_treats_naive_as_utc(proc):
    g = proc.build_master_grid(
        datetime(2025, 10, 16, 22, 0),
        datetime(2025, 10, 16, 22, 3),
    )
    assert str(g.tz) == "UTC"
    assert len(g) == 4
    assert g[0] == pd.Timestamp("2025-10-16 22:00:00", tz="UTC")
    assert g[-1] == pd.Timestamp("2025-10-16 22:03:00", tz="UTC")


@pytest.mark.parametrize(
    "ticker,expected",
    [
        ("UXM25 Index", (2025, 6, 18)),
        ("UXZ25 Index", (2025, 12, 17)),
        ("UXF26 Index", (2026, 1, 21)),
        ("UXM24 Index", (2024, 6, 18)),  # Juneteenth rollback
    ],
)
def test_settlement_for_ticker(ticker, expected):
    settled = DataProcessor._settlement_for_ticker(ticker)
    assert (settled.year, settled.month, settled.day) == expected


@pytest.mark.parametrize(
    "bad_ticker",
    [
        "VXM25 Index",      # wrong root
        "UXM5 Index",       # single-digit year (deprecated form)
        "UXM25",            # missing 'Index'
        "UX25 Index",       # missing month code
    ],
)
def test_settlement_for_ticker_rejects_malformed(bad_ticker):
    with pytest.raises(ValueError):
        DataProcessor._settlement_for_ticker(bad_ticker)
