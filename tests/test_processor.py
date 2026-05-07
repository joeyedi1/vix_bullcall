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


# --------------------------------------------------------------------------- #
# align_option_quotes — schema, sizes, derived flags, staleness                #
# --------------------------------------------------------------------------- #


def _option_ticks(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a tick frame with the expected schema."""
    return pd.DataFrame(rows, columns=["ticker", "time", "type", "value", "size"])


def test_align_option_quotes_schema(proc, grid_6min):
    """Empty input still produces the OptionQuote-aligned schema."""
    out = proc.align_option_quotes(_option_ticks([]), grid_6min)
    assert list(out.columns) == [
        "bid", "ask", "bid_size", "ask_size", "last_trade",
        "quote_age_seconds", "last_trade_age_seconds",
        "is_locked", "is_crossed",
    ]
    assert len(out) == len(grid_6min)
    assert out["is_locked"].dtype == bool
    assert out["is_crossed"].dtype == bool


def test_align_option_quotes_picks_last_per_minute_and_ffills(proc, grid_6min):
    """Last BID/ASK in a minute wins; sizes track values; absent minutes ffill."""
    ticks = _option_ticks([
        # minute 0: bid 1.40 size 50, ask 1.50 size 60 (each updated mid-minute)
        ("VIX X", "2025-10-16 22:00:30", "BID", 1.40, 50),
        ("VIX X", "2025-10-16 22:00:45", "ASK", 1.50, 60),
        # minute 2: bid bumps to 1.45 (size 30); no new ask
        ("VIX X", "2025-10-16 22:02:10", "BID", 1.45, 30),
        # minute 3: ask drops to 1.48 (size 10)
        ("VIX X", "2025-10-16 22:03:50", "ASK", 1.48, 10),
    ])
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    out = proc.align_option_quotes(ticks, grid_6min)
    # Minute 0: fresh bid+ask
    assert out.loc["2025-10-16 22:00:00+00:00", "bid"] == pytest.approx(1.40)
    assert out.loc["2025-10-16 22:00:00+00:00", "ask"] == pytest.approx(1.50)
    assert out.loc["2025-10-16 22:00:00+00:00", "bid_size"] == 50
    assert out.loc["2025-10-16 22:00:00+00:00", "ask_size"] == 60
    # Minute 1: ffill — same as minute 0
    assert out.loc["2025-10-16 22:01:00+00:00", "bid"] == pytest.approx(1.40)
    assert out.loc["2025-10-16 22:01:00+00:00", "ask"] == pytest.approx(1.50)
    # Minute 2: bid updated, ask ffilled
    assert out.loc["2025-10-16 22:02:00+00:00", "bid"] == pytest.approx(1.45)
    assert out.loc["2025-10-16 22:02:00+00:00", "bid_size"] == 30
    assert out.loc["2025-10-16 22:02:00+00:00", "ask"] == pytest.approx(1.50)
    # Minute 3: ask updated
    assert out.loc["2025-10-16 22:03:00+00:00", "ask"] == pytest.approx(1.48)
    assert out.loc["2025-10-16 22:03:00+00:00", "ask_size"] == 10
    # Minute 5: both still ffilled to last seen
    assert out.loc["2025-10-16 22:05:00+00:00", "bid"] == pytest.approx(1.45)
    assert out.loc["2025-10-16 22:05:00+00:00", "ask"] == pytest.approx(1.48)


def test_align_option_quotes_is_locked_and_is_crossed(proc, grid_6min):
    """Derived flags fire only when both sides are non-null AND match the
    NBBO state."""
    ticks = _option_ticks([
        # minute 0: normal NBBO bid<ask
        ("VIX X", "2025-10-16 22:00:30", "BID", 1.40, 50),
        ("VIX X", "2025-10-16 22:00:45", "ASK", 1.50, 60),
        # minute 2: locked — bid bumps to 1.50 (== ask)
        ("VIX X", "2025-10-16 22:02:10", "BID", 1.50, 25),
        # minute 4: crossed — bid bumps further to 1.55 (> ask 1.50)
        ("VIX X", "2025-10-16 22:04:30", "BID", 1.55, 10),
    ])
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    out = proc.align_option_quotes(ticks, grid_6min)
    # Minute 0: normal — neither flag
    assert not out.loc["2025-10-16 22:00:00+00:00", "is_locked"]
    assert not out.loc["2025-10-16 22:00:00+00:00", "is_crossed"]
    # Minute 2: locked
    assert bool(out.loc["2025-10-16 22:02:00+00:00", "is_locked"])
    assert not out.loc["2025-10-16 22:02:00+00:00", "is_crossed"]
    # Minute 4: crossed (bid > ask after bid bump)
    assert not out.loc["2025-10-16 22:04:00+00:00", "is_locked"]
    assert bool(out.loc["2025-10-16 22:04:00+00:00", "is_crossed"])


def test_align_option_quotes_flags_false_when_either_side_missing(
    proc, grid_6min,
):
    """is_locked/is_crossed must be False (not NaN) on minutes with one
    side missing, even though `bid == ask` could be vacuously true if both
    were NaN."""
    ticks = _option_ticks([
        # bid only — ask remains NaN through the whole grid
        ("VIX X", "2025-10-16 22:01:00", "BID", 1.40, 50),
    ])
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    out = proc.align_option_quotes(ticks, grid_6min)
    assert (out["is_locked"] == False).all()  # noqa: E712
    assert (out["is_crossed"] == False).all()  # noqa: E712


def test_align_option_quotes_quote_age_is_max_of_sides(proc, grid_6min):
    """quote_age_seconds = max(bid_age, ask_age) so a staleness gate trips
    on the older side."""
    ticks = _option_ticks([
        ("VIX X", "2025-10-16 22:00:00", "BID", 1.40, 50),
        ("VIX X", "2025-10-16 22:00:00", "ASK", 1.50, 60),
        # bid is bumped at minute 3; ask stays stale
        ("VIX X", "2025-10-16 22:03:00", "BID", 1.42, 50),
    ])
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    out = proc.align_option_quotes(ticks, grid_6min)
    # Minute 3: bid_age=0 (fresh), ask_age=180 (3 minutes since 22:00)
    assert out.loc["2025-10-16 22:03:00+00:00", "quote_age_seconds"] == 180.0
    # Minute 4: bid_age=60, ask_age=240 → max=240
    assert out.loc["2025-10-16 22:04:00+00:00", "quote_age_seconds"] == 240.0
    # Minute 5: bid_age=120, ask_age=300 → max=300
    assert out.loc["2025-10-16 22:05:00+00:00", "quote_age_seconds"] == 300.0


def test_align_option_quotes_quote_age_nan_until_both_sides_observed(
    proc, grid_6min,
):
    """If only one side has been observed, the NBBO doesn't exist yet —
    quote_age_seconds is NaN, not the age of the lone observed side."""
    ticks = _option_ticks([
        ("VIX X", "2025-10-16 22:01:00", "BID", 1.40, 50),
        # No ASK ever observed
    ])
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    out = proc.align_option_quotes(ticks, grid_6min)
    assert out["quote_age_seconds"].isna().all()


def test_align_option_quotes_last_trade_age_independent(proc, grid_6min):
    """last_trade_age_seconds tracks TRADE events independently of BBO."""
    ticks = _option_ticks([
        ("VIX X", "2025-10-16 22:00:00", "BID", 1.40, 50),
        ("VIX X", "2025-10-16 22:00:00", "ASK", 1.50, 60),
        ("VIX X", "2025-10-16 22:02:00", "TRADE", 1.45, 5),
    ])
    ticks["time"] = pd.to_datetime(ticks["time"], utc=True)
    out = proc.align_option_quotes(ticks, grid_6min)
    # Minute 2: trade age 0; minute 5: trade age 180
    assert out.loc["2025-10-16 22:02:00+00:00", "last_trade_age_seconds"] == 0.0
    assert out.loc["2025-10-16 22:05:00+00:00", "last_trade_age_seconds"] == 180.0
    # Minute 0/1: NaN before first trade
    assert pd.isna(out.loc["2025-10-16 22:00:00+00:00", "last_trade_age_seconds"])


# --------------------------------------------------------------------------- #
# process_vix_index_options — multi-contract panel assembly                   #
# --------------------------------------------------------------------------- #


def test_process_vix_index_options_panel_shape(tmp_path, proc):
    """Two contracts, two parquets, expect MultiIndex (timestamp, contract_id)."""
    raw_root = tmp_path / "data" / "raw"
    d = raw_root / "blpapi" / "vix_index_options_quotes"
    d.mkdir(parents=True)
    vintage = "20260505T100000Z"

    def _save(ticker: str, rows: list[dict]) -> None:
        df = pd.DataFrame(rows)
        df["ticker"] = ticker
        # safe_shard_key strips spaces/slashes to underscores; the loader
        # recovers the ticker from the in-data column, not the filename.
        safe = "VIX_US_05_19_26_C20_Index" if "20" in ticker else "VIX_US_05_19_26_C21_Index"
        df.to_parquet(d / f"{safe}_{vintage}.parquet", index=False)

    rows_a = [
        {"time": pd.Timestamp("2026-04-01 14:00:30", tz="UTC"),
         "type": "BID", "value": 1.40, "size": 50},
        {"time": pd.Timestamp("2026-04-01 14:00:45", tz="UTC"),
         "type": "ASK", "value": 1.50, "size": 60},
    ]
    rows_b = [
        {"time": pd.Timestamp("2026-04-01 14:00:30", tz="UTC"),
         "type": "BID", "value": 1.10, "size": 30},
        {"time": pd.Timestamp("2026-04-01 14:00:45", tz="UTC"),
         "type": "ASK", "value": 1.20, "size": 25},
    ]
    _save("VIX US 05/19/26 C20 Index", rows_a)
    _save("VIX US 05/19/26 C21 Index", rows_b)

    p = DataProcessor(raw_root=raw_root)
    panel = p.process_vix_index_options()

    assert isinstance(panel.index, pd.MultiIndex)
    assert panel.index.names == ["timestamp", "contract_id"]
    expected_cols = {
        "bid", "ask", "bid_size", "ask_size", "last_trade",
        "quote_age_seconds", "last_trade_age_seconds",
        "is_locked", "is_crossed",
    }
    assert set(panel.columns) == expected_cols
    # Both contracts present
    contract_ids = set(panel.index.get_level_values("contract_id"))
    assert contract_ids == {
        "VIX US 05/19/26 C20 Index",
        "VIX US 05/19/26 C21 Index",
    }
    # Spot-check values for one contract
    row_a = panel.loc[(pd.Timestamp("2026-04-01 14:00:00", tz="UTC"),
                       "VIX US 05/19/26 C20 Index")]
    assert row_a["bid"] == pytest.approx(1.40)
    assert row_a["ask"] == pytest.approx(1.50)
    assert row_a["bid_size"] == 50
    assert row_a["ask_size"] == 60


def test_process_vix_index_options_returns_empty_when_no_shards(tmp_path):
    raw_root = tmp_path / "data" / "raw"
    p = DataProcessor(raw_root=raw_root)
    out = p.process_vix_index_options()
    assert out.empty


# --------------------------------------------------------------------------- #
# _load_shards — recovers ticker from in-data column                          #
# --------------------------------------------------------------------------- #


def test_load_shards_recovers_multitoken_ticker_from_data_column(tmp_path):
    """`safe_shard_key` mangles the VIX option ticker; the loader must
    recover it from the in-data `ticker` column rather than the filename."""
    raw_root = tmp_path / "data" / "raw"
    d = raw_root / "blpapi" / "vix_index_options_quotes"
    d.mkdir(parents=True)
    vintage = "20260505T100000Z"
    ticker = "VIX US 05/19/26 C20 Index"  # has spaces AND slashes
    df = pd.DataFrame({
        "time": [pd.Timestamp("2026-04-01 14:00:30", tz="UTC")],
        "type": ["BID"], "value": [1.4], "size": [50],
        "ticker": [ticker],
    })
    df.to_parquet(d / f"VIX_US_05_19_26_C20_Index_{vintage}.parquet", index=False)

    p = DataProcessor(raw_root=raw_root)
    shards = p._load_shards("vix_index_options_quotes", vintage)
    assert ticker in shards
    assert len(shards) == 1


# --------------------------------------------------------------------------- #
# process_vix_options_daily — pivots long-form daily shards into the         #
# ChainIVProvider-shaped MultiIndex panel.                                    #
# --------------------------------------------------------------------------- #


def _save_daily_shard(
    out_dir, ticker: str, vintage: str,
    rows: list[tuple],   # (date, field, value)
) -> None:
    """Helper: write one per-contract daily parquet in long form."""
    df = pd.DataFrame(
        [
            {"date": pd.Timestamp(d_), "ticker": ticker, "field": f, "value": v}
            for (d_, f, v) in rows
        ]
    )
    safe = ticker.replace(" ", "_").replace("/", "_")
    df.to_parquet(out_dir / f"{safe}_{vintage}.parquet", index=False)


def test_process_vix_options_daily_schema_and_pivot(tmp_path):
    """Per-(date, field) values pivoted into columns; (expiry, right,
    strike) added from the parsed ticker; MultiIndex names exact."""
    from datetime import date as _date

    raw = tmp_path / "data" / "raw"
    d = raw / "blpapi" / "vix_index_options_daily"
    d.mkdir(parents=True)
    vintage = "20260505T100000Z"

    _save_daily_shard(
        d, "VIX US 04/15/26 C20 Index", vintage,
        [
            ("2026-04-01", "IVOL_LAST", 60.0),
            ("2026-04-01", "PX_BID",     1.40),
            ("2026-04-01", "PX_ASK",     1.50),
            ("2026-04-02", "IVOL_LAST", 61.0),
            ("2026-04-02", "PX_BID",     1.42),
            ("2026-04-02", "PX_ASK",     1.52),
        ],
    )

    p = DataProcessor(raw_root=raw)
    panel = p.process_vix_options_daily()

    assert isinstance(panel.index, pd.MultiIndex)
    assert panel.index.names == ["date", "expiry", "right", "strike"]
    for col in ("IVOL_LAST", "PX_BID", "PX_ASK"):
        assert col in panel.columns

    # Day-1 row: pivot resolved correctly
    key = (_date(2026, 4, 1), _date(2026, 4, 15), "C", 20.0)
    assert key in panel.index
    assert panel.loc[key, "IVOL_LAST"] == 60.0
    assert panel.loc[key, "PX_BID"] == pytest.approx(1.40)
    assert panel.loc[key, "PX_ASK"] == pytest.approx(1.50)

    # Day-2 row: independent
    assert panel.loc[(_date(2026, 4, 2), _date(2026, 4, 15), "C", 20.0), "PX_BID"] == pytest.approx(1.42)


def test_process_vix_options_daily_normalizes_active_ticker_date_to_soq_wednesday(
    tmp_path,
):
    """Settled tickers encode SOQ Wednesday; active tickers encode the
    Tuesday last-trade. The panel must normalize ALL expiries to the
    SOQ Wednesday so ChainIVProvider lookups (keyed on product.expiry.date()
    = SOQ Wed) succeed regardless of contract state at pull time.
    """
    from datetime import date as _date

    raw = tmp_path / "data" / "raw"
    d = raw / "blpapi" / "vix_index_options_daily"
    d.mkdir(parents=True)
    vintage = "20260505T100000Z"

    # SETTLED: April 2026 monthly — Bloomberg ticker date = SOQ Wed 4/15.
    _save_daily_shard(
        d, "VIX US 04/15/26 C20 Index", vintage,
        [("2026-04-01", "IVOL_LAST", 60.0),
         ("2026-04-01", "PX_BID", 1.40),
         ("2026-04-01", "PX_ASK", 1.50)],
    )
    # ACTIVE: May 2026 monthly — Bloomberg ticker date = Tuesday 5/19,
    # SOQ Wednesday is 5/20 (vx_settlement_date(2026, 5)).
    _save_daily_shard(
        d, "VIX US 05/19/26 P18 Index", vintage,
        [("2026-04-01", "IVOL_LAST", 55.0),
         ("2026-04-01", "PX_BID", 0.80),
         ("2026-04-01", "PX_ASK", 0.85)],
    )

    panel = DataProcessor(raw_root=raw).process_vix_options_daily()

    # Settled — ticker date already Wed; expiry index unchanged.
    settled = (_date(2026, 4, 1), _date(2026, 4, 15), "C", 20.0)
    assert settled in panel.index
    assert panel.loc[settled, "IVOL_LAST"] == 60.0

    # Active — ticker date 5/19 (Tue); expiry NORMALIZED to 5/20 (Wed SOQ).
    active = (_date(2026, 4, 1), _date(2026, 5, 20), "P", 18.0)
    assert active in panel.index, "active expiry should normalize to SOQ Wed"
    assert panel.loc[active, "IVOL_LAST"] == 55.0
    # The Tuesday key should NOT be present.
    bad = (_date(2026, 4, 1), _date(2026, 5, 19), "P", 18.0)
    assert bad not in panel.index


def test_process_vix_options_daily_skips_unparseable_tickers(tmp_path):
    """Weeklies / AM-PM variants whose ticker form doesn't match the
    standard regex are silently skipped (same convention as the ingestion
    `filter_chain`)."""
    from datetime import date as _date

    raw = tmp_path / "data" / "raw"
    d = raw / "blpapi" / "vix_index_options_daily"
    d.mkdir(parents=True)
    vintage = "20260505T100000Z"

    _save_daily_shard(
        d, "VIX US 04/15/26 C20 Index", vintage,
        [("2026-04-01", "IVOL_LAST", 60.0),
         ("2026-04-01", "PX_BID", 1.40),
         ("2026-04-01", "PX_ASK", 1.50)],
    )
    # An obviously-non-conforming ticker (still saved by the ingestion
    # path because it carries the column).
    _save_daily_shard(
        d, "GARBAGE TICKER", vintage,
        [("2026-04-01", "IVOL_LAST", 99.0)],
    )

    panel = DataProcessor(raw_root=raw).process_vix_options_daily()
    # Standard one survives, garbage one dropped.
    assert (_date(2026, 4, 1), _date(2026, 4, 15), "C", 20.0) in panel.index
    assert len(panel) == 1


def test_process_vix_options_daily_returns_empty_when_no_shards(tmp_path):
    raw = tmp_path / "data" / "raw"
    panel = DataProcessor(raw_root=raw).process_vix_options_daily()
    assert panel.empty


def test_process_vix_options_daily_handles_multiple_strikes_and_rights(tmp_path):
    """A realistic multi-contract pull: 2 strikes × 2 rights × 1 expiry."""
    from datetime import date as _date

    raw = tmp_path / "data" / "raw"
    d = raw / "blpapi" / "vix_index_options_daily"
    d.mkdir(parents=True)
    vintage = "20260505T100000Z"

    for strike in (18, 20):
        for right_letter, mid in [("C", 1.5), ("P", 0.9)]:
            _save_daily_shard(
                d, f"VIX US 04/15/26 {right_letter}{strike} Index", vintage,
                [
                    ("2026-04-01", "IVOL_LAST", 50.0 + strike),
                    ("2026-04-01", "PX_BID", mid - 0.05),
                    ("2026-04-01", "PX_ASK", mid + 0.05),
                ],
            )

    panel = DataProcessor(raw_root=raw).process_vix_options_daily()
    assert len(panel) == 4
    # Spot-check one
    key = (_date(2026, 4, 1), _date(2026, 4, 15), "C", 18.0)
    assert panel.loc[key, "IVOL_LAST"] == 68.0
    # Index sorted
    assert list(panel.index.get_level_values("right").unique()) == ["C", "P"]


def test_process_vix_options_daily_panel_satisfies_chain_iv_provider(tmp_path):
    """End-to-end: the panel produced here must construct + serve a
    ChainIVProvider lookup without further massaging. This is the
    integration contract."""
    from datetime import date as _date, datetime as _dt, timezone as _tz

    from vix_spread.pricing.black76 import Black76Pricer
    from vix_spread.pricing.forward_selector import Forward
    from vix_spread.pricing.leg_iv import ChainIVProvider, LegIVSource
    from vix_spread.products.vix_index_option import VIXIndexOption

    raw = tmp_path / "data" / "raw"
    d = raw / "blpapi" / "vix_index_options_daily"
    d.mkdir(parents=True)
    vintage = "20260505T100000Z"

    _save_daily_shard(
        d, "VIX US 04/15/26 C20 Index", vintage,
        [("2026-04-01", "IVOL_LAST", 70.0),
         ("2026-04-01", "PX_BID", 1.40),
         ("2026-04-01", "PX_ASK", 1.50)],
    )

    panel = DataProcessor(raw_root=raw).process_vix_options_daily()
    pricer = Black76Pricer()
    provider = ChainIVProvider(panel, pricer)

    settlement = _dt(2026, 4, 15, 14, 30, tzinfo=_tz.utc)
    product = VIXIndexOption(
        contract_root="VIX",
        expiry=settlement,                # SOQ Wed — matches normalized expiry index
        settlement_event=settlement,
        strike=20.0,
        right="call",
    )
    forward = Forward(
        value=22.0, selection_method="settlement_date_match",
        model_risk_flag=False, settlement_date=settlement,
    )
    iv = provider.get(
        product=product,
        as_of=_dt(2026, 4, 1, 14, 0, tzinfo=_tz.utc),
        forward=forward, risk_free_rate=0.04,
    )
    assert iv.source == LegIVSource.VENDOR
    assert iv.value == pytest.approx(0.70)  # 70.0 / 100
