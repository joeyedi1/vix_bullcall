"""DataProcessor — load vintage-tagged raw shards, align to a 1-min master grid.

Phase-2 entry point. Consumes the per-contract parquet shards produced by
`scripts/pull_data.py` (one shard per ticker, one vintage per pull) and
emits a `ProcessedContract` per VX futures contract: 1-min OHLCV bars and
per-minute last-known BBO/trade snapshots, both aligned to a common UTC
master grid. The settlement-date for each contract is resolved through
`expiry_calendar.vx_settlement_date` so downstream code (ForwardSelector,
the option-chain joiner) can key on settlement date rather than ticker
strings (ARCHITECTURE §4.1, §6.1).

Design choices
--------------
1. **UTC master grid.** Bloomberg returns UTC timestamps. We retain UTC
   throughout the processed layer; conversion to exchange-local CT happens
   only at display time. This avoids the daylight-savings ambiguity at the
   alignment layer.

2. **No session masking in this scaffold.** CFE VX runs ~23h/day during
   the trading week (RTH 08:30-15:15 CT + ETH 15:30-15:15 next day with a
   15-min daily halt). The grid spans every minute between observed first
   and last events; downstream consumers apply session masks (e.g. RTH-only
   for liquidity gates) on top. Reason: liquidity / staleness gating is a
   FillEngine concern, not a data-layer concern.

3. **Sparse-event reindexing.** The two raw schemas are different and
   handled separately:
     - 1-min OHLCV bars: Bloomberg returns rows only for minutes with at
       least one event. Reindex to grid; missing minutes get NaN OHLC and
       volume=0. We do NOT forward-fill OHLCV — a missing minute is missing,
       not "same close as last minute". That distinction matters for
       volume-weighted analytics later.
     - Tick BBO/trade events: pivot by event type (BID/ASK/TRADE), take the
       LAST value per minute, forward-fill across the grid to produce a
       continuous "last-known quote" series. This forms the input for
       `OptionQuote` synthetic NBBO construction (ARCHITECTURE §5).
       Forward-filling without a staleness signal would leak old data into
       fresh decisions; we expose `*_age_seconds` columns so the FillEngine
       can apply `is_stale(max_age_seconds=...)` gates.

4. **Latest vintage only.** `process_vx_futures()` reads the most recent
   vintage by default. Multi-vintage replay (for audit / FeatureAvailability)
   is a separate consumer pattern and is NOT part of this scaffold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .expiry_calendar import CODE_TO_MONTH, vx_settlement_date


_TICKER_RE = re.compile(r"^UX([A-Z])(\d{2})\s+Index$")


@dataclass(frozen=True)
class ProcessedContract:
    """One VX contract, fully aligned to the master 1-minute grid.

    Attributes
    ----------
    ticker
        Bloomberg security string, e.g. "UXM25 Index".
    settlement_date
        CFE final settlement date, resolved via the expiry calendar.
    bars
        DataFrame indexed on the master grid; columns:
        open, high, low, close, volume, numEvents.
        NaN OHLC and volume=0 on minutes with no Bloomberg bar.
    quotes
        DataFrame indexed on the master grid; columns:
        last_bid, last_ask, last_trade, bid_age_seconds, ask_age_seconds,
        trade_age_seconds. Forward-filled within the grid; ages-since-last-
        update populated so downstream FillEngine staleness gates work.
        `None` if no quote shard was available for this ticker.
    """
    ticker: str
    settlement_date: date
    bars: pd.DataFrame
    quotes: pd.DataFrame | None


class DataProcessor:
    """Loads raw VX-futures shards and returns per-contract aligned panels."""

    SOURCE = "blpapi"
    OHLCV_PRODUCT = "vx_futures_ohlcv"
    QUOTES_PRODUCT = "vx_futures_quotes"
    GRID_FREQ = "1min"
    GRID_FREQ_SECONDS = 60

    def __init__(self, raw_root: Path | str = "data/raw") -> None:
        self.raw_root = Path(raw_root)

    # ---------------------------------------------------------------- #
    # Vintage discovery & shard loading                                #
    # ---------------------------------------------------------------- #

    def latest_vintage(self, product: str) -> str:
        d = self.raw_root / self.SOURCE / product
        if not d.exists():
            raise FileNotFoundError(f"No raw directory: {d}")
        files = list(d.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {d}")
        # Vintage is the trailing "YYYYMMDDTHHMMSSZ" component of the stem.
        vintages = {f.stem.rsplit("_", 1)[-1] for f in files}
        return max(vintages)

    def load_ohlcv_shards(self, vintage: str | None = None) -> dict[str, pd.DataFrame]:
        return self._load_shards(self.OHLCV_PRODUCT, vintage)

    def load_quote_shards(self, vintage: str | None = None) -> dict[str, pd.DataFrame]:
        return self._load_shards(self.QUOTES_PRODUCT, vintage)

    def _load_shards(self, product: str, vintage: str | None) -> dict[str, pd.DataFrame]:
        if vintage is None:
            vintage = self.latest_vintage(product)
        d = self.raw_root / self.SOURCE / product
        out: dict[str, pd.DataFrame] = {}
        for f in sorted(d.glob(f"*_{vintage}.parquet")):
            ticker_safe = f.stem.removesuffix(f"_{vintage}")
            # Reverse the BaseDataFetcher safe-shard transformation: the
            # ticker had its single space replaced with underscore, so
            # restore the FIRST underscore.
            ticker = ticker_safe.replace("_", " ", 1)
            out[ticker] = pd.read_parquet(f)
        return out

    # ---------------------------------------------------------------- #
    # Master grid                                                      #
    # ---------------------------------------------------------------- #

    def build_master_grid(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DatetimeIndex:
        """1-minute UTC index from `start` to `end` inclusive.

        Both bounds are floored / ceiled to the minute. Naive timestamps
        are treated as UTC (Bloomberg convention).
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return pd.date_range(
            start=pd.Timestamp(start).floor(self.GRID_FREQ),
            end=pd.Timestamp(end).ceil(self.GRID_FREQ),
            freq=self.GRID_FREQ,
            tz="UTC",
        )

    # ---------------------------------------------------------------- #
    # Per-shard alignment                                              #
    # ---------------------------------------------------------------- #

    def align_bars(
        self,
        bars: pd.DataFrame,
        grid: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Reindex 1-min OHLCV to the master grid.

        Output columns: open, high, low, close, volume, numEvents.
        Missing minutes: NaN OHLC, volume=0, numEvents=0. We deliberately
        do NOT forward-fill OHLC — a missing minute is missing, not
        "previous close repeated".
        """
        cols = ["open", "high", "low", "close", "volume", "numEvents"]
        if bars is None or bars.empty:
            return pd.DataFrame(
                {c: pd.Series(dtype="float64") for c in cols},
                index=grid,
            )
        idx = pd.to_datetime(bars["time"], utc=True)
        df = (
            bars.assign(_idx=idx)
            .drop(columns=["time"], errors="ignore")
            .set_index("_idx")
            .sort_index()
            .loc[:, cols]
        )
        # If multiple raw bars share a minute (rare), keep the last.
        df = df[~df.index.duplicated(keep="last")]
        out = df.reindex(grid)
        out["volume"] = out["volume"].fillna(0).astype("int64")
        out["numEvents"] = out["numEvents"].fillna(0).astype("int64")
        return out

    def align_quotes(
        self,
        ticks: pd.DataFrame,
        grid: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Per-minute last-known quote, forward-filled, with staleness ages.

        Strategy
        --------
        1. Group raw ticks by event type (BID / ASK / TRADE).
        2. For each type: resample to 1-min, take the LAST observation per
           minute, then reindex to the master grid. This gives the
           per-minute "fresh" observation series (NaN where no event in
           that minute).
        3. Forward-fill across the grid to produce a continuous
           last-known-value series.
        4. Compute the seconds-since-last-update for each event type. The
           FillEngine's staleness gate (ARCHITECTURE §5.3) consumes these
           directly via `OptionQuote.is_stale(max_age_seconds=...)`.

        Output columns:
          last_bid, last_ask, last_trade,
          bid_age_seconds, ask_age_seconds, trade_age_seconds.
        """
        cols = [
            "last_bid",
            "last_ask",
            "last_trade",
            "bid_age_seconds",
            "ask_age_seconds",
            "trade_age_seconds",
        ]
        if ticks is None or ticks.empty:
            return pd.DataFrame(
                {c: pd.Series(dtype="float64") for c in cols},
                index=grid,
            )

        idx = pd.to_datetime(ticks["time"], utc=True)
        df = (
            ticks.assign(_idx=idx)
            .drop(columns=["time"], errors="ignore")
            .set_index("_idx")
            .sort_index()
        )

        out = pd.DataFrame(index=grid)
        for event_type, label in (("BID", "bid"), ("ASK", "ask"), ("TRADE", "trade")):
            fresh = (
                df.loc[df["type"] == event_type, "value"]
                .resample(self.GRID_FREQ)
                .last()
                .reindex(grid)
            )
            out[f"last_{label}"] = fresh.ffill()
            out[f"{label}_age_seconds"] = self._age_seconds(fresh)

        return out[cols]

    def _age_seconds(self, fresh: pd.Series) -> pd.Series:
        """Seconds since the last non-null value in `fresh`.

        `fresh` is the per-minute "new observation" series (NaN on minutes
        with no event). The age series is 0 on observation minutes and
        increments by `GRID_FREQ_SECONDS` per minute thereafter; remains
        NaN before the very first observation.
        """
        # Each non-null starts a new "since-update" group. cumsum on the
        # boolean mask labels each group with a unique integer; cumcount
        # within each group counts ticks since the group's first member.
        observed = fresh.notna()
        if not observed.any():
            return pd.Series(np.nan, index=fresh.index)
        group = observed.cumsum()
        ticks_since = fresh.groupby(group).cumcount()
        ages = (ticks_since * self.GRID_FREQ_SECONDS).astype("float64")
        # Pre-first-observation rows: group is 0, mark NaN.
        ages[group == 0] = np.nan
        return ages

    # ---------------------------------------------------------------- #
    # Top-level                                                        #
    # ---------------------------------------------------------------- #

    def process_vx_futures(
        self,
        ohlcv_vintage: str | None = None,
        quotes_vintage: str | None = None,
    ) -> dict[str, ProcessedContract]:
        """Load, align, and pair OHLCV + quotes per contract.

        Returns dict keyed on Bloomberg ticker (e.g. "UXM25 Index").
        OHLCV and quote vintages may differ (independent pulls); both
        default to the latest available for their product.
        """
        ohlcv_shards = self.load_ohlcv_shards(ohlcv_vintage)
        try:
            quote_shards = self.load_quote_shards(quotes_vintage)
        except FileNotFoundError:
            quote_shards = {}

        if not ohlcv_shards and not quote_shards:
            return {}

        spans: list[pd.Timestamp] = []
        for shard_dict in (ohlcv_shards, quote_shards):
            for df in shard_dict.values():
                if "time" in df.columns and len(df):
                    times = pd.to_datetime(df["time"], utc=True)
                    spans.append(times.min())
                    spans.append(times.max())
        if not spans:
            return {}
        grid = self.build_master_grid(min(spans).to_pydatetime(), max(spans).to_pydatetime())

        out: dict[str, ProcessedContract] = {}
        all_tickers = set(ohlcv_shards) | set(quote_shards)
        for ticker in sorted(all_tickers):
            settlement = self._settlement_for_ticker(ticker)
            bars = self.align_bars(ohlcv_shards.get(ticker, pd.DataFrame()), grid)
            quotes_df = quote_shards.get(ticker)
            quotes = self.align_quotes(quotes_df, grid) if quotes_df is not None else None
            out[ticker] = ProcessedContract(
                ticker=ticker,
                settlement_date=settlement,
                bars=bars,
                quotes=quotes,
            )
        return out

    # ---------------------------------------------------------------- #
    # Ticker -> settlement_date                                        #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _settlement_for_ticker(ticker: str) -> date:
        """Resolve `'UXM25 Index'` → June 18 2025 via the expiry calendar."""
        m = _TICKER_RE.match(ticker)
        if m is None:
            raise ValueError(
                f"Cannot parse VX ticker {ticker!r} — expected 'UX{{F..Z}}{{YY}} Index'."
            )
        month_code, yy = m.group(1), int(m.group(2))
        if month_code not in CODE_TO_MONTH:
            raise ValueError(f"Unknown VX month code {month_code!r} in {ticker!r}.")
        # 2-digit year → century-aware. We assume 2000-2099 for VX (the
        # contract didn't exist before 2004).
        year = 2000 + yy
        return vx_settlement_date(year, CODE_TO_MONTH[month_code])
