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
from .vix_index_options import parse_vix_option_ticker


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
    VIX_INDEX_OPTIONS_QUOTES_PRODUCT = "vix_index_options_quotes"
    VIX_INDEX_OPTIONS_DAILY_PRODUCT = "vix_index_options_daily"
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
        """Load all per-contract parquets for `product` at `vintage`.

        Recovers the original Bloomberg ticker from the in-data `ticker`
        column (set at ingestion time by the IntradayTickRequest /
        IntradayBarRequest handlers). Falls back to filename-based recovery
        only for legacy shards that lack the column. The in-data column is
        authoritative for VIX option tickers, whose `safe_shard_key` form
        (e.g. `'VIX_US_05_19_26_C20_Index'`) is not invertible by simple
        underscore-substitution.
        """
        if vintage is None:
            vintage = self.latest_vintage(product)
        d = self.raw_root / self.SOURCE / product
        out: dict[str, pd.DataFrame] = {}
        for f in sorted(d.glob(f"*_{vintage}.parquet")):
            df = pd.read_parquet(f)
            if "ticker" in df.columns and len(df):
                ticker = str(df["ticker"].iloc[0])
            else:
                # Legacy fallback: works only for tickers with a single space
                # (VX futures: 'UXV25 Index'). Will produce the wrong ticker
                # for multi-token symbols (VIX options) — but those shards
                # always carry the column, so this branch isn't reached.
                ticker_safe = f.stem.removesuffix(f"_{vintage}")
                ticker = ticker_safe.replace("_", " ", 1)
            out[ticker] = df
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
    # OptionQuote-shaped alignment                                     #
    #                                                                  #
    # Distinct from `align_quotes` (futures BBO without sizes): pivots #
    # BID/ASK ticks to per-minute (price + size) tuples, derives       #
    # is_locked / is_crossed flags here (Bloomberg doesn't expose them #
    # on the wire), and emits a schema that maps 1:1 onto OptionQuote  #
    # (ARCHITECTURE §5.1) modulo dtype coercion at construction time.  #
    # ---------------------------------------------------------------- #

    def align_option_quotes(
        self,
        ticks: pd.DataFrame,
        grid: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Per-minute NBBO + last-trade for one option contract.

        Output columns (all on `grid`):
            bid, ask, bid_size, ask_size               -- forward-filled
            last_trade                                  -- forward-filled
            quote_age_seconds                           -- max(bid_age, ask_age)
            last_trade_age_seconds
            is_locked   = (bid == ask) & both non-null
            is_crossed  = (bid >  ask) & both non-null

        `quote_age_seconds` reports the staler side of the NBBO so a
        FillEngine staleness gate (`max_age_seconds`) trips on either
        leg's drift, not the average. NaN until both sides have been
        observed at least once — there is no NBBO to age before that.
        """
        cols = [
            "bid", "ask", "bid_size", "ask_size", "last_trade",
            "quote_age_seconds", "last_trade_age_seconds",
            "is_locked", "is_crossed",
        ]
        if ticks is None or ticks.empty:
            empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in cols[:7]}, index=grid)
            empty["is_locked"] = pd.Series(False, index=grid, dtype="bool")
            empty["is_crossed"] = pd.Series(False, index=grid, dtype="bool")
            return empty[cols]

        idx = pd.to_datetime(ticks["time"], utc=True)
        df = (
            ticks.assign(_idx=idx)
            .drop(columns=["time"], errors="ignore")
            .set_index("_idx")
            .sort_index()
        )

        bid_rows = df[df["type"] == "BID"]
        ask_rows = df[df["type"] == "ASK"]
        trade_rows = df[df["type"] == "TRADE"]

        # Last-per-minute price + size for each event type.
        fresh_bid = bid_rows["value"].resample(self.GRID_FREQ).last().reindex(grid)
        fresh_bid_size = (
            bid_rows["size"].resample(self.GRID_FREQ).last().reindex(grid)
            if "size" in bid_rows else pd.Series(np.nan, index=grid)
        )
        fresh_ask = ask_rows["value"].resample(self.GRID_FREQ).last().reindex(grid)
        fresh_ask_size = (
            ask_rows["size"].resample(self.GRID_FREQ).last().reindex(grid)
            if "size" in ask_rows else pd.Series(np.nan, index=grid)
        )
        fresh_trade = trade_rows["value"].resample(self.GRID_FREQ).last().reindex(grid)

        out = pd.DataFrame(index=grid)
        out["bid"] = fresh_bid.ffill()
        out["ask"] = fresh_ask.ffill()
        out["bid_size"] = fresh_bid_size.ffill()
        out["ask_size"] = fresh_ask_size.ffill()
        out["last_trade"] = fresh_trade.ffill()

        # Staleness clocks — NaN before first observation, +60s per minute since.
        bid_age = self._age_seconds(fresh_bid)
        ask_age = self._age_seconds(fresh_ask)
        # max(skipna=False): NaN propagates if either side never observed.
        out["quote_age_seconds"] = pd.concat([bid_age, ask_age], axis=1).max(
            axis=1, skipna=False,
        )
        out["last_trade_age_seconds"] = self._age_seconds(fresh_trade)

        # Derived NBBO state — only meaningful when both sides are present.
        both = out["bid"].notna() & out["ask"].notna()
        out["is_locked"] = (out["bid"] == out["ask"]) & both
        out["is_crossed"] = (out["bid"] > out["ask"]) & both

        return out[cols]

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

    def process_vix_index_options(
        self,
        quotes_vintage: str | None = None,
    ) -> pd.DataFrame:
        """Load all VIX-idx option tick shards; return a unified per-minute
        panel keyed on `(timestamp, contract_id)`.

        Output schema maps 1:1 onto `OptionQuote` (ARCHITECTURE §5.1):
            bid, ask, bid_size, ask_size, last_trade,
            quote_age_seconds, last_trade_age_seconds,
            is_locked, is_crossed.

        Each contract is aligned to its OWN [first-tick, last-tick] minute
        grid (not a global grid) — the cross-contract panel is therefore
        sparse over the union of lifetimes. Querying `panel.loc[(ts, cid)]`
        returns the latest forward-filled NBBO at `ts` for that contract,
        or raises KeyError if `ts` is outside the contract's life.

        `is_locked` / `is_crossed` are derived here per the chain ingestion
        contract: Bloomberg does not expose them as flags on the wire.
        """
        try:
            quote_shards = self._load_shards(
                self.VIX_INDEX_OPTIONS_QUOTES_PRODUCT, quotes_vintage,
            )
        except FileNotFoundError:
            return pd.DataFrame()

        parts: list[pd.DataFrame] = []
        for ticker, ticks in quote_shards.items():
            if ticks is None or ticks.empty:
                continue
            t = pd.to_datetime(ticks["time"], utc=True)
            grid = self.build_master_grid(
                t.min().to_pydatetime(), t.max().to_pydatetime(),
            )
            per = self.align_option_quotes(ticks, grid)
            per["contract_id"] = ticker
            parts.append(per)

        if not parts:
            return pd.DataFrame()

        panel = pd.concat(parts)
        panel.index.name = "timestamp"
        panel = panel.set_index("contract_id", append=True).sort_index()
        return panel

    def process_vix_options_daily(
        self,
        daily_vintage: str | None = None,
    ) -> pd.DataFrame:
        """Load all VIX-idx daily option shards; return a wide pivoted panel
        keyed on `MultiIndex(date, expiry, right, strike)` with columns
        `[IVOL_LAST, PX_BID, PX_ASK, ...]` (whichever fields are present
        in the shards).

        This is the consumer-facing form for `ChainIVProvider`
        (ARCHITECTURE §4.3) — its `__init__` validator checks for that
        exact MultiIndex shape and the `IVOL_LAST / PX_BID / PX_ASK`
        column subset.

        Expiry normalization
        --------------------
        Bloomberg's ticker-date convention is asymmetric:
          - Settled monthlies: ticker date == SOQ Wednesday
            (e.g. `'VIX US 11/19/25 ...'`)
          - Active monthlies:  ticker date == Tuesday last-trade
            (e.g. `'VIX US 05/19/26 ...'`, with SOQ on Wed 5/20)
        We normalize ALL `expiry` index values to the SOQ Wednesday via
        `vx_settlement_date(year, month)` so downstream lookups keyed on
        `product.expiry.date()` succeed regardless of contract state at
        pull time. Without this, settled-month lookups would work but
        active-month lookups would silently miss.

        Tickers that don't match the canonical regex (weeklies, AM/PM
        settlement variants, etc.) are silently skipped — same convention
        as `filter_chain` in the ingestion path.
        """
        try:
            shards = self._load_shards(
                self.VIX_INDEX_OPTIONS_DAILY_PRODUCT, daily_vintage,
            )
        except FileNotFoundError:
            return pd.DataFrame()

        parts: list[pd.DataFrame] = []
        for ticker, df in shards.items():
            try:
                parsed = parse_vix_option_ticker(ticker)
            except ValueError:
                continue
            if df is None or df.empty:
                continue

            soq_wed = vx_settlement_date(
                parsed.expiry_date.year, parsed.expiry_date.month,
            )

            pivot = df.pivot_table(
                index="date", columns="field", values="value", aggfunc="last",
            )
            pivot = pivot.reset_index()
            pivot["date"] = pd.to_datetime(pivot["date"]).dt.date
            pivot["expiry"] = soq_wed
            pivot["right"] = parsed.right
            pivot["strike"] = float(parsed.strike)
            parts.append(pivot)

        if not parts:
            return pd.DataFrame()

        panel = pd.concat(parts, ignore_index=True, sort=False)
        panel = panel.set_index(
            ["date", "expiry", "right", "strike"]
        ).sort_index()
        # Drop the columns axis name introduced by pivot_table for
        # clean __repr__ in downstream consumers.
        panel.columns.name = None
        return panel

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
