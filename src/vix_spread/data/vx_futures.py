"""VX futures intraday ingestion via Bloomberg.

Pulls 1-minute OHLCV bars and tick-level BID/ASK quote data for VX futures
contracts, ONE PARQUET PER CONTRACT. Persists via `BaseDataFetcher._save_raw`
under `data/raw/blpapi/{product}/{ticker}_{vintage}.parquet`. All contracts
in a single CLI invocation share one vintage.

The class overrides `pull(...)` directly (multi-shard pattern from
`base.py`) so the per-contract loop, the tqdm progress bar, and the
incremental save can all live in one place. This avoids holding 12 months
of tick data in memory before writing.

Validation-memo notes:
  - VX futures are the deliverable underlying for `VXFutureOption` and the
    forward input for `VIXIndexOption` via settlement-date match
    (ARCHITECTURE §4.1). Their settlement-date is the join key — NOT a
    generic month index. Do not introduce "front month" / "M1" semantics
    in this layer; that mapping belongs to the calendar module.
  - 1-minute quote data is the input to `OptionQuote` synthetic NBBO
    construction (§5). Last-trade prices are diagnostic-only downstream;
    we still pull them but they must not leak into fill logic.
"""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd
from tqdm import tqdm

from .base import BaseDataFetcher, RawPullManifest, make_vintage

try:  # pragma: no cover - import-time guard, not a runtime branch
    import blpapi  # type: ignore
except ImportError:  # pragma: no cover
    blpapi = None  # type: ignore[assignment]

try:  # pragma: no cover
    import pdblp  # type: ignore
except ImportError:  # pragma: no cover
    pdblp = None  # type: ignore[assignment]


Kind = Literal["ohlcv", "quotes"]


class VXFuturesIntradayFetcher(BaseDataFetcher):
    """Bloomberg ingestion for 1-minute VX futures OHLCV and BBO quotes.

    Connection
    ----------
    pdblp's `BCon` is used for `IntradayBarRequest` (OHLCV bars). Raw
    `blpapi` is used for `IntradayTickRequest` (BID/ASK/TRADE ticks)
    because pdblp's tick coverage is uneven across versions.

    Both wrap the same local Bloomberg terminal endpoint
    (default `localhost:8194`).

    Subclass contract
    -----------------
    `_resolve_tickers(**kwargs)` must return the list of Bloomberg
    securities to pull (e.g. `["UXA Comdty", "UXM5 Comdty"]`). Phase-1
    scaffold leaves this abstract — wiring it requires the VX expiry
    calendar (ARCHITECTURE §10, `data/expiry_calendar.py`).

    Storage
    -------
    Two product tags are written, depending on the `kind` argument:
      - `vx_futures_ohlcv` for IntradayBarRequest output
      - `vx_futures_quotes` for IntradayTickRequest output
    Each contract becomes its own file:
        data/raw/blpapi/{product}/{ticker_safe}_{vintage}.parquet
    e.g. `UXM5_Index_20251201T143012Z.parquet`.
    """

    source = "blpapi"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8194,
        timeout_ms: int = 30_000,
        raw_root: Path | str = "data/raw",
    ) -> None:
        super().__init__(raw_root=raw_root)
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._con: "pdblp.BCon | None" = None

    # ------------------------------------------------------------------ #
    # Connection helpers                                                 #
    # ------------------------------------------------------------------ #

    def _connect_pdblp(self) -> "pdblp.BCon":
        if pdblp is None:
            raise RuntimeError(
                "pdblp not installed. `pip install pdblp blpapi` and ensure "
                "the Bloomberg terminal is running on this machine."
            )
        if self._con is None:
            con = pdblp.BCon(
                host=self.host,
                port=self.port,
                timeout=self.timeout_ms,
                debug=False,
            )
            con.start()
            self._con = con
        return self._con

    def _open_blpapi_session(self) -> "blpapi.Session":
        if blpapi is None:
            raise RuntimeError(
                "blpapi not installed. `pip install blpapi` and ensure the "
                "Bloomberg terminal is running on this machine."
            )
        opts = blpapi.SessionOptions()
        opts.setServerHost(self.host)
        opts.setServerPort(self.port)
        session = blpapi.Session(opts)
        if not session.start():
            raise RuntimeError("blpapi: failed to start session.")
        if not session.openService("//blp/refdata"):
            session.stop()
            raise RuntimeError("blpapi: failed to open //blp/refdata service.")
        return session

    # ------------------------------------------------------------------ #
    # Ticker mapping — abstract until the calendar module exists         #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _resolve_tickers(self, **kwargs: Any) -> list[str]:
        """Return the Bloomberg security strings to pull for this request.

        Phase-1 deliberately leaves this unimplemented. The mapping from
        (settlement_date | contract_code) to a Bloomberg ticker belongs to
        `data/expiry_calendar.py` and is wired in once that module exists.
        """

    # ------------------------------------------------------------------ #
    # Per-ticker request builders                                        #
    # ------------------------------------------------------------------ #

    def _fetch_intraday_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval_minutes: int = 1,
        event_type: str = "TRADE",
    ) -> pd.DataFrame:
        """1-minute OHLCV via raw `blpapi.IntradayBarRequest`.

        We use raw blpapi (not pdblp) because pdblp's `bdib` parser raises
        opaque KeyErrors on two common Bloomberg responses we MUST handle
        gracefully:
          - response with no `barData` element (security has zero bars in
            the requested window — typically because the window precedes
            Bloomberg's intraday history ceiling, ~140d for many products);
          - response with bars but a schema mismatch.
        Returning an empty DataFrame in those cases lets the caller skip
        and continue instead of aborting the whole loop.

        Output columns: time, open, high, low, close, volume, numEvents,
        ticker. (Empty frame has the same columns.)
        """
        cols = ["time", "open", "high", "low", "close", "volume", "numEvents", "ticker"]
        session = self._open_blpapi_session()
        try:
            ref_svc = session.getService("//blp/refdata")
            request = ref_svc.createRequest("IntradayBarRequest")
            request.set("security", ticker)
            request.set("eventType", event_type)
            request.set("interval", interval_minutes)
            request.set("startDateTime", start)
            request.set("endDateTime", end)

            session.sendRequest(request)

            rows: list[dict[str, Any]] = []
            response_errors: list[str] = []
            done = False
            while not done:
                ev = session.nextEvent(timeout=self.timeout_ms)
                for msg in ev:
                    if msg.hasElement("responseError"):
                        err = msg.getElement("responseError")
                        response_errors.append(
                            f"{err.getElementAsString('category')}: "
                            f"{err.getElementAsString('message')}"
                        )
                        continue
                    if not msg.hasElement("barData"):
                        continue
                    bar_data = msg.getElement("barData")
                    if not bar_data.hasElement("barTickData"):
                        continue
                    series = bar_data.getElement("barTickData")
                    for i in range(series.numValues()):
                        item = series.getValue(i)
                        rows.append(
                            {
                                "time": item.getElementAsDatetime("time"),
                                "open": item.getElementAsFloat("open"),
                                "high": item.getElementAsFloat("high"),
                                "low": item.getElementAsFloat("low"),
                                "close": item.getElementAsFloat("close"),
                                "volume": item.getElementAsInteger("volume"),
                                "numEvents": item.getElementAsInteger("numEvents"),
                                "ticker": ticker,
                            }
                        )
                if ev.eventType() == blpapi.Event.RESPONSE:
                    done = True

            if response_errors and not rows:
                raise RuntimeError(
                    f"Bloomberg responseError for {ticker}: "
                    + "; ".join(response_errors)
                )
            if not rows:
                return pd.DataFrame(columns=cols)
            return pd.DataFrame(rows, columns=cols)
        finally:
            session.stop()

    def _fetch_intraday_ticks(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        event_types: Iterable[str] = ("BID", "ASK", "TRADE"),
    ) -> pd.DataFrame:
        """Tick-level BID/ASK/TRADE via raw blpapi IntradayTickRequest."""
        session = self._open_blpapi_session()
        try:
            ref_svc = session.getService("//blp/refdata")
            request = ref_svc.createRequest("IntradayTickRequest")
            request.set("security", ticker)
            request.set("startDateTime", start)
            request.set("endDateTime", end)
            for et in event_types:
                request.append("eventTypes", et)
            request.set("includeConditionCodes", True)
            session.sendRequest(request)

            rows: list[dict[str, Any]] = []
            done = False
            while not done:
                ev = session.nextEvent(timeout=self.timeout_ms)
                for msg in ev:
                    if not msg.hasElement("tickData"):
                        continue
                    tick_data = msg.getElement("tickData").getElement("tickData")
                    for i in range(tick_data.numValues()):
                        item = tick_data.getValue(i)
                        rows.append(
                            {
                                "ticker": ticker,
                                "time": item.getElementAsDatetime("time"),
                                "type": item.getElementAsString("type"),
                                "value": item.getElementAsFloat("value"),
                                "size": (
                                    item.getElementAsInteger("size")
                                    if item.hasElement("size")
                                    else None
                                ),
                            }
                        )
                if ev.eventType() == blpapi.Event.RESPONSE:
                    done = True
            return pd.DataFrame(rows)
        finally:
            session.stop()

    # ------------------------------------------------------------------ #
    # Multi-shard pull — one parquet per contract, with tqdm.            #
    # ------------------------------------------------------------------ #

    def pull(
        self,
        *,
        start: datetime,
        end: datetime,
        kind: Kind,
        **kwargs: Any,
    ) -> list[RawPullManifest]:
        if kind not in ("ohlcv", "quotes"):
            raise ValueError(
                f"unknown kind={kind!r}; expected 'ohlcv' or 'quotes'."
            )
        tickers = self._resolve_tickers(**kwargs)
        if not tickers:
            raise ValueError("_resolve_tickers returned no tickers.")

        pulled_at = datetime.now(timezone.utc)
        vintage = make_vintage(pulled_at)
        product = f"vx_futures_{kind}"

        manifests: list[RawPullManifest] = []
        bar = tqdm(tickers, desc=f"VX {kind}", unit="contract")
        for ticker in bar:
            bar.set_postfix_str(ticker, refresh=True)
            try:
                df = (
                    self._fetch_intraday_bars(ticker, start, end)
                    if kind == "ohlcv"
                    else self._fetch_intraday_ticks(ticker, start, end)
                )
            except Exception as exc:  # don't let one bad contract abort the loop
                tqdm.write(f"  [error] {ticker}: {type(exc).__name__}: {exc}")
                continue

            if df is None or len(df) == 0:
                tqdm.write(f"  [skip] {ticker}: no rows returned.")
                continue

            df["_pulled_at"] = pulled_at
            df["_vintage"] = vintage
            path = self._save_raw(df, product, vintage, shard_key=ticker)
            tqdm.write(f"  [{ticker}] {len(df):,} rows -> {path.name}")
            manifests.append(
                RawPullManifest(
                    source=self.source,
                    product=product,
                    vintage=vintage,
                    path=path,
                    pulled_at=pulled_at,
                    row_count=len(df),
                    extra={"shard_key": ticker, "kind": kind},
                )
            )
        return manifests
