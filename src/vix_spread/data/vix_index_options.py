"""VIX index option chain ingestion (Cboe VIX-suffix product, cash-settled to VRO).

Pulls 1-min NBBO (BID/ASK/TRADE ticks via `IntradayTickRequest`) and daily
chain-reference fields (`IVOL_LAST`, `OPEN_INT`, etc. via `bdh`) for every
qualifying contract. Output rows downstream feed `OptionQuote`
(ARCHITECTURE §5.1) and `ChainIVProvider` (§4.3).

Two products written, one parquet per (contract):

    data/raw/blpapi/vix_index_options_quotes/{contract_safe}_{vintage}.parquet
    data/raw/blpapi/vix_index_options_daily/{contract_safe}_{vintage}.parquet

The chain is enumerated via Bloomberg's `OPT_CHAIN` field on `VIX Index`,
then filtered by:
  - DTE window: keeps expiries with `pull_start + dte_min <= expiry <= pull_end + dte_max`
    days. The default (7 - 90 days) excludes 0-6 DTE expiry-week pinning
    noise per the strategy spec.
  - Strike window: keeps strikes within ±N strikes of the ATM reference.
    ATM reference defaults to the median spot VIX over the pull window
    (computed from the latest `vix_history_daily` vintage if available;
    overridable via `--atm-price`).

Limitations (surface to user)
-----------------------------
1. `OPT_CHAIN` returns currently-listed contracts. Already-expired contracts
   in our window (e.g. Nov 2025 VIX expiry, expired 6 months ago) may be
   missing. Manual ticker construction by Cboe strike rules is the fallback;
   currently NOT implemented in this scaffold — surface as a follow-up if
   the OPT_CHAIN coverage proves insufficient.
2. Bloomberg's intraday-history ceiling (~140-200 days for VX futures, may
   be similar for options) caps how far back tick data is retrievable.
3. Bloomberg ticker format for VIX options on this terminal is assumed to
   match `'VIX <M/D/YY> <C|P><strike> Index'`. Confirm empirically on
   first run via the OPT_CHAIN response.
"""
from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from .base import BaseDataFetcher, RawPullManifest, make_vintage
from .expiry_calendar import vx_settlement_date

try:  # pragma: no cover
    import blpapi  # type: ignore
except ImportError:  # pragma: no cover
    blpapi = None  # type: ignore[assignment]

try:  # pragma: no cover
    import pdblp  # type: ignore
except ImportError:  # pragma: no cover
    pdblp = None  # type: ignore[assignment]


VIX_OPT_TICKER_RE = re.compile(
    r"^VIX\s+"
    r"(?:US\s+)?"                       # optional 'US ' from some Bloomberg outputs
    r"(\d{1,2})/(\d{1,2})/(\d{2,4})\s+"  # date M/D/YY or M/D/YYYY
    r"([CP])"                            # right
    r"(\d+(?:\.\d+)?)\s+"                # strike
    r"Index$"
)

DAILY_FIELDS_DEFAULT: tuple[str, ...] = (
    # Bloomberg `bdh` does not expose BID_SIZE / ASK_SIZE historically for
    # options on this terminal (ValueError([]) on those fields, confirmed
    # 2026-05-04). Sizes are captured at tick level via the `size` field
    # on BID/ASK ticks, so omitting them here is lossless for downstream
    # OptionQuote construction.
    "PX_BID",
    "PX_ASK",
    "PX_LAST",
    "OPEN_INT",
    "PX_VOLUME",
    "IVOL_LAST",
)


@dataclass(frozen=True)
class VIXOptionContract:
    """Parsed VIX index option contract metadata."""
    ticker: str
    expiry_date: date
    right: str   # 'C' or 'P'
    strike: float


def parse_vix_option_ticker(ticker: str) -> VIXOptionContract:
    """Parse 'VIX 11/19/25 C20 Index' -> (expiry, right, strike).

    Raises ValueError if the ticker form is not recognized — used to filter
    out non-standard items in OPT_CHAIN responses (e.g. weeklies if Bloomberg
    encodes them differently).
    """
    m = VIX_OPT_TICKER_RE.match(ticker.strip())
    if m is None:
        raise ValueError(f"Unrecognized VIX option ticker: {ticker!r}")
    mo, d, y, right, strike = m.groups()
    yyyy = int(y) if int(y) > 1000 else 2000 + int(y)
    return VIXOptionContract(
        ticker=ticker,
        expiry_date=date(yyyy, int(mo), int(d)),
        right=right,
        strike=float(strike),
    )


# --------------------------------------------------------------------------- #
# Manual chain reconstruction (for already-settled expiries that have dropped #
# out of OPT_CHAIN). Used by `enumerate_historical_chain` below.              #
# --------------------------------------------------------------------------- #


def cboe_vix_strike_grid(lo: float = 5.0, hi: float = 200.0) -> list[float]:
    """Generous-superset Cboe VIX option strike grid.

    Empirical Cboe convention (verified against the live OPT_CHAIN response):
      - 0.5 increments at low strikes (5 - 30)
      - 1.0 increments mid (30 - 50)
      - 2.5 increments above (50 - 100)
      - 5.0 increments at high strikes (100 - 200)

    Bands overlap at the boundaries (30, 50, 100) so integer strikes are
    not dropped at transitions; deduplication is handled by the set in the
    accumulator. Caller MUST validate each candidate against Bloomberg
    (`_validate_tickers_batch`) — strikes never listed will fail with
    `BAD_SEC` and are silently dropped.
    """
    bands = (
        (5.0, 30.0, 0.5),
        (30.0, 50.0, 1.0),
        (50.0, 100.0, 2.5),
        (100.0, 200.0, 5.0),
    )
    out: set[float] = set()
    for band_lo, band_hi, step in bands:
        a = max(band_lo, lo)
        b = min(band_hi, hi)
        if a > b + 1e-9:
            continue
        n_steps = int(round((b - a) / step))
        for i in range(n_steps + 1):
            s = round(a + i * step, 2)
            if lo - 1e-9 <= s <= hi + 1e-9:
                out.add(s)
    return sorted(out)


def vix_option_ticker_date(year: int, month: int) -> date:
    """Bloomberg-ticker date for the SETTLED monthly VIX option whose
    underlying VX cycle is `(year, month)`. **Returns the SOQ Wednesday.**

    Bloomberg's ticker-date convention is asymmetric across the contract
    lifecycle (verified empirically 2026-05-05):

      - **Active / currently-listed** monthlies use the Tuesday LAST-TRADE
        date (e.g. `VIX US 05/19/26 C20 Index` for the May 2026 monthly
        whose SOQ is Wed May 20). These tickers come from `OPT_CHAIN` and
        do NOT require manual construction — `resolve_chain` discovers
        them directly.
      - **Already-settled** monthlies use the Wednesday SOQ date
        (e.g. `VIX US 04/15/26 C20 Index` for the April 2026 monthly that
        settled Wed Apr 15). The Tuesday form returns BAD_SEC. This
        function exists to construct THESE tickers for historical
        backfill via `enumerate_historical_chain`.

    Holiday rollback: if VX SOQ Wednesday rolled to Tuesday (e.g. Juneteenth
    2024-06-19), `vx_settlement_date` returns the rolled-back Tuesday and
    this function returns the same Tuesday. Empirically untested for the
    rollback case; surface as a follow-up if a Juneteenth-month historical
    backfill returns 0 validated contracts.
    """
    return vx_settlement_date(year, month)


def construct_vix_option_ticker(ticker_date: date, right: str, strike: float) -> str:
    """`(date(2026, 5, 19), 'C', 20.0)` -> `'VIX US 05/19/26 C20 Index'`.

    Strike formatting matches the OPT_CHAIN-emitted form: integer strikes
    drop the decimal (`20.0` → `'20'`), half-strikes keep one decimal
    (`20.5` → `'20.5'`, `42.5` → `'42.5'`). Format `:g` handles both.
    """
    if right not in ('C', 'P'):
        raise ValueError(f"right must be 'C' or 'P', got {right!r}")
    mo, d, yy = ticker_date.month, ticker_date.day, ticker_date.year % 100
    strike_str = f"{strike:g}"
    return f"VIX US {mo:02d}/{d:02d}/{yy:02d} {right}{strike_str} Index"


class VIXIndexOptionsChainFetcher(BaseDataFetcher):
    """Pulls 1-min NBBO ticks + daily chain reference for VIX index options.

    Multi-shard: one parquet per (contract, kind) pair, all sharing one
    vintage tag per CLI invocation. Override `pull` so the per-contract
    loop, tqdm progress, and incremental save all live in one place.
    """

    source = "blpapi"
    UNDERLYING_TICKER = "VIX Index"

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

    # ---------------------------------------------------------------- #
    # Connection helpers                                               #
    # ---------------------------------------------------------------- #

    def _connect_pdblp(self) -> "pdblp.BCon":
        if pdblp is None:
            raise RuntimeError(
                "pdblp not installed. `pip install pdblp blpapi` and ensure "
                "the Bloomberg terminal is running on this machine."
            )
        if self._con is None:
            con = pdblp.BCon(
                host=self.host, port=self.port,
                timeout=self.timeout_ms, debug=False,
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

    # ---------------------------------------------------------------- #
    # Chain discovery                                                  #
    # ---------------------------------------------------------------- #

    def resolve_chain(self) -> list[str]:
        """Return all option tickers listed under `OPT_CHAIN` for VIX Index."""
        session = self._open_blpapi_session()
        try:
            svc = session.getService("//blp/refdata")
            req = svc.createRequest("ReferenceDataRequest")
            req.append("securities", self.UNDERLYING_TICKER)
            req.append("fields", "OPT_CHAIN")
            session.sendRequest(req)

            tickers: list[str] = []
            done = False
            while not done:
                ev = session.nextEvent(timeout=self.timeout_ms)
                for msg in ev:
                    if not msg.hasElement("securityData"):
                        continue
                    arr = msg.getElement("securityData")
                    for i in range(arr.numValues()):
                        sd = arr.getValue(i)
                        if sd.hasElement("securityError"):
                            err = sd.getElement("securityError")
                            raise RuntimeError(
                                f"Bloomberg error on {self.UNDERLYING_TICKER}: "
                                f"{err.getElementAsString('message')}"
                            )
                        if not sd.hasElement("fieldData"):
                            continue
                        fd = sd.getElement("fieldData")
                        if not fd.hasElement("OPT_CHAIN"):
                            continue
                        chain = fd.getElement("OPT_CHAIN")
                        for j in range(chain.numValues()):
                            entry = chain.getValue(j)
                            # OPT_CHAIN is an array of structs with one
                            # element "Security Description".
                            if entry.hasElement("Security Description"):
                                tickers.append(
                                    entry.getElementAsString("Security Description")
                                )
                if ev.eventType() == blpapi.Event.RESPONSE:
                    done = True
            return tickers
        finally:
            session.stop()

    def filter_chain(
        self,
        chain: Iterable[str],
        *,
        pull_start: datetime,
        pull_end: datetime,
        dte_min: int = 7,
        dte_max: int = 90,
        atm_price: float,
        atm_window: int = 20,
    ) -> list[VIXOptionContract]:
        """Filter the chain to expiries in the DTE window and strikes in
        the ATM ± `atm_window` band (by sorted strike-distance)."""
        parsed: list[VIXOptionContract] = []
        for tk in chain:
            try:
                parsed.append(parse_vix_option_ticker(tk))
            except ValueError:
                continue  # skip weeklies/non-standard formats

        expiry_lo = pull_start.date() + timedelta(days=dte_min)
        expiry_hi = pull_end.date() + timedelta(days=dte_max)
        in_window = [c for c in parsed if expiry_lo <= c.expiry_date <= expiry_hi]

        # Group by (expiry, right) and keep the (atm_window*2 + 1) strikes
        # closest to atm_price.
        keep: list[VIXOptionContract] = []
        keys = {(c.expiry_date, c.right) for c in in_window}
        for exp, rt in sorted(keys):
            bucket = sorted(
                (c for c in in_window if c.expiry_date == exp and c.right == rt),
                key=lambda c: abs(c.strike - atm_price),
            )[: atm_window * 2 + 1]
            keep.extend(bucket)
        return sorted(keep, key=lambda c: (c.expiry_date, c.right, c.strike))

    # ---------------------------------------------------------------- #
    # Historical-chain reconstruction                                  #
    #                                                                  #
    # `OPT_CHAIN` returns only currently-listed contracts, so already- #
    # settled monthlies (e.g. Nov 2025 - Apr 2026 in our backtest      #
    # window) drop out. We reconstruct them by:                        #
    #   1. Computing the Bloomberg ticker date for the expiry month.   #
    #   2. Generating Cboe-spec candidate strikes (generous superset). #
    #   3. Validating each candidate via ReferenceDataRequest in       #
    #      batches; BAD_SEC = was never listed, silently dropped.      #
    # The result is the actually-listed contract set for that expiry.  #
    # ---------------------------------------------------------------- #

    def _validate_tickers_batch(
        self,
        candidates: list[str],
        chunk_size: int = 250,
    ) -> set[str]:
        """Return the subset of `candidates` that resolve on Bloomberg.

        Bloomberg's `ReferenceDataRequest` accepts hundreds of securities
        per request; we chunk at `chunk_size` to stay well under any cap.
        Each chunk requests `NAME` only — the lightest possible field. A
        candidate with a `securityError` element is treated as BAD_SEC and
        dropped; everything else is considered listed.
        """
        if not candidates:
            return set()
        valid: set[str] = set()
        session = self._open_blpapi_session()
        try:
            svc = session.getService("//blp/refdata")
            for i in range(0, len(candidates), chunk_size):
                chunk = candidates[i : i + chunk_size]
                req = svc.createRequest("ReferenceDataRequest")
                for s in chunk:
                    req.append("securities", s)
                req.append("fields", "NAME")
                session.sendRequest(req)

                done = False
                while not done:
                    ev = session.nextEvent(timeout=self.timeout_ms)
                    for msg in ev:
                        if not msg.hasElement("securityData"):
                            continue
                        arr = msg.getElement("securityData")
                        for j in range(arr.numValues()):
                            sd = arr.getValue(j)
                            sec = sd.getElementAsString("security")
                            if sd.hasElement("securityError"):
                                continue
                            valid.add(sec)
                    if ev.eventType() == blpapi.Event.RESPONSE:
                        done = True
            return valid
        finally:
            session.stop()

    def enumerate_historical_chain(
        self,
        year: int,
        month: int,
        *,
        strike_lo: float = 5.0,
        strike_hi: float = 200.0,
    ) -> list[VIXOptionContract]:
        """Reconstruct the (year, month) standard-monthly VIX option chain
        by candidate construction + Bloomberg validation.

        Returns the list of contracts that Bloomberg recognizes today.
        Empty list = either Bloomberg has no record of any strike for
        this expiry (unlikely for standard monthlies) or the ticker form
        on this terminal differs from `'VIX US M/D/YY {C|P}{strike} Index'`.
        """
        ticker_date = vix_option_ticker_date(year, month)
        strikes = cboe_vix_strike_grid(strike_lo, strike_hi)
        candidates: list[tuple[str, VIXOptionContract]] = []
        for strike in strikes:
            for right in ("C", "P"):
                tk = construct_vix_option_ticker(ticker_date, right, strike)
                candidates.append(
                    (
                        tk,
                        VIXOptionContract(
                            ticker=tk,
                            expiry_date=ticker_date,
                            right=right,
                            strike=strike,
                        ),
                    )
                )
        valid_tickers = self._validate_tickers_batch([t for t, _ in candidates])
        return sorted(
            (c for t, c in candidates if t in valid_tickers),
            key=lambda c: (c.right, c.strike),
        )

    # ---------------------------------------------------------------- #
    # Per-contract data pulls                                          #
    # ---------------------------------------------------------------- #

    def _fetch_intraday_ticks(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        event_types: Iterable[str] = ("BID", "ASK", "TRADE"),
    ) -> pd.DataFrame:
        """1-min NBBO + trade tick stream via raw `IntradayTickRequest`.

        Returns BID/ASK ticks WITH `size` so DataProcessor can produce
        per-minute `bid_size`/`ask_size` snapshots for `OptionQuote`.
        Empty frame on no-data (Bloomberg history ceiling) — caller skips.
        """
        cols = ["ticker", "time", "type", "value", "size"]
        session = self._open_blpapi_session()
        try:
            svc = session.getService("//blp/refdata")
            req = svc.createRequest("IntradayTickRequest")
            req.set("security", ticker)
            req.set("startDateTime", start)
            req.set("endDateTime", end)
            for et in event_types:
                req.append("eventTypes", et)
            req.set("includeConditionCodes", True)
            session.sendRequest(req)

            rows: list[dict[str, Any]] = []
            response_errors: list[str] = []
            done = False
            while not done:
                ev = session.nextEvent(timeout=self.timeout_ms)
                for msg in ev:
                    if msg.hasElement("responseError"):
                        e = msg.getElement("responseError")
                        response_errors.append(
                            f"{e.getElementAsString('category')}: "
                            f"{e.getElementAsString('message')}"
                        )
                        continue
                    if not msg.hasElement("tickData"):
                        continue
                    tick_data = msg.getElement("tickData").getElement("tickData")
                    for i in range(tick_data.numValues()):
                        item = tick_data.getValue(i)
                        rows.append({
                            "ticker": ticker,
                            "time": item.getElementAsDatetime("time"),
                            "type": item.getElementAsString("type"),
                            "value": item.getElementAsFloat("value"),
                            "size": (
                                item.getElementAsInteger("size")
                                if item.hasElement("size") else None
                            ),
                        })
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

    def _fetch_daily(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        fields: tuple[str, ...] = DAILY_FIELDS_DEFAULT,
    ) -> pd.DataFrame:
        """Per-contract daily chain reference via pdblp `bdh`.

        Returns long-form `(date, ticker, field, value)`. Empty frame on
        no-data (e.g. contract not yet listed at start date).
        """
        cols = ["date", "ticker", "field", "value"]
        con = self._connect_pdblp()
        wide = con.bdh(
            [ticker], list(fields),
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if wide.empty:
            return pd.DataFrame(columns=cols)
        long = (
            wide.stack(level=0, future_stack=True)
            .stack(level=0, future_stack=True)
            .rename("value")
            .reset_index()
        )
        long.columns = ["date", "ticker", "field", "value"]
        return long[cols]

    # ---------------------------------------------------------------- #
    # Multi-shard pull                                                 #
    # ---------------------------------------------------------------- #

    def pull(
        self,
        *,
        start: datetime,
        end: datetime,
        contracts: list[VIXOptionContract],
        kinds: tuple[str, ...] = ("quotes", "daily"),
        **_: Any,
    ) -> list[RawPullManifest]:
        """Pull tick + daily for each contract; one parquet per shard.

        `contracts` is the pre-filtered list (use `resolve_chain` +
        `filter_chain` upstream). `kinds` defaults to both products;
        narrow to `("quotes",)` or `("daily",)` for incremental runs.
        """
        from datetime import timezone as _tz
        pulled_at = datetime.now(_tz.utc)
        vintage = make_vintage(pulled_at)

        manifests: list[RawPullManifest] = []
        bar = tqdm(contracts, desc="VIX-idx options", unit="contract")
        for c in bar:
            bar.set_postfix_str(c.ticker, refresh=True)
            for kind in kinds:
                product = f"vix_index_options_{kind}"
                try:
                    if kind == "quotes":
                        df = self._fetch_intraday_ticks(c.ticker, start, end)
                    elif kind == "daily":
                        df = self._fetch_daily(c.ticker, start, end)
                    else:
                        raise ValueError(f"unknown kind={kind!r}")
                except Exception as exc:
                    tqdm.write(f"  [error/{kind}] {c.ticker}: {type(exc).__name__}: {exc}")
                    continue

                if df is None or len(df) == 0:
                    tqdm.write(f"  [skip/{kind}] {c.ticker}: no rows.")
                    continue

                df["_pulled_at"] = pulled_at
                df["_vintage"] = vintage
                path = self._save_raw(df, product, vintage, shard_key=c.ticker)
                tqdm.write(f"  [{c.ticker}/{kind}] {len(df):,} rows -> {path.name}")
                manifests.append(
                    RawPullManifest(
                        source=self.source,
                        product=product,
                        vintage=vintage,
                        path=path,
                        pulled_at=pulled_at,
                        row_count=len(df),
                        extra={"shard_key": c.ticker, "kind": kind},
                    )
                )
        return manifests
