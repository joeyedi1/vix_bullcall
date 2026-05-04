"""VX option-on-futures chain ingestion (CFE OOF, physically settled into one VX future).

DATA-BLOCKED on the current Bloomberg terminal (verified 2026-05-04): VX OOF
chains are not subscribed — `OPT_CHAIN` returns 0 entries on every VX future
form, and direct ticker construction (`UXM26C 25 Index`, `VX 06/17/26 C25
Index`, ~20 variants) all return `Unknown/Invalid Security`. The same
terminal also rejects ES options-on-futures and SPX index options, so this
is a subscription-scope limitation, not a code bug. The scaffold below is
correct and ARCH-aligned; activate it once a data source becomes available
(different terminal, Cboe DataShop, OPRA archive, etc.). The Phase-2
type-system guards (`VXFutureOption.deliverable_vx` REQUIRED, mixed-product
spread rejection) remain intact independent of data availability.

Distinct from `vix_index_options.py` per ARCHITECTURE §2.1 / §6.1: separate
ingestion path, separate schema, separate storage prefix. Cross-mixing the
two products is the single largest source of pricing error flagged in the
validation memo.

For each VX futures contract we already pulled (UXV25..UXK26), the OOF
chain is enumerated via `OPT_CHAIN` on the future's ticker, then filtered
by DTE (option expiry == VX future settlement date for standard CFE OOF,
so this is largely a sanity filter) and ATM ± strike window using the
deliverable VX future's price as the ATM reference.

Two products written, one parquet per (contract):

    data/raw/blpapi/vx_future_options_quotes/{contract_safe}_{vintage}.parquet
    data/raw/blpapi/vx_future_options_daily/{contract_safe}_{vintage}.parquet

Limitations
-----------
1. Bloomberg ticker format for VX OOF on this terminal is assumed to match
   `'<UXyyMM><C|P> <strike> Index'` (e.g. `'UXM25C 25.0 Index'`) or a
   close variant. The first-run OPT_CHAIN response is the source of truth;
   adjust `_VX_OOF_TICKER_RE` if needed.
2. Already-expired VX OOF contracts may be missing from `OPT_CHAIN`. Manual
   ticker construction is the fallback (NOT implemented in this scaffold).
3. Bloomberg's intraday-history ceiling caps tick coverage to ~140-200d
   depending on product.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from .base import BaseDataFetcher, RawPullManifest, make_vintage
from .expiry_calendar import CODE_TO_MONTH, vx_settlement_date

try:  # pragma: no cover
    import blpapi  # type: ignore
except ImportError:  # pragma: no cover
    blpapi = None  # type: ignore[assignment]

try:  # pragma: no cover
    import pdblp  # type: ignore
except ImportError:  # pragma: no cover
    pdblp = None  # type: ignore[assignment]


# Common VX OOF ticker forms observed on Bloomberg:
#   "UXM25C 25.0 Index"
#   "UXM5C 25 Comdty"           (legacy single-digit-year form)
# This regex prioritizes the modern 2-digit-year form; legacy is also handled.
_VX_OOF_TICKER_RE = re.compile(
    r"^UX([A-Z])(\d{1,2})([CP])\s+"      # month-code, year(1-2 digits), right
    r"(\d+(?:\.\d+)?)\s+"                # strike
    r"(?:Index|Comdty)$"
)

DAILY_FIELDS_DEFAULT: tuple[str, ...] = (
    # BID_SIZE / ASK_SIZE are unavailable in `bdh` for options (confirmed
    # against `VIX US 05/19/26 C10 Index` 2026-05-04). Sizes captured at
    # tick level instead via the BID/ASK tick `size` field.
    "PX_BID",
    "PX_ASK",
    "PX_LAST",
    "OPEN_INT",
    "PX_VOLUME",
    "IVOL_LAST",
)


@dataclass(frozen=True)
class VXOptionContract:
    """Parsed VX option-on-futures contract metadata."""
    ticker: str
    underlying_vx_ticker: str    # e.g. 'UXM25 Index'
    deliverable_settlement_date: date
    right: str                    # 'C' or 'P'
    strike: float


def parse_vx_option_ticker(ticker: str) -> VXOptionContract:
    """Parse 'UXM25C 25.0 Index' -> contract metadata.

    The deliverable VX future's ticker and settlement date are derived
    from the month-code + year embedded in the option ticker. This is the
    single point where option ↔ deliverable VX is wired (consumed downstream
    when constructing `VXFutureOption(deliverable_vx=...)`).
    """
    m = _VX_OOF_TICKER_RE.match(ticker.strip())
    if m is None:
        raise ValueError(f"Unrecognized VX option ticker: {ticker!r}")
    month_code, yy, right, strike = m.groups()
    if month_code not in CODE_TO_MONTH:
        raise ValueError(f"Unknown VX month code {month_code!r} in {ticker!r}.")
    yy_int = int(yy)
    year = 2000 + yy_int if yy_int >= 10 or len(yy) == 2 else 2020 + yy_int
    settlement = vx_settlement_date(year, CODE_TO_MONTH[month_code])
    return VXOptionContract(
        ticker=ticker,
        underlying_vx_ticker=f"UX{month_code}{year % 100:02d} Index",
        deliverable_settlement_date=settlement,
        right=right,
        strike=float(strike),
    )


class VXFutureOptionsChainFetcher(BaseDataFetcher):
    """Pulls 1-min NBBO ticks + daily reference for VX option-on-futures.

    Multi-shard: one parquet per (contract, kind). Pulls the chain for each
    underlying VX future passed in, unions the results, then iterates with
    tqdm just like `VIXIndexOptionsChainFetcher`.
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

    # ---------------------------------------------------------------- #
    # Connection                                                       #
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
    # Chain discovery — one OPT_CHAIN per underlying VX future          #
    # ---------------------------------------------------------------- #

    def resolve_chain(self, vx_future_tickers: list[str]) -> dict[str, list[str]]:
        """Returns `{vx_future_ticker -> [option_ticker, ...]}` from
        `OPT_CHAIN` for each underlying."""
        out: dict[str, list[str]] = {underlying: [] for underlying in vx_future_tickers}
        session = self._open_blpapi_session()
        try:
            svc = session.getService("//blp/refdata")
            req = svc.createRequest("ReferenceDataRequest")
            for u in vx_future_tickers:
                req.append("securities", u)
            req.append("fields", "OPT_CHAIN")
            session.sendRequest(req)

            done = False
            while not done:
                ev = session.nextEvent(timeout=self.timeout_ms)
                for msg in ev:
                    if not msg.hasElement("securityData"):
                        continue
                    arr = msg.getElement("securityData")
                    for i in range(arr.numValues()):
                        sd = arr.getValue(i)
                        underlying = sd.getElementAsString("security")
                        if sd.hasElement("securityError"):
                            err = sd.getElement("securityError")
                            tqdm.write(
                                f"  [chain-error] {underlying}: "
                                f"{err.getElementAsString('message')}"
                            )
                            continue
                        if not sd.hasElement("fieldData"):
                            continue
                        fd = sd.getElement("fieldData")
                        if not fd.hasElement("OPT_CHAIN"):
                            continue
                        chain = fd.getElement("OPT_CHAIN")
                        for j in range(chain.numValues()):
                            entry = chain.getValue(j)
                            if entry.hasElement("Security Description"):
                                out[underlying].append(
                                    entry.getElementAsString("Security Description")
                                )
                if ev.eventType() == blpapi.Event.RESPONSE:
                    done = True
            return out
        finally:
            session.stop()

    def filter_chain(
        self,
        chain_by_underlying: dict[str, list[str]],
        *,
        pull_start: datetime,
        pull_end: datetime,
        dte_min: int = 7,
        dte_max: int = 90,
        atm_prices: dict[str, float],
        atm_window: int = 10,
    ) -> list[VXOptionContract]:
        """Filter chains by DTE-of-deliverable-settlement and ATM ± window.

        `atm_prices` maps each VX future ticker -> reference price (median
        over its lifetime in the pull window, computed upstream from the
        existing `vx_futures_ohlcv` shards).
        """
        expiry_lo = pull_start.date() + timedelta(days=dte_min)
        expiry_hi = pull_end.date() + timedelta(days=dte_max)

        keep: list[VXOptionContract] = []
        for underlying, tickers in chain_by_underlying.items():
            atm = atm_prices.get(underlying)
            if atm is None:
                continue
            parsed: list[VXOptionContract] = []
            for tk in tickers:
                try:
                    parsed.append(parse_vx_option_ticker(tk))
                except ValueError:
                    continue
            in_window = [
                c for c in parsed
                if expiry_lo <= c.deliverable_settlement_date <= expiry_hi
            ]
            keys = {(c.deliverable_settlement_date, c.right) for c in in_window}
            for sd, rt in sorted(keys):
                bucket = sorted(
                    (c for c in in_window
                     if c.deliverable_settlement_date == sd and c.right == rt),
                    key=lambda c: abs(c.strike - atm),
                )[: atm_window * 2 + 1]
                keep.extend(bucket)
        return sorted(
            keep,
            key=lambda c: (c.deliverable_settlement_date, c.right, c.strike),
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
        contracts: list[VXOptionContract],
        kinds: tuple[str, ...] = ("quotes", "daily"),
        **_: Any,
    ) -> list[RawPullManifest]:
        from datetime import timezone as _tz
        pulled_at = datetime.now(_tz.utc)
        vintage = make_vintage(pulled_at)

        manifests: list[RawPullManifest] = []
        bar = tqdm(contracts, desc="VX OOF", unit="contract")
        for c in bar:
            bar.set_postfix_str(c.ticker, refresh=True)
            for kind in kinds:
                product = f"vx_future_options_{kind}"
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
