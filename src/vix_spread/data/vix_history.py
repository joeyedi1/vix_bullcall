"""Daily-close history for VIX Index and front-month VX futures (M1, M2).

This pull is the input panel for the regime classifier (ARCHITECTURE §3).
It is NOT a Black-76 forward input — spot VIX is permanently forbidden as a
pricing forward (validation memo, ARCHITECTURE §4.1). The daily series
exist only to feed:

  - HMM emissions (VIX-level, log-returns, etc.)
  - Curve-slope features `F_far / F_near − 1` (M1/M2)
  - Diagnostics / overlays in the reporting layer

Persisted under `data/raw/blpapi/vix_history_daily/{vintage}.parquet` via
`BaseDataFetcher`.

Phase-1 scaffold: connection + request boilerplate are wired; the specific
ticker mapping (which Bloomberg security represents the M1 / M2 generic VX
futures roll, vs the calendar-aware specific contract) is left abstract.
"""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .base import BaseDataFetcher

try:  # pragma: no cover
    import pdblp  # type: ignore
except ImportError:  # pragma: no cover
    pdblp = None  # type: ignore[assignment]


LogicalSeries = Literal["vix_index", "vx_m1", "vx_m2"]


class VIXHistoryFetcher(BaseDataFetcher):
    """Bloomberg ingestion for daily VIX Index + M1/M2 VX futures close.

    Subclass contract
    -----------------
    `_resolve_tickers()` returns a mapping
    `{logical_name -> bloomberg_ticker}` covering at minimum
    `{'vix_index', 'vx_m1', 'vx_m2'}`. Phase-1 leaves this abstract.

    Storage
    -------
    Product tag: `vix_history_daily`.
    Long-form schema written to parquet:
      - date           (datetime64[ns], NY exchange close convention)
      - logical        (str: 'vix_index' | 'vx_m1' | 'vx_m2' | ...)
      - ticker         (str: the Bloomberg security used)
      - field          (str: e.g. 'PX_LAST')
      - value          (float)
      - _pulled_at     (UTC pull timestamp, added by BaseDataFetcher)
      - _vintage       (vintage tag, added by BaseDataFetcher)
    """

    source = "blpapi"

    DEFAULT_FIELDS: tuple[str, ...] = ("PX_LAST",)

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
    # Connection                                                         #
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

    # ------------------------------------------------------------------ #
    # Ticker mapping — abstract                                          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _resolve_tickers(self) -> dict[LogicalSeries, str]:
        """Return `{logical_name -> bloomberg_ticker}`.

        Must include at least `vix_index`, `vx_m1`, `vx_m2`. Phase-1
        deliberately leaves this unimplemented — the M1/M2 generic-vs-roll
        choice has audit consequences (see ARCHITECTURE §3.4 sign
        convention) and belongs to a dedicated mapping module.
        """

    # ------------------------------------------------------------------ #
    # BaseDataFetcher hook                                               #
    # ------------------------------------------------------------------ #

    def _fetch(
        self,
        *,
        start: datetime,
        end: datetime,
        fields: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> tuple[str, pd.DataFrame]:
        mapping = self._resolve_tickers()
        if not mapping:
            raise ValueError("_resolve_tickers returned an empty mapping.")
        for required in ("vix_index", "vx_m1", "vx_m2"):
            if required not in mapping:
                raise ValueError(
                    f"_resolve_tickers must provide '{required}'; "
                    f"got keys={list(mapping)}."
                )

        fields = fields or self.DEFAULT_FIELDS
        tickers = list(mapping.values())

        con = self._connect_pdblp()
        wide = con.bdh(
            tickers,
            list(fields),
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )

        long = (
            wide.stack(level=0, future_stack=True)
            .stack(level=0, future_stack=True)
            .rename("value")
            .reset_index()
        )
        long.columns = ["date", "ticker", "field", "value"]

        rev_map = {v: k for k, v in mapping.items()}
        long["logical"] = long["ticker"].map(rev_map)

        long = long[["date", "logical", "ticker", "field", "value"]]
        return "vix_history_daily", long
