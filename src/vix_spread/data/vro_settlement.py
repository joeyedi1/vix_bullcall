"""VRO Index daily SOQ prints.

The VRO is the special opening-quotation print used to cash-settle VIX index
options on each Wednesday-morning expiry. It is the load-bearing input to
`SettlementMarket.vro_for(expiry)` (ARCHITECTURE §6.2 / validation-memo
§12.7) — `VIXIndexOption.settlement_value` MUST consume the actual VRO
print for the option's expiry; spot VIX close, theoretical Black-76 value,
and Tuesday VX settle are all forbidden substitutes.

Bloomberg ticker
----------------
`VRO Index` is the published series. Confirm against
ReferenceDataRequest(NAME) on first run if uncertain.

Storage
-------
Single product `vro_settlement`, single parquet per pull (small daily
series — no per-shard split needed). Long-form schema:
    date  field  value  _pulled_at  _vintage
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseDataFetcher

try:  # pragma: no cover
    import pdblp  # type: ignore
except ImportError:  # pragma: no cover
    pdblp = None  # type: ignore[assignment]


class VROSettlementFetcher(BaseDataFetcher):
    """Daily VRO (VIX SOQ) prints via pdblp `bdh`."""

    source = "blpapi"
    BLOOMBERG_TICKER = "VRO Index"
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

    def _connect(self) -> "pdblp.BCon":
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

    def _fetch(
        self,
        *,
        start: datetime,
        end: datetime,
        fields: tuple[str, ...] | None = None,
        **_: Any,
    ) -> tuple[str, pd.DataFrame]:
        fields = fields or self.DEFAULT_FIELDS
        con = self._connect()
        wide = con.bdh(
            [self.BLOOMBERG_TICKER],
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
        long = long[["date", "ticker", "field", "value"]]
        return "vro_settlement", long
