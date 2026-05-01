"""BaseDataFetcher — vintage-tagged raw-pull persistence.

Every concrete data fetcher in this codebase MUST subclass `BaseDataFetcher`.
The base class enforces the on-disk layout and audit contract from
ARCHITECTURE §6.3:

    data/raw/{source}/{product}/{shard_key_}{vintage}.parquet

`vintage` is the UTC pull timestamp embedded in BOTH the filename and a
`_pulled_at` column on every row. All shards within a single `pull(...)` call
share one vintage and one `_pulled_at`. Replays use the vintage tag to
reconstruct exactly what was knowable at any historical decision time
(`FeatureAvailability` validator, ARCHITECTURE §6.4).

Sharding
--------
Subclasses choose between two override points:
  - `_fetch(**kwargs) -> (product, dataframe)` for single-shard pulls
    (the default `pull()` template handles vintage stamping + parquet write).
  - `pull(**kwargs) -> list[RawPullManifest]` directly for multi-shard pulls
    (one parquet per shard). Override `pull()` and use
    `_save_raw(..., shard_key=...)` for filename disambiguation.

Either way `pull` returns a list of manifests — single-shard pulls just
return a 1-item list. This unification keeps the CLI uniform and avoids
union return types.
"""
from __future__ import annotations

import re
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RAW_ROOT = Path("data/raw")


def make_vintage(now: datetime | None = None) -> str:
    """UTC pull-timestamp string used for the vintage filename component.

    Format: YYYYMMDDTHHMMSSZ (sortable, filename-safe, unambiguously UTC).
    """
    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return ts.strftime("%Y%m%dT%H%M%SZ")


def safe_shard_key(key: str) -> str:
    """Filename-safe form of a shard key.

    Bloomberg ticker `'UXM5 Index'` -> `'UXM5_Index'`. Any character outside
    `[A-Za-z0-9._-]` becomes `_`; leading/trailing underscores are stripped.
    Empty input or input that sanitizes to empty raises ValueError.
    """
    if not key:
        raise ValueError("shard_key must be non-empty.")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", key).strip("_")
    if not cleaned:
        raise ValueError(f"shard_key {key!r} is empty after sanitization.")
    return cleaned


@dataclass(frozen=True)
class RawPullManifest:
    """Returned by `BaseDataFetcher.pull`. Records exactly one written file.

    These are the rows that populate `data/raw/_manifest.jsonl` (ARCHITECTURE
    §10) and feed the FeatureAvailability validator's as-of map.
    """
    source: str
    product: str
    vintage: str
    path: Path
    pulled_at: datetime
    row_count: int
    extra: dict[str, Any] = field(default_factory=dict)


class BaseDataFetcher(ABC):
    """Abstract base for all raw-data ingestion.

    Subclass contract:
      - Set the class attribute `source` (e.g. "blpapi", "fred", "cboe").
      - Implement EITHER `_fetch` (single-shard) OR override `pull` directly
        (multi-shard). See module docstring.

    The on-disk layout is enforced here, not by convention. Subclasses must
    NOT bypass `_save_raw` — doing so breaks the vintage-replay contract.
    """

    source: str  # set by subclass

    def __init__(self, raw_root: Path | str = DEFAULT_RAW_ROOT) -> None:
        if not getattr(type(self), "source", None):
            raise TypeError(
                f"{type(self).__name__} must set class attribute `source`."
            )
        self.raw_root = Path(raw_root)

    def _fetch(self, **kwargs: Any) -> tuple[str, pd.DataFrame]:
        """Single-shard fetch: `(product_tag, dataframe)`.

        The default raises. Subclasses doing single-shard pulls override
        this; subclasses doing multi-shard pulls override `pull` instead and
        leave this default in place.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must either implement `_fetch` "
            f"(single-shard) or override `pull` (multi-shard)."
        )

    def pull(self, **kwargs: Any) -> list[RawPullManifest]:
        """Single-shard template: fetch -> stamp -> save -> 1-item manifest list."""
        pulled_at = datetime.now(timezone.utc)
        vintage = make_vintage(pulled_at)
        product, df = self._fetch(**kwargs)

        if df is None or len(df) == 0:
            raise ValueError(
                f"{type(self).__name__}._fetch returned empty frame for "
                f"product={product!r} — refusing to write a zero-row vintage."
            )

        df = df.copy()
        df["_pulled_at"] = pulled_at
        df["_vintage"] = vintage

        path = self._save_raw(df, product, vintage)
        return [
            RawPullManifest(
                source=self.source,
                product=product,
                vintage=vintage,
                path=path,
                pulled_at=pulled_at,
                row_count=len(df),
            )
        ]

    def _save_raw(
        self,
        df: pd.DataFrame,
        product: str,
        vintage: str,
        *,
        shard_key: str = "",
    ) -> Path:
        out_dir = self.raw_root / self.source / product
        out_dir.mkdir(parents=True, exist_ok=True)
        if shard_key:
            out = out_dir / f"{safe_shard_key(shard_key)}_{vintage}.parquet"
        else:
            out = out_dir / f"{vintage}.parquet"
        df.to_parquet(out, index=False)
        return out
