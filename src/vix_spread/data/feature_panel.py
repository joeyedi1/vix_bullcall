"""FeaturePanel — daily-frequency input panel for the regime layer.

ARCHITECTURE §6.2 / §6.4. A typed wrapper around a daily DataFrame plus a
per-column `as_of_map` giving the release timestamp for every row of every
column. The release-boundary rule — a feature with vintage `v` may be
consumed by a decision at `as_of >= v` — is enforced by `slice_causal`,
which is the only sanctioned reader for downstream walk-forward fitters.
Walking the panel by hand and forgetting the as_of cut is the whole class
of bugs this module exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class FeaturePanel:
    """Daily panel + per-column as-of map.

    Attributes
    ----------
    dates
        Tz-aware DatetimeIndex (one row per observation date).
    features
        DataFrame indexed on `dates`. One column per feature.
    as_of_map
        `{column_name -> Series}` aligned to `dates`. Each entry gives the
        release timestamp (when the value first became knowable) for that
        column at that row. For end-of-day prints whose value is published
        at the close, `as_of_map[col][date] == close_of(date)`.
    """
    dates: pd.DatetimeIndex
    features: pd.DataFrame
    as_of_map: dict[str, pd.Series]

    def __post_init__(self) -> None:
        if self.dates.tz is None:
            raise TypeError("FeaturePanel.dates must be tz-aware.")
        if not self.features.index.equals(self.dates):
            raise ValueError("features.index must equal dates.")
        for col in self.features.columns:
            if col not in self.as_of_map:
                raise ValueError(f"as_of_map missing column {col!r}.")
            if not self.as_of_map[col].index.equals(self.dates):
                raise ValueError(f"as_of_map[{col!r}].index must equal dates.")

    def slice_causal(self, column: str, as_of: datetime) -> pd.Series:
        """Return values of `column` for rows whose `as_of_effective <= as_of`.

        This is the release-boundary rule: a value is admissible iff it
        was knowable at `as_of`. Rows with later release timestamps are
        dropped silently — the caller cannot accidentally observe them.
        """
        eff = self.as_of_map[column]
        mask = eff <= pd.Timestamp(as_of)
        return self.features.loc[mask, column]
