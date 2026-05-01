"""Strict-causality broadcaster: daily EOD signal → 1-minute grid.

ARCHITECTURE §7.2 timing rule. A signal stamped at the close of day T may
inform decisions only on bars STRICTLY AFTER that close. The mapping is
implemented via `pd.merge_asof` with `direction='backward'` and
`allow_exact_matches=False`, so the bar whose timestamp equals the daily
signal's `as_of` carries the PRIOR signal — never the same-timestamp one.
That is the structural defense against the "same-minute fill" leak.

Phase-3 scope: timestamp-strict, not session-snapped. A future iteration
can replace the merge with a snap-to-next-CFE-session-open once the
session calendar is in the repo. Until then, "next 1-min bar after
close" is a strict superset of "next session open" and is provably
leak-free.
"""
from __future__ import annotations

import pandas as pd


def broadcast_daily_to_minute(
    daily_signals: pd.DataFrame,
    minute_grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Map each minute in `minute_grid` to the most recent row of
    `daily_signals` whose `as_of` (the index) is STRICTLY before that
    minute. Bars before the first daily signal carry NaN.

    Parameters
    ----------
    daily_signals
        DataFrame indexed on tz-aware as_of timestamps. Columns are the
        signal payload (state_label, p_low, etc.) carried as-is to the
        output.
    minute_grid
        Tz-aware DatetimeIndex of 1-minute bar timestamps (e.g., the
        master grid produced by DataProcessor.build_master_grid).

    Returns
    -------
    DataFrame indexed on `minute_grid`, with all `daily_signals` columns
    plus an `as_of_effective` column carrying the source signal's `as_of`
    (NaT for bars before the first signal). Useful for downstream audit:
    every minute-bar consumer can verify which daily decision it acted on.
    """
    if minute_grid.tz is None:
        raise TypeError("minute_grid must be tz-aware.")
    if daily_signals.index.tz is None:
        raise TypeError("daily_signals.index must be tz-aware.")
    if 'as_of_effective' in daily_signals.columns:
        raise ValueError(
            "daily_signals already contains 'as_of_effective'; the "
            "broadcaster reserves this column."
        )

    daily = daily_signals.sort_index().copy()
    daily['as_of_effective'] = daily.index

    sorted_grid = minute_grid.sort_values()
    minute_df = pd.DataFrame(index=sorted_grid)
    out = pd.merge_asof(
        minute_df,
        daily,
        left_index=True,
        right_index=True,
        direction='backward',
        allow_exact_matches=False,  # strict shift: as_of itself never matches
    )
    return out.reindex(minute_grid)
