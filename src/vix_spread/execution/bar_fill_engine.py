"""MinuteBarFillEngine — single-instrument minute-bar fill simulator.

Sits one layer below the ARCH §5 spread fill engine. Operates on the
merged 1-minute panel produced by `DataProcessor` + the regime
broadcaster (and whatever strategy layer writes the per-minute action
column). Cross-the-spread mechanics only: BUY at `last_ask`, SELL at
`last_bid`. Mid / last-trade / close fills are structurally absent —
there is no flag to enable them. The §5 spread engine, when added,
will compose two leg-fills produced by this class.

Per-side staleness gate (§5.3): a BUY rejects on `ask_age_seconds`
> threshold, a SELL on `bid_age_seconds` > threshold. The opposite
side's age clock is irrelevant — a quote you aren't crossing cannot
make you mis-fill.

ARCH §7.2 strict-shift causality is the CALLER'S contract. The action
column at minute *t* means "execute against this bar's quotes." The
broadcaster has already shifted the upstream signal by one minute, so
the engine intentionally does NO further shift — double-shifting would
silently delay every fill by an extra minute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    'last_bid', 'last_ask',
    'bid_age_seconds', 'ask_age_seconds',
    'action',
)


class MinuteBarFillEngine:
    """Per-minute single-leg cross-the-spread fill simulator.

    Parameters
    ----------
    max_quote_age_seconds
        A fill is rejected if the age clock on the SIDE BEING CROSSED
        (ask for BUY, bid for SELL) exceeds this threshold. NaN age is
        treated as effectively infinite (also rejected). Default 300s.
    """

    def __init__(self, max_quote_age_seconds: float = 300.0) -> None:
        self.max_quote_age_seconds = float(max_quote_age_seconds)

    def execute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return an augmented copy of `panel` with three new columns:

          - `fill_price`       float — the cross price; NaN where no fill.
          - `fill_side`        'BUY' | 'SELL' | NaN.
          - `rejection_reason` str | NaN — code if a non-NaN action was
            rejected; NaN otherwise (including for NaN action rows).

        Required columns: `last_bid`, `last_ask`, `bid_age_seconds`,
        `ask_age_seconds`, `action`. Other columns (OHLCV, regime
        signals, broadcaster passengers) pass through untouched.

        Rejection priority on a BUY:
          1. `invalid_action`  — action is non-NaN but not 'BUY' / 'SELL'.
          2. `no_ask`          — `last_ask` is NaN or <= 0.
          3. `stale_quote`     — `ask_age_seconds` > `max_quote_age_seconds`
                                 (NaN treated as infinite).

        Same on SELL with the bid-side fields.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
        if missing:
            raise KeyError(f"panel missing required columns: {missing}")

        out = panel.copy()
        action = out['action']

        is_action = action.notna()
        is_buy = action.eq('BUY').fillna(False)
        is_sell = action.eq('SELL').fillna(False)
        is_invalid = is_action & ~(is_buy | is_sell)

        out['fill_price'] = np.nan
        out['fill_side'] = pd.Series(pd.NA, index=out.index, dtype='object')
        out['rejection_reason'] = pd.Series(pd.NA, index=out.index,
                                            dtype='object')

        # Invalid sentinel — caller bug, not a market condition.
        out.loc[is_invalid, 'rejection_reason'] = 'invalid_action'

        # BUY: cross at last_ask after no_ask / stale_ask gates.
        ask_finite = out['last_ask'].notna() & (out['last_ask'] > 0)
        no_ask = is_buy & ~ask_finite
        ask_age = out['ask_age_seconds'].fillna(np.inf)
        stale_ask = is_buy & ask_finite & (ask_age > self.max_quote_age_seconds)
        ok_buy = is_buy & ask_finite & ~stale_ask

        out.loc[no_ask, 'rejection_reason'] = 'no_ask'
        out.loc[stale_ask, 'rejection_reason'] = 'stale_quote'
        out.loc[ok_buy, 'fill_price'] = out.loc[ok_buy, 'last_ask']
        out.loc[ok_buy, 'fill_side'] = 'BUY'

        # SELL: cross at last_bid after no_bid / stale_bid gates.
        bid_finite = out['last_bid'].notna() & (out['last_bid'] > 0)
        no_bid = is_sell & ~bid_finite
        bid_age = out['bid_age_seconds'].fillna(np.inf)
        stale_bid = is_sell & bid_finite & (bid_age > self.max_quote_age_seconds)
        ok_sell = is_sell & bid_finite & ~stale_bid

        out.loc[no_bid, 'rejection_reason'] = 'no_bid'
        out.loc[stale_bid, 'rejection_reason'] = 'stale_quote'
        out.loc[ok_sell, 'fill_price'] = out.loc[ok_sell, 'last_bid']
        out.loc[ok_sell, 'fill_side'] = 'SELL'

        return out
