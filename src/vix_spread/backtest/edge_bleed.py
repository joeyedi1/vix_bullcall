"""Edge-bleed diagnostic (ARCHITECTURE §8.3).

Quantifies execution drag: the difference between Black-76 theoretical
spread value and the executed debit, per filled trade. The distribution
of bleed is the headline summary of "what the strategy left on the
table to microstructure".

Sign convention (matches `SpreadEvaluation.edge_bleed`):
  - `edge_bleed = theoretical - executed_debit`
  - **POSITIVE**: paid LESS than fair value — favourable (rare for
    debit spreads crossed against a non-zero NBBO width).
  - **NEGATIVE**: paid MORE than fair value — the typical case;
    quantifies the bid-ask edge bleed at entry.

Pairing trades with decisions
-----------------------------
`CompletedTrade` carries the actual T+1 fill but NOT the original
`SpreadEvaluation.theoretical` from T. We pair by joining
`completed_trades` against `decisions_log` on
`(spread.long_leg.strike, spread.short_leg.strike, spread.long_leg.expiry)`
and picking the most recent enter-decision strictly before the fill
timestamp. This is robust because:

  - Each fill comes from exactly one decision (single-attempt at T+1).
  - The strike+expiry tuple uniquely identifies the spread within an
    `as_of` minute (the selector picks one spread per evaluation).
  - The pairing is decision -> trade, not the other way (multiple skip
    decisions can precede a single trade).

A trade whose paired decision can't be found in `decisions_log` is
silently skipped — that's expected for backtests where `decisions_log`
was filtered upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .results import BacktestResults, CompletedTrade

if TYPE_CHECKING:
    from vix_spread.products.spread import BullCallSpread
    from vix_spread.strategy.strategy import StrategyDecision


# --------------------------------------------------------------------------- #
# Entry + audit dataclasses                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EdgeBleedEntry:
    """Per-trade edge-bleed audit row."""
    decision_as_of: datetime           # T (signal time)
    fill_timestamp: datetime           # T+1 (entry fill)
    spread: "BullCallSpread"
    size: int
    theoretical_preview: float         # SpreadEvaluation.theoretical.value at T
    executed_debit: float              # ExecutedFill.debit_per_spread at T+1
    edge_bleed_per_spread: float       # theoretical - executed
    edge_bleed_dollars: float          # × multiplier × size
    iv_long_source: str                # 'vendor' | 'b76_inverted'
    iv_short_source: str


@dataclass(frozen=True)
class EdgeBleedAudit:
    """Per-trade edge-bleed distribution + summary.

    Build via `from_results(results)`. Inspect via `to_dataframe()`,
    `summary()`, or the per-dimension breakdowns (by IV source, by
    expiry).
    """
    entries: list[EdgeBleedEntry]

    # ---------------------------------------------------------------- #
    # Constructors                                                     #
    # ---------------------------------------------------------------- #

    @classmethod
    def from_results(cls, results: BacktestResults) -> "EdgeBleedAudit":
        """Pair each `CompletedTrade.entry_fill` with the originating
        `StrategyDecision` (matched on spread strikes + expiry, with the
        most recent decision strictly before the fill timestamp).
        """
        # Group enter-decisions with evaluations by spread identity.
        by_key: dict[tuple, list["StrategyDecision"]] = {}
        for d in results.decisions_log:
            if d.action != "enter" or d.spread is None or d.evaluation is None:
                continue
            key = _spread_key(d.spread)
            by_key.setdefault(key, []).append(d)
        for v in by_key.values():
            v.sort(key=lambda d: d.as_of)

        entries: list[EdgeBleedEntry] = []
        for trade in results.completed_trades:
            sp = trade.spread
            key = _spread_key(sp)
            candidates = by_key.get(key, ())
            fill_ts = trade.entry_fill.timestamp if trade.entry_fill else None
            if fill_ts is None or not candidates:
                continue
            # Most recent decision strictly before the fill.
            matching = [d for d in candidates if d.as_of < fill_ts]
            if not matching:
                continue
            decision = matching[-1]

            theoretical = float(decision.evaluation.theoretical.value)
            executed = float(trade.entry_fill.debit_per_spread)
            per_spread_bleed = theoretical - executed
            multiplier = sp.long_leg.option_multiplier()
            dollar_bleed = per_spread_bleed * multiplier * trade.size

            entries.append(
                EdgeBleedEntry(
                    decision_as_of=decision.as_of,
                    fill_timestamp=fill_ts,
                    spread=sp,
                    size=trade.size,
                    theoretical_preview=theoretical,
                    executed_debit=executed,
                    edge_bleed_per_spread=per_spread_bleed,
                    edge_bleed_dollars=dollar_bleed,
                    iv_long_source=decision.evaluation.iv_long.source.value,
                    iv_short_source=decision.evaluation.iv_short.source.value,
                )
            )
        return cls(entries=entries)

    # ---------------------------------------------------------------- #
    # Aggregates                                                       #
    # ---------------------------------------------------------------- #

    def per_spread_distribution(self) -> pd.Series:
        """Series of per-spread bleed values (theoretical − executed)."""
        if not self.entries:
            return pd.Series(dtype="float64", name="edge_bleed_per_spread")
        return pd.Series(
            [e.edge_bleed_per_spread for e in self.entries],
            name="edge_bleed_per_spread",
        )

    def dollars_distribution(self) -> pd.Series:
        """Series of dollar-bleed values (per-spread × multiplier × size)."""
        if not self.entries:
            return pd.Series(dtype="float64", name="edge_bleed_dollars")
        return pd.Series(
            [e.edge_bleed_dollars for e in self.entries],
            name="edge_bleed_dollars",
        )

    def summary(self) -> dict[str, float]:
        """Headline numbers: total / mean / median / std / p5 / p95
        of dollar bleed; same six for per-spread bleed."""
        if not self.entries:
            return {
                "n_trades": 0,
                "total_dollars": 0.0,
                "mean_dollars": float("nan"),
                "median_dollars": float("nan"),
                "std_dollars": float("nan"),
                "p5_dollars": float("nan"),
                "p95_dollars": float("nan"),
                "mean_per_spread": float("nan"),
                "median_per_spread": float("nan"),
            }
        dol = self.dollars_distribution().to_numpy()
        per = self.per_spread_distribution().to_numpy()
        return {
            "n_trades": int(len(self.entries)),
            "total_dollars": float(dol.sum()),
            "mean_dollars": float(dol.mean()),
            "median_dollars": float(np.median(dol)),
            "std_dollars": float(dol.std(ddof=0)),
            "p5_dollars": float(np.percentile(dol, 5)),
            "p95_dollars": float(np.percentile(dol, 95)),
            "mean_per_spread": float(per.mean()),
            "median_per_spread": float(np.median(per)),
        }

    def by_iv_source_breakdown(self) -> pd.DataFrame:
        """Group entries by `(iv_long_source, iv_short_source)` pair.
        Surfaces whether bleed differs systematically between vendor-IV
        trades and B76-inverted trades (e.g., the inversion fallback
        might produce different bleed magnitudes if the chain mid
        diverges from the actual fill price)."""
        if not self.entries:
            return pd.DataFrame(
                columns=["iv_long_source", "iv_short_source",
                         "n_trades", "mean_dollars", "total_dollars"]
            )
        df = self.to_dataframe()
        grp = df.groupby(["iv_long_source", "iv_short_source"], as_index=False)
        return grp["edge_bleed_dollars"].agg(
            n_trades="count",
            mean_dollars="mean",
            total_dollars="sum",
        )

    def to_dataframe(self) -> pd.DataFrame:
        cols = [
            "decision_as_of", "fill_timestamp",
            "long_strike", "short_strike", "expiry",
            "size",
            "theoretical_preview", "executed_debit",
            "edge_bleed_per_spread", "edge_bleed_dollars",
            "iv_long_source", "iv_short_source",
        ]
        if not self.entries:
            return pd.DataFrame(columns=cols)
        rows = [
            {
                "decision_as_of": e.decision_as_of,
                "fill_timestamp": e.fill_timestamp,
                "long_strike": float(e.spread.long_leg.strike),
                "short_strike": float(e.spread.short_leg.strike),
                "expiry": e.spread.long_leg.expiry,
                "size": int(e.size),
                "theoretical_preview": e.theoretical_preview,
                "executed_debit": e.executed_debit,
                "edge_bleed_per_spread": e.edge_bleed_per_spread,
                "edge_bleed_dollars": e.edge_bleed_dollars,
                "iv_long_source": e.iv_long_source,
                "iv_short_source": e.iv_short_source,
            }
            for e in self.entries
        ]
        return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _spread_key(spread: "BullCallSpread") -> tuple:
    """Tuple identifier used to pair decisions with trades."""
    return (
        float(spread.long_leg.strike),
        float(spread.short_leg.strike),
        spread.long_leg.expiry,
    )
