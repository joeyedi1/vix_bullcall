"""Phase-6 diagnostic audit trails (ARCHITECTURE §8.3).

Two audit trails for interpreting backtest results:

  - `RegimeAuditTrail` — per-refit HMM stability metrics. Captures
    transition matrices, emission means/variances, label maps, and
    implied stationary distributions across walk-forward refits. The
    consumer asks "did the HMM see consistent regimes across refits?"
    and "what did the model learn that made it call X-VIX 'high-vol'?".

  - `ForwardSelectionAudit` — distribution of forward-selection methods
    across decisions. Reports the breakdown between
    `settlement_date_match` (preferred), `put_call_parity` (when wired),
    `interpolated` (fallback; model-risk-flagged). The consumer asks
    "what fraction of trades used the fallback forward?".

Both trails are READ-ONLY views over data the engine already produces
(`WalkForwardRegimeFitter.refit_log`, `StrategyDecision.evaluation`).
Building them after the fact keeps the engine itself instrument-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import pandas as pd

from vix_spread.pricing.forward_selector import Forward

if TYPE_CHECKING:
    from vix_spread.regime.walk_forward import WalkForwardRegimeFitter
    from vix_spread.strategy.strategy import StrategyDecision


ForwardMethodTag = Literal[
    "settlement_date_match", "put_call_parity", "interpolated",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _implied_stationary(transmat: np.ndarray) -> np.ndarray:
    """Solve `π P = π`, `Σπ = 1` via the left-eigenvector with eigenvalue 1.

    Defensive copy of `HMMSpec.implied_stationary` logic so the audit
    module is standalone; if the HMM transmat has no eigenvalue at 1
    (degenerate / ill-conditioned), returns a uniform distribution and
    relies on the caller to treat it as a smell.
    """
    T = np.asarray(transmat, dtype=float)
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    if abs(eigvals[idx].real - 1.0) > 1e-3:
        return np.ones(T.shape[0]) / T.shape[0]
    vec = np.maximum(eigvecs[:, idx].real, 0.0)
    s = vec.sum()
    if s <= 0:
        return np.ones(T.shape[0]) / T.shape[0]
    return vec / s


def _frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm of (A - B). 0 ⇒ identical; small ⇒ stable refit."""
    return float(np.sqrt(np.sum((A - B) ** 2)))


# --------------------------------------------------------------------------- #
# RegimeAuditTrail                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegimeRefitEntry:
    """One refit snapshot — exactly what the HMM learned at this `as_of`."""
    as_of: datetime
    n_obs: int
    transmat: np.ndarray           # (n_states, n_states)
    means: np.ndarray              # (n_states,)
    variances: np.ndarray          # (n_states,)
    label_map: tuple[int, ...]     # raw_idx -> stable_label
    stationary: np.ndarray         # (n_states,) — implied stationary


@dataclass(frozen=True)
class RegimeAuditTrail:
    """Per-refit HMM stability + emission diagnostics.

    Useful for diagnosing "why did the HMM classify VIX X-Y as high-vol":
    inspect `emission_means_per_state` and `emission_variances_per_state`
    to see what cluster each state represents.

    Between-refit diagnostics:
      - `label_map_consistent` — did the stable label index ↔ raw state
        index mapping stay constant? A flip means downstream signals were
        relabeled silently.
      - `transmat_frobenius_diffs` — Frobenius distance between
        consecutive refits' transition matrices. Spikes indicate refit
        instability.
      - `stationary_drifts` — L2 drift in the implied stationary
        distribution between consecutive refits.
    """
    refits: list[RegimeRefitEntry]

    @classmethod
    def from_walk_forward_fitter(
        cls, fitter: "WalkForwardRegimeFitter",
    ) -> "RegimeAuditTrail":
        """Capture the trail from a fitter's accumulated `refit_log`."""
        entries = []
        for r in fitter.refit_log:
            transmat = np.asarray(r["transmat"], dtype=float)
            entries.append(
                RegimeRefitEntry(
                    as_of=r["as_of"],
                    n_obs=int(r["n_obs"]),
                    transmat=transmat,
                    means=np.asarray(r["means"]).reshape(-1),
                    variances=np.asarray(r["covars"]).reshape(-1),
                    label_map=tuple(r["label_map"]),
                    stationary=_implied_stationary(transmat),
                )
            )
        return cls(refits=entries)

    # ---------------------------------------------------------------- #
    # Stability diagnostics                                            #
    # ---------------------------------------------------------------- #

    def label_map_consistent(self) -> bool:
        """True iff every refit's label_map is identical to the first.

        A flip means the HMM's raw state-0 and state-1 swapped roles
        between refits — silent signal-series inversion if downstream
        code keys on `state_label`.
        """
        if not self.refits:
            return True
        first = self.refits[0].label_map
        return all(r.label_map == first for r in self.refits)

    def label_map_flip_dates(self) -> list[datetime]:
        """`as_of` timestamps of refits whose `label_map` differs from
        the first refit's. Empty list when `label_map_consistent()` is
        True. The first refit is never in the list (it's the reference).
        """
        if not self.refits:
            return []
        first = self.refits[0].label_map
        return [r.as_of for r in self.refits if r.label_map != first]

    def transmat_frobenius_diffs(self) -> pd.Series:
        """Frobenius distance between consecutive refits' transition
        matrices. Index = the LATER refit's `as_of`; value = distance to
        the previous refit. First entry is NaN (no predecessor)."""
        if len(self.refits) <= 1:
            return pd.Series(dtype="float64", name="transmat_frobenius")
        vals = [float("nan")]
        for prev, curr in zip(self.refits, self.refits[1:]):
            vals.append(_frobenius_distance(prev.transmat, curr.transmat))
        return pd.Series(
            vals,
            index=pd.DatetimeIndex([r.as_of for r in self.refits]),
            name="transmat_frobenius",
        )

    def stationary_drifts(self) -> pd.Series:
        """L2 norm of (stationary_t - stationary_{t-1}) per consecutive
        refit pair. First entry NaN."""
        if len(self.refits) <= 1:
            return pd.Series(dtype="float64", name="stationary_drift")
        vals = [float("nan")]
        for prev, curr in zip(self.refits, self.refits[1:]):
            d = float(np.linalg.norm(curr.stationary - prev.stationary))
            vals.append(d)
        return pd.Series(
            vals,
            index=pd.DatetimeIndex([r.as_of for r in self.refits]),
            name="stationary_drift",
        )

    def emission_summary(self) -> pd.DataFrame:
        """Per-refit emission means / variances per stable state label.

        Columns: `mean_state_0`, `var_state_0`, `mean_state_1`, `var_state_1`,
        ... (n_states columns × 2 for mean/var). Index = refit `as_of`.

        Each refit applies its own `label_map` so the column "state_0"
        always refers to the lower-variance state (by_emission_variance
        rule), even when the raw index swaps between refits.
        """
        if not self.refits:
            return pd.DataFrame()
        n_states = len(self.refits[0].label_map)
        cols = []
        for state in range(n_states):
            cols.extend([f"mean_state_{state}", f"var_state_{state}"])
        rows = []
        for r in self.refits:
            row = {}
            for raw_idx, stable_label in enumerate(r.label_map):
                row[f"mean_state_{stable_label}"] = float(r.means[raw_idx])
                row[f"var_state_{stable_label}"] = float(r.variances[raw_idx])
            rows.append(row)
        return pd.DataFrame(
            rows,
            index=pd.DatetimeIndex([r.as_of for r in self.refits]),
            columns=cols,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """One-row-per-refit overview: as_of, n_obs, stationary[k], +
        transmat Frobenius diff vs prior refit."""
        if not self.refits:
            return pd.DataFrame()
        n_states = len(self.refits[0].stationary)
        diffs = self.transmat_frobenius_diffs()
        rows = []
        for r in self.refits:
            row = {"as_of": r.as_of, "n_obs": r.n_obs}
            for s in range(n_states):
                row[f"stationary_{s}"] = float(r.stationary[s])
            row["transmat_diff_vs_prev"] = float(
                diffs.get(pd.Timestamp(r.as_of), float("nan"))
            )
            rows.append(row)
        return pd.DataFrame(rows).set_index("as_of")


# --------------------------------------------------------------------------- #
# ForwardSelectionAudit                                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ForwardSelectionEntry:
    """One forward-selection event recorded for audit."""
    as_of: datetime
    forward_value: float
    selection_method: ForwardMethodTag
    model_risk_flag: bool
    long_strike: float
    short_strike: float
    settlement_date: datetime


@dataclass(frozen=True)
class ForwardSelectionAudit:
    """Distribution of forward-selection methods across decisions.

    Reports the fraction of trades using each forward source, flags any
    `model_risk_flag=True` decisions (the `interpolated` fallback), and
    surfaces the raw audit entries for ad-hoc analysis.

    Reads from `StrategyDecision.evaluation.forward` — every decision
    that produced an evaluation contributes one entry. Decisions that
    skipped before evaluation (regime/curve filter or no_spread) are
    NOT in the audit since no forward was selected.
    """
    entries: list[ForwardSelectionEntry]

    @classmethod
    def from_decisions(
        cls, decisions: Iterable["StrategyDecision"],
    ) -> "ForwardSelectionAudit":
        out: list[ForwardSelectionEntry] = []
        for d in decisions:
            if d.evaluation is None or d.spread is None:
                continue
            fwd = d.evaluation.forward
            out.append(
                ForwardSelectionEntry(
                    as_of=d.as_of,
                    forward_value=float(fwd.value),
                    selection_method=fwd.selection_method,
                    model_risk_flag=bool(fwd.model_risk_flag),
                    long_strike=float(d.spread.long_leg.strike),
                    short_strike=float(d.spread.short_leg.strike),
                    settlement_date=d.spread.long_leg.settlement_event,
                )
            )
        return cls(entries=out)

    # ---------------------------------------------------------------- #
    # Aggregates                                                       #
    # ---------------------------------------------------------------- #

    def method_breakdown(self) -> pd.Series:
        """Counts per `selection_method`, sorted by method name for
        stable column ordering."""
        if not self.entries:
            return pd.Series(dtype="int64", name="count")
        counts = pd.Series(
            [e.selection_method for e in self.entries]
        ).value_counts().rename("count")
        return counts.sort_index()

    def method_fractions(self) -> pd.Series:
        """Fraction of audit entries per selection_method."""
        counts = self.method_breakdown()
        if counts.empty:
            return counts.astype("float64").rename("fraction")
        total = int(counts.sum())
        return (counts / total).astype("float64").rename("fraction")

    def model_risk_count(self) -> int:
        """Count of audited decisions where `model_risk_flag=True`
        (i.e. the `interpolated` fallback)."""
        return sum(1 for e in self.entries if e.model_risk_flag)

    def to_dataframe(self) -> pd.DataFrame:
        cols = [
            "as_of", "forward_value", "selection_method",
            "model_risk_flag",
            "long_strike", "short_strike", "settlement_date",
        ]
        if not self.entries:
            return pd.DataFrame(columns=cols)
        rows = [
            {
                "as_of": e.as_of,
                "forward_value": e.forward_value,
                "selection_method": e.selection_method,
                "model_risk_flag": e.model_risk_flag,
                "long_strike": e.long_strike,
                "short_strike": e.short_strike,
                "settlement_date": e.settlement_date,
            }
            for e in self.entries
        ]
        return pd.DataFrame(rows, columns=cols)
