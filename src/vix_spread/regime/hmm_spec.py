"""HMMSpec — HMM specification with built-in transition↔stationary check.

ARCHITECTURE §3.2 / §12.6. The validation-memo failure mode flagged here
was that the source backtester declared a transition matrix and a
stationary distribution that were mutually inconsistent (`p_low ≈ 0.745`
implied vs `≈ 0.50` declared). Inconsistency is impossible by
construction here: `transition_matrix` is the source of truth and
`stationary_distribution`, if supplied, is cross-checked against the
implied stationary computed from it.

State labelling is a separate concern handled at the fitter (see
`WalkForwardRegimeFitter._stable_label_map`). `state_label_rule` is
carried on the spec so the user's stated convention travels with the
spec object rather than being rediscovered per fitter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


class HMMSpecificationError(ValueError):
    """Raised when an HMMSpec is internally inconsistent."""


@dataclass(frozen=True)
class HMMSpec:
    n_states: int
    transition_matrix: np.ndarray
    state_label_rule: Literal['by_emission_mean',
                              'by_emission_variance'] = 'by_emission_variance'
    emission_family: Literal['gaussian'] = 'gaussian'
    stationary_distribution: np.ndarray | None = None

    def validate(self, atol: float = 1e-6, stationary_atol: float = 1e-3) -> None:
        T = np.asarray(self.transition_matrix, dtype=float)
        if T.shape != (self.n_states, self.n_states):
            raise HMMSpecificationError(
                f"transition_matrix shape {T.shape} != "
                f"({self.n_states}, {self.n_states})."
            )
        if (T < -atol).any():
            raise HMMSpecificationError("transition_matrix has negative entries.")
        if not np.allclose(T.sum(axis=1), 1.0, atol=atol):
            raise HMMSpecificationError(
                f"transition_matrix rows must sum to 1; got {T.sum(axis=1)}."
            )
        if self.stationary_distribution is not None:
            given = np.asarray(self.stationary_distribution, dtype=float)
            if given.shape != (self.n_states,):
                raise HMMSpecificationError(
                    f"stationary_distribution shape {given.shape} != "
                    f"({self.n_states},)."
                )
            implied = self.implied_stationary()
            if not np.allclose(implied, given, atol=stationary_atol):
                raise HMMSpecificationError(
                    f"stationary_distribution {given} inconsistent with "
                    f"implied {implied} from transition_matrix."
                )

    def implied_stationary(self) -> np.ndarray:
        """Solve `π P = π`, `Σπ = 1` via the left-eigenvector with eigenvalue 1."""
        T = np.asarray(self.transition_matrix, dtype=float)
        eigvals, eigvecs = np.linalg.eig(T.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        if abs(eigvals[idx].real - 1.0) > 1e-3:
            raise HMMSpecificationError(
                f"No eigenvalue near 1 in transition matrix; got {eigvals}."
            )
        vec = np.maximum(eigvecs[:, idx].real, 0.0)
        s = vec.sum()
        if s <= 0:
            raise HMMSpecificationError(
                f"Stationary eigenvector is non-positive: {eigvecs[:, idx].real}."
            )
        return vec / s
