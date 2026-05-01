"""WalkForwardRegimeFitter — strictly causal Gaussian HMM regime classifier.

ARCHITECTURE §3.3. Refits `hmmlearn.GaussianHMM` on a rolling
`lookback_days` window ending at `as_of` and exposes only filtered
posteriors `P(S_t | y_{1:t})`. Smoothed probabilities and Viterbi paths
are not in the public surface; the forward pass is implemented here
explicitly (rather than via `hmmlearn.score_samples`, which returns
smoothed gammas) so there is no quiet path that leaks `P(S_t | y_{1:T})`
into a production signal — the validation-memo failure mode flagged in
§3.1.

State labelling stability across refits is enforced by the spec's
`state_label_rule` (default `by_emission_variance`): after each fit, raw
HMM state indices are remapped so that label 0 is the lowest-variance
state. Without this, a refit can swap state indices and silently flip
the entire signal series (§12.6).
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from vix_spread.utils.errors import LookaheadError

from .base import FittedRegime, RegimeClassifier, RegimeSignal
from .hmm_spec import HMMSpec

if TYPE_CHECKING:
    from vix_spread.data.feature_panel import FeaturePanel


class WalkForwardRegimeFitter(RegimeClassifier):
    """Walk-forward Gaussian HMM. fit_walk_forward refits on the rolling
    `(as_of − lookback_days, as_of]` window; predict_filtered runs the
    forward algorithm and returns a label-stable RegimeSignal."""

    def __init__(
        self,
        spec: HMMSpec,
        feature_column: str = 'log_vix',
        lookback_days: int = 1260,
        cadence: Literal['daily', 'weekly', 'monthly'] = 'weekly',
        n_iter: int = 50,
        random_state: int = 0,
        hypothesis_tag: Literal['contrarian_tail',
                                'breakout_momentum',
                                'curve_normalization'] = 'contrarian_tail',
    ) -> None:
        spec.validate()
        self.spec = spec
        self.feature_column = feature_column
        self.lookback_days = lookback_days
        self.cadence = cadence
        self.n_iter = n_iter
        self.random_state = random_state
        self.hypothesis_tag = hypothesis_tag
        self.refit_log: list[dict] = []

    # ------------------------------------------------------------------ #
    # ABC contract                                                       #
    # ------------------------------------------------------------------ #

    def fit_walk_forward(
        self,
        history: 'FeaturePanel',
        as_of: datetime,
    ) -> FittedRegime:
        """Slice the panel column to rows whose `as_of_effective <= as_of`
        and whose date is in `(as_of − lookback_days, as_of]`. Fit
        GaussianHMM, apply `state_label_rule`, append to refit_log."""
        col = self.feature_column
        causal = history.slice_causal(col, as_of).dropna()
        if causal.empty:
            raise LookaheadError(
                f"No observations at or before as_of={as_of}; cannot fit."
            )
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=self.lookback_days)
        window = causal[causal.index > cutoff]
        min_obs = self.spec.n_states * 5
        if len(window) < min_obs:
            raise ValueError(
                f"Window too short ({len(window)} obs) to fit "
                f"{self.spec.n_states}-state HMM at as_of={as_of} "
                f"(need >= {min_obs})."
            )
        X = window.to_numpy(dtype=float).reshape(-1, 1)

        model = GaussianHMM(
            n_components=self.spec.n_states,
            covariance_type='diag',
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params='mc',
            params='stmc',
        )
        model.startprob_ = self.spec.implied_stationary()
        model.transmat_ = np.asarray(
            self.spec.transition_matrix, dtype=float,
        ).copy()
        model.fit(X)

        label_map = self._stable_label_map(model)

        timestamps = tuple(
            ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
            for ts in window.index
        )
        observations = tuple(window.tolist())

        self.refit_log.append({
            'as_of': as_of,
            'n_obs': len(observations),
            'transmat': model.transmat_.copy(),
            'means': model.means_.copy(),
            'covars': model.covars_.copy(),
            'label_map': label_map,
        })

        return FittedRegime(
            as_of=as_of,
            observation_timestamps=timestamps,
            observations=observations,
            model=model,
            label_map=label_map,
        )

    def predict_filtered(
        self,
        fitted: FittedRegime,
        as_of: datetime,
    ) -> RegimeSignal:
        """Run the forward algorithm on `fitted.observations` and return
        the filtered posterior at the last observation, remapped to stable
        labels. Raises LookaheadError if any fitted timestamp post-dates
        `as_of` (defensive: FittedRegime is publicly constructible)."""
        if any(ts > as_of for ts in fitted.observation_timestamps):
            raise LookaheadError(
                "fitted contains observations with timestamp > as_of; "
                "predict_filtered must consume only y_{1:t}."
            )
        if fitted.model is None or fitted.label_map is None:
            raise ValueError(
                "FittedRegime missing model/label_map; was it produced by "
                "WalkForwardRegimeFitter.fit_walk_forward?"
            )
        X = np.asarray(fitted.observations, dtype=float).reshape(-1, 1)
        raw_filtered = self._forward_filtered(fitted.model, X)
        relabelled = np.zeros(self.spec.n_states, dtype=float)
        for raw_idx, stable in enumerate(fitted.label_map):
            relabelled[stable] = raw_filtered[raw_idx]
        state = int(np.argmax(relabelled))
        return RegimeSignal(
            as_of=as_of,
            filtered_probs=relabelled,
            state_label=state,
            curve_features={},
            hypothesis_tag=self.hypothesis_tag,
            as_of_inputs={self.feature_column: fitted.observation_timestamps[-1]},
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _stable_label_map(self, model: GaussianHMM) -> tuple[int, ...]:
        """raw_state_index -> stable_label (0 = lowest, n-1 = highest by rule)."""
        if self.spec.state_label_rule == 'by_emission_variance':
            key = model.covars_.reshape(model.covars_.shape[0], -1).sum(axis=1)
        elif self.spec.state_label_rule == 'by_emission_mean':
            key = model.means_.reshape(model.means_.shape[0], -1).sum(axis=1)
        else:
            raise ValueError(
                f"Unknown state_label_rule {self.spec.state_label_rule!r}."
            )
        order = np.argsort(key)
        label_map = np.empty(len(order), dtype=int)
        for stable_label, raw_idx in enumerate(order):
            label_map[raw_idx] = stable_label
        return tuple(label_map.tolist())

    @staticmethod
    def _forward_filtered(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
        """Forward algorithm; returns P(S_T | y_{1:T}) — the FILTERED
        posterior at the last observation. No backward pass, no smoothing.

        Computed in log-space for numerical stability. Gaussian emission
        log-density is computed directly from `model.means_` /
        `model.covars_` rather than via `model._compute_log_likelihood`
        to avoid coupling to hmmlearn internals."""
        T = X.shape[0]
        K = model.n_components
        means = model.means_.reshape(K)
        var = model.covars_.reshape(K)
        x = X.reshape(T)
        # log N(x_t | mu_k, var_k) for each t, k
        log_emission = (
            -0.5 * np.log(2.0 * np.pi * var)[None, :]
            - 0.5 * (x[:, None] - means[None, :]) ** 2 / var[None, :]
        )
        log_pi = np.log(np.maximum(model.startprob_, 1e-300))
        log_A = np.log(np.maximum(model.transmat_, 1e-300))

        log_alpha = log_pi + log_emission[0]
        for t in range(1, T):
            # log_alpha_t(j) = logsumexp_i(log_alpha_{t-1}(i) + log_A[i,j])
            #                  + log_emission[t,j]
            mat = log_alpha[:, None] + log_A             # (K, K)
            m = mat.max(axis=0)                          # (K,)
            log_alpha = m + np.log(np.exp(mat - m[None, :]).sum(axis=0)) \
                        + log_emission[t]
        m = log_alpha.max()
        probs = np.exp(log_alpha - m)
        return probs / probs.sum()
