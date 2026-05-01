from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from vix_spread.utils.errors import LookaheadError

from .base import FittedRegime, RegimeClassifier, RegimeSignal

if TYPE_CHECKING:
    from vix_spread.data.feature_panel import FeaturePanel


class MinimalHMMFilter(RegimeClassifier):
    """Spike-phase 2-state HMM stub. NOT a production classifier — its only
    job is to demonstrate that the strict-causal contract is enforceable
    end-to-end. fit_walk_forward slices the panel to timestamps <= as_of;
    predict_filtered emits a RegimeSignal whose state assignment depends
    only on data <= as_of. There is no public smoothing or Viterbi API."""

    def fit_walk_forward(
        self,
        history: 'FeaturePanel',
        as_of: datetime,
    ) -> FittedRegime:
        """Slices history to timestamps <= as_of. Future-stamped rows are
        dropped — they cannot influence any subsequent predict_filtered
        call. Raises LookaheadError if no observation survives the slice
        (cannot fit on an empty causal window)."""
        kept_ts: list[datetime] = []
        kept_obs: list[float] = []
        for ts, val in zip(history.timestamps, history.values):
            if ts <= as_of:
                kept_ts.append(ts)
                kept_obs.append(float(val))
        if not kept_ts:
            raise LookaheadError(
                f"No observations at or before as_of={as_of}; cannot fit."
            )
        return FittedRegime(
            as_of=as_of,
            observation_timestamps=tuple(kept_ts),
            observations=tuple(kept_obs),
        )

    def predict_filtered(
        self,
        fitted: FittedRegime,
        as_of: datetime,
    ) -> RegimeSignal:
        """Spike-phase: filtered probabilities are a deterministic function
        of the observations carried in `fitted`. Defensively re-checks that
        no fitted timestamp exceeds as_of and raises LookaheadError if so —
        FittedRegime is a public dataclass and could be hand-constructed
        with bad timestamps. State 0 is labelled low-vol (fraction of obs
        at or below the sample mean); state 1 is labelled high-vol."""
        if any(ts > as_of for ts in fitted.observation_timestamps):
            raise LookaheadError(
                "fitted contains observations with timestamp > as_of; "
                "predict_filtered must consume only y_{1:t}."
            )
        n = len(fitted.observations)
        mean_obs = sum(fitted.observations) / n
        below = sum(1 for x in fitted.observations if x <= mean_obs) / n
        probs = np.array([below, 1.0 - below], dtype=float)
        state = int(np.argmax(probs))
        return RegimeSignal(
            as_of=as_of,
            filtered_probs=probs,
            state_label=state,
            curve_features={},
            hypothesis_tag='contrarian_tail',
            as_of_inputs={'observations': fitted.observation_timestamps[-1]},
        )
