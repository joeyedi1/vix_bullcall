from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from vix_spread.data.feature_panel import FeaturePanel


@dataclass(frozen=True)
class RegimeSignal:
    """Output of the regime engine. ALL fields are functions of data with
    timestamp <= as_of. The as_of_inputs map is audited by the
    FeatureAvailability validator on every signal generation."""
    as_of: datetime                        # decision timestamp; action is AFTER this
    filtered_probs: np.ndarray             # P(S_t | y_{1:t}), shape (n_states,)
    state_label: int                       # argmax of filtered_probs, label-stable
    curve_features: dict[str, float]       # signed-normalized contango features
    hypothesis_tag: Literal['contrarian_tail',
                            'breakout_momentum',
                            'curve_normalization']
    as_of_inputs: dict[str, datetime]      # per-input as-of timestamps; audited


@dataclass(frozen=True)
class FittedRegime:
    """Spike-phase placeholder for the parameters returned by
    fit_walk_forward. Carries the slice of observations that survived the
    causal cutoff so predict_filtered can reproduce a deterministic signal
    without re-slicing the panel. Production replacement will hold HMM
    transition / emission parameters instead of raw observations."""
    as_of: datetime
    observation_timestamps: tuple[datetime, ...]
    observations: tuple[float, ...]


class RegimeClassifier(ABC):
    """ABC for regime classifiers. Implementations MUST be strictly causal.

    Hard rule: predict_filtered(t) may only consume data with timestamp <= t.
    Smoothed probabilities P(S_t | y_{1:T}) and full-sample Viterbi paths
    are FORBIDDEN as production signals — they are not exposed through any
    public method on this class. Implementations may compute smoothed paths
    internally for diagnostics but must not return them.
    """

    @abstractmethod
    def fit_walk_forward(
        self,
        history: 'FeaturePanel',
        as_of: datetime,
    ) -> FittedRegime:
        """Fit using only data with timestamp <= as_of. The validator inspects
        the panel slice taken and raises LookaheadError on any violation."""

    @abstractmethod
    def predict_filtered(
        self,
        fitted: FittedRegime,
        as_of: datetime,
    ) -> RegimeSignal:
        """Returns filtered probability P(S_t | y_{1:t}) ONLY. Must not
        invoke any internal smoothing routine. Unit-tested by injecting a
        future-stamped spike into the panel and asserting the filtered
        probability for as_of is unchanged."""
