from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from vix_spread.regime.hmm_filter import MinimalHMMFilter


@dataclass(frozen=True)
class _StubPanel:
    """Spike-phase duck-typed FeaturePanel: just enough surface area for
    the causal contract. The real panel lives in
    vix_spread.data.feature_panel and adds an as_of_map per column."""
    timestamps: tuple
    values: tuple


def test_predict_filtered_ignores_future_spike():
    """Validation-memo constraint: predict_filtered(t) consumes only y_{1:t}.
    Injecting a row with timestamp > as_of MUST NOT change the filtered
    probability vector for as_of. The spike value is implausibly large so
    that any leak would be loud."""
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    timestamps = tuple(base + timedelta(days=i) for i in range(10))
    values = (12.1, 13.5, 11.8, 14.2, 12.9, 13.0, 12.5, 14.8, 13.7, 12.3)
    as_of = timestamps[-1]

    clean_panel = _StubPanel(timestamps=timestamps, values=values)

    spike_ts = as_of + timedelta(days=5)
    spike_value = 999.0  # implausibly large; would dominate any leak
    spiked_panel = _StubPanel(
        timestamps=timestamps + (spike_ts,),
        values=values + (spike_value,),
    )

    classifier = MinimalHMMFilter()

    fitted_clean = classifier.fit_walk_forward(clean_panel, as_of=as_of)
    fitted_spiked = classifier.fit_walk_forward(spiked_panel, as_of=as_of)

    signal_clean = classifier.predict_filtered(fitted_clean, as_of=as_of)
    signal_spiked = classifier.predict_filtered(fitted_spiked, as_of=as_of)

    assert np.allclose(signal_clean.filtered_probs, signal_spiked.filtered_probs)
    assert signal_clean.state_label == signal_spiked.state_label
