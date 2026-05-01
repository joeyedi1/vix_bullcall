"""Tests for WalkForwardRegimeFitter.

Three contracts:
  1. Strict causality: a row whose as_of_effective > as_of must NOT
     change the fitted model or the predicted filtered probabilities.
  2. State labelling stability: a label-stable fit on synthetic 2-regime
     data assigns label 0 to the lower-variance state regardless of
     hmmlearn's internal state ordering (which can swap arbitrarily).
  3. predict_filtered raises LookaheadError if a hand-constructed
     FittedRegime carries observations stamped after as_of.
"""
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from vix_spread.data.feature_panel import FeaturePanel
from vix_spread.regime.base import FittedRegime
from vix_spread.regime.hmm_spec import HMMSpec
from vix_spread.regime.walk_forward import WalkForwardRegimeFitter
from vix_spread.utils.errors import LookaheadError


def _two_regime_panel(n: int = 400, seed: int = 7):
    """Synthetic log-VIX panel: low-variance state then high-variance state.
    Long enough to give the HMM something to actually learn."""
    rng = np.random.default_rng(seed)
    n_low = n // 2
    low = rng.normal(loc=2.5, scale=0.05, size=n_low)
    high = rng.normal(loc=2.9, scale=0.30, size=n - n_low)
    obs = np.concatenate([low, high])

    base = datetime(2024, 1, 1, 21, 0, tzinfo=timezone.utc)
    dates = pd.DatetimeIndex(
        [base + timedelta(days=i) for i in range(n)], name='date',
    )
    features = pd.DataFrame({'log_vix': obs}, index=dates)
    as_of_map = {'log_vix': pd.Series(dates, index=dates)}
    return FeaturePanel(dates=dates, features=features, as_of_map=as_of_map)


def _spec():
    return HMMSpec(
        n_states=2,
        transition_matrix=np.array([[0.96, 0.04], [0.117, 0.883]]),
        state_label_rule='by_emission_variance',
    )


@pytest.fixture(autouse=True)
def _silence_hmm_convergence_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_future_row_does_not_change_fit():
    """Add a row whose as_of_effective is AFTER the decision time. The
    release-boundary slice in fit_walk_forward must drop it; the fitted
    model and filtered posterior must be byte-identical to the no-future
    fit. This is the structural anti-leak guarantee of §3.1."""
    panel = _two_regime_panel()
    as_of = panel.dates[-1].to_pydatetime()

    # Same data + a row whose value is implausible AND released later.
    spike_date = panel.dates[-1] + pd.Timedelta(days=1)
    spike_eff = pd.Timestamp(as_of) + pd.Timedelta(days=10)
    dates2 = panel.dates.append(pd.DatetimeIndex([spike_date]))
    feat2 = pd.concat([
        panel.features,
        pd.DataFrame({'log_vix': [999.0]}, index=[spike_date]),
    ])
    eff2 = pd.concat([
        panel.as_of_map['log_vix'],
        pd.Series([spike_eff], index=[spike_date]),
    ])
    spiked = FeaturePanel(dates=dates2, features=feat2,
                          as_of_map={'log_vix': eff2})

    fitter_clean = WalkForwardRegimeFitter(spec=_spec(), random_state=0)
    fitter_spike = WalkForwardRegimeFitter(spec=_spec(), random_state=0)

    fit_clean = fitter_clean.fit_walk_forward(panel, as_of=as_of)
    fit_spike = fitter_spike.fit_walk_forward(spiked, as_of=as_of)

    sig_clean = fitter_clean.predict_filtered(fit_clean, as_of=as_of)
    sig_spike = fitter_spike.predict_filtered(fit_spike, as_of=as_of)

    # Same fitted parameters and same filtered posterior.
    np.testing.assert_array_equal(fit_clean.observations, fit_spike.observations)
    np.testing.assert_allclose(sig_clean.filtered_probs, sig_spike.filtered_probs)
    assert sig_clean.state_label == sig_spike.state_label


def test_state_label_zero_is_lowest_variance():
    """After fit, label 0 must correspond to the lower-variance emission
    regardless of hmmlearn's internal raw-state index assignment."""
    panel = _two_regime_panel()
    as_of = panel.dates[-1].to_pydatetime()
    fitter = WalkForwardRegimeFitter(spec=_spec(), random_state=0)
    fitted = fitter.fit_walk_forward(panel, as_of=as_of)
    model = fitted.model
    raw_variances = model.covars_.reshape(model.covars_.shape[0], -1).sum(axis=1)
    # label_map[raw] = stable_label; the raw with the smallest variance
    # must map to stable label 0.
    raw_min_var = int(np.argmin(raw_variances))
    assert fitted.label_map[raw_min_var] == 0


def test_predict_filtered_rejects_future_stamped_fitted():
    """FittedRegime is publicly constructible; predict_filtered must
    defensively raise LookaheadError if its caller hands over an object
    whose observations post-date the decision."""
    fitter = WalkForwardRegimeFitter(spec=_spec(), random_state=0)
    as_of = datetime(2026, 1, 10, tzinfo=timezone.utc)
    later = as_of + timedelta(days=1)
    bogus = FittedRegime(
        as_of=as_of,
        observation_timestamps=(as_of, later),
        observations=(2.5, 2.6),
        model=object(),       # non-None to bypass the missing-model branch
        label_map=(0, 1),
    )
    with pytest.raises(LookaheadError):
        fitter.predict_filtered(bogus, as_of=as_of)


def test_window_too_short_raises():
    """A panel with fewer than n_states*5 rows in the lookback window
    cannot fit and must raise rather than emit a noisy bogus signal."""
    base = datetime(2026, 1, 1, 21, 0, tzinfo=timezone.utc)
    dates = pd.DatetimeIndex(
        [base + timedelta(days=i) for i in range(3)], name='date',
    )
    features = pd.DataFrame({'log_vix': [2.5, 2.6, 2.7]}, index=dates)
    as_of_map = {'log_vix': pd.Series(dates, index=dates)}
    panel = FeaturePanel(dates=dates, features=features, as_of_map=as_of_map)
    fitter = WalkForwardRegimeFitter(spec=_spec())
    with pytest.raises(ValueError, match="Window too short"):
        fitter.fit_walk_forward(panel, as_of=dates[-1].to_pydatetime())
