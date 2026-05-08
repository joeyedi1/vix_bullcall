"""StrategyHypothesis dataclass + the `contrarian_tail` factory."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from vix_spread.regime.base import RegimeSignal
from vix_spread.strategy.hypothesis import (
    StrategyHypothesis,
    make_contrarian_tail_hypothesis,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


AS_OF = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)


def _signal(
    *,
    state_label: int = 0,
    filtered_probs: tuple[float, float] = (0.85, 0.15),
    curve_features: dict[str, float] | None = None,
) -> RegimeSignal:
    return RegimeSignal(
        as_of=AS_OF,
        filtered_probs=np.array(filtered_probs),
        state_label=state_label,
        curve_features=curve_features or {"slope_30_182": 0.05},
        hypothesis_tag="contrarian_tail",
        as_of_inputs={"log_vix": AS_OF},
    )


# --------------------------------------------------------------------------- #
# StrategyHypothesis dataclass                                                #
# --------------------------------------------------------------------------- #


def test_hypothesis_dataclass_is_frozen_and_carries_required_fields():
    hyp = StrategyHypothesis(
        name="contrarian_tail",
        entry_regime_filter=lambda s: True,
        entry_curve_filter=lambda f: True,
        expected_holding_period=timedelta(days=21),
    )
    assert hyp.name == "contrarian_tail"
    assert hyp.entry_regime_filter(_signal()) is True
    assert hyp.entry_curve_filter({"slope_30_182": 0.05}) is True
    assert hyp.expected_holding_period == timedelta(days=21)
    # Frozen — backtest engines can rely on immutability.
    with pytest.raises(Exception):
        hyp.name = "breakout_momentum"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# contrarian_tail factory — defaults                                          #
# --------------------------------------------------------------------------- #


def test_contrarian_tail_passes_when_low_vol_decisive_and_hyper_contango():
    hyp = make_contrarian_tail_hypothesis()
    sig = _signal(
        state_label=0, filtered_probs=(0.85, 0.15),
        curve_features={"slope_30_182": 0.05},      # 5% contango
    )
    assert hyp.entry_regime_filter(sig) is True
    assert hyp.entry_curve_filter(sig.curve_features) is True


def test_contrarian_tail_skips_when_in_high_vol_state():
    hyp = make_contrarian_tail_hypothesis()
    sig = _signal(state_label=1, filtered_probs=(0.15, 0.85))
    assert hyp.entry_regime_filter(sig) is False


def test_contrarian_tail_skips_when_low_vol_prob_below_threshold():
    """Default min_filtered_prob=0.7. State_label is correct (low-vol),
    but the HMM isn't decisive enough."""
    hyp = make_contrarian_tail_hypothesis()
    sig = _signal(state_label=0, filtered_probs=(0.55, 0.45))
    assert hyp.entry_regime_filter(sig) is False


def test_contrarian_tail_skips_when_curve_in_backwardation():
    hyp = make_contrarian_tail_hypothesis()
    assert hyp.entry_curve_filter({"slope_30_182": -0.04}) is False


def test_contrarian_tail_skips_when_contango_below_threshold():
    """Default min_contango_slope=0.03 (3%). 1% contango is not enough."""
    hyp = make_contrarian_tail_hypothesis()
    assert hyp.entry_curve_filter({"slope_30_182": 0.01}) is False


def test_contrarian_tail_skips_when_curve_feature_missing():
    """Curve features dict is missing the expected key — defensive False
    (no assumption about the curve, so no entry)."""
    hyp = make_contrarian_tail_hypothesis()
    assert hyp.entry_curve_filter({}) is False
    assert hyp.entry_curve_filter({"some_other_key": 0.1}) is False


# --------------------------------------------------------------------------- #
# contrarian_tail factory — parameterised                                     #
# --------------------------------------------------------------------------- #


def test_contrarian_tail_respects_custom_state_label():
    """Some HMM seeds put low-vol at label=1 instead of 0."""
    hyp = make_contrarian_tail_hypothesis(low_vol_state_label=1)
    sig = _signal(state_label=1, filtered_probs=(0.15, 0.85))
    assert hyp.entry_regime_filter(sig) is True
    sig_off = _signal(state_label=0, filtered_probs=(0.85, 0.15))
    assert hyp.entry_regime_filter(sig_off) is False


def test_contrarian_tail_respects_custom_thresholds():
    hyp = make_contrarian_tail_hypothesis(
        min_filtered_prob=0.95, min_contango_slope=0.10,
    )
    # 0.85 prob is no longer enough; 5% contango is no longer enough.
    sig = _signal(filtered_probs=(0.85, 0.15),
                  curve_features={"slope_30_182": 0.05})
    assert hyp.entry_regime_filter(sig) is False
    assert hyp.entry_curve_filter(sig.curve_features) is False


def test_contrarian_tail_respects_custom_curve_feature_key():
    hyp = make_contrarian_tail_hypothesis(curve_feature_key="slope_m1_m2")
    assert hyp.entry_curve_filter({"slope_m1_m2": 0.05}) is True
    assert hyp.entry_curve_filter({"slope_30_182": 0.05}) is False


def test_contrarian_tail_rejects_invalid_min_filtered_prob():
    for bad in (-0.01, 0.0, 1.5):
        with pytest.raises(ValueError):
            make_contrarian_tail_hypothesis(min_filtered_prob=bad)


def test_contrarian_tail_returns_hypothesis_with_correct_metadata():
    hyp = make_contrarian_tail_hypothesis(
        expected_holding_period=timedelta(days=14),
    )
    assert hyp.name == "contrarian_tail"
    assert hyp.expected_holding_period == timedelta(days=14)
