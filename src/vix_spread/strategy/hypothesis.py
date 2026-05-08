"""StrategyHypothesis — declared entry-filter contract.

ARCHITECTURE §3.5. The validation memo flagged that "low-vol filter" and
"high-vol entry" describe DIFFERENT strategies. The architecture forces
the user to declare one before signal generation: comparing hypotheses
requires SEPARATE backtest runs, never a flag flipped mid-run. The
hypothesis is constructor-bound on `VIXBullCallSpreadStrategy` and
immutable for the life of the strategy object.

Phase-5 wires the headline `contrarian_tail` hypothesis from
`config/strategy.yaml` (ARCH §10.1):

  hypothesis:
    name: contrarian_tail
    entry_regime_filter: low_vol_with_min_duration
    entry_curve_filter: hyper_contango_30d182d_below_minus_3pct
    expected_holding_period_days: 21

Implementation: factory `make_contrarian_tail_hypothesis` builds the
filters with explicit thresholds. The regime filter requires the HMM
to be in the low-volatility state with a minimum filtered probability;
the curve filter requires the term-structure slope to exceed a contango
threshold (per the §3.4 sign convention: positive slope = contango).

Sign convention reminder
------------------------
ARCH §3.4: `slope = F_far / F_near − 1` — POSITIVE means contango. The
config's "below_minus_3pct" naming reflects an alternate convention
(`F_near / F_far − 1`); the implementation here uses §3.4 throughout.
The threshold default (`min_contango_slope=0.03`) means slope >= 3%.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from vix_spread.regime.base import RegimeSignal


HypothesisName = Literal[
    "contrarian_tail", "breakout_momentum", "curve_normalization",
]


@dataclass(frozen=True)
class StrategyHypothesis:
    """Constructor-bound, immutable entry-filter contract.

    `entry_regime_filter` consumes a `RegimeSignal` and returns True iff
    the regime allows entry. `entry_curve_filter` consumes the signal's
    `curve_features` dict (signed-normalized contango features per
    §3.4) and returns True iff the curve allows entry. BOTH must pass
    for an entry to be considered (logical AND in
    `VIXBullCallSpreadStrategy`).

    `expected_holding_period` informs reporting and exit-policy timing
    but is NOT enforced inside the strategy itself.
    """
    name: HypothesisName
    entry_regime_filter: Callable[["RegimeSignal"], bool]
    entry_curve_filter: Callable[[dict[str, float]], bool]
    expected_holding_period: timedelta


def make_contrarian_tail_hypothesis(
    *,
    low_vol_state_label: int = 0,
    min_filtered_prob: float = 0.7,
    min_contango_slope: float = 0.03,
    curve_feature_key: str = "slope_30_182",
    expected_holding_period: timedelta = timedelta(days=21),
) -> StrategyHypothesis:
    """Build the headline `contrarian_tail` hypothesis (ARCH §10.1).

    Enters when the HMM is decisively in the low-volatility state AND
    the term structure is in hyper-contango — bets on a vol bounce off
    a complacent low.

    Parameters
    ----------
    low_vol_state_label
        The HMM state index that corresponds to "low volatility". Under
        the canonical `state_label_rule='by_emission_variance'`, the
        low-variance state gets label 0; this default matches.
    min_filtered_prob
        Minimum filtered probability of being in the low-vol state for
        the regime filter to pass. Default 0.7 keeps decisive entries
        only — the HMM noise floor is more visible at lower thresholds.
    min_contango_slope
        Minimum `F_far / F_near - 1` (§3.4 sign) for the curve filter
        to pass. Default 0.03 (3%) is the "hyper-contango" threshold
        from the validation memo's regime descriptions.
    curve_feature_key
        Dict key in `RegimeSignal.curve_features` that holds the slope.
        Default `'slope_30_182'` matches the 30-day / 182-day slope
        convention from the regime layer's curve_features module.
    """
    if not (0.0 < min_filtered_prob <= 1.0):
        raise ValueError(
            f"min_filtered_prob must be in (0, 1]; got {min_filtered_prob}."
        )

    def regime_filter(signal: "RegimeSignal") -> bool:
        if signal.state_label != low_vol_state_label:
            return False
        if low_vol_state_label >= len(signal.filtered_probs):
            return False
        return float(signal.filtered_probs[low_vol_state_label]) >= min_filtered_prob

    def curve_filter(features: dict[str, float]) -> bool:
        slope = features.get(curve_feature_key)
        if slope is None:
            return False
        return float(slope) >= min_contango_slope

    return StrategyHypothesis(
        name="contrarian_tail",
        entry_regime_filter=regime_filter,
        entry_curve_filter=curve_filter,
        expected_holding_period=expected_holding_period,
    )
