"""Audit trail tests — RegimeAuditTrail + ForwardSelectionAudit.

Synthetic in-memory tests; no real Bloomberg or HMM fitting needed.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vix_spread.backtest.audit import (
    ForwardSelectionAudit,
    ForwardSelectionEntry,
    RegimeAuditTrail,
    RegimeRefitEntry,
    _frobenius_distance,
    _implied_stationary,
)
from vix_spread.pricing.evaluator import SpreadEvaluation
from vix_spread.pricing.forward_selector import Forward
from vix_spread.pricing.leg_iv import LegIV, LegIVSource
from vix_spread.pricing.theoretical import (
    TheoreticalPrice,
    TheoreticalSpreadPrice,
)
from vix_spread.products.spread import BullCallSpread
from vix_spread.products.vix_index_option import VIXIndexOption
from vix_spread.strategy.strategy import StrategyDecision


SOQ_DT = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _refit_entry(
    *, as_of: datetime, n_obs: int, transmat: list[list[float]],
    means: list[float], variances: list[float],
    label_map: tuple[int, ...],
) -> RegimeRefitEntry:
    transmat_np = np.asarray(transmat, dtype=float)
    return RegimeRefitEntry(
        as_of=as_of,
        n_obs=n_obs,
        transmat=transmat_np,
        means=np.asarray(means),
        variances=np.asarray(variances),
        label_map=label_map,
        stationary=_implied_stationary(transmat_np),
    )


def _spread(long_strike: float = 20.0, short_strike: float = 22.0) -> BullCallSpread:
    return BullCallSpread(
        long_leg=VIXIndexOption(
            contract_root="VIX", expiry=SOQ_DT, settlement_event=SOQ_DT,
            strike=long_strike, right="call",
        ),
        short_leg=VIXIndexOption(
            contract_root="VIX", expiry=SOQ_DT, settlement_event=SOQ_DT,
            strike=short_strike, right="call",
        ),
    )


def _forward(method: str = "settlement_date_match", model_risk: bool = False) -> Forward:
    return Forward(
        value=22.0,
        selection_method=method,
        model_risk_flag=model_risk,
        settlement_date=SOQ_DT,
    )


def _theoretical_price(forward: Forward) -> TheoreticalPrice:
    return TheoreticalPrice(
        value=1.0, delta=0.5, gamma=0.1, vega=0.2, theta=-0.05,
        forward_used=forward, iv_used=0.5,
        T_minutes=10_000.0, is_executable=False, rho=0.0,
    )


def _evaluation(
    spread: BullCallSpread, forward: Forward,
) -> SpreadEvaluation:
    long_tp = _theoretical_price(forward)
    short_tp = _theoretical_price(forward)
    th = TheoreticalSpreadPrice(
        value=long_tp.value - short_tp.value,
        long_leg=long_tp, short_leg=short_tp,
        delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0,
        is_executable=False,
    )
    return SpreadEvaluation(
        spread=spread,
        as_of=datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
        forward=forward,
        iv_long=LegIV(value=0.5, source=LegIVSource.VENDOR),
        iv_short=LegIV(value=0.45, source=LegIVSource.VENDOR),
        theoretical=th,
        fill=None,  # type: ignore[arg-type]  # fill is irrelevant for this audit
    )


def _decision_with_forward(
    as_of: datetime, forward: Forward,
    long_strike: float = 20.0, short_strike: float = 22.0,
) -> StrategyDecision:
    sp = _spread(long_strike, short_strike)
    return StrategyDecision(
        as_of=as_of, action="enter", reason="entered",
        hypothesis_name="test",
        spread=sp,
        evaluation=_evaluation(sp, forward),
        size=1,
    )


# --------------------------------------------------------------------------- #
# Helper math                                                                 #
# --------------------------------------------------------------------------- #


def test_implied_stationary_matches_known_2state():
    """[[0.96, 0.04], [0.117, 0.883]] -> stationary ≈ [0.745, 0.255]
    (validation memo's corrected π)."""
    pi = _implied_stationary(np.array([[0.96, 0.04], [0.117, 0.883]]))
    assert pi[0] == pytest.approx(0.745, abs=1e-3)
    assert pi[1] == pytest.approx(0.255, abs=1e-3)
    assert pi.sum() == pytest.approx(1.0)


def test_frobenius_distance_zero_on_identical_matrices():
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert _frobenius_distance(A, A.copy()) == pytest.approx(0.0)


def test_frobenius_distance_on_known_perturbation():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.1], [0.0, 1.0]])
    assert _frobenius_distance(A, B) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# RegimeAuditTrail                                                            #
# --------------------------------------------------------------------------- #


def _stable_refit(as_of: datetime, n_obs: int = 1000) -> RegimeRefitEntry:
    """A refit with the canonical 2-state low/high vol params."""
    return _refit_entry(
        as_of=as_of, n_obs=n_obs,
        transmat=[[0.96, 0.04], [0.117, 0.883]],
        means=[2.5, 3.5],
        variances=[0.1, 0.3],
        label_map=(0, 1),
    )


def test_regime_audit_trail_empty():
    trail = RegimeAuditTrail(refits=[])
    assert trail.label_map_consistent() is True
    assert trail.transmat_frobenius_diffs().empty
    assert trail.stationary_drifts().empty
    assert trail.emission_summary().empty
    assert trail.to_dataframe().empty


def test_regime_audit_trail_label_map_consistent_when_stable():
    trail = RegimeAuditTrail(refits=[
        _stable_refit(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        _stable_refit(datetime(2026, 2, 1, tzinfo=timezone.utc)),
        _stable_refit(datetime(2026, 3, 1, tzinfo=timezone.utc)),
    ])
    assert trail.label_map_consistent() is True


def test_regime_audit_trail_detects_label_flip():
    """A refit where raw state 0 swapped with raw state 1 — silent
    signal inversion if downstream code keys on raw state index. The
    state_label_rule's job is to PREVENT this from being silent; the
    audit's job is to SURFACE it when prevention fails."""
    flipped = _refit_entry(
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
        n_obs=1000,
        transmat=[[0.883, 0.117], [0.04, 0.96]],  # swapped rows
        means=[3.5, 2.5],                          # swapped order
        variances=[0.3, 0.1],
        label_map=(1, 0),                          # raw 0 -> stable 1
    )
    trail = RegimeAuditTrail(refits=[
        _stable_refit(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        flipped,
    ])
    assert trail.label_map_consistent() is False
    flips = trail.label_map_flip_dates()
    assert flips == [datetime(2026, 2, 1, tzinfo=timezone.utc)]


def test_regime_audit_trail_no_flip_dates_when_consistent():
    trail = RegimeAuditTrail(refits=[
        _stable_refit(datetime(2026, 1, 1, tzinfo=timezone.utc)),
        _stable_refit(datetime(2026, 2, 1, tzinfo=timezone.utc)),
    ])
    assert trail.label_map_flip_dates() == []


def test_regime_audit_trail_transmat_frobenius_diffs():
    r1 = _refit_entry(
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc), n_obs=1000,
        transmat=[[0.96, 0.04], [0.117, 0.883]],
        means=[2.5, 3.5], variances=[0.1, 0.3], label_map=(0, 1),
    )
    r2 = _refit_entry(
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc), n_obs=1000,
        transmat=[[0.96, 0.04], [0.117, 0.883]],
        means=[2.5, 3.5], variances=[0.1, 0.3], label_map=(0, 1),
    )
    r3 = _refit_entry(
        as_of=datetime(2026, 3, 1, tzinfo=timezone.utc), n_obs=1000,
        transmat=[[0.90, 0.10], [0.20, 0.80]],   # shifted
        means=[2.5, 3.5], variances=[0.1, 0.3], label_map=(0, 1),
    )
    trail = RegimeAuditTrail(refits=[r1, r2, r3])
    diffs = trail.transmat_frobenius_diffs()
    assert len(diffs) == 3
    assert pd.isna(diffs.iloc[0])             # first has no predecessor
    assert diffs.iloc[1] == pytest.approx(0.0)  # r1 == r2
    assert diffs.iloc[2] > 0.0                  # r2 -> r3 shift


def test_regime_audit_trail_stationary_drifts():
    r1 = _stable_refit(datetime(2026, 1, 1, tzinfo=timezone.utc))
    r2 = _stable_refit(datetime(2026, 2, 1, tzinfo=timezone.utc))
    r3 = _refit_entry(
        as_of=datetime(2026, 3, 1, tzinfo=timezone.utc), n_obs=1000,
        transmat=[[0.5, 0.5], [0.5, 0.5]],       # stationary [0.5, 0.5]
        means=[2.5, 3.5], variances=[0.1, 0.3], label_map=(0, 1),
    )
    trail = RegimeAuditTrail(refits=[r1, r2, r3])
    drifts = trail.stationary_drifts()
    assert len(drifts) == 3
    assert pd.isna(drifts.iloc[0])
    assert drifts.iloc[1] == pytest.approx(0.0, abs=1e-6)
    # r1 stationary ≈ [0.745, 0.255]; r3 stationary = [0.5, 0.5]
    # L2 = sqrt(0.245^2 + 0.245^2) ≈ 0.346
    assert drifts.iloc[2] == pytest.approx(0.346, abs=1e-2)


def test_regime_audit_trail_emission_summary_stable_label():
    """Emission summary keys on STABLE label, not raw index. So when a
    label_map flips, the same stable column reports the same state's
    parameters regardless of which raw index it now lives at."""
    r1 = _stable_refit(datetime(2026, 1, 1, tzinfo=timezone.utc))
    # Flipped raw indices but same stable-label assignment
    r2 = _refit_entry(
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc), n_obs=1000,
        transmat=[[0.883, 0.117], [0.04, 0.96]],
        means=[3.5, 2.5],                          # swapped raw
        variances=[0.3, 0.1],
        label_map=(1, 0),                          # raw 0 = stable 1
    )
    trail = RegimeAuditTrail(refits=[r1, r2])
    summary = trail.emission_summary()
    assert summary.shape == (2, 4)
    # Stable state 0 (lower-variance) — should be mean 2.5 in BOTH refits.
    assert summary.iloc[0]["mean_state_0"] == pytest.approx(2.5)
    assert summary.iloc[1]["mean_state_0"] == pytest.approx(2.5)
    assert summary.iloc[0]["var_state_0"] == pytest.approx(0.1)
    assert summary.iloc[1]["var_state_0"] == pytest.approx(0.1)
    # Stable state 1 (higher-variance) — mean 3.5 in BOTH.
    assert summary.iloc[0]["mean_state_1"] == pytest.approx(3.5)
    assert summary.iloc[1]["mean_state_1"] == pytest.approx(3.5)


def test_regime_audit_trail_to_dataframe_overview():
    trail = RegimeAuditTrail(refits=[
        _stable_refit(datetime(2026, 1, 1, tzinfo=timezone.utc), n_obs=1000),
        _stable_refit(datetime(2026, 2, 1, tzinfo=timezone.utc), n_obs=1100),
    ])
    df = trail.to_dataframe()
    assert len(df) == 2
    assert set(df.columns) >= {
        "n_obs", "stationary_0", "stationary_1", "transmat_diff_vs_prev",
    }
    assert df.iloc[0]["n_obs"] == 1000
    assert df.iloc[0]["stationary_0"] == pytest.approx(0.745, abs=1e-3)
    assert pd.isna(df.iloc[0]["transmat_diff_vs_prev"])
    assert df.iloc[1]["transmat_diff_vs_prev"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# ForwardSelectionAudit                                                       #
# --------------------------------------------------------------------------- #


def test_forward_selection_audit_empty():
    audit = ForwardSelectionAudit(entries=[])
    assert audit.method_breakdown().empty
    assert audit.method_fractions().empty
    assert audit.model_risk_count() == 0
    assert audit.to_dataframe().empty


def test_forward_selection_audit_collects_from_decisions():
    decisions = [
        _decision_with_forward(
            datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
            _forward("settlement_date_match"),
        ),
        _decision_with_forward(
            datetime(2026, 4, 2, 14, 0, tzinfo=timezone.utc),
            _forward("settlement_date_match"),
        ),
        _decision_with_forward(
            datetime(2026, 4, 3, 14, 0, tzinfo=timezone.utc),
            _forward("put_call_parity"),
        ),
        _decision_with_forward(
            datetime(2026, 4, 4, 14, 0, tzinfo=timezone.utc),
            _forward("interpolated", model_risk=True),
        ),
    ]
    audit = ForwardSelectionAudit.from_decisions(decisions)
    assert len(audit.entries) == 4


def test_forward_selection_audit_skips_decisions_without_evaluation():
    """Skip decisions had no forward (filter rejected before SpreadEvaluator
    ran) — the audit must not synthesize one."""
    skip_decision = StrategyDecision(
        as_of=datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc),
        action="skip", reason="regime_filter",
        hypothesis_name="test",
        spread=None, evaluation=None, size=None,
    )
    enter_decision = _decision_with_forward(
        datetime(2026, 4, 2, 14, 0, tzinfo=timezone.utc),
        _forward("settlement_date_match"),
    )
    audit = ForwardSelectionAudit.from_decisions([skip_decision, enter_decision])
    assert len(audit.entries) == 1
    assert audit.entries[0].selection_method == "settlement_date_match"


def test_forward_selection_audit_method_breakdown_and_fractions():
    decisions = [
        _decision_with_forward(
            datetime(2026, 4, 1, tzinfo=timezone.utc),
            _forward("settlement_date_match"),
        ),
        _decision_with_forward(
            datetime(2026, 4, 2, tzinfo=timezone.utc),
            _forward("settlement_date_match"),
        ),
        _decision_with_forward(
            datetime(2026, 4, 3, tzinfo=timezone.utc),
            _forward("settlement_date_match"),
        ),
        _decision_with_forward(
            datetime(2026, 4, 4, tzinfo=timezone.utc),
            _forward("interpolated", model_risk=True),
        ),
    ]
    audit = ForwardSelectionAudit.from_decisions(decisions)
    counts = audit.method_breakdown()
    assert int(counts["settlement_date_match"]) == 3
    assert int(counts["interpolated"]) == 1
    fractions = audit.method_fractions()
    assert fractions["settlement_date_match"] == pytest.approx(0.75)
    assert fractions["interpolated"] == pytest.approx(0.25)


def test_forward_selection_audit_model_risk_count():
    decisions = [
        _decision_with_forward(
            datetime(2026, 4, 1, tzinfo=timezone.utc),
            _forward("settlement_date_match", model_risk=False),
        ),
        _decision_with_forward(
            datetime(2026, 4, 2, tzinfo=timezone.utc),
            _forward("interpolated", model_risk=True),
        ),
        _decision_with_forward(
            datetime(2026, 4, 3, tzinfo=timezone.utc),
            _forward("interpolated", model_risk=True),
        ),
    ]
    audit = ForwardSelectionAudit.from_decisions(decisions)
    assert audit.model_risk_count() == 2


def test_forward_selection_audit_to_dataframe_schema():
    decisions = [
        _decision_with_forward(
            datetime(2026, 4, 1, tzinfo=timezone.utc),
            _forward("settlement_date_match"),
            long_strike=18.0, short_strike=20.0,
        ),
    ]
    audit = ForwardSelectionAudit.from_decisions(decisions)
    df = audit.to_dataframe()
    assert set(df.columns) == {
        "as_of", "forward_value", "selection_method",
        "model_risk_flag",
        "long_strike", "short_strike", "settlement_date",
    }
    assert len(df) == 1
    assert df.iloc[0]["long_strike"] == 18.0
    assert df.iloc[0]["short_strike"] == 20.0


# --------------------------------------------------------------------------- #
# Smoke: RegimeAuditTrail.from_walk_forward_fitter                            #
# --------------------------------------------------------------------------- #


@dataclass
class _StubFitter:
    """Minimal stand-in for WalkForwardRegimeFitter — exposes the
    `refit_log` attribute the audit consumes."""
    refit_log: list[dict[str, Any]] = field(default_factory=list)


def test_from_walk_forward_fitter_converts_refit_log():
    """Build a stub fitter mimicking WalkForwardRegimeFitter.refit_log
    layout — verify the trail's RegimeRefitEntry is constructed correctly,
    including implied stationary computed from the transmat."""
    fitter = _StubFitter(refit_log=[
        {
            "as_of": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "n_obs": 1000,
            "transmat": np.array([[0.96, 0.04], [0.117, 0.883]]),
            "means": np.array([[2.5], [3.5]]),
            "covars": np.array([[0.1], [0.3]]),
            "label_map": (0, 1),
        }
    ])
    trail = RegimeAuditTrail.from_walk_forward_fitter(fitter)  # type: ignore[arg-type]
    assert len(trail.refits) == 1
    r = trail.refits[0]
    assert r.n_obs == 1000
    assert r.label_map == (0, 1)
    # Stationary derived from transmat (the fitter's refit_log doesn't
    # include it, the audit fills it in).
    assert r.stationary[0] == pytest.approx(0.745, abs=1e-3)
    assert r.stationary[1] == pytest.approx(0.255, abs=1e-3)
