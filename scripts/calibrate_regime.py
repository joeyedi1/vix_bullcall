"""HMM calibration sweep — alternative regime configs at smoke-window start.

ARCH §3.3. Fits one Gaussian HMM per (n_states, lookback_days) cell at
2026-03-20 and prints a side-by-side matrix of:

  - Per-state geometric-mean VIX + ±1σ band (state boundaries)
  - Implied stationary distribution
  - VIX-level state-membership posteriors at probe VIX levels

The point is to test two architectural hypotheses on this specific window:

  (1) Does shortening the lookback to 756 trading days (drops the 2022
      rate-shock spike) shift the high-vol state's mean upward — i.e.
      does the 2022 fat tail bias state-1 emission downward?

  (2) Does a 3-state HMM cleanly separate VIX 18–22 into a "mid" tier,
      reserving "high" for true crisis VIX (>28)? The 2-state model
      currently classifies VIX 20-22 as 99% high-vol — possibly too
      coarse.

Re-uses the smoke runner's `load_vix_feature_panel` data path so the
sweep operates on the exact same observations that the live HMM sees.

Usage:
    python scripts/calibrate_regime.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the smoke runner's panel loader so the sweep sees the same series.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_smoke_backtest import load_vix_feature_panel  # noqa: E402

from vix_spread.regime.hmm_spec import HMMSpec  # noqa: E402
from vix_spread.regime.walk_forward import WalkForwardRegimeFitter  # noqa: E402


SMOKE_START = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
PROBE_VIX_LEVELS = (12, 14, 16, 18, 20, 22, 25, 28, 32)


# --------------------------------------------------------------------------- #
# Config grid                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepConfig:
    label: str
    n_states: int
    lookback_days: int


# Initial transition matrices — hmmlearn re-estimates these during fit
# (`params='stmc'`); the init only seeds the EM iteration.
_TRANSMAT_2 = np.array(
    [[0.96, 0.04],
     [0.117, 0.883]],
    dtype=float,
)
_TRANSMAT_3 = np.array(
    [[0.94, 0.05, 0.01],
     [0.05, 0.90, 0.05],
     [0.01, 0.05, 0.94]],
    dtype=float,
)


def _spec_for(n_states: int) -> HMMSpec:
    if n_states == 2:
        return HMMSpec(
            n_states=2, transition_matrix=_TRANSMAT_2,
            state_label_rule="by_emission_variance",
        )
    if n_states == 3:
        return HMMSpec(
            n_states=3, transition_matrix=_TRANSMAT_3,
            state_label_rule="by_emission_variance",
        )
    raise ValueError(f"Unsupported n_states={n_states}")


CONFIGS = (
    SweepConfig(label="2-state, 1260d (baseline)", n_states=2, lookback_days=1260),
    SweepConfig(label="2-state,  756d (drops 2022)", n_states=2, lookback_days=756),
    SweepConfig(label="3-state, 1260d",             n_states=3, lookback_days=1260),
    SweepConfig(label="3-state,  756d",             n_states=3, lookback_days=756),
)


# --------------------------------------------------------------------------- #
# Per-config fit                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepResult:
    config: SweepConfig
    n_obs: int
    # All arrays below are in STABLE-LABEL order (state 0 = lowest variance).
    log_mu: np.ndarray            # (n_states,)
    log_var: np.ndarray           # (n_states,)
    stationary: np.ndarray        # (n_states,) — implied from estimated transmat


def _implied_stationary(transmat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(transmat.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    raw = eigvecs[:, idx].real
    if raw.sum() < 0:
        raw = -raw
    vec = np.maximum(raw, 0.0)
    s = vec.sum()
    if s <= 0:
        return np.ones(transmat.shape[0]) / transmat.shape[0]
    return vec / s


def _fit_one(cfg: SweepConfig, feature_panel) -> SweepResult:
    spec = _spec_for(cfg.n_states)
    fitter = WalkForwardRegimeFitter(
        spec=spec, feature_column="log_vix",
        lookback_days=cfg.lookback_days, cadence="weekly",
        random_state=0, hypothesis_tag="contrarian_tail",
    )
    fitter.fit_walk_forward(feature_panel, SMOKE_START)
    r = fitter.refit_log[-1]
    raw_means = np.asarray(r["means"]).reshape(-1)
    raw_vars = np.asarray(r["covars"]).reshape(-1)
    transmat = np.asarray(r["transmat"], dtype=float)
    label_map = r["label_map"]
    stable_mu = np.empty(cfg.n_states)
    stable_var = np.empty(cfg.n_states)
    for raw_idx, stable_label in enumerate(label_map):
        stable_mu[stable_label] = float(raw_means[raw_idx])
        stable_var[stable_label] = float(raw_vars[raw_idx])
    raw_pi = _implied_stationary(transmat)
    stable_pi = np.empty(cfg.n_states)
    for raw_idx, stable_label in enumerate(label_map):
        stable_pi[stable_label] = float(raw_pi[raw_idx])
    return SweepResult(
        config=cfg, n_obs=int(r["n_obs"]),
        log_mu=stable_mu, log_var=stable_var, stationary=stable_pi,
    )


# --------------------------------------------------------------------------- #
# Membership posteriors                                                        #
# --------------------------------------------------------------------------- #


def _posteriors(result: SweepResult, vix_levels) -> pd.DataFrame:
    """For each VIX in `vix_levels`, return P(state_k | VIX) using
    Gaussian emission × stationary prior. Columns: state_0, state_1, ...
    """
    x = np.log(np.asarray(vix_levels, dtype=float))
    mu = result.log_mu[None, :]
    var = result.log_var[None, :]
    log_emission = -0.5 * np.log(2.0 * np.pi * var) \
                   - 0.5 * (x[:, None] - mu) ** 2 / var
    log_prior = np.log(np.maximum(result.stationary, 1e-300))[None, :]
    log_p = log_emission + log_prior
    m = log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p - m)
    p /= p.sum(axis=1, keepdims=True)
    cols = [f"state_{k}" for k in range(result.config.n_states)]
    return pd.DataFrame(p, index=list(vix_levels), columns=cols)


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #


def _print_state_matrix(results: list[SweepResult]) -> None:
    print("=" * 86)
    print("STATE BOUNDARIES — geo-mean VIX, ±1σ band, stationary mass per state")
    print("=" * 86)
    header = (
        f"{'config':<30s}{'n_obs':>7s}{'state':>7s}"
        f"{'geo-VIX':>10s}{'-1σ':>10s}{'+1σ':>10s}{'π_stat':>10s}"
    )
    print(header)
    print("-" * 86)
    for res in results:
        for k in range(res.config.n_states):
            label = res.config.label if k == 0 else ""
            n_obs = res.n_obs if k == 0 else ""
            mu = float(res.log_mu[k])
            sd = float(np.sqrt(res.log_var[k]))
            geo = np.exp(mu)
            lo = np.exp(mu - sd)
            hi = np.exp(mu + sd)
            pi = float(res.stationary[k])
            n_str = f"{n_obs:>7,d}" if isinstance(n_obs, int) else f"{'':>7s}"
            print(
                f"{label:<30s}{n_str}{k:>7d}"
                f"{geo:>10.2f}{lo:>10.2f}{hi:>10.2f}{pi:>10.4f}"
            )
        print("-" * 86)
    print()


def _print_membership_matrix(results: list[SweepResult]) -> None:
    print("=" * 86)
    print("VIX-LEVEL → STATE MEMBERSHIP (prior = stationary)")
    print("=" * 86)
    for res in results:
        post = _posteriors(res, PROBE_VIX_LEVELS)
        print(f"\n  {res.config.label}")
        print("  " + "-" * 60)
        cols = list(post.columns)
        head = "    VIX " + "".join(f"{c:>12s}" for c in cols) + "      verdict"
        print(head)
        for vix in PROBE_VIX_LEVELS:
            row = post.loc[vix]
            probs = "".join(f"{row[c]:>12.4f}" for c in cols)
            verdict_idx = int(np.argmax(row.values))
            verdict = f"state_{verdict_idx}"
            print(f"    {vix:>3d}{probs}     {verdict}")
    print()


def main() -> int:
    print("=" * 86)
    print(f"HMM CALIBRATION SWEEP @ as_of {SMOKE_START.date()}")
    print("=" * 86)
    print("Underlying:  log(VIX) daily close")
    print("Label rule:  by_emission_variance (state 0 = lowest variance)")
    print(f"Configs:     {len(CONFIGS)}")
    print(f"Probe VIX:   {list(PROBE_VIX_LEVELS)}")
    print()

    print("Loading VIX feature panel ...", flush=True)
    panel = load_vix_feature_panel()
    print(
        f"  panel: {panel.features.shape}, "
        f"{panel.dates.min().date()} → {panel.dates.max().date()}"
    )
    print()

    print("Fitting configs ...", flush=True)
    t0 = time.time()
    results: list[SweepResult] = []
    for cfg in CONFIGS:
        try:
            res = _fit_one(cfg, panel)
        except Exception as exc:
            print(f"  {cfg.label}: FAILED — {type(exc).__name__}: {exc}")
            continue
        results.append(res)
        print(f"  {cfg.label}: n_obs={res.n_obs}")
    print(f"  done in {time.time() - t0:.1f}s")
    print()

    if not results:
        print("No configs converged; aborting.")
        return 1

    _print_state_matrix(results)
    _print_membership_matrix(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
