# VIX Bull Call Spread Backtester

A causally-strict, regime-aware backtesting engine for VIX option spreads.
Built to evaluate a bull-call spread strategy across volatility regimes with
production-grade execution realism — no spot-VIX-as-forward shortcuts, no
mid-price fills masquerading as base-case P&L, no smoothed HMM probabilities
leaking into live signals.

The detailed design rationale and every validation-memo constraint that
shapes the architecture live in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Why this exists

A prior VIX-spread backtester produced an attractive equity curve that
collapsed under audit: it priced options against spot VIX rather than the
forward, used mid-price fills, and ran an HMM whose declared transition
matrix and stationary distribution were mutually inconsistent. The
[validation memo](ARCHITECTURE.md) enumerated the failure modes; this
repository is the from-scratch rebuild that enforces every one of them
*by construction* rather than by code-review convention.

Where the prior tool reported P&L that didn't exist, this one reports a
0.00% hit rate when the thesis doesn't hold — and surfaces that rejection
as a first-class number, not a footnote.

---

## Architectural pillars

### 1. Calibrated 3-state regime HMM

Walk-forward Gaussian HMM on `log(VIX)` with a 1260-day rolling lookback
([src/vix_spread/regime/walk_forward.py](src/vix_spread/regime/walk_forward.py)).
A calibration sweep
([scripts/calibrate_regime.py](scripts/calibrate_regime.py)) compared
2-state and 3-state specs across 1260d/756d lookbacks and selected the
3-state configuration because it cleanly resolves VIX 16–18 into a
distinct "mid" regime rather than lumping it with true high-vol crises:

| state | label | geo-mean VIX | role |
|---|---|---:|---|
| 0 | low-vol | ~13.5 | contrarian_tail entry zone |
| 1 | mid | ~16.5 | rejected — no decisive thesis |
| 2 | high-vol | ~22.2 | crisis regime |

State labels are pinned via `by_emission_variance` so that lowest-variance
state is always label 0, even when the HMM's raw indices swap between
refits. `RegimeAuditTrail.label_map_flip_dates()`
([src/vix_spread/backtest/audit.py](src/vix_spread/backtest/audit.py))
detects and reports those raw-index swaps — a silent signal-inversion
bug that destroyed a calendar year of returns in the prior tool.

Only filtered posteriors `P(S_t | y_{1:t})` are exposed publicly. The
forward pass is implemented in-house rather than via
`hmmlearn.score_samples` so there is no quiet path that leaks smoothed
gammas into a production signal.

### 2. ChainIVProvider with Black-76 inversion fallback

`ChainIVProvider` ([src/vix_spread/pricing/leg_iv.py](src/vix_spread/pricing/leg_iv.py))
returns the per-leg implied vol used by `SpreadEvaluator`. When the
Bloomberg chain row has `IVOL_LAST` populated, it's used directly and
tagged `source='vendor'`. When `IVOL_LAST` is missing — common at deep
strikes — the provider inverts Black-76 from the chain's last trade
price and tags the result `source='b76_inverted'`. Both source tags
travel through to `SpreadEvaluation`, the decision log, and
`EdgeBleedAudit.by_iv_source_breakdown()` so bleed magnitude can be
attributed cleanly to vendor-IV vs inverted-IV decisions.

A `FlatVolError` is raised when the long and short legs land at
identical IVs — the validation-memo failure mode where a flat-vol panel
made every credit-spread evaluation look identically attractive.

### 3. Strict T → T+1 causal FillEngine

`FillEngine.attempt_fill` ([src/vix_spread/execution/fill_engine.py](src/vix_spread/execution/fill_engine.py))
fires exactly once at T+1 with the NBBO known at T+1. A decision made
at minute T cannot fill at minute T; the broadcaster
([src/vix_spread/regime/broadcaster.py](src/vix_spread/regime/broadcaster.py))
uses `pd.merge_asof(..., direction='backward', allow_exact_matches=False)`
to guarantee this. Test
[test_walk_forward.py::test_pending_entry_at_T_fills_at_T_plus_1](tests/test_walk_forward.py)
locks it down.

Three fill modes are supported:
- `SYNTHETIC_BIDASK` — cross the spread (long leg @ ask, short leg @ bid). Base case.
- `MIDPOINT` — both legs at midpoint. Requires explicit
  `accept_midpoint_optimism=True` opt-in; never the default.
- `SYNTHETIC_PLUS_SLIPPAGE` — synthetic with a configurable tick-per-leg
  slippage layer. Stress sensitivity.

### 4. Liquidity gates with rejection-as-first-class

`LiquidityGates` ([src/vix_spread/execution/liquidity_gates.py](src/vix_spread/execution/liquidity_gates.py))
evaluates every spread *before* the fill engine prices it. Gates include
`reject_no_bid_short_leg` (the critical one — no bid on the short leg
makes the spread unhedgeable), `reject_locked_or_crossed`, per-leg
spread-pct caps, quote-age limits, and order-size-vs-displayed bounds.

Rejections are logged as `RejectedOrder` objects in `BacktestResults`
alongside `CompletedTrade`s. A backtest that produces 477 trades and 163
rejections shows both numbers — there is no path that quietly fills a
spread that should have been rejected.

### 5. Three-Scenario Execution Reporting

Every backtest can run under all three fill modes in one invocation
([src/vix_spread/backtest/reporting.py](src/vix_spread/backtest/reporting.py)).
The output matrix
([format_three_scenario_summary](src/vix_spread/backtest/reporting.py))
puts base / optimistic / stressed side-by-side so the bid-ask edge bleed
(optimistic − base) and the slippage tail (stressed − base) surface
directly in the headline numbers. The base case is always the synthetic
bid-ask figure; the spec explicitly forbids reporting midpoint as the
headline.

`EdgeBleedAudit`
([src/vix_spread/backtest/edge_bleed.py](src/vix_spread/backtest/edge_bleed.py))
gives the per-trade view: pairs each `CompletedTrade.entry_fill` with
the originating `StrategyDecision` and reports the per-spread and dollar
distribution of `theoretical − executed`.

---

## Repository layout

```
src/vix_spread/
  data/            # Bloomberg fetchers, daily/intraday processors, FeaturePanel
  products/        # VIXIndexOption, VXFutureOption (abstract Product hierarchy)
  pricing/         # Black-76 pricer, forward selector, ChainIVProvider, evaluator
  regime/          # HMMSpec, walk-forward fitter, daily→minute broadcaster
  execution/       # FillEngine, LiquidityGates, ExitEngine, OptionQuote
  strategy/        # StrategyHypothesis, SpreadSelector, FixedRiskSizer, strategy
  backtest/        # WalkForwardBacktest, results, reporting, audit, edge_bleed
  utils/           # Errors (LookaheadError), settlement calendar helpers

scripts/
  pull_data.py            # Bloomberg pulls: VIX/UX history, VX futures OHLCV/quotes
  pull_options_chain.py   # Bloomberg pulls: VIX-index option chain, VX OOF chain, VRO
  calibrate_regime.py     # HMM config sweep (n_states × lookback)
  run_smoke_backtest.py   # End-to-end smoke run: Mar 20 – May 1, 2026
  check_ivol_coverage.py  # Vendor IV coverage diagnostic on the chain panel

tests/                    # 322 passing tests
ARCHITECTURE.md           # Authoritative design doc (validation-memo enforced)
PHASE_5_COMPLIANCE.md     # Compliance mapping for Phase-5 build
```

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate           # PowerShell
pip install -r requirements.txt
```

Bloomberg data pulls additionally require:
- A running Bloomberg terminal on the local machine (default `localhost:8194`)
- The `blpapi` Python package (NOT in `requirements.txt`; install per Bloomberg's
  instructions)
- `pdblp` for the `bdh`-style daily fetches

The smoke runner and tests work entirely off the parquet files already
written under `data/raw/blpapi/` — no live terminal needed.

---

## Quick start

### Data pulls (Bloomberg required)

```powershell
# Regime panel: VIX index + UX1/UX2 generics, daily close.
python scripts/pull_data.py vix-history --start 2010-01-01 --end 2026-05-01

# 1-min OHLCV for the deliverable VX future.
python scripts/pull_data.py vx-futures --kind ohlcv `
    --start 2025-10-16T00:00:00 --end 2026-05-01T23:59:00 `
    --contract 2026-05

# VIX-index option chain (NBBO + daily) over the smoke window.
python scripts/pull_options_chain.py vix-index `
    --start 2025-10-16T00:00:00 --end 2026-05-01T23:59:00

# VRO settlement history (cheap; run first to validate terminal connectivity).
python scripts/pull_options_chain.py vro --start 2010-01-01 --end 2026-05-01
```

### Diagnostic sweeps

```powershell
# HMM calibration: 2-state vs 3-state × 1260d vs 756d lookback at smoke start.
python scripts/calibrate_regime.py

# Regime diagnostic dump: quarterly refits 2024-Q1 → 2026-Q1 with
# emission summary, Frobenius stability, label-flip dates, VIX-level
# membership posteriors.
python scripts/run_smoke_backtest.py --regime-diagnostic

# Vendor IV coverage on the chain panel.
python scripts/check_ivol_coverage.py
```

### Smoke backtest

```powershell
# Headline: contrarian_tail hypothesis, 3-state HMM, base-case fills.
python scripts/run_smoke_backtest.py

# Three-scenario matrix (base / optimistic / stressed).
python scripts/run_smoke_backtest.py --three-scenario

# Trade-machinery stress test: targets state_2 instead of state_0 so
# the fill / sizing / reporting layers exercise on a non-target regime.
# Not a backtest of the contrarian_tail thesis.
python scripts/run_smoke_backtest.py --three-scenario --high-vol-stress

# Permissive: always-true hypothesis. Smoke-grade only — lets every
# minute attempt a fill so the trade engine itself can be debugged.
python scripts/run_smoke_backtest.py --three-scenario --permissive
```

### Test suite

```powershell
python -m pytest -q
```

322 tests covering: Black-76 pricing, settlement-value computation,
forward selection, ChainIVProvider with B76 inversion, spread
evaluation, FillEngine (all three modes), LiquidityGates, ExitEngine,
WalkForwardRegimeFitter, broadcaster causality, walk-forward backtest
no-lookahead invariants, strategy hypothesis filters, sizing, spread
selection, reporting, edge bleed audit, regime audit, forward-selection
audit, and product-type separation.

---

## Status

The engineering build is complete. The 3-state HMM is calibrated and
wired into the smoke runner; the trade machinery is verified to fire
correctly when given a valid regime signal (the `--high-vol-stress`
diagnostic produced 477 base-case trades with edge bleed cleanly
quantified). The contrarian_tail hypothesis correctly refuses to trade
in the Mar–May 2026 window because every day sits in the mid- or
high-vol regime — exactly the discipline the architecture was built to
enforce.

The next phase is alpha research: identifying the historical windows
where the contrarian_tail thesis should fire, sourcing intraday data
within the Bloomberg 140-day tick ceiling, and exploring alternative
hypotheses (e.g., `breakout_momentum`, `curve_normalization`) against
the same execution stack.
