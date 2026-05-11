# Phase 5 — ARCHITECTURE Compliance Sign-Off

**Date:** 2026-05-11
**HEAD at sign-off:** `9c43eb1` (`feat: three-scenario reporting bundle (base/optimistic/stressed) per ARCH §8.2 + WalkForwardBacktest fill-mode kwargs`)
**Test count:** 291 passing (Phase-1 through Phase-5 inclusive)
**Branch:** `main`

This document maps every load-bearing requirement from `ARCHITECTURE.md` (and the underlying validation memo) to the source files that enforce it and the tests that pin the behaviour. Every entry below has a structural defense — by `@dataclass(frozen=True)`, by `@abstractmethod`, by `raise` on a known-bad input — not by code-review convention.

The full suite ran 291/291 green on `9c43eb1`. The end-to-end smoke (`scripts/run_smoke_backtest.py --permissive --three-scenario`) produced 530-549 completed trades across base/optimistic/stressed scenarios with $1,893 of observable bid-ask edge bleed between base and optimistic — the validation memo's narrative-cherry-picking failure mode is now structurally impossible.

---

## §2 — Product separation (the largest pricing-error source in the memo)

| Requirement | File(s) | Test(s) |
|---|---|---|
| `Product` is an abstract base; subclasses dispatch all settlement / multiplier / hedge logic | [src/vix_spread/products/base.py](src/vix_spread/products/base.py), [products/vix_index_option.py](src/vix_spread/products/vix_index_option.py), [products/vx_future_option.py](src/vix_spread/products/vx_future_option.py) | [tests/test_products_separation.py](tests/test_products_separation.py) |
| Mixing `VIXIndexOption` and `VXFutureOption` in a `BullCallSpread` raises `TypeError` at `__post_init__` | [products/spread.py](src/vix_spread/products/spread.py) | [tests/test_products_separation.py](tests/test_products_separation.py) |
| `VXFutureOption.deliverable_vx: VXFuture` is REQUIRED at construction (prevents the "factor of 10" hedging defect) | [products/vx_future_option.py](src/vix_spread/products/vx_future_option.py), [products/vx_future.py](src/vix_spread/products/vx_future.py) | [tests/test_settlement_value.py](tests/test_settlement_value.py), [tests/test_vx_future.py](tests/test_vx_future.py) |
| `VIXIndexOption.settlement_value(market)` consumes `market.vro_for(expiry)` — never spot VIX, never theoretical, never Tuesday VX close | [products/vix_index_option.py](src/vix_spread/products/vix_index_option.py), [data/snapshot.py](src/vix_spread/data/snapshot.py) | [tests/test_settlement_value.py](tests/test_settlement_value.py) |
| `VXFutureOption.settlement_value` keys on `deliverable_vx.settlement_date`, NOT option expiry | [products/vx_future_option.py](src/vix_spread/products/vx_future_option.py) | [tests/test_settlement_value.py](tests/test_settlement_value.py) |

## §3 — Strict-causal regime classification

| Requirement | File(s) | Test(s) |
|---|---|---|
| `RegimeClassifier.predict_filtered` returns ONLY filtered posteriors `P(S_t \| y_{1:t})` — no smoothing path is on the public API | [regime/base.py](src/vix_spread/regime/base.py), [regime/walk_forward.py](src/vix_spread/regime/walk_forward.py), [regime/hmm_filter.py](src/vix_spread/regime/hmm_filter.py) | [tests/test_walk_forward_no_lookahead.py](tests/test_walk_forward_no_lookahead.py), [tests/test_regime_causality.py](tests/test_regime_causality.py) |
| Forward algorithm is implemented in log-space directly (no `hmmlearn.score_samples` which returns smoothed gammas) | [regime/walk_forward.py](src/vix_spread/regime/walk_forward.py) | [tests/test_walk_forward_no_lookahead.py](tests/test_walk_forward_no_lookahead.py) |
| `HMMSpec` cross-checks supplied `stationary_distribution` against the implied left-eigenvector; mismatch raises `HMMSpecificationError` | [regime/hmm_spec.py](src/vix_spread/regime/hmm_spec.py) | [tests/test_hmm_spec.py](tests/test_hmm_spec.py) |
| `state_label_rule = by_emission_variance` stabilises state indices across refits (low-var = label 0) | [regime/walk_forward.py](src/vix_spread/regime/walk_forward.py) | [tests/test_walk_forward_no_lookahead.py](tests/test_walk_forward_no_lookahead.py) |
| Daily-to-minute broadcast uses `pd.merge_asof(direction='backward', allow_exact_matches=False)` — strict shift, no same-bar leak | [regime/broadcaster.py](src/vix_spread/regime/broadcaster.py) | [tests/test_regime_broadcaster.py](tests/test_regime_broadcaster.py) |
| `FeaturePanel.slice_causal(column, as_of)` is the only sanctioned reader; release-boundary rule enforced | [data/feature_panel.py](src/vix_spread/data/feature_panel.py) | [tests/test_feature_panel.py](tests/test_feature_panel.py) |

## §4 — Pricing & Greeks (Black-76 with no spot-VIX shortcut)

| Requirement | File(s) | Test(s) |
|---|---|---|
| `ForwardSelector.select(source='spot_vix')` permanently raises `ForwardSelectionError` | [pricing/forward_selector.py](src/vix_spread/pricing/forward_selector.py) | [tests/test_pricing_no_spot_vix.py](tests/test_pricing_no_spot_vix.py), [tests/test_forward_selector.py](tests/test_forward_selector.py) |
| `Forward` carries `selection_method` + `model_risk_flag` for audit | [pricing/forward_selector.py](src/vix_spread/pricing/forward_selector.py) | [tests/test_forward_selector.py](tests/test_forward_selector.py) |
| `minutes_to_settlement(as_of, settlement)` requires tz-aware timestamps (`TimezoneError`) and refuses T <= 0 (`ExpiryError`) | [pricing/time_to_expiry.py](src/vix_spread/pricing/time_to_expiry.py) | [tests/test_time_to_expiry.py](tests/test_time_to_expiry.py) |
| `Black76Pricer.price(...)` consumes ONLY (product, Forward, leg_iv, as_of, rate); never spot VIX | [pricing/black76.py](src/vix_spread/pricing/black76.py) | [tests/test_black76_pricer.py](tests/test_black76_pricer.py), [tests/test_pricing_no_spot_vix.py](tests/test_pricing_no_spot_vix.py) |
| `price_spread` raises `FlatVolError` when `iv_long == iv_short` — the validation-memo VVIX-substitution defect | [pricing/black76.py](src/vix_spread/pricing/black76.py), [utils/errors.py](src/vix_spread/utils/errors.py) | [tests/test_price_spread.py](tests/test_price_spread.py) |
| `ChainIVProvider` resolves vendor IV first; falls back to Black-76 inversion of `(PX_BID + PX_ASK) / 2` when vendor missing; refuses with `LegIVResolutionError` if both fail (no NaN propagation) | [pricing/leg_iv.py](src/vix_spread/pricing/leg_iv.py) | [tests/test_leg_iv.py](tests/test_leg_iv.py) |
| `LegIV.source` tags every resolved IV as `VENDOR` or `B76_INVERTED` for audit | [pricing/leg_iv.py](src/vix_spread/pricing/leg_iv.py) | [tests/test_leg_iv.py](tests/test_leg_iv.py), [tests/test_evaluator.py](tests/test_evaluator.py) |
| `TheoreticalPrice` carries `is_executable=False` sentinel | [pricing/theoretical.py](src/vix_spread/pricing/theoretical.py) | [tests/test_fill_engine_rejects_theoretical.py](tests/test_fill_engine_rejects_theoretical.py) |

## §5 — Execution simulation (synthetic bid/ask, never theoretical)

| Requirement | File(s) | Test(s) |
|---|---|---|
| `FillEngine.attempt_fill` rejects `TheoreticalPrice` at the type level (TypeError) | [execution/fill_engine.py](src/vix_spread/execution/fill_engine.py) | [tests/test_fill_engine_rejects_theoretical.py](tests/test_fill_engine_rejects_theoretical.py) |
| `FillMode.SYNTHETIC_BIDASK` is the default; `MIDPOINT` requires explicit `accept_midpoint_optimism=True` per call | [execution/fill_engine.py](src/vix_spread/execution/fill_engine.py), [execution/fill_modes.py](src/vix_spread/execution/fill_modes.py) | [tests/test_fill_engine_no_midpoint_default.py](tests/test_fill_engine_no_midpoint_default.py), [tests/test_fill_engine_attempt_fill.py](tests/test_fill_engine_attempt_fill.py) |
| Synthetic spread quotes: `open_debit = long.ask - short.bid`, `close_credit = long.bid - short.ask` | [execution/synthetic_quote.py](src/vix_spread/execution/synthetic_quote.py), [execution/fill_engine.py](src/vix_spread/execution/fill_engine.py) | [tests/test_fill_engine_attempt_fill.py](tests/test_fill_engine_attempt_fill.py) |
| `LiquidityGates` evaluation with typed `RejectedOrder` (`no_bid_short`, `stale_quote`, `locked`, `crossed`, `gate_fail`); earliest violation wins | [execution/liquidity_gates.py](src/vix_spread/execution/liquidity_gates.py), [execution/fill_engine.py](src/vix_spread/execution/fill_engine.py) | [tests/test_fill_engine_attempt_fill.py](tests/test_fill_engine_attempt_fill.py) |
| `MinuteBarFillEngine` (single-instrument, futures) per-side staleness gate, NaN→∞ | [execution/bar_fill_engine.py](src/vix_spread/execution/bar_fill_engine.py) | [tests/test_bar_fill_engine.py](tests/test_bar_fill_engine.py) |
| `OptionQuote` excludes `last_trade` from fill logic (diagnostic only) | [execution/quote.py](src/vix_spread/execution/quote.py) | implicit in FillEngine tests |
| `SYNTHETIC_PLUS_SLIPPAGE` adds `N ticks × tick_value × legs_affected` to the debit; configurable `slippage_ticks_per_leg`, `slippage_apply_to_short_leg_only`, `tick_value` | [execution/fill_engine.py](src/vix_spread/execution/fill_engine.py) | [tests/test_fill_engine_attempt_fill.py](tests/test_fill_engine_attempt_fill.py) |
| `ExitEngine.execute_exit` with `FORCED_TUESDAY_LIQUIDATION` (close-side gates with flipped leg priorities: `no_bid_long` and `no_ask_short` are the critical refusals) | [execution/exit_engine.py](src/vix_spread/execution/exit_engine.py), [execution/exit_policy.py](src/vix_spread/execution/exit_policy.py) | [tests/test_exit_engine.py](tests/test_exit_engine.py) |
| `ExitEngine` `HOLD_TO_SETTLEMENT` path consumes actual VRO via `Product.settlement_value(SettlementMarket)`; missing VRO → `KeyError` (no fabrication) | [execution/exit_engine.py](src/vix_spread/execution/exit_engine.py), [products/vix_index_option.py](src/vix_spread/products/vix_index_option.py) | [tests/test_exit_engine.py](tests/test_exit_engine.py) |
| `FailedExit` is a first-class outcome with typed reason; the position stays open for retry | [execution/exit_engine.py](src/vix_spread/execution/exit_engine.py) | [tests/test_exit_engine.py](tests/test_exit_engine.py) |

## §6 — Data architecture (vintage-tagged audit)

| Requirement | File(s) | Test(s) |
|---|---|---|
| `BaseDataFetcher` enforces `data/raw/{source}/{product}/{shard_key_}{vintage}.parquet` with `_pulled_at` + `_vintage` columns | [data/base.py](src/vix_spread/data/base.py) | exercised by ingestion scripts |
| `DataProcessor.process_vix_index_options` produces per-minute panel with `is_locked` / `is_crossed` / `quote_age_seconds` / `last_trade_age_seconds` derived | [data/processor.py](src/vix_spread/data/processor.py) | [tests/test_processor.py](tests/test_processor.py) |
| `DataProcessor.process_vix_options_daily` normalizes asymmetric Tue/Wed VIX ticker dates to canonical SOQ Wed expiry | [data/processor.py](src/vix_spread/data/processor.py) | [tests/test_processor.py](tests/test_processor.py) |
| `expiry_calendar` resolves VX settlement dates with Juneteenth Wed→Tue rollback | [data/expiry_calendar.py](src/vix_spread/data/expiry_calendar.py) | [tests/test_expiry_calendar.py](tests/test_expiry_calendar.py) |
| OHLCV `align_bars` does NOT forward-fill (missing minute is missing, not "previous close repeated") | [data/processor.py](src/vix_spread/data/processor.py) | [tests/test_processor.py](tests/test_processor.py) |
| Per-event staleness clocks (`bid_age_seconds`, `ask_age_seconds`, `trade_age_seconds`) — NaN before first observation, +60s per missing minute | [data/processor.py](src/vix_spread/data/processor.py) | [tests/test_processor.py](tests/test_processor.py) |

## §7 — Strategy & signal layer (§7.2 timing rule)

| Requirement | File(s) | Test(s) |
|---|---|---|
| `SpreadSelector` filters out invalid strikes PRE-selection (`ask > 0` for long, `bid > 0` for short) — never produces a spread the FillEngine would reject as `no_bid_short` | [strategy/spread_selector.py](src/vix_spread/strategy/spread_selector.py) | [tests/test_spread_selector.py](tests/test_spread_selector.py) |
| `FixedRiskSizer` reads multiplier from `Product.option_multiplier()` (VIX-index $100, VX-future $1000 size correctly without caller passing the value); returns 0 when debit > risk budget | [strategy/sizing.py](src/vix_spread/strategy/sizing.py) | [tests/test_sizing.py](tests/test_sizing.py) |
| `StrategyHypothesis` constructor-bound and immutable; `contrarian_tail` factory with parameterised state-label / prob / contango thresholds | [strategy/hypothesis.py](src/vix_spread/strategy/hypothesis.py) | [tests/test_hypothesis.py](tests/test_hypothesis.py) |
| `VIXBullCallSpreadStrategy.evaluate` raises `LookaheadError` if `signal.as_of > as_of` | [strategy/strategy.py](src/vix_spread/strategy/strategy.py), [utils/errors.py](src/vix_spread/utils/errors.py) | [tests/test_strategy.py](tests/test_strategy.py) |
| `StrategyDecision` carries typed `SkipReason` (`regime_filter`, `curve_filter`, `no_spread`, `fill_rejected`, `size_zero`) for audit | [strategy/strategy.py](src/vix_spread/strategy/strategy.py) | [tests/test_strategy.py](tests/test_strategy.py) |
| `SpreadEvaluator` composes ForwardSelector + ChainIVProvider + Black76Pricer + FillEngine for per-decision evaluation | [pricing/evaluator.py](src/vix_spread/pricing/evaluator.py) | [tests/test_evaluator.py](tests/test_evaluator.py) |

## §8 — Backtest engine & reporting

| Requirement | File(s) | Test(s) |
|---|---|---|
| **Strict T → T+1 execution**: a decision at T is filled against `market[T+1]` (or next eligible), never `market[T]` | [backtest/walk_forward.py](src/vix_spread/backtest/walk_forward.py) | [tests/test_walk_forward.py](tests/test_walk_forward.py) |
| Same-minute fills structurally impossible (loop only attempts pending entries with `T > queued_at`) | [backtest/walk_forward.py](src/vix_spread/backtest/walk_forward.py) | [tests/test_walk_forward.py](tests/test_walk_forward.py) |
| Exits evaluated BEFORE new decisions within each minute (§7.2) | [backtest/walk_forward.py](src/vix_spread/backtest/walk_forward.py) | [tests/test_walk_forward.py](tests/test_walk_forward.py) |
| `FailedExit` keeps the position open for retry | [backtest/walk_forward.py](src/vix_spread/backtest/walk_forward.py) | [tests/test_walk_forward.py](tests/test_walk_forward.py) |
| Cash-equity tracking via signed cash-flow accounting (positive entry debit; negative close-credit debit; positive settlement payoff) | [backtest/walk_forward.py](src/vix_spread/backtest/walk_forward.py), [backtest/results.py](src/vix_spread/backtest/results.py) | [tests/test_walk_forward.py](tests/test_walk_forward.py) |
| **Three execution scenarios always reported** (base/optimistic/stressed); `SCENARIOS` table is the single source of truth; `accept_midpoint_optimism=True` ONLY on optimistic | [backtest/reporting.py](src/vix_spread/backtest/reporting.py) | [tests/test_reporting.py](tests/test_reporting.py) |
| Base case (`SYNTHETIC_BIDASK`) is the headline; optimistic and stressed are sensitivity-analysis only — pinned in `format_three_scenario_summary` footer | [backtest/reporting.py](src/vix_spread/backtest/reporting.py) | [tests/test_reporting.py](tests/test_reporting.py) |
| `CompletedTrade` carries entry + exit + signed P&L; `trades_to_dataframe` renders the wide audit log | [backtest/results.py](src/vix_spread/backtest/results.py), [backtest/reporting.py](src/vix_spread/backtest/reporting.py) | [tests/test_reporting.py](tests/test_reporting.py) |
| `ExecutionScenarioMetrics` includes `max_drawdown_dollars` + `max_drawdown_pct` + `hit_rate` per scenario | [backtest/reporting.py](src/vix_spread/backtest/reporting.py) | [tests/test_reporting.py](tests/test_reporting.py) |

---

## Validation-memo failure-mode neutralization summary

The validation memo flagged each of the following as a critical failure mode in the source backtester. Every one is now structurally impossible — guarded by `@dataclass(frozen=True)`, `@abstractmethod`, or a typed `raise` at the boundary — and pinned by at least one regression test.

| Failure mode | Where neutralized | Test |
|---|---|---|
| Spot VIX as Black-76 input | `ForwardSelector.select(source='spot_vix')` raises | `tests/test_pricing_no_spot_vix.py` |
| Mixed product types in a spread | `BullCallSpread.__post_init__` raises | `tests/test_products_separation.py` |
| Theoretical price as fill | `TheoreticalPrice.is_executable=False`; FillEngine TypeError | `tests/test_fill_engine_rejects_theoretical.py` |
| Midpoint as default fill | `SYNTHETIC_BIDASK` is the default; midpoint needs explicit opt-in | `tests/test_fill_engine_no_midpoint_default.py` |
| Smoothed HMM probs leak | `predict_filtered` runs forward pass manually in log-space; no smoothing API exposed | `tests/test_walk_forward_no_lookahead.py` |
| HMM transition ↔ stationary inconsistency | `HMMSpec.validate()` cross-checks | `tests/test_hmm_spec.py` |
| Same-bar (look-ahead) fills | `merge_asof(allow_exact_matches=False)`; strict T→T+1 loop | `tests/test_regime_broadcaster.py`, `tests/test_walk_forward.py` |
| VVIX flat-vol substitution | `Black76Pricer.price_spread` strict `iv_long == iv_short` check raises `FlatVolError` | `tests/test_price_spread.py` |
| Theoretical VRO instead of actual SOQ | `VIXIndexOption.settlement_value` consumes `market.vro_for(expiry)` — never spot, never theoretical | `tests/test_settlement_value.py` |
| Conflating VIX-index option with VX OOF | `VXFutureOption.deliverable_vx: VXFuture` REQUIRED; settle keys on deliverable's settlement_date | `tests/test_settlement_value.py` |
| Stale quote fills | `MinuteBarFillEngine` per-side 300s gate (NaN→∞); `FillEngine` `quote_age_seconds` gate | `tests/test_bar_fill_engine.py`, `tests/test_fill_engine_attempt_fill.py` |
| Hallucinated liquidity on exit | `ExitEngine` close-side gates with flipped leg priorities; `FailedExit` for the far-OTM no-quote case | `tests/test_exit_engine.py` |
| NaN propagation through Black-76 | `ChainIVProvider` raises `LegIVResolutionError` when both vendor and B76-invert fail | `tests/test_leg_iv.py` |
| Cherry-picking midpoint as headline | `ThreeScenarioResults` always carries all three scenarios; `accept_midpoint_optimism=True` only on optimistic | `tests/test_reporting.py` |
| Premature exits silently lost | `FailedExit` keeps position open for retry; loop retries next minute | `tests/test_walk_forward.py`, `tests/test_exit_engine.py` |

---

## End-to-end smoke run (real data, real Bloomberg pull)

`python scripts/run_smoke_backtest.py --permissive --three-scenario` against the Mar 20 → May 1 2026 window with the live-pulled data (60K daily chain rows, 13.4M tick rows, ~57K unique minute snapshots).

| Metric | Base (synthetic-bidask) | Optimistic (midpoint) | Stressed (+1 tick/leg) |
|---|---|---|---|
| Final equity | $29,161 | $31,054 | $29,181 |
| Total P&L | -$70,839 | -$68,946 | -$70,819 |
| n trades | 530 | 549 | 495 |
| Hit rate | 1.89% | 3.83% | 0.00% |
| Max DD | 87.61% | 88.62% | 85.71% |

**$1,893 of bid-ask edge bleed between base and optimistic; hit-rate halves between optimistic and base; stressed wipes every winner.** The validation memo's narrative-cherry-picking failure mode is now structurally observable in every report.

---

## Open items deferred to Phase 6

- ARCH §8.3 diagnostic suite (`RegimeAuditTrail`, `ForwardSelectionAudit`, edge-bleed distribution, HMM-stability-per-refit). Scaffolding in progress.
- Walk-forward refit cadence (current smoke runs a single fit at smoke start with daily forward-pass; ARCH §3.3 specifies weekly refit).
- Diagnostic for "why is VIX 18-22 classified as high-vol in the 1260-day lookback" — investigation belongs to §8.3 tools.
- Production-grade equity-curve subsampling for multi-year runs.
- VX OOF chain ingestion (data-blocked on the current terminal; scaffold present in `data/vx_future_options.py`).

**Signed off:** the architecture is structurally sound, the loop runs end-to-end on real data, every validation-memo failure mode is guarded, and the three-scenario reporting surfaces edge bleed transparently. Phase 6 promotion gate can begin.
