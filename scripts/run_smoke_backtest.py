"""End-to-end smoke backtest — VIX bull call spread, 2026-03-20 to 2026-05-01.

Wires the entire stack against real ingested data:
  - DataProcessor for the chain panel (ChainIVProvider input) and
    per-minute NBBO panel (FillEngine quotes).
  - WalkForwardRegimeFitter for the HMM regime signal (single fit at
    smoke start, then daily forward-pass for filtered probabilities —
    the smoke-grade simplification of weekly walk-forward refit).
  - VIXBullCallSpreadStrategy with `contrarian_tail` hypothesis and
    FixedRiskSizer @ 0.5%.
  - WalkForwardBacktest with strict T → T+1 execution.

Outputs a console summary with starting/final equity, completed trade
count, rejection breakdown, and decision-skip-reason histogram so we
can see whether the hypothesis even fired during this window.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vix_spread.backtest.reporting import (  # noqa: E402
    format_three_scenario_summary,
    run_three_scenarios,
)
from vix_spread.backtest.walk_forward import WalkForwardBacktest  # noqa: E402
from vix_spread.data.feature_panel import FeaturePanel  # noqa: E402
from vix_spread.data.processor import DataProcessor  # noqa: E402
from vix_spread.data.snapshot import VIXSnapshot  # noqa: E402
from vix_spread.execution.exit_engine import ExitEngine  # noqa: E402
from vix_spread.execution.exit_policy import ExitPolicy  # noqa: E402
from vix_spread.execution.fill_engine import FillEngine  # noqa: E402
from vix_spread.execution.fill_modes import FillMode  # noqa: E402
from vix_spread.execution.liquidity_gates import LiquidityGates  # noqa: E402
from vix_spread.execution.quote import OptionQuote  # noqa: E402
from vix_spread.pricing.black76 import Black76Pricer  # noqa: E402
from vix_spread.pricing.evaluator import SpreadEvaluator  # noqa: E402
from vix_spread.pricing.forward_selector import ForwardSelector  # noqa: E402
from vix_spread.pricing.leg_iv import ChainIVProvider  # noqa: E402
from vix_spread.regime.base import FittedRegime, RegimeSignal  # noqa: E402
from vix_spread.regime.broadcaster import broadcast_daily_to_minute  # noqa: E402
from vix_spread.regime.hmm_spec import HMMSpec  # noqa: E402
from vix_spread.regime.walk_forward import WalkForwardRegimeFitter  # noqa: E402
from vix_spread.strategy.hypothesis import (  # noqa: E402
    StrategyHypothesis, make_contrarian_tail_hypothesis,
)
from vix_spread.strategy.sizing import FixedRiskSizer  # noqa: E402
from vix_spread.strategy.spread_selector import SpreadSelector  # noqa: E402
from vix_spread.strategy.strategy import VIXBullCallSpreadStrategy  # noqa: E402


SMOKE_START = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
SMOKE_END = datetime(2026, 5, 1, 23, 59, tzinfo=timezone.utc)
STARTING_EQUITY = 100_000.0
MAY_2026_SOQ = date(2026, 5, 20)


# --------------------------------------------------------------------------- #
# Data loaders                                                                #
# --------------------------------------------------------------------------- #


def load_vix_feature_panel() -> FeaturePanel:
    """Long-form vix_history_daily → FeaturePanel with `log_vix`.

    Dates anchored at NY 16:00 close → UTC tz-aware. The panel's
    `as_of_map` records the same close timestamp as the value's
    knowable-from time, so `slice_causal(as_of)` accepts everything
    on or before that close.
    """
    d = REPO_ROOT / "data" / "raw" / "blpapi" / "vix_history_daily"
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No vix_history_daily parquets in {d}")
    df = pd.read_parquet(files[-1])
    vix = df[(df["logical"] == "vix_index") & (df["field"] == "PX_LAST")]
    vix = vix.sort_values("date").drop_duplicates("date", keep="last")
    naive_dates = pd.to_datetime(vix["date"]).dt.normalize()
    ny_close = naive_dates + pd.Timedelta(hours=16)
    dates_utc = (
        ny_close.dt.tz_localize(
            "America/New_York", ambiguous=False, nonexistent="shift_forward",
        ).dt.tz_convert("UTC")
    )
    log_vix = pd.Series(
        np.log(vix["value"].astype(float).values),
        index=pd.DatetimeIndex(dates_utc.values, tz="UTC"),
        name="log_vix",
    )
    features = pd.DataFrame({"log_vix": log_vix})
    as_of_map = {"log_vix": pd.Series(log_vix.index, index=log_vix.index)}
    return FeaturePanel(
        dates=log_vix.index, features=features, as_of_map=as_of_map,
    )


def compute_daily_curve_slope() -> pd.Series:
    """`slope_m1_m2 = (UX2 / UX1) - 1` per day. Positive = contango (ARCH §3.4)."""
    d = REPO_ROOT / "data" / "raw" / "blpapi" / "vix_history_daily"
    files = sorted(d.glob("*.parquet"))
    df = pd.read_parquet(files[-1])
    sub = df[df["field"] == "PX_LAST"][["date", "logical", "value"]]
    sub = sub.drop_duplicates(["date", "logical"], keep="last")
    wide = sub.pivot(index="date", columns="logical", values="value")
    if "vx_m1" not in wide or "vx_m2" not in wide:
        raise ValueError(f"vix_history_daily missing vx_m1/vx_m2; got {wide.columns}")
    slope = wide["vx_m2"] / wide["vx_m1"] - 1.0
    return slope.dropna()


def load_uxk26_close_per_minute() -> pd.Series:
    """UXK26 1-minute close — the May 2026 deliverable VX future, used as
    the forward for the May 20 SOQ Wed expiry."""
    d = REPO_ROOT / "data" / "raw" / "blpapi" / "vx_futures_ohlcv"
    files = sorted(d.glob("UXK26_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No UXK26 OHLCV in {d}")
    df = pd.read_parquet(files[-1])
    df = df.sort_values("time")
    return (
        df.set_index(pd.to_datetime(df["time"], utc=True))["close"]
        .dropna()
        .astype(float)
    )


# --------------------------------------------------------------------------- #
# Per-minute callable factories                                                #
# --------------------------------------------------------------------------- #


def _row_to_optionquote(row, ts: pd.Timestamp, contract_id: str) -> OptionQuote:
    return OptionQuote(
        timestamp=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
        contract_id=contract_id,
        bid=float(row["bid"]),
        ask=float(row["ask"]),
        bid_size=int(row["bid_size"]) if pd.notna(row["bid_size"]) else 0,
        ask_size=int(row["ask_size"]) if pd.notna(row["ask_size"]) else 0,
        last_trade=(
            float(row["last_trade"]) if pd.notna(row["last_trade"]) else None
        ),
        last_trade_age_seconds=(
            float(row["last_trade_age_seconds"])
            if pd.notna(row["last_trade_age_seconds"]) else None
        ),
        is_locked=bool(row["is_locked"]),
        is_crossed=bool(row["is_crossed"]),
        quote_age_seconds=(
            float(row["quote_age_seconds"])
            if pd.notna(row["quote_age_seconds"]) else 9999.0
        ),
    )


def build_quotes_by_minute(quotes_panel: pd.DataFrame) -> dict:
    """`{Timestamp -> {contract_id -> OptionQuote}}` for fast market_at lookup.

    Skips rows where bid OR ask is NaN (no NBBO at this minute for this
    contract — the FillEngine would reject anyway).
    """
    print(f"  building quote dict from {len(quotes_panel):,} rows ...", flush=True)
    out: dict[pd.Timestamp, dict[str, OptionQuote]] = {}
    t0 = time.time()
    for row in quotes_panel.itertuples(index=True):
        ts, cid = row.Index
        bid = row.bid
        ask = row.ask
        if pd.isna(bid) or pd.isna(ask):
            continue
        oq = OptionQuote(
            timestamp=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
            contract_id=cid,
            bid=float(bid),
            ask=float(ask),
            bid_size=int(row.bid_size) if pd.notna(row.bid_size) else 0,
            ask_size=int(row.ask_size) if pd.notna(row.ask_size) else 0,
            last_trade=(
                float(row.last_trade) if pd.notna(row.last_trade) else None
            ),
            last_trade_age_seconds=(
                float(row.last_trade_age_seconds)
                if pd.notna(row.last_trade_age_seconds) else None
            ),
            is_locked=bool(row.is_locked),
            is_crossed=bool(row.is_crossed),
            quote_age_seconds=(
                float(row.quote_age_seconds)
                if pd.notna(row.quote_age_seconds) else 9999.0
            ),
        )
        out.setdefault(ts, {})[cid] = oq
    print(
        f"  built {len(out):,} unique minute snapshots "
        f"in {time.time() - t0:.1f}s", flush=True,
    )
    return out


def build_market_at(
    quotes_by_minute: dict,
    uxk26_close: pd.Series,
    chain_dates: set[date],
):
    def market_at(T):
        # Skip non-trading days — daily chain panel has no entry for
        # weekends/holidays; the ChainIVProvider would raise KeyError
        # on such dates. Filtering at the market_at edge keeps the
        # rest of the engine date-naive.
        if T.weekday() >= 5:
            return None
        if T.date() not in chain_dates:
            return None
        ts = pd.Timestamp(T)
        bucket = quotes_by_minute.get(ts)
        if bucket is None:
            return None
        # Forward for May 20 SOQ = UXK26 last close at or before T.
        try:
            f = uxk26_close.asof(ts)
        except KeyError:
            f = np.nan
        vx_curve = (
            {MAY_2026_SOQ: float(f)}
            if (f is not None and not pd.isna(f) and f > 0)
            else {}
        )
        return VIXSnapshot(
            timestamp=T, vx_curve=vx_curve, options_quotes=bucket,
            risk_free_rate=0.04, vix_spot=None,
        )
    return market_at


def build_signal_at(minute_signals_df: pd.DataFrame, hypothesis_tag: str):
    def signal_at(T):
        ts = pd.Timestamp(T)
        try:
            row = minute_signals_df.loc[ts]
        except KeyError:
            return None
        if pd.isna(row.get("state_label")):
            return None
        as_of_eff = row["as_of_effective"]
        as_of_eff_dt = (
            as_of_eff.to_pydatetime()
            if isinstance(as_of_eff, pd.Timestamp) else as_of_eff
        )
        return RegimeSignal(
            as_of=as_of_eff_dt,
            filtered_probs=np.array([row["p_low"], row["p_high"]]),
            state_label=int(row["state_label"]),
            curve_features={"slope_m1_m2": float(row["slope_m1_m2"])},
            hypothesis_tag=hypothesis_tag,
            as_of_inputs={"log_vix": as_of_eff_dt},
        )
    return signal_at


def build_exit_decider():
    # VIX options trade until 4pm ET (16:00 ET) = 20:00 UTC during EDT.
    # Smoke-grade EOD: 19:30 UTC = 3:30 PM ET on the smoke's last trading
    # day, leaving 30 minutes for the quotes to be available before close.
    smoke_eod = datetime(
        SMOKE_END.year, SMOKE_END.month, SMOKE_END.day,
        19, 30, tzinfo=timezone.utc,
    )

    def exit_decider(pos, T):
        soq_wed = pos.spread.long_leg.settlement_event.date()
        last_trade_tue = soq_wed - timedelta(days=1)
        # Standard forced-Tuesday exit at end-of-RTH on the Tuesday before SOQ.
        if T.date() == last_trade_tue and T.hour >= 19:
            return ExitPolicy.FORCED_TUESDAY_LIQUIDATION
        # Smoke-only: liquidate anything still open near the smoke window's
        # last trading-day close so the report has completed trades. Smoke
        # simplification, not a strategy rule.
        if T >= smoke_eod:
            return ExitPolicy.FORCED_TUESDAY_LIQUIDATION
        return None
    return exit_decider


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--permissive", action="store_true",
        help=(
            "Use an always-true hypothesis instead of contrarian_tail. "
            "Smoke-grade: lets the strategy fire on every minute with a "
            "fillable spread so the trade machinery exercises end-to-end "
            "even when the real regime/curve filter would gate everything."
        ),
    )
    parser.add_argument(
        "--three-scenario", action="store_true",
        help=(
            "Run the backtest THREE times — base (SYNTHETIC_BIDASK), "
            "optimistic (MIDPOINT), stressed (SYNTHETIC_PLUS_SLIPPAGE) — "
            "and emit a side-by-side comparison per ARCH §8.2. "
            "~3x compute; data prep is shared across scenarios."
        ),
    )
    args = parser.parse_args()

    hypothesis_label = (
        "PERMISSIVE (always-true)" if args.permissive else "contrarian_tail"
    )
    print("=" * 70)
    print(f"Smoke backtest — VIX bull call spread ({hypothesis_label})")
    print(f"Window: {SMOKE_START.date()} → {SMOKE_END.date()}")
    print(f"Starting equity: ${STARTING_EQUITY:,.0f}")
    print("=" * 70)
    print()

    print("[1/8] Loading processed panels ...")
    proc = DataProcessor()
    chain_panel = proc.process_vix_options_daily()
    print(f"  chain panel:  {chain_panel.shape}")
    quotes_panel = proc.process_vix_index_options()
    print(f"  quotes panel: {quotes_panel.shape}")

    # Limit quotes to the smoke window (timestamp level).
    ts_lvl = quotes_panel.index.get_level_values("timestamp")
    in_window = (ts_lvl >= pd.Timestamp(SMOKE_START)) & (ts_lvl <= pd.Timestamp(SMOKE_END))
    smoke_quotes = quotes_panel.loc[in_window]
    print(f"  smoke-window quotes: {smoke_quotes.shape}")

    print()
    print("[2/8] Building VIX feature panel ...")
    feature_panel = load_vix_feature_panel()
    print(
        f"  feature panel: {feature_panel.features.shape}, "
        f"{feature_panel.dates.min()} → {feature_panel.dates.max()}"
    )

    print()
    print("[3/8] Computing daily curve slope (M1/M2) ...")
    slope_series = compute_daily_curve_slope()
    print(
        f"  slope series: {len(slope_series):,} days, "
        f"range {slope_series.min():.3f} → {slope_series.max():.3f}, "
        f"median {slope_series.median():.3f}"
    )

    print()
    print("[4/8] Fitting HMM at smoke start ...")
    spec = HMMSpec(
        n_states=2,
        transition_matrix=np.array([[0.96, 0.04], [0.117, 0.883]]),
        state_label_rule="by_emission_variance",
    )
    fitter = WalkForwardRegimeFitter(
        spec=spec, feature_column="log_vix", lookback_days=1260,
        cadence="weekly", random_state=0,
        hypothesis_tag="contrarian_tail",
    )
    fitted_at_start = fitter.fit_walk_forward(feature_panel, as_of=SMOKE_START)
    print(f"  fit on {len(fitted_at_start.observations):,} observations")
    print(f"  label_map: {fitted_at_start.label_map}")

    print()
    print("[5/8] Generating daily regime + curve signals ...")
    smoke_dates = pd.date_range(
        SMOKE_START.date(), SMOKE_END.date(), freq="D",
    )
    daily_rows = []
    for d in smoke_dates:
        d_close = pd.Timestamp(d).tz_localize(None) + pd.Timedelta(hours=16)
        d_close_utc = (
            d_close.tz_localize(
                "America/New_York", ambiguous=False, nonexistent="shift_forward",
            ).tz_convert("UTC").to_pydatetime()
        )
        log_vix_through_d = (
            feature_panel.slice_causal("log_vix", d_close_utc).dropna()
        )
        if len(log_vix_through_d) < 5:
            continue
        try:
            new_fitted = FittedRegime(
                as_of=d_close_utc,
                observation_timestamps=tuple(log_vix_through_d.index),
                observations=tuple(log_vix_through_d.tolist()),
                model=fitted_at_start.model,
                label_map=fitted_at_start.label_map,
            )
            sig = fitter.predict_filtered(new_fitted, d_close_utc)
        except Exception as exc:
            print(f"  signal at {d_close_utc}: {type(exc).__name__}: {exc}")
            continue
        slope = slope_series.asof(pd.Timestamp(d))
        daily_rows.append(
            {
                "as_of": pd.Timestamp(d_close_utc),
                "state_label": sig.state_label,
                "p_low": float(sig.filtered_probs[0]),
                "p_high": float(sig.filtered_probs[1]),
                "slope_m1_m2": float(slope) if not pd.isna(slope) else 0.0,
            }
        )
    daily_signals = pd.DataFrame(daily_rows).set_index("as_of")
    print(f"  daily signals: {len(daily_signals)}")
    print(f"  state_label distribution: {dict(Counter(daily_signals['state_label']))}")
    print(
        f"  p_low range:    "
        f"{daily_signals['p_low'].min():.3f} → "
        f"{daily_signals['p_low'].max():.3f} "
        f"(median {daily_signals['p_low'].median():.3f})"
    )
    print(
        f"  slope range:    "
        f"{daily_signals['slope_m1_m2'].min():.3f} → "
        f"{daily_signals['slope_m1_m2'].max():.3f} "
        f"(median {daily_signals['slope_m1_m2'].median():.3f})"
    )

    print()
    print("[6/8] Broadcasting daily → minute grid ...")
    minute_grid = pd.date_range(SMOKE_START, SMOKE_END, freq="1min", tz="UTC")
    minute_signals = broadcast_daily_to_minute(daily_signals, minute_grid)
    n_minute_with_signal = minute_signals["state_label"].notna().sum()
    print(f"  minute grid: {len(minute_grid):,} minutes")
    print(f"  with signal: {n_minute_with_signal:,}")

    print()
    print("[7/8] Building per-minute lookups (quotes + UXK26 forward) ...")
    quotes_by_minute = build_quotes_by_minute(smoke_quotes)
    uxk26 = load_uxk26_close_per_minute()
    print(f"  UXK26 close: {len(uxk26):,} bars, "
          f"{uxk26.index.min()} → {uxk26.index.max()}")

    chain_dates = set(chain_panel.index.get_level_values("date").unique())
    market_at = build_market_at(quotes_by_minute, uxk26, chain_dates)
    signal_at = build_signal_at(minute_signals, hypothesis_tag="contrarian_tail")
    exit_decider = build_exit_decider()

    print()
    print("[8/8] Building strategy stack + backtest engine ...")
    pricer = Black76Pricer()
    chain_iv_provider = ChainIVProvider(chain_panel, pricer)
    forward_selector = ForwardSelector()
    fill_engine = FillEngine()
    gates = LiquidityGates(
        max_leg_spread_pct=0.50,
        min_leg_open_interest=0,
        min_leg_volume_today=0,
        min_displayed_size=1,
        max_quote_age_seconds=60.0,
        reject_locked_or_crossed=True,
        reject_no_bid_short_leg=True,
        max_order_size_pct_of_displayed=0.5,
    )
    evaluator = SpreadEvaluator(
        chain_iv_provider=chain_iv_provider,
        forward_selector=forward_selector,
        pricer=pricer,
        fill_engine=fill_engine,
        gates=gates,
    )
    if args.permissive:
        hypothesis = StrategyHypothesis(
            name="contrarian_tail",
            entry_regime_filter=lambda s: True,
            entry_curve_filter=lambda f: True,
            expected_holding_period=timedelta(days=21),
        )
    else:
        hypothesis = make_contrarian_tail_hypothesis(
            curve_feature_key="slope_m1_m2",
        )
    selector = SpreadSelector(
        long_offset=2.0, short_offset=8.0, dte_min=7, dte_max=60,
    )
    sizer = FixedRiskSizer(risk_per_trade_pct=0.005)
    strategy = VIXBullCallSpreadStrategy(
        hypothesis=hypothesis,
        spread_selector=selector,
        evaluator=evaluator,
        sizer=sizer,
        exit_policy=ExitPolicy.FORCED_TUESDAY_LIQUIDATION,
    )
    exit_engine = ExitEngine(gates=gates)

    if args.three_scenario:
        # Per ARCH §8.2: rebuild the engine per scenario, sharing the
        # heavy data prep (market_at / signal_at / exit_decider closures
        # are already built and reused).
        def factory(mode, accept_mid, slip):
            return WalkForwardBacktest(
                strategy=strategy,
                fill_engine=fill_engine,
                exit_engine=exit_engine,
                gates=gates,
                starting_equity=STARTING_EQUITY,
                market_at=market_at,
                signal_at=signal_at,
                exit_decider=exit_decider,
                fill_mode=mode,
                accept_midpoint_optimism=accept_mid,
                slippage_ticks_per_leg=slip,
            )

        print()
        print("Running THREE-SCENARIO backtest ...", flush=True)
        t0 = time.time()
        bundle = run_three_scenarios(factory, minute_grid)
        print(f"  done in {time.time() - t0:.1f}s")

        print()
        print(format_three_scenario_summary(bundle))
        return 0

    backtest = WalkForwardBacktest(
        strategy=strategy,
        fill_engine=fill_engine,
        exit_engine=exit_engine,
        gates=gates,
        starting_equity=STARTING_EQUITY,
        market_at=market_at,
        signal_at=signal_at,
        exit_decider=exit_decider,
        fill_mode=FillMode.SYNTHETIC_BIDASK,
    )

    print()
    print("Running backtest ...", flush=True)
    t0 = time.time()
    results = backtest.run(minute_grid)
    print(f"  done in {time.time() - t0:.1f}s")

    print()
    _print_summary(results)
    return 0


def _print_summary(results) -> None:
    print("=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Starting equity:      ${results.starting_equity:>14,.2f}")
    print(f"Final equity:         ${results.final_equity:>14,.2f}")
    pnl = results.final_equity - results.starting_equity
    pnl_pct = (
        (pnl / results.starting_equity * 100)
        if results.starting_equity else 0.0
    )
    print(f"P&L:                  ${pnl:>+14,.2f}  ({pnl_pct:+.2f}%)")
    print()
    print(f"Decisions logged:     {len(results.decisions_log):,}")
    enter_decisions = [d for d in results.decisions_log if d.action == "enter"]
    skip_decisions = [d for d in results.decisions_log if d.action == "skip"]
    print(f"  enter actions:      {len(enter_decisions):,}")
    print(f"  skip actions:       {len(skip_decisions):,}")
    if skip_decisions:
        print(f"  skip reasons (top 10):")
        for reason, n in Counter(d.reason for d in skip_decisions).most_common(10):
            print(f"    {reason:<24s} {n:>10,}")

    print()
    print(f"Completed trades:     {len(results.completed_trades):,}")
    print(f"Open at end:          {len(results.open_positions):,}")

    if results.completed_trades:
        pnls = [t.pnl for t in results.completed_trades]
        print(f"  P&L per trade:")
        print(f"    total:            ${sum(pnls):>+14,.2f}")
        print(f"    mean:             ${np.mean(pnls):>+14,.2f}")
        print(f"    min:              ${min(pnls):>+14,.2f}")
        print(f"    max:              ${max(pnls):>+14,.2f}")

    print()
    print(f"Rejection log:        {len(results.rejection_log):,} entries")
    if results.rejection_log:
        print(f"  rejection reasons:")
        for reason, n in Counter(
            getattr(item, "reason", "?") for item in results.rejection_log
        ).most_common(10):
            print(f"    {reason:<24s} {n:>10,}")
    print()


if __name__ == "__main__":
    sys.exit(main())
