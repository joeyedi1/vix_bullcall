"""CLI: pull option-chain + VRO data into the vintage-tagged raw layer.

Three subcommands map to the three new fetchers:

  vix-index   VIX index option chain (VIXIndexOptionsChainFetcher)
              Underlying: VIX Index, cash-settled to VRO
              Defaults: DTE 7-90, ATM ± 20 strikes
              Storage: data/raw/blpapi/vix_index_options_{quotes,daily}/...

  vx-future   VX option-on-futures chain (VXFutureOptionsChainFetcher)
              Underlyings: every UX{F..Z}{YY} Index passed in (default: the
              same 8 contracts as the futures pull, V25..K26)
              Defaults: DTE 7-90, ATM ± 10 strikes
              Storage: data/raw/blpapi/vx_future_options_{quotes,daily}/...

  vro         VRO Index daily SOQ prints (VROSettlementFetcher)
              Storage: data/raw/blpapi/vro_settlement/{vintage}.parquet

The three subcommands are independent — call them one at a time so you
can audit each pull in isolation. ATM-reference defaulting reads from the
existing `vix_history_daily` and `vx_futures_ohlcv` shards (median spot
VIX / median VX-future close over the pull window), and can be overridden
via `--atm-price` / `--atm-prices-json`.

Examples
--------
  # 1. VRO history (cheap, run first to validate Bloomberg connectivity).
  python scripts/pull_options_chain.py vro --start 2010-01-01 --end 2026-05-01

  # 2. VIX index option chain over the same window as the futures.
  python scripts/pull_options_chain.py vix-index \\
      --start 2025-10-16T00:00:00 --end 2026-05-01T23:59:00

  # 3. VX OOF chain across all 8 deliverable VX futures.
  python scripts/pull_options_chain.py vx-future \\
      --start 2025-10-16T00:00:00 --end 2026-05-01T23:59:00
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from vix_spread.data.vix_index_options import (  # noqa: E402
    VIXIndexOptionsChainFetcher,
    VIXOptionContract,
)
from vix_spread.data.vix_index_options import (  # noqa: E402
    vix_option_ticker_date,
)
from vix_spread.data.vx_future_options import (  # noqa: E402
    VXFutureOptionsChainFetcher,
    VXOptionContract,
)
from vix_spread.data.vro_settlement import VROSettlementFetcher  # noqa: E402


# Default underlyings for vx-future: the 8 contracts that overlap our window.
DEFAULT_VX_UNDERLYINGS = [
    "UXV25 Index", "UXX25 Index", "UXZ25 Index",
    "UXF26 Index", "UXG26 Index", "UXH26 Index",
    "UXJ26 Index", "UXK26 Index",
]


# --------------------------------------------------------------------------- #
# ATM-reference loaders                                                       #
# --------------------------------------------------------------------------- #


def _median_spot_vix(raw_root: Path, start: datetime, end: datetime) -> float | None:
    """Median PX_LAST for `VIX Index` over the pull window, from the latest
    `vix_history_daily` shard. Returns None if no shard available."""
    d = raw_root / "blpapi" / "vix_history_daily"
    if not d.exists():
        return None
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[-1])
    sub = df[
        (df["logical"] == "vix_index")
        & (df["field"] == "PX_LAST")
        & (df["date"] >= pd.Timestamp(start.date()))
        & (df["date"] <= pd.Timestamp(end.date()))
    ]
    if sub.empty:
        return None
    return float(sub["value"].median())


def _median_vx_close_per_underlying(
    raw_root: Path, vx_tickers: list[str], start: datetime, end: datetime,
) -> dict[str, float]:
    """Median close per VX future ticker, from the latest `vx_futures_ohlcv`
    shards keyed on each ticker's safe-shard-key form."""
    d = raw_root / "blpapi" / "vx_futures_ohlcv"
    if not d.exists():
        return {}
    out: dict[str, float] = {}
    for tk in vx_tickers:
        safe = tk.replace(" ", "_")
        files = sorted(d.glob(f"{safe}_*.parquet"))
        if not files:
            continue
        df = pd.read_parquet(files[-1])
        ts = pd.to_datetime(df["time"], utc=True)
        mask = (
            (ts >= pd.Timestamp(start, tz="UTC"))
            & (ts <= pd.Timestamp(end, tz="UTC"))
            & df["close"].notna()
        )
        sub = df.loc[mask, "close"]
        if not sub.empty:
            out[tk] = float(sub.median())
    return out


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull option-chain + VRO data into data/raw/ with vintage tags.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="localhost", help="blpapi host")
    common.add_argument("--port", type=int, default=8194, help="blpapi port")
    common.add_argument("--raw-root", default="data/raw", help="raw-data root")

    # ---- vro ---- #
    p_vro = sub.add_parser(
        "vro", parents=[common],
        help="Daily VRO Index (VIX SOQ) prints — required by SettlementMarket.vro_for.",
    )
    p_vro.add_argument("--start", required=True, type=date.fromisoformat,
                       help="ISO date YYYY-MM-DD")
    p_vro.add_argument("--end", required=True, type=date.fromisoformat,
                       help="ISO date YYYY-MM-DD")

    # ---- vix-index ---- #
    p_vix = sub.add_parser(
        "vix-index", parents=[common],
        help="VIX index option chain (1-min ticks + daily IV/OI/volume).",
    )
    p_vix.add_argument("--start", required=True, type=datetime.fromisoformat,
                       help="ISO datetime YYYY-MM-DDTHH:MM:SS")
    p_vix.add_argument("--end", required=True, type=datetime.fromisoformat,
                       help="ISO datetime YYYY-MM-DDTHH:MM:SS")
    p_vix.add_argument("--dte-min", type=int, default=7,
                       help="minimum days-to-expiry (default 7 — excludes pin week)")
    p_vix.add_argument("--dte-max", type=int, default=90,
                       help="maximum days-to-expiry (default 90)")
    p_vix.add_argument("--atm-window", type=int, default=20,
                       help="strikes each side of ATM (default 20)")
    p_vix.add_argument("--atm-price", type=float, default=None,
                       help="ATM reference price; default = median spot VIX over the window")
    p_vix.add_argument("--kinds", default="quotes,daily",
                       help="comma-sep: 'quotes', 'daily', or both (default both)")
    p_vix.add_argument("--limit", type=int, default=None,
                       help="optional cap on contracts (smoke testing)")

    # ---- vix-index-historical ---- #
    p_hist = sub.add_parser(
        "vix-index-historical", parents=[common],
        help=(
            "Daily backfill for already-settled VIX index option monthly "
            "expiries (drops out of OPT_CHAIN once the contract has settled). "
            "Reconstructs the chain by Cboe strike-grid construction + "
            "ReferenceDataRequest validation."
        ),
    )
    p_hist.add_argument("--start", required=True, type=date.fromisoformat,
                        help="ISO date YYYY-MM-DD for the daily series start")
    p_hist.add_argument("--end", required=True, type=date.fromisoformat,
                        help="ISO date YYYY-MM-DD for the daily series end")
    p_hist.add_argument(
        "--months", required=True,
        help="comma-sep YYYY-MM list, e.g. '2025-11,2025-12,2026-01,2026-02,2026-03,2026-04'",
    )
    p_hist.add_argument("--atm-window", type=int, default=20,
                        help="strikes each side of ATM (default 20)")
    p_hist.add_argument("--atm-price", type=float, default=None,
                        help="ATM reference price; default = median spot VIX over window")
    p_hist.add_argument("--strike-lo", type=float, default=5.0,
                        help="candidate-grid low bound (default 5.0)")
    p_hist.add_argument("--strike-hi", type=float, default=200.0,
                        help="candidate-grid high bound (default 200.0)")

    # ---- vx-future ---- #
    p_vx = sub.add_parser(
        "vx-future", parents=[common],
        help="VX option-on-futures chain (1-min ticks + daily IV/OI/volume).",
    )
    p_vx.add_argument("--start", required=True, type=datetime.fromisoformat,
                      help="ISO datetime YYYY-MM-DDTHH:MM:SS")
    p_vx.add_argument("--end", required=True, type=datetime.fromisoformat,
                      help="ISO datetime YYYY-MM-DDTHH:MM:SS")
    p_vx.add_argument(
        "--vx-underlyings",
        default=",".join(DEFAULT_VX_UNDERLYINGS),
        help="comma-sep VX future tickers (default: V25..K26)",
    )
    p_vx.add_argument("--dte-min", type=int, default=7)
    p_vx.add_argument("--dte-max", type=int, default=90)
    p_vx.add_argument("--atm-window", type=int, default=10,
                      help="strikes each side of ATM (default 10)")
    p_vx.add_argument(
        "--atm-prices-json",
        type=str, default=None,
        help='JSON dict {"UXM25 Index": 18.0, ...}; default = median close per shard',
    )
    p_vx.add_argument("--kinds", default="quotes,daily",
                      help="comma-sep: 'quotes', 'daily', or both")
    p_vx.add_argument("--limit", type=int, default=None,
                      help="optional cap on contracts (smoke testing)")

    return parser


# --------------------------------------------------------------------------- #
# Subcommand handlers                                                         #
# --------------------------------------------------------------------------- #


def _print_manifests(label: str, manifests: list[Any]) -> None:
    if not manifests:
        print(f"[{label}] no files written.")
        return
    total_rows = sum(m.row_count for m in manifests)
    vintages = {m.vintage for m in manifests}
    vintage_str = next(iter(vintages)) if len(vintages) == 1 else f"{len(vintages)} vintages"
    print(f"[{label}] wrote {len(manifests)} file(s), {total_rows:,} rows total "
          f"(vintage={vintage_str}).")
    for m in manifests:
        print(f"  - {m.path}  ({m.row_count:,} rows)")


def _split_kinds(s: str) -> tuple[str, ...]:
    kinds = tuple(k.strip() for k in s.split(",") if k.strip())
    bad = [k for k in kinds if k not in ("quotes", "daily")]
    if bad:
        raise SystemExit(f"--kinds must be 'quotes' and/or 'daily'; got {bad}")
    if not kinds:
        raise SystemExit("--kinds may not be empty")
    return kinds


def _cmd_vro(args: argparse.Namespace) -> int:
    if args.start > args.end:
        raise SystemExit(f"--start ({args.start}) is after --end ({args.end})")
    fetcher = VROSettlementFetcher(
        host=args.host, port=args.port, raw_root=args.raw_root,
    )
    manifests = fetcher.pull(start=args.start, end=args.end)
    _print_manifests("vro", manifests)
    return 0


def _cmd_vix_index(args: argparse.Namespace) -> int:
    if args.start >= args.end:
        raise SystemExit(f"--start ({args.start}) is not before --end ({args.end})")
    kinds = _split_kinds(args.kinds)

    raw_root = Path(args.raw_root)
    atm = args.atm_price
    if atm is None:
        atm = _median_spot_vix(raw_root, args.start, args.end)
        if atm is None:
            raise SystemExit(
                "--atm-price not provided and no vix_history_daily shard found. "
                "Run `pull_data.py vix-history` first or pass --atm-price."
            )
        print(f"[vix-index] atm_price = {atm:.2f} (median spot VIX from existing shard)")

    fetcher = VIXIndexOptionsChainFetcher(
        host=args.host, port=args.port, raw_root=args.raw_root,
    )
    print(f"[vix-index] resolving OPT_CHAIN on {fetcher.UNDERLYING_TICKER}...")
    chain = fetcher.resolve_chain()
    print(f"[vix-index] discovered {len(chain)} chain entries")
    contracts = fetcher.filter_chain(
        chain,
        pull_start=args.start, pull_end=args.end,
        dte_min=args.dte_min, dte_max=args.dte_max,
        atm_price=atm, atm_window=args.atm_window,
    )
    if args.limit:
        contracts = contracts[: args.limit]
    print(f"[vix-index] filtered to {len(contracts)} contracts; pulling kinds={kinds}")

    manifests = fetcher.pull(
        start=args.start, end=args.end, contracts=contracts, kinds=kinds,
    )
    _print_manifests("vix-index", manifests)
    return 0


def _parse_months(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            y_str, m_str = tok.split("-")
            y, m = int(y_str), int(m_str)
        except (ValueError, AttributeError) as exc:
            raise SystemExit(f"--months entries must be YYYY-MM, got {tok!r}") from exc
        if not (1 <= m <= 12):
            raise SystemExit(f"month out of range in {tok!r}")
        out.append((y, m))
    if not out:
        raise SystemExit("--months may not be empty")
    return out


def _cmd_vix_index_historical(args: argparse.Namespace) -> int:
    if args.start > args.end:
        raise SystemExit(f"--start ({args.start}) is after --end ({args.end})")

    months = _parse_months(args.months)
    raw_root = Path(args.raw_root)

    atm = args.atm_price
    if atm is None:
        atm = _median_spot_vix(
            raw_root,
            datetime.combine(args.start, datetime.min.time()),
            datetime.combine(args.end, datetime.min.time()),
        )
        if atm is None:
            raise SystemExit(
                "--atm-price not provided and no vix_history_daily shard found. "
                "Run `pull_data.py vix-history` first or pass --atm-price."
            )
        print(f"[vix-index-historical] atm_price = {atm:.2f} (median spot VIX)")

    fetcher = VIXIndexOptionsChainFetcher(
        host=args.host, port=args.port, raw_root=args.raw_root,
    )

    # Per-month: enumerate via Cboe strike grid + Bloomberg validation,
    # then ATM-window-filter the survivors.
    contracts: list[VIXOptionContract] = []
    for y, m in months:
        ticker_date = vix_option_ticker_date(y, m)
        print(f"[vix-index-historical] enumerating {y}-{m:02d} (ticker date {ticker_date}) ...")
        listed = fetcher.enumerate_historical_chain(
            y, m, strike_lo=args.strike_lo, strike_hi=args.strike_hi,
        )
        print(f"  -> {len(listed)} contracts validated by Bloomberg")
        if not listed:
            continue
        # ATM-window filter, per (right) bucket.
        for right in ("C", "P"):
            bucket = sorted(
                (c for c in listed if c.right == right),
                key=lambda c: abs(c.strike - atm),
            )[: args.atm_window * 2 + 1]
            contracts.extend(bucket)

    if not contracts:
        raise SystemExit("No historical contracts validated. "
                         "Check ticker form / Bloomberg subscription.")
    contracts.sort(key=lambda c: (c.expiry_date, c.right, c.strike))
    print(f"[vix-index-historical] total filtered contracts: {len(contracts)}")

    # Daily-only — quotes won't be available for already-settled options
    # given the ~45-day Bloomberg intraday-history ceiling.
    start_dt = datetime.combine(args.start, datetime.min.time())
    end_dt = datetime.combine(args.end, datetime.max.time().replace(microsecond=0))
    manifests = fetcher.pull(
        start=start_dt, end=end_dt, contracts=contracts, kinds=("daily",),
    )
    _print_manifests("vix-index-historical", manifests)
    return 0


def _cmd_vx_future(args: argparse.Namespace) -> int:
    if args.start >= args.end:
        raise SystemExit(f"--start ({args.start}) is not before --end ({args.end})")
    kinds = _split_kinds(args.kinds)

    underlyings = [s.strip() for s in args.vx_underlyings.split(",") if s.strip()]
    raw_root = Path(args.raw_root)

    if args.atm_prices_json:
        atm_prices = json.loads(args.atm_prices_json)
    else:
        atm_prices = _median_vx_close_per_underlying(
            raw_root, underlyings, args.start, args.end,
        )
        missing = [u for u in underlyings if u not in atm_prices]
        if missing:
            raise SystemExit(
                f"No vx_futures_ohlcv shards for {missing}. "
                f"Run `pull_data.py vx-futures --kind ohlcv` first or pass --atm-prices-json."
            )
        for u, p in atm_prices.items():
            print(f"[vx-future] atm[{u}] = {p:.2f} (median close from shard)")

    fetcher = VXFutureOptionsChainFetcher(
        host=args.host, port=args.port, raw_root=args.raw_root,
    )
    print(f"[vx-future] resolving OPT_CHAIN for {len(underlyings)} underlyings...")
    chain_by_u = fetcher.resolve_chain(underlyings)
    total_chain = sum(len(v) for v in chain_by_u.values())
    print(f"[vx-future] discovered {total_chain} chain entries across {len(chain_by_u)} underlyings")
    contracts = fetcher.filter_chain(
        chain_by_u,
        pull_start=args.start, pull_end=args.end,
        dte_min=args.dte_min, dte_max=args.dte_max,
        atm_prices=atm_prices, atm_window=args.atm_window,
    )
    if args.limit:
        contracts = contracts[: args.limit]
    print(f"[vx-future] filtered to {len(contracts)} contracts; pulling kinds={kinds}")

    manifests = fetcher.pull(
        start=args.start, end=args.end, contracts=contracts, kinds=kinds,
    )
    _print_manifests("vx-future", manifests)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "vro":
        return _cmd_vro(args)
    if args.cmd == "vix-index":
        return _cmd_vix_index(args)
    if args.cmd == "vix-index-historical":
        return _cmd_vix_index_historical(args)
    if args.cmd == "vx-future":
        return _cmd_vx_future(args)
    raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
