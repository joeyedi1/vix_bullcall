"""CLI: pull Bloomberg data into the vintage-tagged raw layer.

Wires the abstract Phase-1 fetchers (`VIXHistoryFetcher`,
`VXFuturesIntradayFetcher`) to concrete ticker mappings and persists each
pull under `data/raw/{source}/{product}/{vintage}.parquet` via
`BaseDataFetcher.pull(...)`. The script does not touch the on-disk layout
itself — it relies on the base class for vintage stamping.

Subcommands
-----------
  vix-history   Daily close for VIX Index + UX1/UX2 generics (regime panel).
  vx-futures    1-min OHLCV or BBO ticks for specific monthly VX contracts,
                resolved through the expiry calendar
                (UXM5 Index = June 2025, ...).

Examples
--------
  # Regime-panel history, 2010-01-01 through today.
  python scripts/pull_data.py vix-history --start 2010-01-01 --end 2025-12-31

  # 1-min OHLCV for the June 2025 VX contract.
  python scripts/pull_data.py vx-futures --kind ohlcv \
      --start 2025-04-01T00:00:00 --end 2025-06-18T23:59:00 \
      --contract 2025-06

  # 1-min BBO ticks for every 2025 monthly contract.
  python scripts/pull_data.py vx-futures --kind quotes \
      --start 2025-01-01T00:00:00 --end 2025-12-31T23:59:00 \
      --year 2025

The CLI does not specify Bloomberg credentials — it expects a running
Bloomberg terminal on the local machine reachable at host:port (default
localhost:8194).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vix_spread.data.expiry_calendar import vx_bloomberg_ticker  # noqa: E402
from vix_spread.data.vix_history import VIXHistoryFetcher  # noqa: E402
from vix_spread.data.vx_futures import VXFuturesIntradayFetcher  # noqa: E402


# --------------------------------------------------------------------------- #
# Concrete fetchers — Phase-1 ticker mapping wired here.                      #
# These are CLI-private; promote to their own modules when a second consumer  #
# appears (per CLAUDE.md "Simplicity First").                                 #
# --------------------------------------------------------------------------- #


class VIXHistoryGenericsFetcher(VIXHistoryFetcher):
    """VIX Index + UX1/UX2 generics for the regime panel.

    Generics are intentional here: the regime layer (ARCHITECTURE §3) consumes
    a continuous M1/M2 series. Settlement-date-specific contracts belong to
    the ForwardSelector path (§4.1) and use a different mapping.
    """

    def _resolve_tickers(self) -> dict[str, str]:
        return {
            "vix_index": "VIX Index",
            "vx_m1": "UX1 Index",
            "vx_m2": "UX2 Index",
        }


class VXFuturesByMonthFetcher(VXFuturesIntradayFetcher):
    """Resolves to specific monthly contracts via the expiry calendar.

    The caller passes `contracts=[(year, month), ...]`; this class converts
    each to its Bloomberg security via `vx_bloomberg_ticker`. The expiry
    calendar is the canonical mapping — no string concatenation here.
    """

    def _resolve_tickers(
        self,
        *,
        contracts: list[tuple[int, int]] | None = None,
        **_: Any,
    ) -> list[str]:
        if not contracts:
            raise ValueError(
                "vx-futures requires at least one --contract YYYY-MM (or --year YYYY)."
            )
        return [vx_bloomberg_ticker(y, m) for y, m in contracts]


# --------------------------------------------------------------------------- #
# Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def _parse_contract(s: str) -> tuple[int, int]:
    try:
        y_str, m_str = s.split("-")
        y, m = int(y_str), int(m_str)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"--contract must be YYYY-MM, got {s!r}"
        ) from exc
    if not (1 <= m <= 12):
        raise argparse.ArgumentTypeError(f"month must be 1..12, got {m} in {s!r}")
    return y, m


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull Bloomberg data into data/raw/ with vintage tags.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="localhost", help="blpapi host")
    common.add_argument("--port", type=int, default=8194, help="blpapi port")
    common.add_argument(
        "--raw-root",
        default="data/raw",
        help="raw-data root (default: data/raw)",
    )

    p_hist = sub.add_parser(
        "vix-history",
        parents=[common],
        help="Daily close for VIX Index + UX1/UX2 generics (regime panel).",
    )
    p_hist.add_argument(
        "--start",
        required=True,
        type=date.fromisoformat,
        help="ISO date YYYY-MM-DD",
    )
    p_hist.add_argument(
        "--end",
        required=True,
        type=date.fromisoformat,
        help="ISO date YYYY-MM-DD",
    )

    p_vx = sub.add_parser(
        "vx-futures",
        parents=[common],
        help="1-min OHLCV or BBO ticks for specific monthly VX contracts.",
    )
    p_vx.add_argument(
        "--kind",
        required=True,
        choices=["ohlcv", "quotes"],
        help="ohlcv = IntradayBarRequest; quotes = IntradayTickRequest",
    )
    p_vx.add_argument(
        "--start",
        required=True,
        type=datetime.fromisoformat,
        help="ISO datetime YYYY-MM-DDTHH:MM:SS",
    )
    p_vx.add_argument(
        "--end",
        required=True,
        type=datetime.fromisoformat,
        help="ISO datetime YYYY-MM-DDTHH:MM:SS",
    )
    grp = p_vx.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--contract",
        action="append",
        type=_parse_contract,
        metavar="YYYY-MM",
        help="contract month; repeatable (e.g. --contract 2025-06 --contract 2025-07)",
    )
    grp.add_argument(
        "--year",
        type=int,
        help="shortcut: pull every monthly contract of this calendar year",
    )

    return parser


# --------------------------------------------------------------------------- #
# Subcommand handlers                                                         #
# --------------------------------------------------------------------------- #


def _print_manifests(label: str, manifests: list[Any]) -> None:
    if not manifests:
        print(f"[{label}] no files written (all shards empty or errored).")
        return
    total_rows = sum(m.row_count for m in manifests)
    vintages = {m.vintage for m in manifests}
    vintage_str = next(iter(vintages)) if len(vintages) == 1 else f"{len(vintages)} vintages"
    print(
        f"[{label}] wrote {len(manifests)} file(s), "
        f"{total_rows:,} rows total (vintage={vintage_str})."
    )
    for m in manifests:
        print(f"  - {m.path}  ({m.row_count:,} rows)")


def _cmd_vix_history(args: argparse.Namespace) -> int:
    if args.start > args.end:
        raise SystemExit(f"--start ({args.start}) is after --end ({args.end})")
    fetcher = VIXHistoryGenericsFetcher(
        host=args.host, port=args.port, raw_root=args.raw_root,
    )
    manifests = fetcher.pull(start=args.start, end=args.end)
    _print_manifests("vix-history", manifests)
    return 0


def _cmd_vx_futures(args: argparse.Namespace) -> int:
    if args.start >= args.end:
        raise SystemExit(f"--start ({args.start}) is not before --end ({args.end})")

    if args.year is not None:
        contracts = [(args.year, m) for m in range(1, 13)]
    else:
        contracts = list(args.contract)

    fetcher = VXFuturesByMonthFetcher(
        host=args.host, port=args.port, raw_root=args.raw_root,
    )
    manifests = fetcher.pull(
        start=args.start,
        end=args.end,
        kind=args.kind,
        contracts=contracts,
    )
    _print_manifests(f"vx-futures/{args.kind}", manifests)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "vix-history":
        return _cmd_vix_history(args)
    if args.cmd == "vx-futures":
        return _cmd_vx_futures(args)
    raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
