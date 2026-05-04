"""VXFuture — CFE VX futures contract identity.

ARCHITECTURE §4.1 / §4.5. Pure identity + immutable spec; price is NOT
carried here (that lives on market snapshots). The deliverable for a
`VXFutureOption` is one `VXFuture`, set at contract-mapping time and
immutable thereafter — preventing the validation-memo "factor of 10"
hedging defect by making the multiplier a contract-level property
rather than a free-floating constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class VXFuture:
    """One VX futures contract (identity + spec only).

    Attributes
    ----------
    contract_root
        Bloomberg / vendor root, e.g. 'VX' or 'UX'. Carried for audit
        rather than dispatch — pricing/settlement code MUST NOT branch
        on this string (use product subclass instead).
    settlement_date
        CFE final settlement date — the join key into vx_settle_prints
        for option settlement and into vx_curve for forward selection.
    multiplier
        Dollar multiplier per VX point per contract. CFE spec is $1000
        and is immutable; defaulted to that value.
    """
    contract_root: str
    settlement_date: date
    multiplier: float = 1000.0
