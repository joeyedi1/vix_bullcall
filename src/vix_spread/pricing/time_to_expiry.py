"""minutes_to_settlement — exact minute-level calendar time-to-expiry.

ARCHITECTURE §4.2. Returns the year-fraction `T = total_minutes /
525_600` for direct use as the Black-76 time argument. Calendar
(not 252-business) is the convention used by the IV vendor in this
codebase, so the pricer and the surface stay on the same clock — no
252/365 mismatch and no day-count drift across DST transitions.

Both timestamps must be tz-aware exchange-time. `as_of` must be
strictly before `settlement_event`: pricing an option at or after its
settlement event is a NaN-or-worse footgun, raised loudly here rather
than producing a meaningless number that silently propagates through
Greeks and P&L.

Despite the function name, the RETURN value is the YEAR-FRACTION (the
minute count divided by 525,600). The architecture keeps the
"minutes_to_settlement" name because the pre-divided minute count is
the audit-relevant quantity; the year-fraction is the form the pricer
consumes.
"""
from __future__ import annotations

from datetime import datetime

from vix_spread.utils.errors import ExpiryError, TimezoneError


_SECONDS_PER_MINUTE = 60.0
_MINUTES_PER_YEAR = 525_600.0  # 365 days × 24 hours × 60 minutes


def minutes_to_settlement(
    as_of: datetime,
    settlement_event: datetime,
) -> float:
    """Year-fraction time to settlement; Black-76 `T` argument.

    Raises
    ------
    TimezoneError
        If either timestamp is naive.
    ExpiryError
        If `as_of` is at or after `settlement_event`.
    """
    if as_of.tzinfo is None or settlement_event.tzinfo is None:
        raise TimezoneError(
            f"Both timestamps must be tz-aware exchange-time; got "
            f"as_of.tzinfo={as_of.tzinfo!r}, "
            f"settlement_event.tzinfo={settlement_event.tzinfo!r}."
        )
    if as_of >= settlement_event:
        raise ExpiryError(
            f"Cannot price an option at or after its settlement event; "
            f"as_of={as_of} >= settlement_event={settlement_event}."
        )
    seconds = (settlement_event - as_of).total_seconds()
    return seconds / _SECONDS_PER_MINUTE / _MINUTES_PER_YEAR
