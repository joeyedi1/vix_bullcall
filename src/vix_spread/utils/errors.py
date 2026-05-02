class ForwardSelectionError(Exception):
    """Raised when ForwardSelector is asked to use an invalid source for the
    Black-76 forward input. Spot VIX is permanently forbidden by the
    validation memo. Other sources (e.g., interpolated fallback) may raise
    while a phase has not yet wired them in."""


class LookaheadError(Exception):
    """Raised when a strictly-causal component (regime classifier, fitter,
    fill engine) is fed data with timestamp > as_of. Validation-memo
    constraint: production signals must be functions only of P(S_t | y_{1:t}),
    never P(S_t | y_{1:T}); fills must use only quotes available at
    decision time."""


class TimezoneError(Exception):
    """Raised when a timestamp lacks tz info where exchange-time is required.
    All timing math (pricing, fills, broadcasts) operates on tz-aware
    timestamps; naive timestamps cause silent timezone drift and are
    refused at the boundary."""


class ExpiryError(Exception):
    """Raised when an option pricing call is made at or after the option's
    settlement event. Black-76 with T<=0 is undefined; this is a loud
    refusal rather than NaN propagation through the Greeks."""
