from .quote import OptionQuote


class SyntheticSpreadQuote:
    """Constructs a debit-spread synthetic NBBO from two leg quotes.
    For a bull call spread (buy K1 @ ask, sell K2 @ bid; K1 < K2):
        synthetic_ask  (open debit)   = ask(K1) − bid(K2)
        synthetic_bid  (close credit) = bid(K1) − ask(K2)
    These are the conservative-base-case fill levels.
    Midpoint is NEVER used in the base case."""

    @staticmethod
    def open_debit_synthetic(long_q: OptionQuote, short_q: OptionQuote) -> float:
        return long_q.ask - short_q.bid

    @staticmethod
    def close_credit_synthetic(long_q: OptionQuote, short_q: OptionQuote) -> float:
        return long_q.bid - short_q.ask
