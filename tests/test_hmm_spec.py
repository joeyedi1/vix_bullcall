"""Tests for HMMSpec.

Three contracts:
  1. A row-stochastic transition matrix validates clean.
  2. A non-row-stochastic matrix raises HMMSpecificationError.
  3. Declaring transition_matrix and a stationary_distribution that are
     mutually inconsistent raises (the validation-memo defect).
  4. implied_stationary recovers the known fixed point of a 2-state chain.
"""
import numpy as np
import pytest

from vix_spread.regime.hmm_spec import HMMSpec, HMMSpecificationError


def test_valid_transition_matrix_validates():
    spec = HMMSpec(
        n_states=2,
        transition_matrix=np.array([[0.96, 0.04], [0.117, 0.883]]),
    )
    spec.validate()  # no raise


def test_non_row_stochastic_raises():
    spec = HMMSpec(
        n_states=2,
        transition_matrix=np.array([[0.5, 0.4], [0.117, 0.883]]),
    )
    with pytest.raises(HMMSpecificationError):
        spec.validate()


def test_inconsistent_stationary_raises():
    """ARCH §3.2 / validation-memo defect: declaring [0.5, 0.5] alongside
    a transition matrix whose true stationary is ~[0.745, 0.255] must
    raise — silently accepting both is the failure mode."""
    spec = HMMSpec(
        n_states=2,
        transition_matrix=np.array([[0.96, 0.04], [0.117, 0.883]]),
        stationary_distribution=np.array([0.5, 0.5]),
    )
    with pytest.raises(HMMSpecificationError):
        spec.validate()


def test_consistent_stationary_validates():
    T = np.array([[0.96, 0.04], [0.117, 0.883]])
    spec_only_T = HMMSpec(n_states=2, transition_matrix=T)
    implied = spec_only_T.implied_stationary()
    spec = HMMSpec(
        n_states=2,
        transition_matrix=T,
        stationary_distribution=implied,
    )
    spec.validate()  # no raise


def test_implied_stationary_known_chain():
    """For T = [[0.96, 0.04], [0.117, 0.883]], the stationary is
    π = (a/(a+b), b/(a+b)) with a = T[1,0] = 0.117, b = T[0,1] = 0.04
    → π_0 ≈ 0.7452, π_1 ≈ 0.2548. Validation memo's expected values."""
    spec = HMMSpec(
        n_states=2,
        transition_matrix=np.array([[0.96, 0.04], [0.117, 0.883]]),
    )
    pi = spec.implied_stationary()
    assert pi.shape == (2,)
    assert pi.sum() == pytest.approx(1.0)
    assert pi[0] == pytest.approx(0.117 / (0.117 + 0.04), abs=1e-4)
    assert pi[1] == pytest.approx(0.04 / (0.117 + 0.04), abs=1e-4)
