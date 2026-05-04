"""Tests for VXFuture (ARCHITECTURE §4.1 / §4.5)."""
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from vix_spread.products.vx_future import VXFuture


def test_construct_with_required_fields():
    vx = VXFuture(contract_root="VX", settlement_date=date(2026, 6, 17))
    assert vx.contract_root == "VX"
    assert vx.settlement_date == date(2026, 6, 17)
    assert vx.multiplier == 1000.0


def test_multiplier_default_is_1000():
    """CFE spec: $1000 per VX point per contract. Hard-coded default."""
    vx = VXFuture(contract_root="VX", settlement_date=date(2026, 6, 17))
    assert vx.multiplier == 1000.0


def test_frozen_attributes_cannot_be_reassigned():
    """Identity is immutable post-construction (validation-memo
    'factor of 10' guard)."""
    vx = VXFuture(contract_root="VX", settlement_date=date(2026, 6, 17))
    with pytest.raises(FrozenInstanceError):
        vx.multiplier = 100.0  # type: ignore[misc]
