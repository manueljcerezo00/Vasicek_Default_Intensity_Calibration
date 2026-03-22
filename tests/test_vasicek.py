import math

import numpy as np

from vasicek_poisson.models.vasicek import VasicekFactor


def make_model() -> VasicekFactor:
    return VasicekFactor(kappa=0.7, theta=0.02, sigma=0.01)


def test_b_zero():
    model = make_model()
    assert model.B(0.0) == 0.0


def test_a_zero():
    model = make_model()
    assert model.A(0.0) == 1.0


def test_discount_factor_zero_maturity():
    model = make_model()
    assert model.discount_factor(state=0.03, maturity=0.0) == 1.0


def test_discount_factor_positive():
    model = make_model()
    price = model.discount_factor(state=0.03, maturity=2.0)
    assert price > 0.0


def test_discount_factor_monotone_in_maturity():
    model = make_model()
    state = 0.03
    p_short = model.discount_factor(state=state, maturity=0.5)
    p_mid = model.discount_factor(state=state, maturity=1.0)
    p_long = model.discount_factor(state=state, maturity=3.0)
    assert p_short >= p_mid >= p_long


def test_discount_factor_decreases_with_state():
    model = make_model()
    maturity = 2.0
    p_low = model.discount_factor(state=0.01, maturity=maturity)
    p_high = model.discount_factor(state=0.05, maturity=maturity)
    assert p_high < p_low


def test_yield_consistency():
    model = make_model()
    maturity = 5.0
    state = 0.025
    price = model.discount_factor(state=state, maturity=maturity)
    y = model.yield_to_maturity(state=state, maturity=maturity)
    assert np.isclose(y, -math.log(price) / maturity)


def test_small_maturity_is_well_behaved():
    model = make_model()
    maturity = 1.0e-6
    price = model.discount_factor(state=0.03, maturity=maturity)
    assert math.isfinite(price)
    assert price > 0.0
