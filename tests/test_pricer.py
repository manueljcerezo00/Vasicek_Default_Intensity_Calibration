import numpy as np

from vasicek_poisson.models.intensity import IntensityModel
from vasicek_poisson.models.vasicek import VasicekFactor
from vasicek_poisson.pricing.bond_pricer import (
    credit_spread,
    price_zc_defaultable,
    price_zc_riskfree,
    price_zc_spread,
    yield_defaultable,
    yield_riskfree,
)


def make_rf_model() -> VasicekFactor:
    return VasicekFactor(kappa=0.6, theta=0.02, sigma=0.01)


def make_intensity_model() -> IntensityModel:
    return IntensityModel.from_params(kappa=0.9, theta=0.03, sigma=0.015)


def test_defaultable_price_factorization():
    rf_model = make_rf_model()
    int_model = make_intensity_model()

    r_t = 0.02
    lambda_t = 0.03
    maturity = 4.0

    p_rf = price_zc_riskfree(r_t, maturity, rf_model)
    p_spread = price_zc_spread(lambda_t, maturity, int_model)
    p_def = price_zc_defaultable(r_t, lambda_t, maturity, rf_model, int_model)

    assert np.isclose(p_def, p_rf * p_spread)


def test_defaultable_price_below_riskfree():
    rf_model = make_rf_model()
    int_model = make_intensity_model()

    r_t = 0.02
    lambda_t = 0.05
    maturity = 3.0

    p_rf = price_zc_riskfree(r_t, maturity, rf_model)
    p_def = price_zc_defaultable(r_t, lambda_t, maturity, rf_model, int_model)

    assert p_def <= p_rf


def test_credit_spread_definition():
    rf_model = make_rf_model()
    int_model = make_intensity_model()

    r_t = 0.015
    lambda_t = 0.02
    maturity = 2.0

    y_rf = yield_riskfree(r_t, maturity, rf_model)
    y_def = yield_defaultable(r_t, lambda_t, maturity, rf_model, int_model)
    s = credit_spread(r_t, lambda_t, maturity, rf_model, int_model)

    assert np.isclose(s, y_def - y_rf)


def test_spread_factor_at_zero_maturity():
    int_model = make_intensity_model()
    p_spread = price_zc_spread(lambda_t=0.04, maturity=0.0, model=int_model)
    assert p_spread == 1.0


def test_higher_lambda_lowers_spread_price():
    int_model = make_intensity_model()
    maturity = 5.0
    p_low = price_zc_spread(lambda_t=0.01, maturity=maturity, model=int_model)
    p_high = price_zc_spread(lambda_t=0.05, maturity=maturity, model=int_model)
    assert p_high < p_low
