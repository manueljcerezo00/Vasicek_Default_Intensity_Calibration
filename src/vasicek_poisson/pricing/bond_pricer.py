from __future__ import annotations

import math

from vasicek_poisson.models.intensity import IntensityModel
from vasicek_poisson.models.vasicek import VasicekFactor


def price_zc_riskfree(r_t: float, maturity: float, model: VasicekFactor) -> float:
    """
    P_rf(t, T) = E_t[exp(-∫ r_s ds)].
    """
    return model.discount_factor(r_t, maturity)


def price_zc_spread(
    lambda_t: float, maturity: float, model: VasicekFactor | IntensityModel
) -> float:
    """
    P_spread(t, T) = E_t[exp(-∫ λ_s ds)].
    """
    if isinstance(model, IntensityModel):
        return model.survival_probability(lambda_t, maturity)
    return model.discount_factor(lambda_t, maturity)


def price_zc_defaultable(
    r_t: float,
    lambda_t: float,
    maturity: float,
    rf_model: VasicekFactor,
    intensity_model: VasicekFactor | IntensityModel,
) -> float:
    """
    P_def(t, T) = P_rf(t, T) * P_spread(t, T).
    """
    return price_zc_riskfree(r_t, maturity, rf_model) * price_zc_spread(
        lambda_t, maturity, intensity_model
    )


def yield_riskfree(r_t: float, maturity: float, model: VasicekFactor) -> float:
    price = price_zc_riskfree(r_t, maturity, model)
    if maturity == 0.0:
        raise ValueError("yield is undefined at maturity 0.")
    return -math.log(price) / maturity


def yield_defaultable(
    r_t: float,
    lambda_t: float,
    maturity: float,
    rf_model: VasicekFactor,
    intensity_model: VasicekFactor | IntensityModel,
) -> float:
    price = price_zc_defaultable(r_t, lambda_t, maturity, rf_model, intensity_model)
    if maturity == 0.0:
        raise ValueError("yield is undefined at maturity 0.")
    return -math.log(price) / maturity


def credit_spread(
    r_t: float,
    lambda_t: float,
    maturity: float,
    rf_model: VasicekFactor,
    intensity_model: VasicekFactor | IntensityModel,
) -> float:
    return yield_defaultable(r_t, lambda_t, maturity, rf_model, intensity_model) - yield_riskfree(
        r_t, maturity, rf_model
    )
