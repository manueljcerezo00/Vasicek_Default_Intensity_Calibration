from __future__ import annotations

from dataclasses import dataclass

from vasicek_poisson.models.vasicek import VasicekFactor


@dataclass(frozen=True)
class IntensityModel:
    """
    Thin wrapper around VasicekFactor to represent default intensity.
    """

    factor: VasicekFactor

    @classmethod
    def from_params(cls, kappa: float, theta: float, sigma: float) -> "IntensityModel":
        return cls(factor=VasicekFactor(kappa=kappa, theta=theta, sigma=sigma))

    def survival_probability(self, lambda_t: float, maturity: float) -> float:
        """
        S(t, T) = E_t[exp(-∫_t^T λ_s ds)]
               = A(Δ) * exp(-B(Δ) * lambda_t)
        """
        return self.factor.discount_factor(lambda_t, maturity)
