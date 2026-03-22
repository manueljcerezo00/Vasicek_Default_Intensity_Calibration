from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VasicekFactor:
    """
    Vasicek / Ornstein-Uhlenbeck factor x_t with dynamics:

        dx_t = kappa (theta - x_t) dt + sigma dW_t

    The discount factor for the integral of x_t over [t, T] is affine:

        P_x(t, T) = E_t[exp(-∫_t^T x_s ds)]
                 = A(Δ) * exp(-B(Δ) x_t),  Δ = T - t

    with

        B(Δ) = (1 - exp(-kappa Δ)) / kappa

        log A(Δ) = (theta - sigma^2 / (2 kappa^2)) * (B(Δ) - Δ)
                   - (sigma^2 * B(Δ)^2) / (4 kappa)
    """

    kappa: float
    theta: float
    sigma: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if self.sigma < 0.0:
            raise ValueError("sigma must be nonnegative.")

    @staticmethod
    def _validate_maturity(maturity: float) -> None:
        if maturity < 0.0:
            raise ValueError("maturity must be nonnegative.")

    def B(self, maturity: float) -> float:
        """
        B(Δ) = (1 - exp(-kappa Δ)) / kappa
        """
        self._validate_maturity(maturity)
        if maturity == 0.0:
            return 0.0
        return (1.0 - math.exp(-self.kappa * maturity)) / self.kappa

    def log_A(self, maturity: float) -> float:
        """
        log A(Δ) for E_t[exp(-∫ x_s ds)] under Vasicek OU dynamics.
        """
        self._validate_maturity(maturity)
        if maturity == 0.0:
            return 0.0
        b = self.B(maturity)
        k = self.kappa
        sigma2 = self.sigma * self.sigma
        term1 = (self.theta - sigma2 / (2.0 * k * k)) * (b - maturity)
        term2 = (sigma2 * b * b) / (4.0 * k)
        return term1 - term2

    def A(self, maturity: float) -> float:
        """
        A(Δ) = exp(log A(Δ)).
        """
        return math.exp(self.log_A(maturity))

    def discount_factor(self, state: float, maturity: float) -> float:
        """
        P_x(t, T) = A(Δ) * exp(-B(Δ) * x_t).
        Returns 1.0 when maturity == 0.
        """
        self._validate_maturity(maturity)
        if maturity == 0.0:
            return 1.0
        return self.A(maturity) * math.exp(-self.B(maturity) * state)

    def yield_to_maturity(self, state: float, maturity: float) -> float:
        """
        y(Δ) = -(1/Δ) * log P_x(t, T).
        """
        self._validate_maturity(maturity)
        if maturity == 0.0:
            raise ValueError("yield is undefined at maturity 0.")
        price = self.discount_factor(state, maturity)
        return -math.log(price) / maturity
