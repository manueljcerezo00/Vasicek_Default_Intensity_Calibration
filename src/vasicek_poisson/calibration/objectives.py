from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from vasicek_poisson.models.vasicek import VasicekFactor


LossType = Literal["price", "yield"]


@dataclass(frozen=True)
class FitStateResult:
    state: float
    objective: float
    success: bool
    n_obs: int


def yields_to_prices(
    yields_pct: Iterable[float], maturities: Iterable[float]
) -> np.ndarray:
    """
    Convert continuously-compounded yields (in percent) to prices:

        P = exp(-y * T / 100)
    """
    y = np.asarray(list(yields_pct), dtype=float)
    t = np.asarray(list(maturities), dtype=float)
    return np.exp(-y * t / 100.0)


def prices_to_yields(prices: Iterable[float], maturities: Iterable[float]) -> np.ndarray:
    """
    Convert prices to continuously-compounded yields (decimal).

        y = -log(P) / T
    """
    p = np.asarray(list(prices), dtype=float)
    t = np.asarray(list(maturities), dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("All maturities must be positive to compute yields.")
    return -np.log(p) / t


def price_errors(
    state: float,
    maturities: Iterable[float],
    prices_obs: Iterable[float],
    model: VasicekFactor,
    weights: Iterable[float] | None = None,
) -> np.ndarray:
    """
    Price residuals: model - observed.
    """
    t = np.asarray(list(maturities), dtype=float)
    p_obs = np.asarray(list(prices_obs), dtype=float)
    p_model = np.array([model.discount_factor(state, m) for m in t], dtype=float)
    errors = p_model - p_obs
    if weights is None:
        return errors
    w = np.asarray(list(weights), dtype=float)
    return errors * w


def objective_sse(
    state: float,
    maturities: Iterable[float],
    prices_obs: Iterable[float],
    model: VasicekFactor,
    weights: Iterable[float] | None = None,
) -> float:
    """
    Sum of squared errors in price space.
    """
    errs = price_errors(state, maturities, prices_obs, model, weights=weights)
    return float(np.sum(errs * errs))


def yield_errors(
    state: float,
    maturities: Iterable[float],
    yields_obs: Iterable[float],
    model: VasicekFactor,
    weights: Iterable[float] | None = None,
) -> np.ndarray:
    """
    Yield residuals: model - observed (yields in decimal units).
    """
    t = np.asarray(list(maturities), dtype=float)
    y_obs = np.asarray(list(yields_obs), dtype=float)
    if np.any(t <= 0.0):
        raise ValueError("All maturities must be positive for yield-space loss.")
    y_model = np.array([model.yield_to_maturity(state, m) for m in t], dtype=float)
    errors = y_model - y_obs
    if weights is None:
        return errors
    w = np.asarray(list(weights), dtype=float)
    return errors * w


def objective_sse_yield(
    state: float,
    maturities: Iterable[float],
    yields_obs: Iterable[float],
    model: VasicekFactor,
    weights: Iterable[float] | None = None,
) -> float:
    """
    Sum of squared errors in yield space.
    """
    errs = yield_errors(state, maturities, yields_obs, model, weights=weights)
    return float(np.sum(errs * errs))


def maturity_weights(
    maturities: Iterable[float],
    scheme: Literal["inverse_maturity"] = "inverse_maturity",
    eps: float = 1.0e-6,
) -> np.ndarray:
    """
    Default weighting for NSS-smoothed curves: downweight long maturities.

    scheme="inverse_maturity": w(T) = 1 / max(T, eps)
    """
    t = np.asarray(list(maturities), dtype=float)
    if scheme != "inverse_maturity":
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    return 1.0 / np.maximum(t, eps)


def fit_state(
    maturities: Iterable[float],
    obs: Iterable[float],
    model: VasicekFactor,
    state_guess: float,
    bounds: tuple[float, float] | None = None,
    weights: Iterable[float] | None = None,
    weight_scheme: Literal["inverse_maturity"] | None = "inverse_maturity",
    loss: LossType = "price",
    maxiter: int | None = None,
) -> FitStateResult:
    """
    Fit a single state x_t by minimizing SSE between model and observed prices or yields.
    """
    t = np.asarray(list(maturities), dtype=float)
    obs_arr = np.asarray(list(obs), dtype=float)
    if weights is None and weight_scheme is not None:
        weights = maturity_weights(t, scheme=weight_scheme)

    if loss == "price":
        def _objective_scalar(x: float) -> float:
            return objective_sse(x, t, obs_arr, model, weights=weights)
    elif loss == "yield":
        def _objective_scalar(x: float) -> float:
            return objective_sse_yield(x, t, obs_arr, model, weights=weights)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    if bounds is None:
        res = minimize_scalar(
            _objective_scalar,
            method="brent",
            options=None if maxiter is None else {"maxiter": maxiter},
        )
    else:
        res = minimize_scalar(
            _objective_scalar,
            bounds=bounds,
            method="bounded",
            options=None if maxiter is None else {"maxiter": maxiter},
        )

    return FitStateResult(
        state=float(res.x),
        objective=float(res.fun),
        success=bool(res.success),
        n_obs=int(len(t)),
    )
