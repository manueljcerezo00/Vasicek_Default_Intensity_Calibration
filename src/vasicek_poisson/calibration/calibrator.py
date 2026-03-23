from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vasicek_poisson.calibration.objectives import fit_state, yields_to_prices
from vasicek_poisson.calibration.results import (
    GlobalParamResult,
    IntensityFitResult,
    RiskFreeFitResult,
)
from vasicek_poisson.models.intensity import IntensityModel
from vasicek_poisson.models.vasicek import VasicekFactor
from vasicek_poisson.pricing.bond_pricer import price_zc_riskfree


@dataclass(frozen=True)
class PanelAlignment:
    date: pd.Timestamp
    maturities: np.ndarray
    aaa_yields: np.ndarray
    all_yields: np.ndarray


def _coerce_panel(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["TIME_PERIOD"] = pd.to_datetime(frame["TIME_PERIOD"], errors="coerce")
    frame["maturity_years"] = pd.to_numeric(frame["maturity_years"], errors="coerce")
    frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    frame = frame.dropna(subset=["TIME_PERIOD", "maturity_years", "OBS_VALUE"])
    return frame


def align_panels(
    aaa_df: pd.DataFrame,
    all_df: pd.DataFrame,
    date: pd.Timestamp,
) -> PanelAlignment:
    """
    Align AAA and ALL panels on a given date and common maturities.
    """
    aaa = _coerce_panel(aaa_df)
    allb = _coerce_panel(all_df)

    aaa = aaa[aaa["TIME_PERIOD"] == pd.Timestamp(date)]
    allb = allb[allb["TIME_PERIOD"] == pd.Timestamp(date)]

    merged = aaa.merge(
        allb,
        on=["TIME_PERIOD", "maturity_years"],
        suffixes=("_aaa", "_all"),
        how="inner",
    )
    merged = merged.sort_values("maturity_years")

    return PanelAlignment(
        date=pd.Timestamp(date),
        maturities=merged["maturity_years"].to_numpy(dtype=float),
        aaa_yields=merged["OBS_VALUE_aaa"].to_numpy(dtype=float),
        all_yields=merged["OBS_VALUE_all"].to_numpy(dtype=float),
    )


def fit_riskfree_for_date(
    aaa_df: pd.DataFrame,
    date: pd.Timestamp,
    model: VasicekFactor,
    state_guess: float,
    state_bounds: tuple[float, float] | None = None,
    weight_scheme: str | None = "inverse_maturity",
    loss: str = "price",
    inner_maxiter: int | None = None,
) -> RiskFreeFitResult:
    """
    Fit r_t to AAA yields for a given date.
    """
    frame = _coerce_panel(aaa_df)
    panel = frame[frame["TIME_PERIOD"] == pd.Timestamp(date)]
    panel = panel.sort_values("maturity_years")

    maturities = panel["maturity_years"].to_numpy(dtype=float)
    yields_pct = panel["OBS_VALUE"].to_numpy(dtype=float)
    if loss == "price":
        obs = yields_to_prices(yields_pct, maturities)
    elif loss == "yield":
        obs = yields_pct / 100.0
    else:
        raise ValueError(f"Unknown loss: {loss}")

    fit = fit_state(
        maturities=maturities,
        obs=obs,
        model=model,
        state_guess=state_guess,
        bounds=state_bounds,
        weight_scheme=weight_scheme,
        loss=loss,
        maxiter=inner_maxiter,
    )

    return RiskFreeFitResult(
        date=pd.Timestamp(date),
        r_t=fit.state,
        objective=fit.objective,
        success=fit.success,
        n_obs=fit.n_obs,
    )


def _iter_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    frame = _coerce_panel(df)
    dates = sorted(frame["TIME_PERIOD"].dropna().unique())
    return [pd.Timestamp(d) for d in dates]


def fit_riskfree_global_params(
    aaa_df: pd.DataFrame,
    kappa_guess: float,
    theta_guess: float,
    sigma_guess: float,
    state_guess: float,
    param_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    state_bounds: tuple[float, float] | None = None,
    weight_scheme: str | None = "inverse_maturity",
    loss: str = "price",
    inner_maxiter: int | None = None,
    outer_maxiter: int | None = None,
) -> tuple[GlobalParamResult, pd.DataFrame]:
    """
    Jointly fit global Vasicek parameters by profiling out r_t per date.
    Returns the fitted parameters and a DataFrame with per-date r_t.
    """
    from scipy.optimize import minimize

    frame = _coerce_panel(aaa_df)
    dates = _iter_dates(frame)
    panels: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []
    for d in dates:
        panel = frame[frame["TIME_PERIOD"] == d].sort_values("maturity_years")
        maturities = panel["maturity_years"].to_numpy(dtype=float)
        yields_pct = panel["OBS_VALUE"].to_numpy(dtype=float)
        if loss == "price":
            obs = yields_to_prices(yields_pct, maturities)
        elif loss == "yield":
            obs = yields_pct / 100.0
        else:
            raise ValueError(f"Unknown loss: {loss}")
        panels.append((d, maturities, obs))

    def _objective(params: np.ndarray) -> float:
        kappa, theta, sigma = params.tolist()
        model = VasicekFactor(kappa=kappa, theta=theta, sigma=sigma)
        total = 0.0
        for d, maturities, obs in panels:
            fit = fit_state(
                maturities=maturities,
                obs=obs,
                model=model,
                state_guess=state_guess,
                bounds=state_bounds,
                weight_scheme=weight_scheme,
                loss=loss,
                maxiter=inner_maxiter,
            )
            total += fit.objective
        return float(total)

    x0 = np.array([kappa_guess, theta_guess, sigma_guess], dtype=float)
    bounds = list(param_bounds) if param_bounds is not None else None

    res = minimize(
        _objective,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options=None if outer_maxiter is None else {"maxiter": outer_maxiter},
    )

    fitted_model = VasicekFactor(
        kappa=float(res.x[0]), theta=float(res.x[1]), sigma=float(res.x[2])
    )

    rows = []
    for d, maturities, obs in panels:
        fit = fit_state(
            maturities=maturities,
            obs=obs,
            model=fitted_model,
            state_guess=state_guess,
            bounds=state_bounds,
            weight_scheme=weight_scheme,
            loss=loss,
            maxiter=inner_maxiter,
        )
        rows.append(
            {
                "TIME_PERIOD": d,
                "r_t": fit.state,
                "objective": fit.objective,
                "success": fit.success,
                "n_obs": int(len(maturities)),
            }
        )

    states_df = pd.DataFrame(rows).sort_values("TIME_PERIOD")
    return (
        GlobalParamResult(
            kappa=fitted_model.kappa,
            theta=fitted_model.theta,
            sigma=fitted_model.sigma,
            objective=float(res.fun),
            success=bool(res.success),
            n_dates=len(dates),
        ),
        states_df,
    )


def fit_intensity_for_date(
    aaa_df: pd.DataFrame,
    all_df: pd.DataFrame,
    date: pd.Timestamp,
    rf_model: VasicekFactor,
    intensity_model: VasicekFactor | IntensityModel,
    r_t: float,
    state_guess: float,
    state_bounds: tuple[float, float] | None = None,
    weight_scheme: str | None = "inverse_maturity",
    loss: str = "price",
    inner_maxiter: int | None = None,
) -> IntensityFitResult:
    """
    Fit lambda_t to the conditional spread panel:

        P_spread_obs = P_def_obs / P_rf_model
    """
    aligned = align_panels(aaa_df, all_df, date)
    maturities = aligned.maturities

    if isinstance(intensity_model, IntensityModel):
        model = intensity_model.factor
    else:
        model = intensity_model

    if loss == "price":
        p_def_obs = yields_to_prices(aligned.all_yields, maturities)
        p_rf_model = np.array(
            [price_zc_riskfree(r_t, m, rf_model) for m in maturities], dtype=float
        )
        obs = p_def_obs / p_rf_model
    elif loss == "yield":
        y_def_obs = aligned.all_yields / 100.0
        y_rf_model = np.array(
            [rf_model.yield_to_maturity(r_t, m) for m in maturities], dtype=float
        )
        obs = y_def_obs - y_rf_model
    else:
        raise ValueError(f"Unknown loss: {loss}")

    fit = fit_state(
        maturities=maturities,
        obs=obs,
        model=model,
        state_guess=state_guess,
        bounds=state_bounds,
        weight_scheme=weight_scheme,
        loss=loss,
        maxiter=inner_maxiter,
    )

    return IntensityFitResult(
        date=aligned.date,
        lambda_t=fit.state,
        objective=fit.objective,
        success=fit.success,
        n_obs=fit.n_obs,
    )


def fit_intensity_global_params(
    aaa_df: pd.DataFrame,
    all_df: pd.DataFrame,
    rf_model: VasicekFactor,
    rf_states: pd.DataFrame,
    kappa_guess: float,
    theta_guess: float,
    sigma_guess: float,
    state_guess: float,
    param_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
    state_bounds: tuple[float, float] | None = None,
    weight_scheme: str | None = "inverse_maturity",
    loss: str = "price",
    inner_maxiter: int | None = None,
    outer_maxiter: int | None = None,
) -> tuple[GlobalParamResult, pd.DataFrame]:
    """
    Jointly fit global intensity parameters by profiling out lambda_t per date.
    Requires risk-free states per date.
    """
    from scipy.optimize import minimize

    rf_states = rf_states.copy()
    rf_states["TIME_PERIOD"] = pd.to_datetime(rf_states["TIME_PERIOD"], errors="coerce")
    rf_states = rf_states.dropna(subset=["TIME_PERIOD", "r_t"])
    rf_map = {pd.Timestamp(r.TIME_PERIOD): float(r.r_t) for r in rf_states.itertuples()}

    dates = [d for d in _iter_dates(aaa_df) if d in rf_map]

    aaa_frame = _coerce_panel(aaa_df)
    all_frame = _coerce_panel(all_df)
    panels: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = []
    for d in dates:
        aligned = align_panels(aaa_frame, all_frame, d)
        maturities = aligned.maturities
        if loss == "price":
            p_def_obs = yields_to_prices(aligned.all_yields, maturities)
            p_rf_model = np.array(
                [price_zc_riskfree(rf_map[d], m, rf_model) for m in maturities], dtype=float
            )
            obs = p_def_obs / p_rf_model
        elif loss == "yield":
            y_def_obs = aligned.all_yields / 100.0
            y_rf_model = np.array(
                [rf_model.yield_to_maturity(rf_map[d], m) for m in maturities], dtype=float
            )
            obs = y_def_obs - y_rf_model
        else:
            raise ValueError(f"Unknown loss: {loss}")
        panels.append((d, maturities, obs))

    def _objective(params: np.ndarray) -> float:
        kappa, theta, sigma = params.tolist()
        model = VasicekFactor(kappa=kappa, theta=theta, sigma=sigma)
        total = 0.0
        for d, maturities, obs in panels:
            fit = fit_state(
                maturities=maturities,
                obs=obs,
                model=model,
                state_guess=state_guess,
                bounds=state_bounds,
                weight_scheme=weight_scheme,
                loss=loss,
                maxiter=inner_maxiter,
            )
            total += fit.objective
        return float(total)

    x0 = np.array([kappa_guess, theta_guess, sigma_guess], dtype=float)
    bounds = list(param_bounds) if param_bounds is not None else None

    res = minimize(
        _objective,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options=None if outer_maxiter is None else {"maxiter": outer_maxiter},
    )

    fitted_model = VasicekFactor(
        kappa=float(res.x[0]), theta=float(res.x[1]), sigma=float(res.x[2])
    )

    rows = []
    for d, maturities, obs in panels:
        fit = fit_state(
            maturities=maturities,
            obs=obs,
            model=fitted_model,
            state_guess=state_guess,
            bounds=state_bounds,
            weight_scheme=weight_scheme,
            loss=loss,
            maxiter=inner_maxiter,
        )
        rows.append(
            {
                "TIME_PERIOD": d,
                "lambda_t": fit.state,
                "objective": fit.objective,
                "success": fit.success,
                "n_obs": int(len(maturities)),
            }
        )

    states_df = pd.DataFrame(rows).sort_values("TIME_PERIOD")
    return (
        GlobalParamResult(
            kappa=fitted_model.kappa,
            theta=fitted_model.theta,
            sigma=fitted_model.sigma,
            objective=float(res.fun),
            success=bool(res.success),
            n_dates=len(dates),
        ),
        states_df,
    )
