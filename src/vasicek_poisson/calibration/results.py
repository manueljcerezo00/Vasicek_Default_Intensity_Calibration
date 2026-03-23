from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RiskFreeFitResult:
    date: pd.Timestamp
    r_t: float
    objective: float
    success: bool
    n_obs: int


@dataclass(frozen=True)
class IntensityFitResult:
    date: pd.Timestamp
    lambda_t: float
    objective: float
    success: bool
    n_obs: int


@dataclass(frozen=True)
class GlobalParamResult:
    kappa: float
    theta: float
    sigma: float
    objective: float
    success: bool
    n_dates: int
