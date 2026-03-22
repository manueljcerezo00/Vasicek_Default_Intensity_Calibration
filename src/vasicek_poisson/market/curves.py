from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class YieldCurve:
    date: pd.Timestamp
    maturities: np.ndarray
    rates: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TIME_PERIOD": self.date,
                "maturity_years": self.maturities,
                "OBS_VALUE": self.rates,
            }
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame, date: pd.Timestamp) -> "YieldCurve":
        curve = df[df["TIME_PERIOD"] == date].copy()
        curve = curve.sort_values("maturity_years")
        return cls(
            date=pd.Timestamp(date),
            maturities=curve["maturity_years"].to_numpy(dtype=float),
            rates=curve["OBS_VALUE"].to_numpy(dtype=float),
        )

    @classmethod
    def from_arrays(
        cls, date: pd.Timestamp, maturities: Iterable[float], rates: Iterable[float]
    ) -> "YieldCurve":
        return cls(
            date=pd.Timestamp(date),
            maturities=np.asarray(list(maturities), dtype=float),
            rates=np.asarray(list(rates), dtype=float),
        )
