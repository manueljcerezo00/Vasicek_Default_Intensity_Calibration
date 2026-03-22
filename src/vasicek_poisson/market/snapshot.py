from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vasicek_poisson.market.curves import YieldCurve


@dataclass(frozen=True)
class MarketSnapshot:
    date: pd.Timestamp
    risk_free_curve: YieldCurve

    @classmethod
    def from_cleaned_panel(cls, df: pd.DataFrame, date: pd.Timestamp) -> "MarketSnapshot":
        frame = df.copy()
        frame["TIME_PERIOD"] = pd.to_datetime(frame["TIME_PERIOD"], errors="coerce")
        frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
        frame["maturity_years"] = pd.to_numeric(frame["maturity_years"], errors="coerce")
        frame = frame.dropna(subset=["TIME_PERIOD", "OBS_VALUE", "maturity_years"])
        curve = YieldCurve.from_frame(frame, pd.Timestamp(date))
        return cls(date=pd.Timestamp(date), risk_free_curve=curve)
