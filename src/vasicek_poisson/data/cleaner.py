
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    rows_in: int
    rows_out: int
    dropped_missing_critical: int
    dropped_out_of_range: int
    dropped_duplicates: int
    dropped_insufficient_maturities: int    
    
class ECBDataCleaner:
    """
    Clean and  validate parsed ECB data.

    Expected Critical Input:
    -----------------
    TIME_PERIOD: str or datetime-like
    OBS_VALUE: numeric
    maturity_years: numeric (non-negative)

    Option Inputs:
    -----------------
    DATA_TYPE_FM: str (must start with "SR_")
    TITLE: str
    """

    def __init__(
            self,
            min_maturity: float = 0.250, # 3 months
            max_maturity: float = 30.0, # 30 years
            min_maturities: int | None = None, # minimum number of maturities per date (None = no filter)
            duplicate_subset: list[str] | None = None, # columns to check for duplicates (None = all)
    ) -> None:
        self.min_maturity = min_maturity
        self.max_maturity = max_maturity
        self.min_maturities = min_maturities
        self.duplicate_subset = duplicate_subset or ["TIME_PERIOD", "maturity_years"]

    def standardize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of the DataFrame with standardized types.

        TASKS:
        - Convert TIME_PERIOD to datetime
        - Convert OBS_VALUE to numeric (float)
        - Convert maturity_years to numeric (float)
        """
        df = df.copy()
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
        df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")

        if "DATA_TYPE_FM" in df.columns:
            df["DATA_TYPE_FM"] = df["DATA_TYPE_FM"].astype(str)
        if "TITLE" in df.columns:
            df["TITLE"] = df["TITLE"].astype(str)

        return df
    
    def drop_missing_critical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing critical values (TIME_PERIOD, OBS_VALUE, maturity_years).
        """
        df = df.copy()
        critical_cols = ["TIME_PERIOD", "OBS_VALUE", "maturity_years"]
        df = df.dropna(subset=critical_cols)
        return df
    
    def filter_maturity_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with maturity_years outside the specified range.
        """
        df = df.copy()
        df = df[(df["maturity_years"] >= self.min_maturity) & (df["maturity_years"] <= self.max_maturity)]
        return df
    
    def drop_duplicates(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicate rows based on a simple deterministic rule: first occurrence is kept, subsequent duplicates are dropped.
        """
        df = df.copy()
        df = df.drop_duplicates(subset=self.duplicate_subset, keep="first")
    
        return df
    
    def sort_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort the DataFrame by TIME_PERIOD and maturity_years.
        """
        df = df.copy()
        df = df.sort_values(by=["TIME_PERIOD", "maturity_years"], ignore_index=True)
        return df

    def compute_discount_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute zero-coupon prices from spot rates and add as a new column "ZC_price".
        ----------
        It will be later used to check for monotonicity and absence of arbitrage opportunities in the curves.
        """
        df = df.copy()
        df["ZC_price"] = np.exp(-df["OBS_VALUE"] * df["maturity_years"] / 100.0)
        return df
    
    def count_maturities_per_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
         Return a summary DataFrame with one row per date and at least:

        - TIME_PERIOD
        - n_maturities
        """
        df = df.copy()
        summary = df.groupby("TIME_PERIOD").agg(n_maturities=("maturity_years", "nunique")).reset_index()
        return summary
    
    def filter_min_maturities(self, df: pd.DataFrame, min_maturities: int = 30) -> pd.DataFrame:
        """
        Drop dates with fewer than min_maturities available maturities.
        """
        df = df.copy()
        summary = self.count_maturities_per_date(df)
        valid_dates = summary[summary["n_maturities"] >= min_maturities]["TIME_PERIOD"]    
        df = df[df["TIME_PERIOD"].isin(valid_dates)]
    
        return df
    
    def check_monotonicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check that zero-coupon prices are non-increasing with maturity for each date.
        Return a DataFrame with any violations of this condition.
        """
        df = df.copy()
        df = self.compute_discount_factors(df)
        df = df.sort_values(by=["TIME_PERIOD", "maturity_years"], ignore_index=True)
        df["ZC_price_next"] = df.groupby("TIME_PERIOD")["ZC_price"].shift(-1)
        violations = df[df["ZC_price_next"] > df["ZC_price"]]
        return violations[["TIME_PERIOD", "maturity_years", "ZC_price", "ZC_price_next"]]
    
    def summarize_panel(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary DataFrame with one row per date and at least:

        - TIME_PERIOD
        - n_maturities
        - min_maturity
        - max_maturity
        """
        df = df.copy()
        summary = df.groupby("TIME_PERIOD").agg(
            n_maturities=("maturity_years", "count"),
            min_maturity=("maturity_years", "min"),
            max_maturity=("maturity_years", "max"),
            mean_spot=("OBS_VALUE", "mean"),
            median_spot=("OBS_VALUE", "median")
        ).reset_index()
        return summary
    
    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """
        Perform the full cleaning pipeline and return the cleaned DataFrame along with a report of the cleaning steps.

        Order:
        1. Standardize types
        2. Drop rows with missing critical values
        3. Filter rows with maturity_years outside the specified range
        4. Drop duplicates
        5. Sort panel
        6. Filter dates with insufficient maturities


        Returns:
        - cleaned DataFrame
        - CleaningReport with counts of rows in, rows out, and reasons for dropping
        """
        df = df.copy()
        report = CleaningReport(
            rows_in=len(df),
            rows_out=0,
            dropped_missing_critical=0,
            dropped_out_of_range=0,
            dropped_duplicates=0,
            dropped_insufficient_maturities=0
        )
        n_0 = len(df)
        df = self.standardize_types(df)
        n_1 = len(df)
        report.dropped_missing_critical = n_0 - n_1
        df = self.drop_missing_critical(df)
        n_2 = len(df)
        report.dropped_missing_critical += n_1 - n_2
        df = self.filter_maturity_range(df)
        n_3 = len(df)
        report.dropped_out_of_range = n_2 - n_3
        df = self.drop_duplicates(df)
        n_4 = len(df)
        report.dropped_duplicates = n_3 - n_4
        df = self.sort_panel(df)
        if self.min_maturities is not None:
            df = self.filter_min_maturities(df, min_maturities=self.min_maturities)
            n_5 = len(df)
            report.dropped_insufficient_maturities = n_4 - n_5
        else:
            n_5 = n_4
        report.rows_out = n_5
        return df, report 
    
    
