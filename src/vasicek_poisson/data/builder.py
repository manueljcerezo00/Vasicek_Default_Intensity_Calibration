from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vasicek_poisson.data.cleaner import ECBDataCleaner, CleaningReport
from vasicek_poisson.data.parser import ECBDataParser


@dataclass
class BuildResult:
    spot: pd.DataFrame
    cleaned: pd.DataFrame
    report: CleaningReport


class ECBDataBuilder:
    def __init__(
        self,
        parser: ECBDataParser | None = None,
        cleaner: ECBDataCleaner | None = None,
    ) -> None:
        self.parser = parser or ECBDataParser()
        self.cleaner = cleaner or ECBDataCleaner()

    def build_spot_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.parser.keep_core_columns(self.parser.filter_spot_rates(df))
        out = self.parser.add_maturity_years(out)
        return out

    def build_cleaned_spot_panel(self, df: pd.DataFrame) -> BuildResult:
        spot = self.build_spot_panel(df)
        cleaned, report = self.cleaner.clean(spot)
        return BuildResult(spot=spot, cleaned=cleaned, report=report)
