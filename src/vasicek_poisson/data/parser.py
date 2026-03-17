import re
import pandas as pd


class ECBDataParser:
    @staticmethod
    def filter_spot_rates(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = out[out["DATA_TYPE_FM"].astype(str).str.startswith("SR_")]
        return out

    @staticmethod
    def keep_core_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["TIME_PERIOD", "DATA_TYPE_FM", "OBS_VALUE", "TITLE"]
        existing = [c for c in cols if c in df.columns]
        return df[existing].copy()

    @staticmethod
    def maturity_code_to_years(code: str) -> float:
        label = code.replace("SR_", "")

        match = re.fullmatch(r"(?:(\d+)Y)?(?:(\d+)M)?", label)
        if match is None or (match.group(1) is None and match.group(2) is None):
            raise ValueError(f"Unsupported maturity code: {code}")

        years = int(match.group(1)) if match.group(1) is not None else 0
        months = int(match.group(2)) if match.group(2) is not None else 0

        return years + months / 12.0

    @classmethod
    def add_maturity_years(cls, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["maturity_years"] = out["DATA_TYPE_FM"].apply(cls.maturity_code_to_years)
        return out