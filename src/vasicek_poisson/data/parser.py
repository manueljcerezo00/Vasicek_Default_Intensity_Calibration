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