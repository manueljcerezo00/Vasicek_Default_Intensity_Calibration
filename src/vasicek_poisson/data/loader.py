from pathlib import Path
import pandas as pd


class ECBDataLoader:
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)

    def load_csv(self, filename: str) -> pd.DataFrame:
        path = self.raw_data_dir / filename
        return pd.read_csv(path)

    def load_pair(self, aaa_filename: str, all_filename: str):
        df_aaa = self.load_csv(aaa_filename)
        df_all = self.load_csv(all_filename)
        return df_aaa, df_all