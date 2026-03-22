import argparse
import json

from vasicek_poisson.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    AAA_FILENAME,
    ALL_BONDS_FILENAME,
)
from vasicek_poisson.data.builder import ECBDataBuilder
from vasicek_poisson.data.cleaner import ECBDataCleaner
from vasicek_poisson.data.loader import ECBDataLoader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-maturity", type=float, default=0.25)
    parser.add_argument("--max-maturity", type=float, default=30.0)
    parser.add_argument("--min-maturities", type=int, default=None)
    args = parser.parse_args()

    loader = ECBDataLoader(RAW_DATA_DIR)
    cleaner = ECBDataCleaner(
        min_maturity=args.min_maturity,
        max_maturity=args.max_maturity,
        min_maturities=args.min_maturities,
    )
    builder = ECBDataBuilder(cleaner=cleaner)

    df_aaa, df_all = loader.load_pair(AAA_FILENAME, ALL_BONDS_FILENAME)

    aaa_result = builder.build_cleaned_spot_panel(df_aaa)
    all_result = builder.build_cleaned_spot_panel(df_all)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    aaa_result.cleaned.to_csv(PROCESSED_DATA_DIR / "aaa_spot_clean.csv", index=False)
    all_result.cleaned.to_csv(PROCESSED_DATA_DIR / "all_spot_clean.csv", index=False)

    report = {
        "aaa": aaa_result.report.__dict__,
        "all": all_result.report.__dict__,
        "min_maturity": args.min_maturity,
        "max_maturity": args.max_maturity,
        "min_maturities": args.min_maturities,
    }
    (PROCESSED_DATA_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("AAA cleaned sample:")
    print(aaa_result.cleaned.head())
    print()
    print("ALL cleaned sample:")
    print(all_result.cleaned.head())
    print()
    print("Cleaning report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
