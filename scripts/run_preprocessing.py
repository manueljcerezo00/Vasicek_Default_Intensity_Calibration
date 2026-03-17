from vasicek_poisson.config import RAW_DATA_DIR, AAA_FILENAME, ALL_BONDS_FILENAME
from vasicek_poisson.data.loader import ECBDataLoader
from vasicek_poisson.data.parser import ECBDataParser


def main() -> None:
    loader = ECBDataLoader(RAW_DATA_DIR)
    parser = ECBDataParser()

    df_aaa, df_all = loader.load_pair(AAA_FILENAME, ALL_BONDS_FILENAME)

    aaa_spot = parser.keep_core_columns(parser.filter_spot_rates(df_aaa))
    all_spot = parser.keep_core_columns(parser.filter_spot_rates(df_all))

    print("AAA raw shape:", df_aaa.shape)
    print("ALL raw shape:", df_all.shape)
    print("AAA spot shape:", aaa_spot.shape)
    print("ALL spot shape:", all_spot.shape)
    print()
    print("AAA spot sample:")
    print(aaa_spot.head())
    print()
    print("ALL spot sample:")
    print(all_spot.head())


if __name__ == "__main__":
    main()