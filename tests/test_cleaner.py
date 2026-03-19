import numpy as np
import pandas as pd

from vasicek_poisson.data.cleaner import ECBDataCleaner


def make_cleaner():
    return ECBDataCleaner(min_maturity=0.25, max_maturity=30.0)


def test_standardize_types():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": ["2026-01-02", "2026-01-03"],
            "OBS_VALUE": ["2.5", "3.1"],
            "maturity_years": ["1.0", "2.0"],
        }
    )

    out = cleaner.standardize_types(df)

    assert pd.api.types.is_datetime64_any_dtype(out["TIME_PERIOD"])
    assert pd.api.types.is_numeric_dtype(out["OBS_VALUE"])
    assert pd.api.types.is_numeric_dtype(out["maturity_years"])


def test_drop_missing_critical():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": ["2026-01-02", None, "2026-01-04"],
            "OBS_VALUE": [2.0, 2.1, None],
            "maturity_years": [1.0, 2.0, 3.0],
        }
    )

    out = cleaner.drop_missing_critical(df)

    assert len(out) == 1
    assert out.iloc[0]["maturity_years"] == 1.0


def test_filter_maturity_range():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": pd.to_datetime(["2026-01-02"] * 4),
            "OBS_VALUE": [1.0, 2.0, 3.0, 4.0],
            "maturity_years": [0.1, 0.25, 10.0, 35.0],
        }
    )

    out = cleaner.filter_maturity_range(df)

    assert len(out) == 2
    assert set(out["maturity_years"]) == {0.25, 10.0}


def test_drop_duplicates():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"]),
            "OBS_VALUE": [2.0, 2.0, 2.5],
            "maturity_years": [1.0, 1.0, 1.0],
        }
    )

    out = cleaner.drop_duplicates(df)

    assert len(out) == 2


def test_sort_panel():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": pd.to_datetime(["2026-01-03", "2026-01-02", "2026-01-02"]),
            "OBS_VALUE": [2.5, 2.0, 1.8],
            "maturity_years": [2.0, 5.0, 1.0],
        }
    )

    out = cleaner.sort_panel(df)

    expected_dates = list(pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"]))
    expected_maturities = [1.0, 5.0, 2.0]

    assert list(out["TIME_PERIOD"]) == expected_dates
    assert list(out["maturity_years"]) == expected_maturities


def test_compute_discount_factors():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": pd.to_datetime(["2026-01-02"]),
            "OBS_VALUE": [2.0],
            "maturity_years": [1.0],
        }
    )

    out = cleaner.compute_discount_factors(df)

    expected = np.exp(-2.0 * 1.0 / 100.0)
    assert np.isclose(out.loc[0, "ZC_price"], expected)


def test_count_maturities_per_date():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": pd.to_datetime(
                ["2026-01-02", "2026-01-02", "2026-01-03"]
            ),
            "OBS_VALUE": [2.0, 2.1, 2.2],
            "maturity_years": [1.0, 2.0, 1.0],
        }
    )

    out = cleaner.count_maturities_per_date(df)

    assert len(out) == 2
    assert out.loc[out["TIME_PERIOD"] == pd.Timestamp("2026-01-02"), "n_maturities"].iloc[0] == 2
    assert out.loc[out["TIME_PERIOD"] == pd.Timestamp("2026-01-03"), "n_maturities"].iloc[0] == 1


def test_filter_min_maturities():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": pd.to_datetime(
                ["2026-01-02", "2026-01-02", "2026-01-03"]
            ),
            "OBS_VALUE": [2.0, 2.1, 2.2],
            "maturity_years": [1.0, 2.0, 1.0],
        }
    )

    out = cleaner.filter_min_maturities(df, min_maturities=2)

    assert set(out["TIME_PERIOD"]) == {pd.Timestamp("2026-01-02")}


def test_check_monotonicity_detects_violation():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": pd.to_datetime(
                ["2026-01-02", "2026-01-02", "2026-01-02"]
            ),
            "maturity_years": [1.0, 2.0, 3.0],
            "ZC_price": [0.99, 0.98, 0.985],
            "OBS_VALUE": [-np.log(0.99)/1.0, -np.log(0.98)/2.0, -np.log(0.985)/3.0],  # needed for compute_discount_factors
        }
    )

    out = cleaner.check_monotonicity(df)

    assert len(out) == 1
    assert out.iloc[0]["TIME_PERIOD"] == pd.Timestamp("2026-01-02")




def test_clean_pipeline_and_report():
    cleaner = make_cleaner()

    df = pd.DataFrame(
        {
            "TIME_PERIOD": ["2026-01-02", "2026-01-02", None, "2026-01-03"],
            "OBS_VALUE": ["2.0", "2.0", "3.0", "4.0"],
            "maturity_years": ["1.0", "1.0", "2.0", "40.0"],
        }
    )

    out, report = cleaner.clean(df)

    assert len(out) == 1
    assert report.rows_in == 4
    assert report.rows_out == 1
    assert report.dropped_missing_critical == 1
    assert report.dropped_out_of_range == 1
    assert report.dropped_duplicates == 1