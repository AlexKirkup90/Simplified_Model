import streamlit as st
import sys, pathlib, types
import pandas as pd
import pytest

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_clean_extreme_moves(monkeypatch):
    messages = []
    monkeypatch.setattr(backend.st, "info", lambda msg: messages.append(msg))

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"A": [1.0, 0.4, 5.0, 1.0, 1.0]}, index=idx)
    cleaned, mask = backend.clean_extreme_moves(
        df, max_daily_move=0.30, min_price=1.0, zscore_threshold=1.0
    )

    expected = pd.DataFrame({"A": [1.0] * 5}, index=idx)
    pd.testing.assert_frame_equal(cleaned, expected)

    expected_mask = pd.DataFrame(
        {"A": [False, True, True, False, False]}, index=idx
    )
    pd.testing.assert_frame_equal(mask, expected_mask)

    assert messages == ["🧹 Data cleaning: Fixed 2 extreme price moves across all stocks"]


def test_fill_missing_data(monkeypatch):
    messages = []
    monkeypatch.setattr(backend.st, "info", lambda msg: messages.append(msg))

    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"A": [1.0, None, 3.0]}, index=idx)

    filled, mask = backend.fill_missing_data(df, max_gap_days=3)

    expected = pd.DataFrame({"A": [1.0, 1.0, 3.0]}, index=idx)
    pd.testing.assert_frame_equal(filled, expected)

    expected_mask = pd.DataFrame({"A": [False, True, False]}, index=idx)
    pd.testing.assert_frame_equal(mask, expected_mask)

    assert messages == ["🔧 Data filling: Filled 1 missing data points with interpolation"]


def test_validate_and_clean_market_data(monkeypatch):
    messages = []
    monkeypatch.setattr(backend.st, "info", lambda msg: messages.append(msg))

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "A": [1.0, 1.0, 5.0, 1.0, None],
            "B": [None, None, None, None, None],
        },
        index=idx,
    )
    cleaned, alerts, mask = backend.validate_and_clean_market_data(df)

    expected = pd.DataFrame({"A": [1.0] * 5}, index=idx)
    pd.testing.assert_frame_equal(cleaned, expected)

    expected_mask = pd.DataFrame(
        {"A": [False, False, True, False, True]}, index=idx
    )
    pd.testing.assert_frame_equal(mask, expected_mask)

    assert alerts == [
        "Removed 1 stocks with >20% missing data",
        "Data shape: (5, 2) → (5, 1)",
    ]

    assert messages == [
        "🧹 Data cleaning: Fixed 1 extreme price moves across all stocks",
        "🔧 Data filling: Filled 1 missing data points with interpolation",
    ]

