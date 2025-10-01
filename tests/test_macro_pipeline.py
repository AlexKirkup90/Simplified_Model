import pathlib
import sys
import types

import numpy as np
import pandas as pd
import streamlit as st

st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend  # noqa: E402


def test_sanitize_series_dataframe():
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "Adj Close": [100.0, 101.0, 102.0],
            "Close": [99.0, 100.0, 101.0],
            "Volume": [1, 2, 3],
        },
        index=idx,
    )

    sanitized = backend._sanitize_series(df)
    assert isinstance(sanitized, pd.Series)
    assert sanitized.index.equals(idx)
    assert sanitized.tolist() == [100.0, 101.0, 102.0]


def test_macro_neutral_when_missing(monkeypatch):
    idx = pd.date_range("2021-01-01", periods=60, freq="B")
    prices = pd.DataFrame({"AAA": 100.0 + np.arange(len(idx), dtype=float)}, index=idx)

    def fake_macro(start, end):
        return {
            "vix": pd.Series(dtype=float),
            "vix3m": pd.Series(dtype=float),
            "hy_oas": pd.Series(dtype=float),
            "vix_status": "missing",
            "vix3m_status": "missing",
            "hy_oas_status": "missing",
        }

    monkeypatch.setattr(backend, "fetch_macro_series", fake_macro)

    metrics = backend.compute_regime_metrics(prices)
    assert metrics["vix_ts_status"] == "missing"
    assert metrics["vix_ts_score"] == 50.0
    assert metrics["vix_term_structure"] == 1.0
    assert metrics["hy_oas_status"] == "missing"
    assert metrics["hy_oas_score"] == 50.0
