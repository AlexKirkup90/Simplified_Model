import pandas as pd
import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_fetch_market_data_uses_parquet_cache(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_download(tickers, start=None, end=None, auto_adjust=True, progress=False):
        calls["count"] += 1
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        df = pd.DataFrame({("Close", tickers[0]): [1.0, 2.0, 3.0]}, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    monkeypatch.setattr(backend.yf, "download", fake_download)
    monkeypatch.setattr(backend, "PARQUET_CACHE_DIR", tmp_path)

    backend.fetch_market_data.clear()
    tickers = ["AAA"]
    df1 = backend.fetch_market_data(tickers, "2020-01-01", "2020-01-03")

    # Clear Streamlit cache to force function execution
    backend.fetch_market_data.clear()
    df2 = backend.fetch_market_data(tickers, "2020-01-01", "2020-01-03")

    assert calls["count"] == 1
    assert df1.equals(df2)
