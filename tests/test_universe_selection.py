import sys, pathlib, types
from datetime import date
import pandas as pd
import streamlit as st

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend

def _patch_yf(monkeypatch, spy_ret, qqq_ret):
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    def fake_download(tickers, start, end, auto_adjust=True, progress=False):
        if isinstance(tickers, str):
            tickers_list = [tickers]
        else:
            tickers_list = tickers
        data = {}
        for t in tickers_list:
            start_price = 100.0
            if t == "SPY":
                end_price = 100.0 * (1 + spy_ret)
            elif t == "QQQ":
                end_price = 100.0 * (1 + qqq_ret)
            else:
                end_price = 100.0
            data[t] = [start_price, end_price]
        df = pd.DataFrame(data, index=dates)
        df.columns = pd.MultiIndex.from_product([["Close"], df.columns])
        return df
    monkeypatch.setattr(backend.yf, "download", fake_download)


def test_select_universe_nasdaq(monkeypatch):
    _patch_yf(monkeypatch, spy_ret=0.01, qqq_ret=0.10)
    assert backend.select_optimal_universe(date(2024, 1, 5)) == "NASDAQ100+"


def test_select_universe_sp500(monkeypatch):
    _patch_yf(monkeypatch, spy_ret=0.05, qqq_ret=0.06)
    assert backend.select_optimal_universe(date(2024, 1, 5)) == "S&P500 (All)"


def test_select_universe_hybrid(monkeypatch):
    _patch_yf(monkeypatch, spy_ret=-0.05, qqq_ret=-0.02)
    assert backend.select_optimal_universe(date(2024, 1, 5)) == "Hybrid Top150"
