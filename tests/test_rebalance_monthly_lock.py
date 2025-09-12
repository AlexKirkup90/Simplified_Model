import pandas as pd
import streamlit as st
import sys, pathlib, types
from datetime import date
from unittest.mock import patch
import pytest

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def _mock_env(monkeypatch):
    def fake_get_universe(choice):
        tickers = ["AAA", "BBB"]
        sectors = {"AAA": "Tech", "BBB": "Tech"}
        return tickers, sectors, "Label"

    def fake_fetch_price_volume(tickers, start, end):
        idx = pd.date_range("2024-06-01", periods=10, freq="B")
        close = pd.DataFrame(100.0, index=idx, columns=tickers)
        vol = pd.DataFrame(1000.0, index=idx, columns=tickers)
        return close, vol

    def fake_fetch_fundamental_metrics(tickers):
        return pd.DataFrame(index=tickers)

    def fake_fundamental_quality_filter(df, min_profitability, max_leverage):
        return df.index.tolist()

    def fake_build_weights(close, params, sectors_map):
        return pd.Series({"AAA": 0.6, "BBB": 0.4})

    monkeypatch.setattr(backend, "get_universe", fake_get_universe)
    monkeypatch.setattr(backend, "fetch_price_volume", fake_fetch_price_volume)
    monkeypatch.setattr(backend, "fetch_fundamental_metrics", fake_fetch_fundamental_metrics)
    monkeypatch.setattr(backend, "fundamental_quality_filter", fake_fundamental_quality_filter)
    monkeypatch.setattr(backend, "_build_isa_weights_fixed", fake_build_weights)


def test_first_trading_day_saves(monkeypatch):
    _mock_env(monkeypatch)
    monkeypatch.setattr(backend, "is_rebalance_today", lambda today, idx: True)
    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]

    with patch.object(backend, "save_portfolio_to_gist") as spg, \
         patch.object(backend, "save_current_portfolio") as scp:
        result = backend.generate_live_portfolio_isa_monthly(preset, None, as_of=date(2024, 6, 3))
        if len(result) == 4:
            disp, raw, price_index, decision = result
        else:
            disp, raw, decision = result
            price_index = st.session_state.get("latest_price_index")
        assert raw is not None and not raw.empty
        saved = backend.save_portfolio_if_rebalance(raw, price_index)
        assert saved is True
        spg.assert_called_once()
        scp.assert_called_once()


def test_mid_month_no_save(monkeypatch):
    _mock_env(monkeypatch)
    monkeypatch.setattr(backend, "is_rebalance_today", lambda today, idx: False)
    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]

    with patch.object(backend, "save_portfolio_to_gist") as spg, \
         patch.object(backend, "save_current_portfolio") as scp:
        result = backend.generate_live_portfolio_isa_monthly(preset, None, as_of=date(2024, 6, 17))
        if len(result) == 4:
            disp, raw, price_index, decision = result
        else:
            disp, raw, decision = result
            price_index = st.session_state.get("latest_price_index")
        assert raw is not None and not raw.empty
        saved = backend.save_portfolio_if_rebalance(raw, price_index)
        assert saved is False
        spg.assert_not_called()
        scp.assert_not_called()
