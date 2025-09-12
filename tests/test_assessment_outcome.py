import streamlit as st
import sys, pathlib, types, json
import pandas as pd
from datetime import date

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_record_assessment_outcome_two_months(monkeypatch):
    st.session_state.clear()

    log_store = {"df": pd.DataFrame(columns=["date", "metrics", "settings", "portfolio_ret", "benchmark_ret"])}

    monkeypatch.setattr(backend, "load_assess_log", lambda: log_store["df"].copy())
    monkeypatch.setattr(backend, "save_assess_log", lambda df: log_store.update(df=df.copy()))

    def fake_get_universe(choice):
        return ["AAA", "BBB"], {"AAA": "Tech", "BBB": "Tech"}, "label"

    def fake_fetch_for_assess(tickers, start, end):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame(100.0, index=dates, columns=tickers)

    def fake_compute_regime_metrics(prices):
        return {
            "breadth_pos_6m": 0.7,
            "qqq_vol_10d": 0.02,
            "vix_term_structure": 1.1,
            "hy_oas": 4.0,
            "qqq_above_200dma": 1.0,
        }

    def fake_get_market_regime():
        return "Risk-On", {}

    monkeypatch.setattr(backend, "get_universe", fake_get_universe)
    monkeypatch.setattr(backend, "fetch_market_data", fake_fetch_for_assess)
    monkeypatch.setattr(backend, "compute_regime_metrics", fake_compute_regime_metrics)
    monkeypatch.setattr(backend, "get_market_regime", fake_get_market_regime)

    backend.assess_market_conditions(date(2024, 1, 5))
    backend.assess_market_conditions(date(2024, 2, 5))

    port_df = pd.DataFrame({"Weight": [0.5, 0.5]}, index=["AAA", "BBB"])
    monkeypatch.setattr(backend, "load_previous_portfolio", lambda: port_df)

    price_map = {
        "2024-01-05": pd.DataFrame(
            {
                "AAA": [100, 110],
                "BBB": [100, 90],
                "QQQ": [100, 105],
            },
            index=pd.to_datetime(["2024-01-05", "2024-02-05"]),
        ),
        "2024-02-05": pd.DataFrame(
            {
                "AAA": [110, 120],
                "BBB": [90, 100],
                "QQQ": [105, 100],
            },
            index=pd.to_datetime(["2024-02-05", "2024-03-05"]),
        ),
    }

    def fake_fetch_for_outcome(tickers, start, end):
        df = price_map[start]
        return df.reindex(columns=tickers)

    monkeypatch.setattr(backend, "fetch_market_data", fake_fetch_for_outcome)

    backend.record_assessment_outcome(date(2024, 1, 5))
    backend.record_assessment_outcome(date(2024, 2, 5))

    log_df = log_store["df"]
    assert len(log_df) == 2
    assert log_df["metrics"].notna().all()
    assert log_df["settings"].notna().all()
    assert log_df["portfolio_ret"].notna().all()
    assert log_df["benchmark_ret"].notna().all()
