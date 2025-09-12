import pandas as pd
import numpy as np
import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def _make_prices():
    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    data = {
        "AAA": 100 + np.arange(len(idx)) * 0.1,
        "BBB": 80 + np.arange(len(idx)) * 0.05,
        "CCC": 50 + np.arange(len(idx)) * 0.2,
    }
    return pd.DataFrame(data, index=idx)


def test_explain_portfolio_changes_has_descriptor():
    prev_df = pd.DataFrame({"Weight": [0.05, 0.04]}, index=["AAA", "BBB"])
    curr_df = pd.DataFrame({"Weight": [0.06, 0.05]}, index=["AAA", "CCC"])
    prices = _make_prices()
    params = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    expl = backend.explain_portfolio_changes(prev_df, curr_df, prices, params)

    assert "Why" in expl.columns
    assert set(["Buy", "Sell", "Rebalance"]).issubset(set(expl["Action"]))
    assert expl.loc[expl["Action"] == "Buy", "Why"].iloc[0].startswith("Buying")
    assert expl.loc[expl["Action"] == "Sell", "Why"].iloc[0].startswith("Selling")
    assert expl.loc[expl["Action"] == "Rebalance", "Why"].iloc[0].startswith("Rebalancing")
