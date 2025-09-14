import numpy as np
import pandas as pd
import streamlit as st
import sys, pathlib, types
from dataclasses import replace

# Provide empty secrets so strategy imports don't fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from strategy_core import HybridConfig, walk_forward_backtest


def test_walk_forward_backtest_with_transaction_costs():
    # create 5 years of business-day prices for three assets
    n = 252 * 5
    dates = pd.bdate_range('2010-01-01', periods=n)
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.005, size=(n, 3))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=['A', 'B', 'C'])

    base_cfg = HybridConfig(momentum_top_n=2, momentum_cap=0.6, mom_weight=1.0, mr_weight=0.0)

    gross = walk_forward_backtest(prices, base_cfg)
    net = walk_forward_backtest(prices, replace(base_cfg, tc_bps=25))

    # Metrics should remain positive but degrade once transaction costs are applied
    assert gross['sharpe'] > net['sharpe'] > 0
    assert gross['sortino'] > net['sortino'] > 0
    assert len(gross['windows']) >= 1
