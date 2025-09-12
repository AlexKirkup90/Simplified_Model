import numpy as np
import pandas as pd
import pytest
import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_get_strategy_health_metrics():
    index = pd.date_range("2022-01-31", periods=24, freq="M")
    returns = pd.Series(np.linspace(0.01, 0.24, 24), index=index)
    benchmark = returns * 1.5

    metrics = backend.get_strategy_health_metrics(returns, benchmark)

    recent_3m = returns.iloc[-3:]
    expected_3m_return = recent_3m.mean()
    expected_3m_sharpe = (recent_3m.mean() * 12) / (recent_3m.std() * np.sqrt(12) + 1e-9)

    recent_6m = returns.iloc[-6:]
    expected_6m_return = recent_6m.mean()
    expected_hit_rate = (recent_6m > 0).mean()

    equity_curve = (1 + returns).cumprod()
    expected_drawdown = (equity_curve / equity_curve.cummax() - 1).iloc[-1]

    recent_vol = returns.iloc[-12:].std() * np.sqrt(12)
    long_vol = returns.std() * np.sqrt(12)
    expected_vol_ratio = recent_vol / (long_vol + 1e-9)

    expected_corr = returns.corr(benchmark)

    assert metrics["recent_3m_return"] == pytest.approx(expected_3m_return)
    assert metrics["recent_3m_sharpe"] == pytest.approx(expected_3m_sharpe)
    assert metrics["recent_6m_return"] == pytest.approx(expected_6m_return)
    assert metrics["hit_rate_6m"] == pytest.approx(expected_hit_rate)
    assert metrics["current_drawdown"] == pytest.approx(expected_drawdown)
    assert metrics["vol_regime_ratio"] == pytest.approx(expected_vol_ratio)
    assert metrics["benchmark_correlation"] == pytest.approx(expected_corr)


def test_diagnose_strategy_issues():
    index = pd.date_range("2022-01-31", periods=12, freq="M")
    returns = pd.Series([0.3] * 6 + [-0.3] * 6, index=index)
    turnover = pd.Series(1.2, index=index)

    issues = backend.diagnose_strategy_issues(returns, turnover)

    assert any("Poor recent performance" in issue for issue in issues)
    assert any("Low hit rate" in issue for issue in issues)
    assert any("Large drawdown" in issue for issue in issues)
    assert any("Excessive turnover" in issue for issue in issues)
    assert any("High volatility" in issue for issue in issues)


def test_diagnose_strategy_issues_none():
    index = pd.date_range("2022-01-31", periods=12, freq="M")
    returns = pd.Series([0.05] * 12, index=index)
    turnover = pd.Series(0.5, index=index)

    issues = backend.diagnose_strategy_issues(returns, turnover)

    assert issues == ["No significant issues detected"]
