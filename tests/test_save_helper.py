import streamlit as st
import types
import pandas as pd
from datetime import date
from unittest.mock import patch
import pytest

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

import backend


def test_save_portfolio_if_rebalance(monkeypatch):
    df = pd.DataFrame({"Weight": [1.0]}, index=["AAPL"])
    price_index = pd.date_range("2024-06-03", periods=5, freq="B")

    # Case: on first trading day -> saving occurs
    with patch.object(backend, "save_portfolio_to_gist") as spg, \
         patch.object(backend, "save_current_portfolio") as scp:
        class RebalanceDate(date):
            @classmethod
            def today(cls):
                return date(2024, 6, 3)

        monkeypatch.setattr(backend, "date", RebalanceDate)
        assert backend.save_portfolio_if_rebalance(df, price_index) is True
        spg.assert_called_once()
        scp.assert_called_once()

    # Case: not a rebalance day -> saving is skipped
    with patch.object(backend, "save_portfolio_to_gist") as spg, \
         patch.object(backend, "save_current_portfolio") as scp:
        class NonRebalanceDate(date):
            @classmethod
            def today(cls):
                return date(2024, 6, 4)

        monkeypatch.setattr(backend, "date", NonRebalanceDate)
        assert backend.save_portfolio_if_rebalance(df, price_index) is False
        spg.assert_not_called()
        scp.assert_not_called()
