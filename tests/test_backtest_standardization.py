import numpy as np
import pandas as pd
import streamlit as st
import types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

import backend


def test_standardize_backtest_payload_handles_pandas_objects():
    idx = pd.date_range("2023-01-31", periods=3, freq="M")
    series = pd.Series([0.01, np.nan, 0.03], index=idx, name="ret")
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]}, index=idx)

    payload = backend.standardize_backtest_payload(
        {
            "series": series,
            "frame": frame,
            "empty": pd.Series(dtype=float),
            "scalar": np.float64(1.25),
        }
    )

    ser_payload = payload["series"]
    assert ser_payload["type"] == "series"
    assert ser_payload["name"] == "ret"
    assert ser_payload["values"][1] is None
    assert ser_payload["index"][0].startswith("2023-01")

    frame_payload = payload["frame"]
    assert frame_payload["type"] == "dataframe"
    assert frame_payload["columns"] == ["a"]
    assert frame_payload["data"][1][0] is None

    empty_payload = payload["empty"]
    assert empty_payload["values"] == []

    assert payload["scalar"] == 1.25
