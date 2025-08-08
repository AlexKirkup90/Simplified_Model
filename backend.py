import pandas as pd
import numpy as np

# ==========================
# ISA Dynamic Top Config
# ==========================
STRATEGY_PRESETS = {}
STRATEGY_PRESETS["ISA Dynamic (0.75)"] = {
    "mom_lb": 15, "mom_topn": 8, "mom_cap": 0.25,
    "mr_lb": 21,  "mr_topn": 3, "mr_ma": 200,
    "mom_w": 0.85, "mr_w": 0.15,
    "trigger": 0.75
}

# =====================================
# Core run function for ISA Dynamic
# =====================================
def run_dynamic_with_log(prices,
                         mom_lb, mom_topn, mom_cap,
                         mr_lb, mr_topn, mr_ma,
                         mom_w, mr_w,
                         trigger_threshold):
    m_rets, m_tno, m_trades = run_momentum_dynamic(
        prices,
        lookback_m=mom_lb,
        top_n=mom_topn,
        cap=mom_cap,
        trigger_ratio=trigger_threshold
    )
    r_rets, r_tno, r_trades = run_meanrev_dynamic(
        prices,
        lookback_days=mr_lb,
        top_n=mr_topn,
        long_ma_days=mr_ma,
        trigger_ratio=trigger_threshold
    )
    hybrid_rets, hybrid_tno = combine_hybrid(
        m_rets, r_rets, m_tno, r_tno,
        mom_w=mom_w, mr_w=mr_w
    )
    trade_log = pd.concat([m_trades, r_trades], ignore_index=True)
    return hybrid_rets, hybrid_tno, trade_log

# =====================================
# Frequency-aware KPI utils
# =====================================
def _infer_periods_per_year(idx: pd.Index) -> float:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 3: return 12.0
    try:
        freq = pd.infer_freq(idx)
    except Exception:
        freq = None
    if freq:
        f = freq.upper()
        if f.startswith(('B','D')): return 252.0
        if f.startswith('W'):       return 52.0
        if f.startswith('M'):       return 12.0
        if f.startswith('Q'):       return 4.0
        if f.startswith(('A','Y')): return 1.0
    deltas = np.diff(idx.view('i8')) / 1e9
    med_days = np.median(deltas) / 86400.0
    if med_days <= 2.5:  return 252.0
    if med_days <= 9:    return 52.0
    if med_days <= 45:   return 12.0
    if med_days <= 150:  return 4.0
    return 1.0

def _freq_label(py: float) -> str:
    if abs(py-252)<1: return "Daily (252py)"
    if abs(py-52)<1:  return "Weekly (52py)"
    if abs(py-12)<.5: return "Monthly (12py)"
    if abs(py-4)<.5:  return "Quarterly (4py)"
    if abs(py-1)<.2:  return "Yearly (1py)"
    return f"{py:.1f}py"

def trades_per_year(trade_log):
    if trade_log is None or len(trade_log) == 0: return 0.0
    years = pd.DatetimeIndex(trade_log['date']).year.nunique()
    return len(trade_log) / years if years > 0 else 0.0

def turnover_per_year(turnover_series):
    if turnover_series is None or len(turnover_series) == 0: return 0.0
    years = pd.DatetimeIndex(turnover_series.index).year.nunique()
    return turnover_series.sum() / years if years > 0 else 0.0

def kpi_row(name, rets, trade_log=None, turnover_series=None):
    r = pd.Series(rets).dropna().astype(float)
    if r.empty:
        return [name, "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    py = _infer_periods_per_year(r.index)
    eq = (1 + r).cumprod(); n = max(len(r), 1)
    ann_ret = eq.iloc[-1] ** (py / n) - 1
    mean_p, std_p = r.mean(), r.std()
    sharpe = (mean_p * py) / (std_p * np.sqrt(py) + 1e-9)
    down_p = r.clip(upper=0).std()
    sortino = (mean_p * py) / (down_p * np.sqrt(py) + 1e-9) if down_p > 0 else np.nan
    dd = (eq/eq.cummax() - 1).min()
    calmar = ann_ret / abs(dd) if dd != 0 else np.nan
    eq_mult = float(eq.iloc[-1])
    tpy  = trades_per_year(trade_log)
    topy = turnover_per_year(turnover_series)
    return [name, _freq_label(py), f"{ann_ret*100:.2f}%", f"{sharpe:.2f}",
            f"{sortino:.2f}" if not np.isnan(sortino) else "N/A",
            f"{calmar:.2f}"  if not np.isnan(calmar)  else "N/A",
            f"{dd*100:.2f}%", f"{tpy:.1f}", f"{topy:.2f}", f"{eq_mult:.2f}x"]
