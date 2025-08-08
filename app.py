from prettytable import PrettyTable
from backend import STRATEGY_PRESETS, run_dynamic_with_log, kpi_row

# --- Inside strategy run block ---
if selected_strategy == "ISA Dynamic (0.75)":
    params = STRATEGY_PRESETS[selected_strategy]
    rets, tno, trades = run_dynamic_with_log(
        prices,
        mom_lb=params['mom_lb'], mom_topn=params['mom_topn'], mom_cap=params['mom_cap'],
        mr_lb=params['mr_lb'],  mr_topn=params['mr_topn'],  mr_ma=params['mr_ma'],
        mom_w=params['mom_w'],  mr_w=params['mr_w'],
        trigger_threshold=params['trigger']
    )
else:
    # existing strategy calls...
    rets, tno, trades = run_existing_strategy(selected_strategy, prices)

# --- KPI Table ---
tbl = PrettyTable()
tbl.field_names = ["Model","Freq","CAGR","Sharpe","Sortino","Calmar","MaxDD","Trades/yr","Turnover/yr","Equity Multiple"]
tbl.add_row(kpi_row(selected_strategy, rets, trades, tno))
st.text(tbl)

# --- What Changed? ---
if trades is not None and not trades.empty:
    st.write("### Changes this period")
    st.dataframe(trades.tail(20))
