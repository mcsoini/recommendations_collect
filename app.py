import os
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import datetime
from dateutil import relativedelta
from streamlit.report_thread import get_report_ctx
from dotenv import load_dotenv
from sqlalchemy import create_engine


dict_trends = {-1: "\u25E2", 0: "\u25A0", 1: "\u25E5"}

st.set_page_config(page_title="Stock Recommendations", layout="wide", page_icon="🧊")

try:
    session_id = get_report_ctx().session_id
    print(session_id)
except Exception as e:  # we are not running streamlit
    session_id = -1

@st.cache
def get_data(session_id):

    load_dotenv()

    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
    POSTGRES_IP = os.getenv('POSTGRES_IP')

    engine = create_engine(f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_IP}:5432/maindb")

    table_names = ["targets", "trends", "companies"]

    return tuple(pd.read_sql(table_name, engine).copy() for table_name in table_names)

df_targets, df_trends, df_companies_0 = get_data(session_id=session_id)

df_targets = df_targets.copy()
df_trends = df_trends.copy()
df_companies_0 = df_companies_0.copy()

data_datetime = max(df_targets.datetime.max(),
                    df_trends.datetime.max(),
                    df_companies_0.datetime.max())


st.sidebar.markdown("Collection of stock market analyst recommendations shamelessly "
                    "ripped from a public website and "
                    "presented in a neat sortable table. Enjoy."
                    "\n\n-------------")

st.sidebar.markdown("#### Filters")

start_time = st.sidebar.slider(
    "Use analyst results of the last X months:",
     value=2, step=1, min_value=0, max_value=12)
select_date = pd.to_datetime((datetime.date.today() 
                              + relativedelta.relativedelta(months=-start_time)
                              ).strftime("%Y%m"), format="%Y%m")
limit_month_name = select_date.strftime('%B %Y')

min_pos_share = st.sidebar.slider(
    f"Minimum positive share (%{dict_trends[1]}):",
     value=0, step=1, min_value=0, max_value=100)


df_targets = df_targets.join(df_companies_0.set_index("id_wkn").name, on="id_wkn")
df_trends = df_trends.join(df_companies_0.set_index("id_wkn").name, on="id_wkn")

df_trends.cat2 = df_trends.cat2.replace(dict_trends)

df_trends.date = pd.to_datetime(df_trends.date, format="%d.%m.%y")
df_targets.date = pd.to_datetime(df_targets.date, format="%d.%m.%y")
df_trends = df_trends.set_index("date").sort_index().loc[limit_month_name:].reset_index()
df_targets = df_targets.set_index("date").sort_index().loc[limit_month_name:].reset_index()

df_cat2_group = (df_trends.pivot_table(index="name", columns="cat2", values="analyst", aggfunc=len)
                          .fillna(0)).assign(Count=lambda df: df.sum(axis=1)).astype(int)

min_count = st.sidebar.slider(
    f"Minimum number of recommendations (Count):",
     value=0, step=1, min_value=0, max_value=int(df_cat2_group["Count"].max()))

df_cat2_group = df_cat2_group.loc[df_cat2_group.Count >= min_count]
df_companies_0 = df_companies_0.loc[df_companies_0.name.isin(df_cat2_group.index)]

df_trends.date = df_trends.date.dt.date
df_targets.date = df_targets.date.dt.date

df_targets_companies = (df_targets.set_index("name")[["dev_price_target"]] * 100).groupby(level="name").agg(["min", "median", "max"]).applymap('{:,.1f}%'.format)
df_targets_companies["Price targets"] = df_targets_companies.applymap(lambda x: x.rjust(7, " ") + " ").apply("/".join, axis=1)#.apply(lambda row: f"{row:[0]02}", axis=1)
df_cat2_group = df_cat2_group.reset_index().join(df_targets_companies["Price targets"], on="name").set_index(df_cat2_group.index.names)

name_share_positive = f"%{dict_trends[1]}"

# complement columns
df_cat2_group[[c for c in dict_trends.values() if not c in df_cat2_group.columns]] = 0

df_cat2_group["share_positive"] = (df_cat2_group[dict_trends[1]] / df_cat2_group["Count"] * 100)
df_cat2_group = df_cat2_group.loc[df_cat2_group.share_positive >= min_pos_share]
df_cat2_group[name_share_positive] = df_cat2_group.share_positive.apply('{:,.0f}%'.format)


df_cat2_group["Total"] = df_cat2_group[dict_trends.values()].sum(axis=1)

df_companies = df_companies_0.join(df_cat2_group, on="name").rename(columns=str)
df_companies = df_companies[["name"] + list(dict_trends.values()) + ["Count", name_share_positive, "Price targets"]].rename(columns={"name": "Company", 
                             "price_change": "Price change", "price": "Price"})
df_companies = df_companies.loc[~df_companies["Count"].isna()]

df_companies_0 = df_companies_0.set_index("name")



gb = GridOptionsBuilder.from_dataframe(df_companies)

gb.configure_default_column(groupable=True, value=True,
                            enableRowGroup=True, aggFunc="sum", editable=True)
gb.configure_column("Company", suppressSizeToFit=True, 
                    initialWidth=200, width=200, resizable=True, suppressAutoSize=True)
# gb.configure_column("Price change", valueFormatter=JsCode("""
# function bracketsFormatter(params) {
#   return parseFloat(params.value * 100).toFixed(2)+"%";
# }
# """))


gb.configure_selection('single')
gb.configure_grid_options(domLayout='normal')
gridOptions = gb.build()

st.markdown(f"Table containing all {len(df_companies)} companies with available recommendations (since {limit_month_name}).")

aggrid_kwargs = dict(gridOptions=gridOptions,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                data_return_mode=DataReturnMode.FILTERED,
                enable_enterprise_modules=True, height=200,
                theme="dark")

grid_response = AgGrid(df_companies, **aggrid_kwargs)
selected_rows = grid_response["selected_rows"]

if selected_rows:
        select_company = selected_rows[0]["Company"]

        price, price_change, date_price = df_companies_0.loc[select_company, ["price", "price_change", "their_date"]]

        st.subheader(f"Selected: *{select_company}*")
        
        st.markdown(f'<h4>{price} (<h style="color:{"green" if price_change > 0 else "red"};">{price_change*100:+.2f}%</h>) on {date_price}</h4>', unsafe_allow_html=True)

        df_trends_select = df_trends.query("name == @select_company")[["date", "cat2", "cat1", "analyst"]]
        df_targets_select = df_targets.query("name == @select_company")[["date", "price_target", "currency", "dev_price_target", "analyst"]]



        aggrid_kwargs = dict(fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                data_return_mode=DataReturnMode.FILTERED,
                enable_enterprise_modules=True,
                theme="dark")

        st.markdown("___Recommendations table___")
        gb = GridOptionsBuilder.from_dataframe(df_trends_select)
        gridOptionsTrends = gb.build()
        grid_response = AgGrid(df_trends_select, **{**aggrid_kwargs, **{"gridOptions": gridOptionsTrends}})


        st.markdown("___Price targets table___")
        gb = GridOptionsBuilder.from_dataframe(df_targets_select)
        gridOptionsTargets = gb.build()
        grid_response = AgGrid(df_targets_select, **{**aggrid_kwargs, **{"gridOptions": gridOptionsTargets}})

        st.markdown(" ")
else:
        "Please select a row in the table above to show analyst recommendation details."


st.sidebar.markdown("--------------------------")
st.sidebar.markdown("#### Backend log")
st.sidebar.text(f"Date obtained at time: {data_datetime}")
     
