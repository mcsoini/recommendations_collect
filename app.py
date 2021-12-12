import os
import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import datetime
from dateutil import relativedelta

PATH_DATA = os.path.abspath("./data/example")

dict_trends = {-1: "\u25E2", 0: "\u25A0", 1: "\u25E5"}

st.set_page_config(page_title="Stock Recommendations", layout="wide", page_icon="🧊")

st.sidebar.markdown("Stock markt analyst recommendations shamelessly "
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


df_companies_0 = pd.read_pickle(os.path.join(PATH_DATA, "df_companies.pickle"))
df_targets = pd.read_pickle(os.path.join(PATH_DATA, "df_targets.pickle")).join(df_companies_0.set_index("id_wkn").name, on="id_wkn")
df_trends = pd.read_pickle(os.path.join(PATH_DATA, "df_trends.pickle")).join(df_companies_0.set_index("id_wkn").name, on="id_wkn")

df_trends.cat2 = df_trends.cat2.replace(dict_trends)

df_trends.date = pd.to_datetime(df_trends.date, format="%d.%m.%y")
df_trends = df_trends.set_index("date").sort_index().loc[limit_month_name:].reset_index()

df_cat2_group = (df_trends.pivot_table(index="name", columns="cat2", values="analyst", aggfunc=len)
                          .fillna(0)).assign(Count=lambda df: df.sum(axis=1)).astype(int)

min_count = st.sidebar.slider(
    f"Minimum number of recommendations (Count):",
     value=0, step=1, min_value=0, max_value=df_cat2_group["Count"].max())

df_cat2_group = df_cat2_group.loc[df_cat2_group.Count >= min_count]
df_companies_0 = df_companies_0.loc[df_companies_0.name.isin(df_cat2_group.index)]

df_trends.date = df_trends.date.dt.date

name_share_positive = f"%{dict_trends[1]}"

df_cat2_group["share_positive"] = (df_cat2_group[dict_trends[1]] / df_cat2_group["Count"] * 100)
df_cat2_group = df_cat2_group.loc[df_cat2_group.share_positive >= min_pos_share]
df_cat2_group[name_share_positive] = df_cat2_group.share_positive.apply('{:,.0f}%'.format)


df_cat2_group["Total"] = df_cat2_group[dict_trends.values()].sum(axis=1)

df_companies = df_companies_0.join(df_cat2_group, on="name").rename(columns=str)
df_companies = df_companies[["name"] + list(dict_trends.values()) + ["Count", name_share_positive]].rename(columns={"name": "Company", 
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
        df_trends_select.assign()
        gb = GridOptionsBuilder.from_dataframe(df_trends_select)
        gridOptionsTrends = gb.build()

        aggrid_kwargs = dict(gridOptions=gridOptionsTrends,
                fit_columns_on_grid_load=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                data_return_mode=DataReturnMode.FILTERED,
                enable_enterprise_modules=True,
                theme="dark")

        grid_response = AgGrid(df_trends_select, **aggrid_kwargs)


else:
        "Please select a row in the table above to show all recommendation ratings"


st.sidebar.markdown("--------------------------")

st.sidebar.markdown("#### Backend log")

log_text = st.sidebar.text_area("", placeholder="""It was the best of times, it was the worst of times, it was
     the age of wisdom, it was the age of foolishness, it was
     the epoch of belief, it was the epoch of incredulity, it
     was the season of Light, it was the season of Darkness, it
     was the spring of hope, it was the winter of despair, (...)""")
     
