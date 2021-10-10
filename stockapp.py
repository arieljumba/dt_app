import plotly
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly_express as px
import streamlit as st
from datetime import datetime

header_row = st.container()
ownership_row = st.container()
financial_row = st.container()
stocks_row = st.container()

with header_row:
    st.title('Company Performance: created by arieljumba')

    company_name_col, from_date_col = st.columns(2)
    # company selection
    company_names = pd.DataFrame({"display_name": ["Apple Inc", "Bank of America Corporation",
                                                   "Amazon.com", "AT&T Inc", "Alphabet Inc",
                                                   "American Airlines", "AstraZeneca PLC."],
                                  "option_name": ["AAPL", "BAC", "AMZN", "T", "GOOG", "AAL", "AZN"]})
    records = company_names.to_dict("records")
    company_name_sel = company_name_col.selectbox('Select Company below', options=records,
                                                  format_func=lambda records: f'{records["display_name"]}')
    # Date picker
    from_date = from_date_col.date_input('Select Report Start date',value = (datetime(2019, 1, 1))

with ownership_row:
    st.write(''' ##### * Brief Company Description * ''')
    stock = yf.Ticker(company_name_sel['option_name'])
    stock_desc = stock.get_info()
    stock_desc = stock_desc['longBusinessSummary']
    st.write(stock_desc)

with financial_row:
    st.write(''' ##### * Balance Sheet Summary * ''')
    balsheet_col, pl_statement_col = st.columns(2)
    stock_balsheet = stock.balance_sheet / 1000000
    stock_balsheet = pd.DataFrame(stock_balsheet)
    stock_balsheet['Item'] = stock_balsheet.index
    stock_balsheet.rename(columns={stock_balsheet.columns[0]: 'Current_Year'}, inplace=True)
    stock_balsheet.rename(columns={stock_balsheet.columns[1]: 'Previous_Year'}, inplace=True)
    stock_balsheet['Annual_Growth'] = stock_balsheet['Current_Year'] - stock_balsheet['Previous_Year']
    stock_balsheet = stock_balsheet[['Item', 'Current_Year', 'Previous_Year', 'Annual_Growth']]
    pd.options.display.float_format = '{0:,.0f}'.format

    fig = go.Figure(data=go.Table(
        header=dict(values=list(stock_balsheet.columns), align='center', fill_color='black', font_color='white'),
        cells=dict(values=[stock_balsheet.Item, stock_balsheet.Current_Year, stock_balsheet.Previous_Year,
                           stock_balsheet.Annual_Growth],
                   align=['left', 'right', 'right', 'right'], format=["", ",", ",", ","])))
    fig.update_layout(margin={'l': 0, 'b': 0, 't': 3, 'r': 10})
    st.write(fig)

with stocks_row:

    st.write(''' ##### * Stock price Movement * ''')
    stock_prices = stock.history(start=from_date)
    stock_prices['Date'] = stock_prices.index
    fig2 = px.line(stock_prices, y=['Close', 'Open', 'High', 'Low'])
    fig2.update_xaxes(dtick='M1',
                      tickformat='%b\n%Y',
                      ticklabelmode='period')
    fig2.update_layout(margin={'l': 0, 'b': 0, 't': 3, 'r': 10})
    st.write(fig2)

    stock_volume = stock.history(start=from_date)
    stock_volume['Date'] = stock_prices.index
    fig3 = px.line(stock_volume, y=['Volume'])
    fig3.update_xaxes(dtick='M1',
                      tickformat='%b\n%Y',
                      ticklabelmode='period')
    fig3.update_layout(margin={'l': 0, 'b': 0, 't': 3, 'r': 10})
    st.write(fig3)
    
    hide_streamlit_style = """
             <style>
             #MainMenu {visibility: hidden;}
             footer {visibility: hidden;}
             footer:after {
             content:'made by arieljumba (arieljumba5@gmail.com)';
             visibility: visible;
             display: block;
             position: relative;
             #background-color: red;
             padding: 5px;
             top: 2px;
             }
             </style>
             """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
