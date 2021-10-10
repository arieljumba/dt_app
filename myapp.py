import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename,sep = ",")
    return taxi_data

with header:
    st.title('Test Project')
    st.write('''This is just a test model for marketing campaign in NYC''')

with dataset:
    st.write('''### The Dataset''')
    market_data = get_data('data/market.csv')

    st.write(market_data.head(5))
    st.write(''' ##### Distribution Data''')
    education_count = pd.DataFrame(market_data['Education'].value_counts())
    st.write(education_count)
    st.write(''' ##### Distribution Chart''')
    st.bar_chart(education_count)

with features:
    st.write('''### Model features''')

with model_training:
    st.write('''
    ### Model Training
    - Here You choose the hyper-parameters of the model and see how the performance changes
    ''')

    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('What is the max depth of the model?', min_value= 10, max_value=100, value=10,step=10)
    n_estimators = sel_col.selectbox('How many trees should there be?',options=[100,200,300,'No Limit'],index = 0)
    input_feature = sel_col.text_input('which feature should be used as the input feature?','ID')

    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    x = market_data[[input_feature]]
    y = market_data[['Income']]
    regr.fit(x,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error is: ')
    disp_col.write(mean_absolute_error(y,prediction))
    disp_col.subheader('Mean squared error is: ')
    disp_col.write(mean_squared_error(y,prediction))
    disp_col.subheader('R2 Score is: ')
    disp_col.write(r2_score(y,prediction))
    
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


