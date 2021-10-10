import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

header_row = st.container()
data_row = st.container()
linearity_row = st.container()
results_row = st.container()


def header(url):
    st.markdown(f'<p style="color:#685ae6;font-size:12px;border-radius:2%;">{url}</p>',
                unsafe_allow_html=True)


with header_row:
    st.title('Designing a Regression model to predict CO2 Emissions')
    header('arieljumba5@gmail.com')


@st.cache
def get_data():
    data = pd.read_csv('data/Fuel_Consumption.csv', encoding='cp1252', index_col=False)
    data.columns = ['Model_Year', 'Make', 'Model', 'Vehicle_Class', 'Engine_Size', 'Cylinders', 'Transmission',
                    'Fuel_Type', 'City', 'Hwy', 'Comb', 'mpg', 'CO2_Emissions', 'CO2_Rating', 'Smog_Rating']
    data = data.dropna(how='all')
    data = data[[
        'Model_Year', 'Engine_Size', 'Cylinders', 'Hwy', 'Comb', 'mpg', 'CO2_Emissions', 'CO2_Rating', 'Smog_Rating']]
    data = data.fillna(0)
    return data


with data_row:
    df = get_data()
    st.write('''Data Description''')
    st.write(df.describe())

with st.sidebar:
    st.write('''
        ## **Data Filters**
        ''')
    st.write('')
    data_split = st.slider('Slide to set the Train:Test Split', min_value=10, max_value=100, value=80, step=10)
    st.write('')
    dependent_var = st.selectbox('Pick Your Preferred Dependent Variables', options=[x for x in df.columns],
                                 index=1)
    st.write('')
    model_type = st.selectbox('Confirm relationship and Select Regression model type', options=['Linear', 'Non-Linear'])
    st.write('')
    form = st.form(key='my-form')
    submit = form.form_submit_button('Submit changes')


with linearity_row:
    st.write('')
    st.write('''
        ## **Relationship Plot**
        ''')
    st.write('Confirm the relationship between the dependent and independent variables below:')

    try:
        fig_1 = plt.figure(figsize=(10, 5))
        plt.scatter(df[dependent_var], df.CO2_Emissions, color='blue')
        plt.xlabel(dependent_var)
        plt.ylabel('CO2_Emissions')
        st.pyplot(fig_1)
    except:
        st.write('Select variable')

with results_row:
    msk = np.random.rand(len(df)) > data_split / 100
    train = df[msk]
    test = df[~msk]

    if model_type == 'Linear':
        model = linear_model.LinearRegression()
        train_x = np.asanyarray(train[[dependent_var]])
        train_y = np.asanyarray(train[['CO2_Emissions']])
        model.fit(train_x, train_y)
        st.write('''
            ## **Model Structure**
            ''')
        if submit:
            st.write('The model Coefficient is: {}'.format(model.coef_))
            st.write('The model Intercept is: {}'.format(model.intercept_))

        test_x = np.asanyarray(test[[dependent_var]])
        test_y = np.asanyarray(test[['CO2_Emissions']])
        pred = model.predict(test_x)

        st.write('''
            ## **Model Plot**
            ''')
        fig_2 = plt.figure(figsize=(10, 5))
        plt.scatter(train[dependent_var], train.CO2_Emissions, color='blue')
        xx = np.arange(0.0, 10.0, 0.1)
        yy = model.coef_[0][0] * xx + model.intercept_[0]
        plt.plot(xx, yy, '-r')
        plt.xlabel(dependent_var)
        plt.ylabel('Emission')
        if submit:
            st.pyplot(fig_2)

        st.write('''
        ## **Accuracy measures**
        ''')
        if submit:
            st.write('MAE: {}'.format(np.mean(np.abs(pred - test_y))))
            st.write('RMSE: {}'.format(np.mean(pred - test_y) ** 2))
            st.write('r2_Score: {}'.format(r2_score(pred, test_y)))
    else:
        from sklearn import linear_model

        model = linear_model.LinearRegression()
        train_x = np.asanyarray(train[[dependent_var]])
        train_y = np.asanyarray(train[['CO2_Emissions']])

        poly = PolynomialFeatures(degree=2)
        train_x = poly.fit_transform(train_x)
        model.fit(train_x, train_y)

        st.write('''
                    ## **Model Structure**
                    ''')
        if submit:
            st.write('The model Coefficients are: {}'.format(model.coef_))
            st.write('The model Intercept is: {}'.format(model.intercept_))
            st.write(model.coef_[0][2])

        test_x = np.asanyarray(test[[dependent_var]])
        test_y = np.asanyarray(test[['CO2_Emissions']])
        test_x = poly.fit_transform((test_x))
        pred = model.predict(test_x)

        fig_2 = plt.figure(figsize=(10, 5))
        plt.scatter(train[dependent_var], train.CO2_Emissions, color='blue')
        xx = np.arange(0.0, 10.0, 0.1)
        yy = model.coef_[0][1] * xx + model.coef_[0][2] * xx * np.power(xx, 2) + model.intercept_[0]
        plt.plot(xx, yy, '-r')
        plt.xlabel('Engine Size')
        plt.ylabel('Emission')
        if submit:
            st.pyplot(fig_2)

        st.write('''
                ## **Accuracy measures**
                ''')
        if submit:
            st.write('MAE: {}'.format(np.mean(np.abs(pred - test_y))))
            st.write('RMSE: {}'.format(np.mean(pred - test_y) ** 2))
            st.write('r2_Score: {}'.format(r2_score(pred, test_y)))

hide_streamlit_style = '''
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
             '''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
