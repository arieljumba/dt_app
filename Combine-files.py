from unittest import skip
import pandas as pd
import streamlit as st
import os
import time

st. set_page_config(layout="wide")
st.header('COMBINE FILES')

raw_path = st.text_input('Paste the folder path where multiple files are stored eg "D:/REPORTS"')
raw_path = raw_path+'\\'

csv_files = []
excel_files = []
files = os.listdir(raw_path)
for file in files:    
    if file.endswith('.csv'):
        csv_files.append(file)
    if file.endswith('.xlsx') or file.endswith('.xls'):
        excel_files.append(file)

st.write('CSV file records: {}'.format(len(csv_files)))
st.write('Excel file records: {}'.format(len(excel_files)))

if raw_path =='':
    convert_comm = st.button('COMBINE FILES',disabled=True, help= 'paste the path names as directed in the texboxes above')
elif '\\' not in raw_path :
    convert_comm = st.button('COMBINE FILES',disabled=True, help= 'paste the correct path format as directed in the texboxes above')
else:
    convert_comm = st.button('COMBINE FILES',disabled=False)
with st.spinner('Wait for it...........'):
    df = pd.DataFrame()
    if convert_comm:
        files = os.listdir(raw_path)
        for file in files:
            if file.endswith('.csv'): 
                try:
                    x = pd.read_csv(raw_path+file,error_bad_lines=False,on_bad_lines=skip)
                    df = df.append(x)
                except:
                    x = pd.read_csv(raw_path+file,encoding='latin1',error_bad_lines=False,on_bad_lines=skip)
                    df = df.append(x)
                continue  
            elif  file.endswith('.xlsx') or file.endswith('.xls'): 
                df = df.append(pd.read_excel(raw_path+file,engine='openpyxl'))
        final_msg = '✅ Files Succesfully Combined'
        st.success(final_msg)
        st.balloons()
        st.snow()

        col1,col2 = st.columns([6,1])
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        col2.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Combined_Data.csv',
            mime='text/csv',
        )

        col1.subheader('Final File')
        col1.dataframe(df.head(20))

hide_streamlit_style = """
             <style>
             #MainMenu {visibility: hidden;}
             footer {visibility: hidden;}
             footer:after {
             content:'Code written by arieljumba (arieljumba5@gmail.com)';
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
