import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(page_title="Chicago Crime Analyser", layout="wide", initial_sidebar_state="expanded")

def background():
    st.markdown("""
        <style>
        .main {
            background-color: #ffe6f0;
            padding: 20px;
        }
        .title {
            font-size: 2.5em;
            color: #333333;
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Arial', sans-serif;
        }
        .subtitle {
            font-size: 1.5em;
            color: #555555;
            text-align: center;
            margin-bottom: 30px;
            font-family: 'Arial', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #e0f7fa;
            padding: 20px;
            border-radius: 8px;
            }
        .stButton>button {
            color: blue;
            background-color: #cce5ff;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            
            font-family: 'Arial', sans-serif;
            font-size: 1em;
        }
        .stButton>button:hover {
            background-color: #cce5ff;
            
        }
        </style>
        """, unsafe_allow_html=True)
background()

st.title('Chicago Crime Type Predictor App')

st.markdown('Give the below inputs to predict crime type')

data=pd.read_csv(r'C:\Users\Lakshmi\Desktop\Chicago\filtered_crime.csv')
data['Block'] = data['Block'].astype(str) #changing the datatype
block_list=data['Block'].unique()
Location_Description=data['Location Description'].unique()
domestic=data['Domestic'].unique()
Beat=data['Beat'].unique()
District=data['District'].unique()
Ward=data['Ward'].unique()
Community_Area=data['Community Area'].unique()
Year=data['Year'].unique()


cola,colb,colc=st.columns(3)

with cola:
    District=st.selectbox('Choose District', District)
with colb:
    dom=st.selectbox('Choose Domestic Type', domestic)
with colc:
    Year=st.selectbox('Choose Year', Year)


col_width=[2,1,2]
col1,col2,col3=st.columns(col_width)

selected_district=data[data['District']==District]

with col1:

    Month=st.number_input('Give Month', min_value=1, max_value=12)
    Day=st.number_input('Give Date', min_value=1, max_value=31)
    Hour=st.number_input('Give Time in 24hours format',min_value=0, max_value=23)
    Beat=st.selectbox('Choose Beat',selected_district['Beat'].unique())

with col3:

    Ward=st.selectbox('Choose Ward', selected_district['Ward'].unique())
    Community_Area=st.selectbox('Choose Community Area', selected_district['Community Area'].unique())
    block=st.selectbox('Choose Block',selected_district['Block'].unique())
    loc_desc=st.selectbox('Choose Location',selected_district['Location Description'].unique())

# transform the input categorical features - block, location description and domestic to label encoder

with open(r'C:\Users\Lakshmi\Desktop\Chicago\encoder_1.pkl', 'rb') as file:
    encoder_block=pickle.load(file)

with open(r'C:\Users\Lakshmi\Desktop\Chicago\encoder_2.pkl', 'rb') as file:
    encoder_loc=pickle.load(file)

with open(r'C:\Users\Lakshmi\Desktop\Chicago\encoder_3.pkl', 'rb') as file:
    encoder_dom=pickle.load(file)




block=encoder_block.transform([block])[0]
loc_desc=encoder_loc.transform([loc_desc])[0]
dom=encoder_dom.transform([dom])[0]

# st.write(type(block))

features=[[block, loc_desc, dom, Beat, District, Ward, Community_Area, Year, Month, Day, Hour]]

# st.write(type(features))
# st.write(features)

if st.button('Predict Crime'):

    

    features=[[block, loc_desc, dom, Beat, District, Ward, Community_Area, Year, Month, Day, Hour]]
    # st.write(features)

    if (District is not None and dom is not None and Year is not None and 
            Month is not None and Day is not None and Hour is not None and Beat is not None and
            Ward is not None and Community_Area is not None and block is not None and loc_desc is not None):

            

            try:
                

                    
                    with open(r'C:\Users\Lakshmi\Desktop\Chicago\predictive_model_1.pkl','rb') as file:
                        model=pickle.load(file) 
                    crime=model.predict(features)
                    with open(r'C:\Users\Lakshmi\Desktop\Chicago\encoder_4.pkl','rb') as file:
                         primary_type=pickle.load(file)

                    crime=primary_type.inverse_transform([crime])[0]
                    st.write(f'Crime Type may occur at given Location and Time is: {crime}')

            except Exception as e:

                st.write('Please provide all inputs')
                st.write(f'Error: {e}')