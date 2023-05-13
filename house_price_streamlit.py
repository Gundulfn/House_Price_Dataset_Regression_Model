import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from category_encoders import TargetEncoder
import pickle

st.set_page_config(page_title="House Price Estimation Model")

st.write("## House Price Dataset - Regression Analysis")
df = pd.read_csv(r"house_price.csv")

with st.sidebar:
    add_radio = st.radio(
        "Please Choose A Process.",
        ("Data Preview", "House Price Estimation")
    )



if add_radio == "Data Preview":
    a = st.radio("##### Please Choose", ("Head", "Tail"))
    if a == "Head":
        st.table(df.head())
        
    if a == "Tail":
        st.table(df.tail())

    
    option = st.selectbox(
        '### Please Choose A Variable That You Want to Examine',
        df.columns.to_list())
    
    arr = df[option]
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)
    st.pyplot(fig)

    st.write("### The Columns That Won't Evaluate To Improve The Results:")

    st.write("Street-", "Alley-", "LotShape-", "Utilities-", "Condition2-", "RoofMatl-", "BsmtFinType2-", "Heating-", "KitchenQual-", 
            "Functional-", "FireplaceQu-", "GarageQual-", "GarageCond-",  "PoolQC-", "Fence-", "MiscFeature-","OverallCond-", "GarageYrBlt-",
             "BsmtFinSF2-", "LowQualFinSF-", "KitchenAbvGr-", "EnclosedPorch-", "3SsnPorch-", "ScreenPorch-", "PoolArea-","MiscVal-", "MoSold-",
             "YrSold-", "BsmtUnfSF-", "2ndFlrSF-", "BsmtFullBath-", "BsmtHalfBath-", "HalfBath-", 'WoodDeckSF-', 'OpenPorchSF')

    image = Image.open(r"feature_importance.png")
    st.image(image ,width=800)
    st.write(" #### The Best 25 Determinative Variables For House Price")
    
if add_radio == "House Price Estimation":

    variables_list=['GrLivArea', '1stFlrSF', 'TotalBsmtSF', 'LotArea', 'BsmtFinSF1', 'GarageArea',
                    'LotFrontage', 'MasVnrArea', 'YearRemodAdd', 'YearBuilt', 'Neighborhood', 'OverallQual',
                    'TotRmsAbvGrd', 'Exterior1st', 'BsmtFinType1', 'BedroomAbvGr', 'Fireplaces', 'SaleCondition',
                    'BsmtExposure', 'HouseStyle', 'Exterior2nd', 'Condition1', 'LotConfig', 'FullBath', 'BsmtQual']

    box_desc_list = ['Physical locations within Ames city limits', 'Exterior covering on house',
                     'Rating of basement finished area', 'Condition of sale', 'Refers to walkout or garden level walls',
                     'Style of dwelling', 'Exterior covering on house (if more than one material)', 'Proximity to various conditions',
                     'Lot configuration', 'Evaluates the height of the basement']
                     
      
    slider_desc_list = ['Above grade (ground) living area square feet', "First Floor square feet", "Total square feet of basement area",
                        'Lot size in square feet', 'Type 1 finished square feet', 'Size of garage in square feet',
                        'Linear feet of street connected to property', 'Masonry veneer area in square feet', 'How much years passed from add ?',
                        'How much years passed from built', 'Rates the overall material and finish of the house', 'Total rooms above grade (does not include bathrooms)',
                        'Bedrooms above grade (does NOT include basement bedrooms)', 'Number of fireplaces', 'Full bathrooms above grade']

                        

    box_list = []
    slider_list = []

   
    for var in range(len(variables_list)):
        if df[variables_list[var]].dtype == "object":
            box_list.append(variables_list[var])
        elif df[variables_list[var]].dtype != "object":
            slider_list.append(variables_list[var])

    box_overall_dict = {}
    slider_overall_dict = {}

    # Creating dictionary for value names and their descriptions
    for var1, var2 in zip(box_list, box_desc_list):
        box_overall_dict.update({var1: var2})

    for var1, var2 in zip(slider_list, slider_desc_list):
        slider_overall_dict.update({var1: var2})

    # Displaying box and slider with functions
    def showing_box(var, desc):
            cycle_option = list(df[var].unique())#
            box = st.sidebar.selectbox(label= f"{desc}", options=cycle_option)
            return box

    def showing_slider(var, desc):
            slider = st.sidebar.slider(label= f"{desc}", min_value=round(df[var].min()), max_value=round(df[var].max()))
            return slider


    # Collecting user inputs in dictionaries
    box_dict = {}
    slider_dict = {}

    for key, value in box_overall_dict.items():
        box_dict.update({key: showing_box(key, value)})

    for key, value in slider_overall_dict.items():
        slider_dict.update({key: showing_slider(key, value)})


    # Keeping inputs in a dic
    input_dict = {**box_dict, **slider_dict}
    dictf = pd.DataFrame(input_dict, index=[0])
    #df = df.append(dictf, ignore_index= True) 
    df = pd.concat([df, dictf], ignore_index=True)

    delete = ["Street", "Alley", "LotShape", "Utilities", "Condition2", "RoofMatl", "BsmtFinType2", "Heating", "KitchenQual", 
          "Functional", "FireplaceQu", "GarageQual", "GarageCond",  "PoolQC", "Fence", "MiscFeature",
         "OverallCond", "GarageYrBlt", "BsmtFinSF2", "LowQualFinSF", "KitchenAbvGr", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
          "MiscVal", "MoSold", "YrSold", "BsmtUnfSF", "2ndFlrSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath", 'WoodDeckSF', 'OpenPorchSF']

    
    # Drop uncessary variables

    for i in delete:
        df.drop(labels= i, axis=1, inplace=True)
        
    df.drop("Id", inplace=True,axis=1)
    df.drop("SalePrice", inplace=True,axis=1)
    
    with open(r"Target_Encoder.sav", 'rb') as f:
        target_encoder = pickle.load(f,  encoding='latin1’)
    #target_encoder = pickle.load(open(r"Target_Encoder.sav", 'rb'))
    
    df3 = pd.DataFrame(target_encoder.transform(df),index = df.index,columns = df.columns)

    # Selecting only last row. (User input data)
    newdata=pd.DataFrame(df3.iloc[[-1]])

    # Load already trained model (XGBoost)
    with open(r"regression_model.sav", 'rb') as f:
        lr = pickle.load(f,  encoding='latin1’) 
                                     
    #lr = pickle.load(open(r"regression_model.sav", 'rb'))
    
    
    ypred = lr.predict(newdata)
    st.write("### The value of the house which the features you choose:")
    st.title(str(np.round(ypred[0]))+" $")

    image = Image.open(r"house.png")
    st.image(image ,width=800)
    
    st.write("### The Results of XGBRegressor Model")
    st.write('#### R^2: % 90.70')
    st.write('#### Mean Squared Error:  597870495.98')
    st.write('#### Root Mean Squared Error:   24451.39')
