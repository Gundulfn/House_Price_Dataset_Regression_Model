import streamlit as st
import pandas as pd
import plotly.express as px
import plotly as pt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore", category=Warning)
from bokeh.plotting import figure

st.write("House Price Dataset - Regresyon Analizi")
df = pd.read_csv(r"C:\Users\Asus\Desktop\kodluyoruz\hafta_2\house_price.csv")
#st.table(df)

with st.sidebar:
    add_radio = st.radio(
        "Bir Aşama Seçiniz.",
        ("Data Ön İnceleme", "Data Ön İşleme", "Model Kurulumu")
    )



if add_radio == "Data Ön İnceleme":
    a = st.radio("Lütfen Seçiniz", ("Head", "Tail"))
    if a == "Head":
        st.table(df.head())
        
    if a == "Tail":
        st.table(df.tail())

    st.table(df.info())
    
    option = st.selectbox(
        'Lütfen İncelemek İstediğiniz Değişkeni Seçiniz',
        df.columns.to_list())

    arr = df[option]
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)
    st.pyplot(fig)