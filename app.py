import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing

st.title('🏠House Price prediction using ML')
st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6DY2zSdvSmVBgTcuCW8Ia8VCMPnx5WvN87Q&s')

