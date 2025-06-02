# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import joblib
from streamlit_autorefresh import st_autorefresh
import datetime
import pandas_ta as ta
from stock_list import STOCK_LIST
from live import render_live_dashboard
from prediction import render_prediction_dashboard

# Theme color schemes
THEMES = {
    'Dark': {
        'background': '#121212', 'text': '#E0E0E0', 'primary': '#BB86FC', 'secondary': '#03DAC6', 'danger': '#CF6679',
        'plot_bg': '#121212', 'paper_bg': '#121212', 'font': '#E0E0E0'
    },
    'Light': {
        'background': '#F5F5F5', 'text': '#222', 'primary': '#6200EE', 'secondary': '#018786', 'danger': '#B00020',
        'plot_bg': '#F5F5F5', 'paper_bg': '#F5F5F5', 'font': '#222'
    },
    'Blue': {
        'background': '#1e3a8a', 'text': '#E0E0E0', 'primary': '#60A5FA', 'secondary': '#38BDF8', 'danger': '#F87171',
        'plot_bg': '#1e3a8a', 'paper_bg': '#1e3a8a', 'font': '#E0E0E0'
    }
}

# Set Streamlit page config
st.set_page_config(page_title="Smart Stock Price Prediction", page_icon="📊", layout="wide")

# Sidebar
with st.sidebar:
    st.image("logo.png", width=120)
    st.title("Preferences")
    theme_choice = st.selectbox("Select Theme", list(THEMES.keys()), index=0)
    theme = THEMES[theme_choice]
    st.markdown(f"<hr style='border:1px solid {theme['secondary']}'>", unsafe_allow_html=True)
    dashboard_mode = st.radio("Dashboard Mode", ["Live Dashboard", "Prediction Dashboard"])
    stock = st.selectbox("Select Stock", [""] + STOCK_LIST)
    if dashboard_mode == "Prediction Dashboard" and stock:
        model_preference = st.selectbox("Select Model", [
            "LSTM", "Linear Regression", "Random Forest", "Gradient Boosting", "SVR", "KNN", "Logistic Regression"
        ])
    else:
        model_preference = None
    st.markdown(f"<hr style='border:1px solid {theme['secondary']}'>", unsafe_allow_html=True)
    st.caption("App by Your Company")

# Dynamic CSS for theme
st.markdown(f"""
    <style>
        body, .stApp, .main, .block-container {{
            background-color: {theme['background']} !important;
            color: {theme['text']} !important;
        }}
        .stSidebar {{
            background-color: {theme['background']} !important;
        }}
        .stButton>button {{
            background-color: {theme['secondary']} !important;
            color: {theme['background']} !important;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        }}
        .stButton>button:hover {{
            background-color: {theme['primary']} !important;
            color: {theme['background']} !important;
        }}
        .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {{
            background-color: {theme['background']} !important;
            color: {theme['text']} !important;
        border-radius: 5px;
        }}
        .stMarkdown, .stDataFrame, .stTable, .stSubheader, .stHeader, .stText, .stTitle {{
            color: {theme['text']} !important;
        }}
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# Main area
st.title("Smart Stock Price Predictor 📊")
st.markdown(f"<span style='color:{theme['primary']};font-size:20px;'>A professional-grade app for stock price prediction using various ML models.</span>", unsafe_allow_html=True)

if not stock:
    st.warning("Please select a stock to begin.")
    st.stop()

if dashboard_mode == "Live Dashboard":
    render_live_dashboard(stock, theme)
elif dashboard_mode == "Prediction Dashboard":
    if model_preference:
        render_prediction_dashboard(stock, model_preference, theme)
