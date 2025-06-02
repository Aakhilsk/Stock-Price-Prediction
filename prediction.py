import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import plotly.graph_objects as go
import joblib
import os

def render_prediction_dashboard(stock, model_preference, theme):
    start = '2013-01-01'
    end = '2023-12-31'
    with st.spinner('Downloading stock data...'):
        df = yf.download(stock, start, end)
    if df.empty:
        st.error("No historical data available for this stock.")
        return
    st.subheader(f'{stock} Data (2013-2023)')
    st.dataframe(df, use_container_width=True)
    st.subheader('Closing Price vs Time')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color=theme['primary'])))
    fig.update_layout(
        title='Closing Price vs Time',
        xaxis_title='Date',
        yaxis_title='Closing Price (INR)',
        template='plotly_white',
        plot_bgcolor=theme['plot_bg'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(color=theme['font'])
    )
    st.plotly_chart(fig, use_container_width=True)
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color=theme['primary'])))
    fig2.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100 Days MA', line=dict(color=theme['secondary'])))
    fig2.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200 Days MA', line=dict(color=theme['danger'])))
    fig2.update_layout(
        title='Closing Price with 100 & 200 Days MA',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        template='plotly_white',
        plot_bgcolor=theme['plot_bg'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(color=theme['font'])
    )
    st.plotly_chart(fig2, use_container_width=True)
    with st.spinner('Preparing data...'):
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        x_train, y_train = [], []
        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i - 100:i])
            y_train.append(data_training_array[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)
    model_path = None
    if model_preference == "LSTM":
        model_path = 'my_model.keras'
    elif model_preference == "Linear Regression":
        model_path = 'linear_regression_model.joblib'
    elif model_preference == "Random Forest":
        model_path = 'random_forest_model.joblib'
    elif model_preference == "Gradient Boosting":
        model_path = 'gradient_boosting_model.joblib'
    elif model_preference == "SVR":
        model_path = 'svr_model.joblib'
    elif model_preference == "KNN":
        model_path = 'knn_model.joblib'
    elif model_preference == "Logistic Regression":
        model_path = 'logistic_regression_model.joblib'
    if model_preference == "LSTM":
        with st.spinner('Loading/Training LSTM model...'):
            if model_path and os.path.exists(model_path):
                model = load_model(model_path)
                st.success('Pre-trained LSTM model loaded!')
            else:
                model = Sequential()
                model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=60, activation='relu', return_sequences=True))
                model.add(Dropout(0.3))
                model.add(LSTM(units=80, activation='relu', return_sequences=True))
                model.add(Dropout(0.4))
                model.add(LSTM(units=120, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=50)
                model.save(model_path)
                st.success('LSTM model trained and saved!')
            y_predicted = model.predict(x_test)
            scale_factor = 1 / 0.0005928
            y_predicted = y_predicted * scale_factor
            y_test_scaled = y_test * scale_factor
            st.subheader('LSTM Prediction vs Original')
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=np.arange(len(y_test_scaled)), y=y_test_scaled, mode='lines', name='Original', line=dict(color=theme['secondary'])))
            fig3.add_trace(go.Scatter(x=np.arange(len(y_predicted)), y=y_predicted.flatten(), mode='lines', name='Predicted', line=dict(color=theme['primary'])))
            fig3.update_layout(title='LSTM Prediction vs Original', xaxis_title='Index', yaxis_title='Price', plot_bgcolor=theme['plot_bg'], paper_bgcolor=theme['paper_bg'], font=dict(color=theme['font']))
            st.plotly_chart(fig3, use_container_width=True)
    elif model_preference == "Linear Regression":
        with st.spinner('Training Linear Regression model...'):
            lin_reg = LinearRegression()
            lin_reg.fit(x_train.reshape(x_train.shape[0], -1), y_train)
            lin_reg_train_pred = lin_reg.predict(x_train.reshape(x_train.shape[0], -1))
            lin_reg_test_pred = lin_reg.predict(x_test.reshape(x_test.shape[0], -1))
            st.subheader('Linear Regression Model Metrics')
            st.write(f"R² (Train): {r2_score(y_train, lin_reg_train_pred):.4f} | R² (Test): {r2_score(y_test, lin_reg_test_pred):.4f}")
            st.write(f"RMSE (Train): {np.sqrt(mean_squared_error(y_train, lin_reg_train_pred)):.4f} | RMSE (Test): {np.sqrt(mean_squared_error(y_test, lin_reg_test_pred)):.4f}")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True', line=dict(color=theme['secondary'])))
            fig4.add_trace(go.Scatter(x=np.arange(len(lin_reg_test_pred)), y=lin_reg_test_pred, mode='lines', name='Predicted', line=dict(color=theme['primary'])))
            fig4.update_layout(title='Linear Regression Predictions', xaxis_title='Index', yaxis_title='Price', plot_bgcolor=theme['plot_bg'], paper_bgcolor=theme['paper_bg'], font=dict(color=theme['font']))
            st.plotly_chart(fig4, use_container_width=True)
    elif model_preference == "Random Forest":
        with st.spinner('Training Random Forest model...'):
            model_rf = RandomForestRegressor()
            model_rf.fit(x_train.reshape(x_train.shape[0], -1), y_train)
            y_pred_rf_train = model_rf.predict(x_train.reshape(x_train.shape[0], -1))
            y_pred_rf_test = model_rf.predict(x_test.reshape(x_test.shape[0], -1))
            st.subheader('Random Forest Model Metrics')
            st.write(f"R² (Train): {r2_score(y_train, y_pred_rf_train):.4f} | R² (Test): {r2_score(y_test, y_pred_rf_test):.4f}")
            st.write(f"RMSE (Train): {np.sqrt(mean_squared_error(y_train, y_pred_rf_train)):.4f} | RMSE (Test): {np.sqrt(mean_squared_error(y_test, y_pred_rf_test)):.4f}")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True', line=dict(color=theme['secondary'])))
            fig5.add_trace(go.Scatter(x=np.arange(len(y_pred_rf_test)), y=y_pred_rf_test, mode='lines', name='Predicted', line=dict(color=theme['primary'])))
            fig5.update_layout(title='Random Forest Predictions', xaxis_title='Index', yaxis_title='Price', plot_bgcolor=theme['plot_bg'], paper_bgcolor=theme['paper_bg'], font=dict(color=theme['font']))
            st.plotly_chart(fig5, use_container_width=True)
    elif model_preference == "Gradient Boosting":
        with st.spinner('Training Gradient Boosting model...'):
            model_gbm = GradientBoostingRegressor()
            model_gbm.fit(x_train.reshape(x_train.shape[0], -1), y_train)
            y_pred_gbm_train = model_gbm.predict(x_train.reshape(x_train.shape[0], -1))
            y_pred_gbm_test = model_gbm.predict(x_test.reshape(x_test.shape[0], -1))
            st.subheader('Gradient Boosting Model Metrics')
            st.write(f"R² (Train): {r2_score(y_train, y_pred_gbm_train):.4f} | R² (Test): {r2_score(y_test, y_pred_gbm_test):.4f}")
            st.write(f"RMSE (Train): {np.sqrt(mean_squared_error(y_train, y_pred_gbm_train)):.4f} | RMSE (Test): {np.sqrt(mean_squared_error(y_test, y_pred_gbm_test)):.4f}")
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True', line=dict(color=theme['secondary'])))
            fig6.add_trace(go.Scatter(x=np.arange(len(y_pred_gbm_test)), y=y_pred_gbm_test, mode='lines', name='Predicted', line=dict(color=theme['primary'])))
            fig6.update_layout(title='Gradient Boosting Predictions', xaxis_title='Index', yaxis_title='Price', plot_bgcolor=theme['plot_bg'], paper_bgcolor=theme['paper_bg'], font=dict(color=theme['font']))
            st.plotly_chart(fig6, use_container_width=True)
    elif model_preference == "SVR":
        with st.spinner('Training SVR model...'):
            model_svr = SVR(kernel='rbf')
            model_svr.fit(x_train.reshape(x_train.shape[0], -1), y_train)
            y_pred_svr_train = model_svr.predict(x_train.reshape(x_train.shape[0], -1))
            y_pred_svr_test = model_svr.predict(x_test.reshape(x_test.shape[0], -1))
            st.subheader('SVR Model Metrics')
            st.write(f"R² (Train): {r2_score(y_train, y_pred_svr_train):.4f} | R² (Test): {r2_score(y_test, y_pred_svr_test):.4f}")
            st.write(f"RMSE (Train): {np.sqrt(mean_squared_error(y_train, y_pred_svr_train)):.4f} | RMSE (Test): {np.sqrt(mean_squared_error(y_test, y_pred_svr_test)):.4f}")
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True', line=dict(color=theme['secondary'])))
            fig7.add_trace(go.Scatter(x=np.arange(len(y_pred_svr_test)), y=y_pred_svr_test, mode='lines', name='Predicted', line=dict(color=theme['primary'])))
            fig7.update_layout(title='SVR Predictions', xaxis_title='Index', yaxis_title='Price', plot_bgcolor=theme['plot_bg'], paper_bgcolor=theme['paper_bg'], font=dict(color=theme['font']))
            st.plotly_chart(fig7, use_container_width=True)
    elif model_preference == "KNN":
        with st.spinner('Training KNN model...'):
            model_knn = KNeighborsRegressor()
            model_knn.fit(x_train.reshape(x_train.shape[0], -1), y_train)
            y_pred_knn_train = model_knn.predict(x_train.reshape(x_train.shape[0], -1))
            y_pred_knn_test = model_knn.predict(x_test.reshape(x_test.shape[0], -1))
            st.subheader('KNN Model Metrics')
            st.write(f"R² (Train): {r2_score(y_train, y_pred_knn_train):.4f} | R² (Test): {r2_score(y_test, y_pred_knn_test):.4f}")
            st.write(f"RMSE (Train): {np.sqrt(mean_squared_error(y_train, y_pred_knn_train)):.4f} | RMSE (Test): {np.sqrt(mean_squared_error(y_test, y_pred_knn_test)):.4f}")
            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True', line=dict(color=theme['secondary'])))
            fig8.add_trace(go.Scatter(x=np.arange(len(y_pred_knn_test)), y=y_pred_knn_test, mode='lines', name='Predicted', line=dict(color=theme['primary'])))
            fig8.update_layout(title='KNN Predictions', xaxis_title='Index', yaxis_title='Price', plot_bgcolor=theme['plot_bg'], paper_bgcolor=theme['paper_bg'], font=dict(color=theme['font']))
            st.plotly_chart(fig8, use_container_width=True)
    elif model_preference == "Logistic Regression":
        with st.spinner('Training Logistic Regression model...'):
            y_train_binary = np.where(y_train > np.mean(y_train), 1, 0)
            y_test_binary = np.where(y_test > np.mean(y_train), 1, 0)
            model_logistic = LogisticRegression()
            model_logistic.fit(x_train.reshape(x_train.shape[0], -1), y_train_binary)
            y_pred_logistic_train = model_logistic.predict(x_train.reshape(x_train.shape[0], -1))
            y_pred_logistic_test = model_logistic.predict(x_test.reshape(x_test.shape[0], -1))
            st.subheader('Logistic Regression Model Metrics')
            st.write(f"Accuracy (Train): {accuracy_score(y_train_binary, y_pred_logistic_train):.4f} | Accuracy (Test): {accuracy_score(y_test_binary, y_pred_logistic_test):.4f}")
            cm_train = confusion_matrix(y_train_binary, y_pred_logistic_train)
            cm_test = confusion_matrix(y_test_binary, y_pred_logistic_test)
            st.write('Confusion Matrix (Train)')
            st.dataframe(pd.DataFrame(cm_train), use_container_width=True)
            st.write('Confusion Matrix (Test)')
            st.dataframe(pd.DataFrame(cm_test), use_container_width=True) 