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

# Set page configuration
st.set_page_config(page_title="TCS Stock Price Prediction", page_icon="📊")
st.header('TCS Stock Price Predictor 📊')

# Select stock
stock_options = ['', 'TCS.NS']
stock = st.selectbox('Select Stock 🔽', stock_options)

start = '2013-01-01'
end = '2023-12-31'

# Check if a stock is selected
if stock:
    # Downloading data
    df = yf.download(stock, start, end)

    # Describing Data
    st.subheader(f'{stock} Data from 2013 to 2023 (10 years)')
    st.write(df)

    # Visualization
    st.subheader('Closing Price vs Time chart ')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
    fig.update_layout(title='Closing Price vs Time',
                      xaxis_title='Date',
                      yaxis_title='Closing Price (INR)',
                      template='plotly_white')
    st.plotly_chart(fig)

    # 100 and 200 days moving averages
    st.subheader('Closing Price vs Time chart with "100" and "200" days Moving Avg.')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))

    # Adding 100 days moving average
    ma100 = df['Close'].rolling(100).mean()
    fig2.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100 Days Moving Avg.'))

    # Adding 200 days moving average
    ma200 = df['Close'].rolling(200).mean()
    fig2.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200 Days Moving Avg.'))

    fig2.update_layout(title='Closing Price vs Time with 100 and 200 Days Moving Avg.',
                       xaxis_title='Date',
                       yaxis_title='Price (INR)',
                       template='plotly_white')
    st.plotly_chart(fig2)

    # Splitting Data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i - 100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model_path = 'my_model.keras'

    if os.path.exists(model_path):
        # Load the pre-trained model
        model = load_model(model_path)
        st.write('Pre-trained model loaded successfully! ✅')
    else:
        # Train the model
        st.write("Model not found. Training the model...")
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True,
                       input_shape=(x_train.shape[1], 1)))
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

        # Save the trained model
        model.save(model_path)
        st.write("Model trained and saved successfully! ✅")

    if 'model' in locals():
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        st.text("Loading...")

        # Perform prediction
        y_predicted = model.predict(x_test)

        # Apply scale factor to predicted prices
        scale_factor = 1 / 0.0005928
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Display completion message
        st.text("Loading is... Done! 🎉")
        # Visualization
        st.markdown('<h3 style="color: blue; text-decoration: underline red;">Prediction vs Original</h3>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=y_test, mode='lines', name='Original Price'))
        fig2.add_trace(go.Scatter(x=df.index, y=y_predicted.flatten(),line=dict(color='orange'), mode='lines', name='Predicted Price'))
        fig2.update_layout(title='Prediction vs Original',
                        xaxis_title='Time',
                        yaxis_title='Price',
                        template='plotly_white')
        st.plotly_chart(fig2)
        def loading():
            st.text("Loading...")

    # Display loading message
        loading()
        #------------------------------------------------------- Liner Regression -----------------------------------------
        # Import necessary libraries
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        # Prepare the training data
        x_train = []
        y_train = []

        for i in range(100, data_training_array.shape[0]):
            x_train.append(data_training_array[i-100:i])
            y_train.append(data_training_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Prepare the testing data
        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Create and train the Linear Regression model
        lin_reg = LinearRegression()
        lin_reg.fit(x_train.reshape(x_train.shape[0], -1), y_train)

        # Make predictions for linear regression
        lin_reg_train_pred = lin_reg.predict(x_train.reshape(x_train.shape[0], -1))
        lin_reg_test_pred = lin_reg.predict(x_test.reshape(x_test.shape[0], -1))

        # Evaluate the model

        # Print metrics
        st.markdown('<h3 style="color: Olive; text-decoration: underline red;">Linear Regression Model Metrics:</h3>', unsafe_allow_html=True)
        st.write("""
        | Metric          | Training Data | Testing Data |
        |-----------------|---------------|--------------|
        | **R^2**         | {:.4f}        | {:.4f}       |
        | **RMSE**        | {:.4f}        | {:.4f}       |
        """.format(r2_score(y_train, lin_reg_train_pred),
                r2_score(y_test, lin_reg_test_pred),
                np.sqrt(mean_squared_error(y_train, lin_reg_train_pred)),
                np.sqrt(mean_squared_error(y_test, lin_reg_test_pred))))


        # Plot predictions for Linear Regression
        # Replace the Matplotlib code with Plotly code
        st.subheader('Linear Regression Predictions')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True Values', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=np.arange(len(lin_reg_test_pred)), y=lin_reg_test_pred, mode='lines', name='Predicted Values', line=dict(color='green')))
        fig3.update_layout(xaxis_title='Data Point Index',
                        yaxis_title='Price',
                        title='Linear Regression Predictions',
                        template='plotly_white')
        st.plotly_chart(fig3)
        
        
        #-------------------------------------------------------Random Forest Regressor -----------------------------------------
        loading()
        
        from sklearn.ensemble import RandomForestRegressor

        # Create and train the Random Forest Regressor model
        model_rf = RandomForestRegressor()
        model_rf.fit(x_train.reshape(x_train.shape[0], -1), y_train)

        # Make predictions for Random Forest Regressor
        y_pred_rf_train = model_rf.predict(x_train.reshape(x_train.shape[0], -1))
        y_pred_rf_test = model_rf.predict(x_test.reshape(x_test.shape[0], -1))

        # Evaluate the model
        st.markdown('<h3 style="color: green; text-decoration: underline red;">Random Forest Regressor Model Metrics:</h3>', unsafe_allow_html=True)
        st.write("""
        | Metric          | Training Data | Testing Data |
        |-----------------|---------------|--------------|
        | **R^2**         | {:.4f}        | {:.4f}       |
        | **RMSE**        | {:.4f}        | {:.4f}       |
        """.format(r2_score(y_train, y_pred_rf_train),
                r2_score(y_test, y_pred_rf_test),
                np.sqrt(mean_squared_error(y_train, y_pred_rf_train)),
                np.sqrt(mean_squared_error(y_test, y_pred_rf_test))))


        # Plot predictions for Random Forest Regressor
        st.subheader('Random Forest Regressor Predictions')
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True Values', line=dict(color='blue')))
        fig4.add_trace(go.Scatter(x=np.arange(len(y_pred_rf_test)), y=y_pred_rf_test, mode='lines', name='Predicted Values (Random Forest)', line=dict(color='green')))
        fig4.update_layout(xaxis_title='Data Point Index',
                        yaxis_title='Price',
                        title='Random Forest Regressor Predictions',
                        template='plotly_white')
        st.plotly_chart(fig4)
        
        st.text('Loading done.')
        #------------------------------------------------------- Gradient Boosting Regressor -----------------------------------------
        loading()
        from sklearn.ensemble import GradientBoostingRegressor

        # Create and train the Gradient Boosting Regressor model
        model_gbm = GradientBoostingRegressor()
        model_gbm.fit(x_train.reshape(x_train.shape[0], -1), y_train)

        # Make predictions for Gradient Boosting Regressor
        y_pred_gbm_train = model_gbm.predict(x_train.reshape(x_train.shape[0], -1))
        y_pred_gbm_test = model_gbm.predict(x_test.reshape(x_test.shape[0], -1))

        # Evaluate the model
        st.markdown('<h3 style="color: purple; text-decoration: underline red;">Gradient Boosting Regressor Model Metrics:</h3>', unsafe_allow_html=True)
        st.write("""
        | Metric          | Training Data | Testing Data |
        |-----------------|---------------|--------------|
        | **R^2**         | {:.4f}        | {:.4f}       |
        | **RMSE**        | {:.4f}        | {:.4f}       |
        """.format(r2_score(y_train, y_pred_gbm_train),
                r2_score(y_test, y_pred_gbm_test),
                np.sqrt(mean_squared_error(y_train, y_pred_gbm_train)),
                np.sqrt(mean_squared_error(y_test, y_pred_gbm_test))))


        # Plot predictions for Gradient Boosting Regressor
        st.subheader('Gradient Boosting Regressor Predictions')
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True Values', line=dict(color='blue')))
        fig5.add_trace(go.Scatter(x=np.arange(len(y_pred_gbm_test)), y=y_pred_gbm_test, mode='lines', name='Predicted Values (Gradient Boosting)', line=dict(color='green')))
        fig5.update_layout(xaxis_title='Data Point Index',
                        yaxis_title='Price',
                        title='Gradient Boosting Regressor Predictions',
                        template='plotly_white')
        st.plotly_chart(fig5)

        #-------------------------------------------------------Support Vector Regression (SVR):-----------------------------------------
        
        from sklearn.svm import SVR

        # Create and train the SVR model
        model_svr = SVR(kernel='rbf')
        model_svr.fit(x_train.reshape(x_train.shape[0], -1), y_train)

        # Make predictions for SVR
        y_pred_svr_train = model_svr.predict(x_train.reshape(x_train.shape[0], -1))
        y_pred_svr_test = model_svr.predict(x_test.reshape(x_test.shape[0], -1))

        # Support Vector Regression Model Metrics
        st.markdown('<h3 style="color: orange; text-decoration: underline red;">Support Vector Regression Model Metrics:</h3>', unsafe_allow_html=True)
        st.write("""
        | Metric          | Training Data | Testing Data |
        |-----------------|---------------|--------------|
        | **R^2**         | {:.4f}        | {:.4f}       |
        | **RMSE**        | {:.4f}        | {:.4f}       |
        """.format(r2_score(y_train, y_pred_svr_train),
                r2_score(y_test, y_pred_svr_test),
                np.sqrt(mean_squared_error(y_train, y_pred_svr_train)),
                np.sqrt(mean_squared_error(y_test, y_pred_svr_test))))

        # Plot predictions for SVR
        st.subheader('Support Vector Regression Predictions')
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True Values', line=dict(color='blue')))
        fig6.add_trace(go.Scatter(x=np.arange(len(y_pred_svr_test)), y=y_pred_svr_test, mode='lines', name='Predicted Values (SVR)', line=dict(color='green')))
        fig6.update_layout(xaxis_title='Data Point Index',
                        yaxis_title='Price',
                        title='Support Vector Regression Predictions',
                        template='plotly_white')
        st.plotly_chart(fig6)

        #------------------------------------------------------- K-Nearest Neighbors (KNN) model -----------------------------------------
        
        from sklearn.neighbors import KNeighborsRegressor

        # Create and train the KNN model
        model_knn = KNeighborsRegressor()
        model_knn.fit(x_train.reshape(x_train.shape[0], -1), y_train)

        # Make predictions for KNN
        y_pred_knn_train = model_knn.predict(x_train.reshape(x_train.shape[0], -1))
        y_pred_knn_test = model_knn.predict(x_test.reshape(x_test.shape[0], -1))

        # Evaluate the model
        # K-Nearest Neighbors Model Metrics
        st.markdown('<h3 style="color: DarkViolet; text-decoration: underline red;">K-Nearest Neighbors Model Metrics:</h3>', unsafe_allow_html=True)
        st.write("""
        | Metric          | Training Data | Testing Data |
        |-----------------|---------------|--------------|
        | **R^2**         | {:.4f}        | {:.4f}       |
        | **RMSE**        | {:.4f}        | {:.4f}       |
        """.format(r2_score(y_train, y_pred_knn_train),
                r2_score(y_test, y_pred_knn_test),
                np.sqrt(mean_squared_error(y_train, y_pred_knn_train)),
                np.sqrt(mean_squared_error(y_test, y_pred_knn_test))))

        # Plot predictions for KNN
        st.subheader('K-Nearest Neighbors Predictions')
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True Values', line=dict(color='blue')))
        fig6.add_trace(go.Scatter(x=np.arange(len(y_pred_knn_test)), y=y_pred_knn_test, mode='lines', name='Predicted Values (KNN)', line=dict(color='green')))
        fig6.update_layout(xaxis_title='Data Point Index',
                        yaxis_title='Price',
                        title='K-Nearest Neighbors Predictions',
                        template='plotly_white')
        st.plotly_chart(fig6)

        #------------------------------------------------------- Liner Regression -----------------------------------------
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Convert target variable to binary labels
        y_train_binary = np.where(y_train > np.mean(y_train), 1, 0)
        y_test_binary = np.where(y_test > np.mean(y_train), 1, 0)

        # Create and train the Logistic Regression model
        model_logistic = LogisticRegression()
        model_logistic.fit(x_train.reshape(x_train.shape[0], -1), y_train_binary)

        # Make predictions for Logistic Regression
        y_pred_logistic_train = model_logistic.predict(x_train.reshape(x_train.shape[0], -1))
        y_pred_logistic_test = model_logistic.predict(x_test.reshape(x_test.shape[0], -1))

        # Evaluate the model
        # Logistic Regression Model Metrics
        st.markdown('<h3 style="color: magenta; text-decoration: underline red;">Logistic Regression Model Metrics:</h3>', unsafe_allow_html=True)
        st.write("""
        | Metric             | Training Data | Testing Data |
        |--------------------|---------------|--------------|
        | **Accuracy**       | {:.4f}        | {:.4f}       |
        """.format(accuracy_score(y_train_binary, y_pred_logistic_train),
                accuracy_score(y_test_binary, y_pred_logistic_test)))

        # Create confusion matrix for training set
        cm_train = confusion_matrix(y_train_binary, y_pred_logistic_train)

        # Create confusion matrix for testing set
        cm_test = confusion_matrix(y_test_binary, y_pred_logistic_test)

        # Plot confusion matrices
        st.subheader('Confusion Matrix (Training)')
        fig_train, ax_train = plt.subplots(figsize=(12, 6))
        sns.heatmap(cm_train, annot=True, cmap='Blues', fmt='g', ax=ax_train)
        ax_train.set_title('Confusion Matrix (Training)')
        ax_train.set_xlabel('Predicted')
        ax_train.set_ylabel('True')
        st.pyplot(fig_train)

        st.subheader('Confusion Matrix (Testing)')
        fig_test, ax_test = plt.subplots(figsize=(12, 6))
        sns.heatmap(cm_test, annot=True, cmap='Blues', fmt='g', ax=ax_test)
        ax_test.set_title('Confusion Matrix (Testing)')
        ax_test.set_xlabel('Predicted')
        ax_test.set_ylabel('True')
        st.pyplot(fig_test)



        
        

        
        


else:
    st.subheader("⛔ Please select a stock!!.")
