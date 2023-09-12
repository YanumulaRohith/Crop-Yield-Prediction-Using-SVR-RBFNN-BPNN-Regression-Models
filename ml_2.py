import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the data set
df = pd.read_csv("crop_yield_data.csv", encoding='unicode_escape')
df = df[:100]

# Normalize the data set
attributes = ['Rain Fall (mm)', 'Fertilizer(urea) (kg/acre)', 
              'Temperature (°C)', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
for attr in attributes:
    mean = df[attr].mean()
    std = df[attr].std()
    df[attr] = (df[attr] - mean) / std

# Display the normalized data set
print(df.head(100))

# Split the data into training and testing sets
X = df[attributes]
y = df['Yeild (Q/acre)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)

# Support Vector Regression (SVR) model
svr_model = SVR(kernel='rbf', C=15, gamma=0.1)  # Adjust C and gamma
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)
svr_r2 = r2_score(y_test, svr_predictions)
svr_mse = mean_squared_error(y_test, svr_predictions)
svr_rmse = np.sqrt(svr_mse)
print('SVR R2:', svr_r2)
print('SVR MSE:', svr_mse)
print('SVR RMSE:', svr_rmse)

# Radial Basis Function Neural Network (RBFNN) model
rbfnn_model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),  # Increase hidden units
    Dense(32, activation='relu'),  # Add another hidden layer
    Dense(1)
])
rbfnn_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))  # Adjust learning rate
rbfnn_model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)  # Increase epochs
rbfnn_predictions = rbfnn_model.predict(X_test).flatten()
rbfnn_r2 = r2_score(y_test, rbfnn_predictions)
rbfnn_mse = mean_squared_error(y_test, rbfnn_predictions)
rbfnn_rmse = np.sqrt(rbfnn_mse)
print('RBFNN R2:', rbfnn_r2)
print('RBFNN MSE:', rbfnn_mse)
print('RBFNN RMSE:', rbfnn_rmse)

# Back Propagation Neural Network (BPNN) model
bpnn_model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),  # Increase hidden units
    Dense(32, activation='relu'),  # Add another hidden layer
    Dense(1)
])
bpnn_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))  # Adjust learning rate
bpnn_model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)  # Increase epochs
bpnn_predictions = bpnn_model.predict(X_test).flatten()
bpnn_r2 = r2_score(y_test, bpnn_predictions)
bpnn_mse = mean_squared_error(y_test, bpnn_predictions)
bpnn_rmse = np.sqrt(bpnn_mse)
print('BPNN R2:', bpnn_r2)
print('BPNN MSE:', bpnn_mse)
print('BPNN RMSE:', bpnn_rmse)
