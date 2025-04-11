# Importing necessary libraries for data manipulation, visualization, and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM

# Loading the dataset
df = pd.read_csv(r"C:\gold_price_prediction\Gold Price (2013-2023).csv")

# Dropping unnecessary columns (Volume and Change %) as they won't be used in the model
df.drop(['Vol.','Change %'], axis=1, inplace=True)

# Converting the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sorting the dataset by date to ensure the chronological order
df.sort_values(by='Date', inplace=True)

# Resetting the index after sorting
df.reset_index(drop=True, inplace=True)

# Removing commas from numerical columns and converting the columns to float for analysis
NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
df[NumCols] = df[NumCols].astype('float64')

# Checking for duplicate rows and missing values
df.duplicated().sum()  # Checking for duplicates
df.isnull().sum().sum()  # Checking for missing values

# Defining test size using data from 2022
test_size = df[df.Date.dt.year == 2022].shape[0]  # Using data from 2022 as the test set

# Scaling the 'Price' column to normalize the values between 0 and 1
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))

window_size = 60  # Setting the window size for the LSTM model

# Preparing the training data (all data before 2022)
train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1, 1))

# Creating the x_train and y_train datasets with a sliding window of 'window_size'
x_train = []
y_train = []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-60:i, 0])  # Creating the input features (last 60 data points)
    y_train.append(train_data[i, 0])  # The target variable (the next data point)

# Preparing the test data (the last 'test_size' data points, plus 60 data points for windowing)
test_data = df.Price[-test_size-60:]
test_data = scaler.transform(test_data.values.reshape(-1,1))

# Creating the x_test and y_test datasets with a sliding window of 'window_size'
x_test = []
y_test = []
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])

# Converting the data lists into NumPy arrays for LSTM input
x_train = np.array(x_train)
x_test  = np.array(x_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)

# Reshaping the input data to be compatible with LSTM layer (3D input for LSTM)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test  = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Reshaping the output data to match the model's output
y_train = np.reshape(y_train, (-1,1))
y_test  = np.reshape(y_test, (-1,1))

# Function to define the LSTM model
def define_model():
    # Input layer with shape based on the window size (60 days)
    input1 = Input(shape=(window_size,1))
    
    # Three LSTM layers with dropout for regularization
    x = LSTM(units = 64, return_sequences=True)(input1)  
    x = Dropout(0.2)(x)
    x = LSTM(units = 64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units = 64)(x)
    x = Dropout(0.2)(x)
    
    # Dense layers for final output prediction
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)  # Output layer with one node for the predicted price

    # Compiling the model with the Adam optimizer and mean squared error loss function
    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    
    # Printing the model summary to check the architecture
    model.summary()
    
    return model

# Initializing the model
model = define_model()

# Training the model with 150 epochs and a batch size of 32
history = model.fit(x_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)

# Evaluating the model on the test data
result = model.evaluate(x_test, y_test)

# Making predictions using the trained model
y_pred = model.predict(x_test)

# Calculating the Mean Absolute Percentage Error (MAPE) for model performance
MAPE = mean_absolute_percentage_error(y_test, y_pred)
Accuracy = 1 - MAPE  # Converting MAPE to accuracy

# Printing the evaluation results
print("Test Loss:", result)
print("Test MAPE:", MAPE)
print("Test Accuracy:", Accuracy)

# Inversely transforming the predicted and actual values to original scale
y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred)

# Plotting the actual vs predicted gold prices over time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-test_size:], y_test_true, label='Actual Price', color='blue')
plt.plot(df['Date'][-test_size:], y_test_pred, label='Predicted Price', color='red')
plt.title('Actual vs Predicted Gold Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
