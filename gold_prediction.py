# 📦 Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from keras import Model
from keras.layers import Input, Dense, Dropout, LSTM

#  Load and preprocess the dataset
df = pd.read_csv(r"C:\Users\msi-pc\Downloads\Gold Price (2013-2023).csv")

# 🧹 Drop irrelevant columns
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

#  Convert 'Date' column to datetime format and sort the data chronologically
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', inplace=True)
df.reset_index(drop=True, inplace=True)

#  Clean numeric columns by removing commas and converting to float
NumCols = df.columns.drop(['Date'])
df[NumCols] = df[NumCols].replace({',': ''}, regex=True).astype('float64')

#  Check for duplicates and missing values (no output here, but helpful during development)
df.duplicated().sum()
df.isnull().sum().sum()

#  Use all 2022 data as the test set (you can modify the year as needed)
test_size = df[df.Date.dt.year == 2022].shape[0]

#  Normalize the price data using MinMaxScaler (fit only on training data ideally)
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))

#  Define time window size for sequence data
window_size = 60

#  Prepare training data (all data before test period)
train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1, 1))

x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])

#  Prepare test data (last portion of the dataset)
test_data = df.Price[-test_size - window_size:]  # include extra window_size points
test_data = scaler.transform(test_data.values.reshape(-1, 1))

x_test, y_test = [], []
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i - window_size:i, 0])
    y_test.append(test_data[i, 0])

#  Convert datasets to NumPy arrays and reshape for LSTM input: (samples, time steps, features)
x_train = np.reshape(np.array(x_train), (len(x_train), window_size, 1))
x_test = np.reshape(np.array(x_test), (len(x_test), window_size, 1))
y_train = np.reshape(np.array(y_train), (-1, 1))
y_test = np.reshape(np.array(y_test), (-1, 1))

# Define the LSTM model architecture
def define_model():
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=64, return_sequences=True)(input1)  # First LSTM layer
    x = Dropout(0.2)(x)  # Dropout for regularization
    x = LSTM(units=64, return_sequences=True)(x)  # Second LSTM layer
    x = Dropout(0.2)(x)
    x = LSTM(units=64)(x)  # Third LSTM layer (final LSTM)
    x = Dropout(0.2)(x)
    
    #  Softmax is not ideal here for regression – changing to ReLU or linear is better
    x = Dense(32, activation='relu')(x)
    dnn_output = Dense(1)(x)  # Output layer for predicting price

    #  Build and compile the model
    model = Model(inputs=input1, outputs=dnn_output)
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()
    
    return model

#  Create and train the model
model = define_model()
history = model.fit(
    x_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=1
)

#  Evaluate the model on test data
result = model.evaluate(x_test, y_test)

#  Make predictions
y_pred = model.predict(x_test)

#  Calculate accuracy metrics
MAPE = mean_absolute_percentage_error(y_test, y_pred)
Accuracy = 1 - MAPE

print("Test Loss:", result)
print("Test MAPE:", MAPE)
print("Test Accuracy:", Accuracy)

#  Inverse transform the scaled predictions back to original prices
y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred)

#  Visualize the predicted vs actual gold prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][-test_size:], y_test_true, label='Actual Price', color='blue')
plt.plot(df['Date'][-test_size:], y_test_pred, label='Predicted Price', color='red')
plt.title('Actual vs Predicted Gold Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
