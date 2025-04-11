# 📈 Gold Price Prediction Using LSTM

This project uses a deep learning LSTM (Long Short-Term Memory) model to predict gold prices based on historical data from 2013 to 2023.

---

## 📚 Overview

- 📅 **Data Source**: Historical gold price data (CSV format)
- 📊 **Model Type**: LSTM (Recurrent Neural Network)
- 🔄 **Scaler**: MinMaxScaler (normalization)
- 🎯 **Loss Function**: Mean Squared Error
- 📈 **Metric**: Mean Absolute Percentage Error (MAPE)
- 📉 **Output**: Actual vs. Predicted Gold Prices

---

## 🧠 Model Architecture

- 3 stacked LSTM layers (64 units each)
- Dropout (0.2) between layers to reduce overfitting
- Dense layer for output regression
- Trained using the `Nadam` optimizer for 150 epochs

---

## 📂 Project Structure

