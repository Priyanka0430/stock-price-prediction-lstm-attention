# 📈 Stock Price Predictor — MLP Neural Network

A production-ready stock price prediction web app built with scikit-learn, deployed on Streamlit Cloud.

## 🔗 Live Demo
👉 [Click Here to Open App](https://stock-price-prediction-lstm-attention-priya.streamlit.app)

## 📌 What It Does
- Downloads real-time stock data using yfinance
- Engineers 20 technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, EMA, SMA)
- Trains a Multi-Layer Perceptron (MLP) neural network on historical prices
- Predicts future stock prices with a sliding window approach
- Generates Buy / Sell / Hold trading signals
- Backtests the strategy vs Buy & Hold benchmark
- Forecasts prices up to 60 days into the future
- Shows feature importance for model interpretability

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| scikit-learn | MLP Neural Network |
| yfinance | Real-time stock data |
| ta | 20 financial indicators |
| Plotly | Interactive charts |
| Streamlit | Web app & deployment |

## 📊 Model Architecture
- Input: Flattened sliding window (lookback × 20 features)
- Hidden layers: 128 → 64 → 32 (configurable)
- Activation: ReLU
- Optimizer: Adam
- Output: Next-day closing price

## 🚀 Run Locally
```bash
