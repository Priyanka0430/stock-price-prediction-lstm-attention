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
git clone https://github.com/priyanka0430/stock-price-prediction-lstm-attention
cd stock-price-prediction-lstm-attention
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## ✨ Key Features
- Works with any stock ticker worldwide (AAPL, TSLA, TCS.NS, RELIANCE.NS)
- Configurable lookback window, epochs, and architecture from sidebar
- Real-time training progress with live loss curve
- Downloadable metrics as JSON

## 👩‍💻 Author
**Priyanka** — [LinkedIn]([https://linkedin.com/in/YOUR-LINKEDIN-HERE](https://www.linkedin.com/in/priyanka-t-s-8704a22a1))
