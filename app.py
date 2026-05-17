"""
Stock Price Prediction — LSTM + Attention Mechanism
Streamlit Web App
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import pickle
import json
import os

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock LSTM Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00D4AA;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.sub-header {
    font-family: 'DM Sans', sans-serif;
    color: #888;
    font-size: 1rem;
    margin-top: 4px;
    margin-bottom: 32px;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #00D4AA33;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00D4AA;
    margin-top: 4px;
}

.signal-buy  { color: #00C853; font-weight: 700; font-family: 'Space Mono', monospace; }
.signal-sell { color: #FF3D57; font-weight: 700; font-family: 'Space Mono', monospace; }
.signal-hold { color: #FFC107; font-weight: 700; font-family: 'Space Mono', monospace; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #00D4AA;
    letter-spacing: 1px;
    border-left: 3px solid #00D4AA;
    padding-left: 12px;
    margin-bottom: 16px;
}

[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #00D4AA22;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions (same logic as your Colab notebook)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'Close', 'Volume',
    'Returns', 'Log_Returns', 'HL_Ratio', 'OC_Ratio',
    'SMA_10', 'SMA_30', 'EMA_12', 'EMA_26', 'Price_SMA10_Ratio',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
    'BB_Width', 'BB_Pos', 'ATR',
    'Volume_Ratio', 'OBV'
]
TARGET_COL_IDX = FEATURE_COLS.index('Close')


@st.cache_data(show_spinner=False)
def load_and_engineer(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    df['Returns']         = df['Close'].pct_change()
    df['Log_Returns']     = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Ratio']        = (df['High'] - df['Low']) / df['Close']
    df['OC_Ratio']        = (df['Close'] - df['Open']) / df['Open']
    df['SMA_10']          = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_30']          = ta.trend.sma_indicator(df['Close'], window=30)
    df['EMA_12']          = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26']          = ta.trend.ema_indicator(df['Close'], window=26)
    df['Price_SMA10_Ratio'] = df['Close'] / df['SMA_10']
    df['RSI']             = ta.momentum.rsi(df['Close'], window=14)
    macd                  = ta.trend.MACD(df['Close'])
    df['MACD']            = macd.macd()
    df['MACD_Signal']     = macd.macd_signal()
    df['MACD_Diff']       = macd.macd_diff()
    bb                    = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_High']         = bb.bollinger_hband()
    df['BB_Low']          = bb.bollinger_lband()
    df['BB_Width']        = (df['BB_High'] - df['BB_Low']) / df['Close']
    df['BB_Pos']          = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'] + 1e-8)
    df['ATR']             = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['Volume_SMA']      = ta.trend.sma_indicator(df['Volume'], window=20)
    df['Volume_Ratio']    = df['Volume'] / (df['Volume_SMA'] + 1)
    df['OBV']             = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df.dropna(inplace=True)
    return df


def prepare_data(df, seq_len, test_split=0.15, val_split=0.15):
    data       = df[FEATURE_COLS].values
    n          = len(data)
    test_size  = int(n * test_split)
    val_size   = int(n * val_split)
    train_size = n - val_size - test_size

    train_data = data[:train_size]
    val_data   = data[train_size:train_size + val_size]
    test_data  = data[train_size + val_size:]

    scaler       = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled   = scaler.transform(val_data)
    test_scaled  = scaler.transform(test_data)

    close_scaler = MinMaxScaler()
    close_scaler.fit(train_data[:, [TARGET_COL_IDX]])

    def make_seq(d, sl):
        X, y = [], []
        for i in range(sl, len(d)):
            X.append(d[i - sl:i])
            y.append(d[i, TARGET_COL_IDX])
        return np.array(X), np.array(y)

    X_train, y_train = make_seq(train_scaled, seq_len)
    X_val,   y_val   = make_seq(val_scaled,   seq_len)
    X_test,  y_test  = make_seq(test_scaled,  seq_len)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler, close_scaler, train_size, val_size)


class AttentionLayer(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W     = layers.Dense(units, use_bias=False)
        self.V     = layers.Dense(1,     use_bias=False)

    def call(self, enc):
        score   = self.V(tf.nn.tanh(self.W(enc)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * enc, axis=1)
        return context, tf.squeeze(weights, -1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg


def build_model(seq_len, n_feat, lstm_units, dropout, lr):
    inp = keras.Input(shape=(seq_len, n_feat))
    x   = inp
    for i, u in enumerate(lstm_units):
        x = layers.LSTM(u, return_sequences=True, dropout=dropout, name=f'lstm_{i}')(x)
        if i < len(lstm_units) - 1:
            x = layers.LayerNormalization(name=f'ln_{i}')(x)
    context, _ = AttentionLayer(64, name='attention')(x)
    x = layers.Dense(64, activation='relu')(context)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss='huber', metrics=['mae'])
    return m


def generate_signals(actual, predicted, threshold=0.005):
    signals = []
    for a, p in zip(actual, predicted):
        pct = (p - a) / a
        signals.append(1 if pct > threshold else (-1 if pct < -threshold else 0))
    return np.array(signals)


def backtest(actual, signals, capital=100_000):
    portfolio, cash, shares, position = [], capital, 0, 0
    for price, sig in zip(actual, signals):
        if sig == 1 and position == 0:
            shares   = int(cash / price)
            cash    -= shares * price
            position = 1
        elif sig == -1 and position == 1:
            cash    += shares * price
            shares   = 0
            position = 0
        portfolio.append(cash + shares * price)
    if shares > 0:
        portfolio[-1] = cash + shares * actual[-1]
    return np.array(portfolio)


def forecast_future(model, last_seq, close_scaler, n_days=30):
    seq   = last_seq.copy()
    preds = []
    for _ in range(n_days):
        p = model.predict(seq[np.newaxis], verbose=0)[0, 0]
        preds.append(p)
        new_row = seq[-1].copy()
        new_row[TARGET_COL_IDX] = p
        seq = np.vstack([seq[1:], new_row])
    return close_scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ticker     = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    start_date = st.date_input("Start Date", value=pd.Timestamp("2018-01-01"))
    end_date   = st.date_input("End Date",   value=pd.Timestamp("2024-12-31"))
    seq_len    = st.slider("Lookback Window (days)", 30, 120, 60)
    epochs     = st.slider("Max Training Epochs", 20, 150, 80)
    threshold  = st.slider("Signal Threshold (%)", 0.1, 2.0, 0.5) / 100
    forecast_days = st.slider("Forecast Days", 7, 60, 30)

    st.markdown("---")
    st.markdown("**LSTM Architecture**")
    lstm_units   = st.multiselect(
        "LSTM Layer Sizes (in order)",
        [32, 64, 128, 256],
        default=[128, 64, 32]
    )
    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.3)
    lr           = st.select_slider(
        "Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}"
    )

    run_btn = st.button("🚀 Train & Predict", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("Built with TensorFlow · yfinance · Plotly")


# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">📈 Stock Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">LSTM + Attention Mechanism · Technical Indicators · Backtesting</p>', unsafe_allow_html=True)

if not run_btn:
    st.info("👈 Configure your settings in the sidebar and click **Train & Predict** to begin.")

    # Feature list preview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔧 Architecture**")
        st.markdown("- Stacked LSTM layers\n- Custom Attention layer\n- LayerNorm + Dropout\n- Huber loss")
    with col2:
        st.markdown("**📊 20 Input Features**")
        st.markdown("- RSI, MACD, Bollinger Bands\n- ATR, OBV, EMA, SMA\n- Volume ratio, Returns\n- HL/OC ratios")
    with col3:
        st.markdown("**📦 Outputs**")
        st.markdown("- Price predictions\n- Buy/Sell/Hold signals\n- Backtest vs benchmark\n- 30-day forecast")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Run pipeline
# ══════════════════════════════════════════════════════════════════════════════

if not lstm_units:
    st.error("Please select at least one LSTM layer size.")
    st.stop()

# 1. Load data
with st.spinner(f"Downloading {ticker} data & engineering features..."):
    try:
        df = load_and_engineer(ticker, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

st.success(f"✅ {ticker} — {len(df)} trading days loaded  |  {len(FEATURE_COLS)} features engineered")

# ── EDA chart ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">TECHNICAL ANALYSIS DASHBOARD</p>', unsafe_allow_html=True)

fig_eda = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=(f'{ticker} Price + Bollinger Bands', 'RSI (14)', 'MACD'),
    row_heights=[0.55, 0.22, 0.23]
)
fig_eda.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'], name='OHLC'
), row=1, col=1)
for col_name, color, name in [
    ('SMA_10','#FFA500','SMA 10'),('SMA_30','#4FC3F7','SMA 30'),
    ('BB_High','#66BB6A','BB High'),('BB_Low','#EF5350','BB Low')
]:
    fig_eda.add_trace(go.Scatter(x=df.index, y=df[col_name], line=dict(color=color,width=1), name=name, opacity=0.8), row=1, col=1)

fig_eda.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#CE93D8', width=1.5), name='RSI'), row=2, col=1)
fig_eda.add_hline(y=70, line_dash='dash', line_color='red',   opacity=0.5, row=2, col=1)
fig_eda.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5, row=2, col=1)

fig_eda.add_trace(go.Scatter(x=df.index, y=df['MACD'],        line=dict(color='#4FC3F7',width=1.2), name='MACD'),   row=3, col=1)
fig_eda.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#FF8A65',width=1.2), name='Signal'), row=3, col=1)
fig_eda.add_trace(go.Bar(x=df.index, y=df['MACD_Diff'], name='Hist', marker_color='#78909C', opacity=0.5),          row=3, col=1)

fig_eda.update_layout(height=650, template='plotly_dark', showlegend=True,
                      xaxis_rangeslider_visible=False, margin=dict(t=40,b=20))
st.plotly_chart(fig_eda, use_container_width=True)

# 2. Prepare data
with st.spinner("Preparing sequences..."):
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, close_scaler, train_size, val_size) = prepare_data(df, seq_len)

# 3. Build & Train model
st.markdown('<p class="section-title">MODEL TRAINING</p>', unsafe_allow_html=True)

model = build_model(seq_len, len(FEATURE_COLS), lstm_units, dropout_rate, lr)

progress_bar = st.progress(0)
status_text  = st.empty()
loss_chart   = st.empty()

train_losses, val_losses = [], []

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6),
]

class StreamlitCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pct = min(int((epoch + 1) / epochs * 100), 100)
        progress_bar.progress(pct)
        status_text.markdown(
            f"**Epoch {epoch+1}** — loss: `{logs.get('loss',0):.4f}` "
            f"| val_loss: `{logs.get('val_loss',0):.4f}` "
            f"| val_mae: `{logs.get('val_mae',0):.4f}`"
        )
        train_losses.append(logs.get('loss', 0))
        val_losses.append(logs.get('val_loss', 0))
        if len(train_losses) > 1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=train_losses, name='Train Loss', line=dict(color='#4FC3F7')))
            fig_loss.add_trace(go.Scatter(y=val_losses,   name='Val Loss',   line=dict(color='#EF5350')))
            fig_loss.update_layout(
                height=220, template='plotly_dark',
                margin=dict(t=20, b=20, l=20, r=20),
                title='Live Training Loss'
            )
            loss_chart.plotly_chart(fig_loss, use_container_width=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=32,
    callbacks=callbacks + [StreamlitCallback()],
    verbose=0
)
progress_bar.progress(100)
status_text.success("✅ Training complete!")

# 4. Evaluate
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred_price  = close_scaler.inverse_transform(y_pred_scaled).flatten()
y_true_price  = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae  = mean_absolute_error(y_true_price, y_pred_price)
rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
mape = np.mean(np.abs((y_true_price - y_pred_price) / (y_true_price + 1e-8))) * 100
r2   = r2_score(y_true_price, y_pred_price)

st.markdown('<p class="section-title">MODEL PERFORMANCE</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "MAE",  f"${mae:.2f}"),
    (c2, "RMSE", f"${rmse:.2f}"),
    (c3, "MAPE", f"{mape:.2f}%"),
    (c4, "R²",   f"{r2:.4f}"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)

# 5. Actual vs Predicted
st.markdown('<p class="section-title" style="margin-top:24px">ACTUAL vs PREDICTED</p>', unsafe_allow_html=True)
test_dates = df.index[train_size + val_size + seq_len:]

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=test_dates, y=y_true_price, name='Actual',    line=dict(color='#4FC3F7', width=2)))
fig_pred.add_trace(go.Scatter(x=test_dates, y=y_pred_price, name='Predicted', line=dict(color='#EF5350', width=2, dash='dot')))
fig_pred.update_layout(height=380, template='plotly_dark', hovermode='x unified', margin=dict(t=20, b=20))
st.plotly_chart(fig_pred, use_container_width=True)

# 6. Signals & Backtest
st.markdown('<p class="section-title">TRADING SIGNALS & BACKTEST</p>', unsafe_allow_html=True)

signals           = generate_signals(y_true_price, y_pred_price, threshold)
strategy_portfolio = backtest(y_true_price, signals)

initial_capital   = 100_000
shares_bh         = int(initial_capital / y_true_price[0])
bh_portfolio      = y_true_price * shares_bh + (initial_capital - shares_bh * y_true_price[0])

strategy_return   = (strategy_portfolio[-1] - initial_capital) / initial_capital * 100
bh_return         = (bh_portfolio[-1]       - initial_capital) / initial_capital * 100

col_a, col_b, col_c, col_d = st.columns(4)
col_a.markdown(f"""<div class="metric-card"><div class="metric-label">STRATEGY RETURN</div>
<div class="metric-value" style="color:{'#00C853' if strategy_return>0 else '#EF5350'}">{strategy_return:.1f}%</div></div>""",
unsafe_allow_html=True)
col_b.markdown(f"""<div class="metric-card"><div class="metric-label">BUY & HOLD RETURN</div>
<div class="metric-value">{bh_return:.1f}%</div></div>""", unsafe_allow_html=True)
col_c.markdown(f"""<div class="metric-card"><div class="metric-label">BUY SIGNALS</div>
<div class="metric-value" style="color:#00C853">{np.sum(signals==1)}</div></div>""", unsafe_allow_html=True)
col_d.markdown(f"""<div class="metric-card"><div class="metric-label">SELL SIGNALS</div>
<div class="metric-value" style="color:#EF5350">{np.sum(signals==-1)}</div></div>""", unsafe_allow_html=True)

fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Portfolio Value', 'Price with Signals'),
                        row_heights=[0.55, 0.45])
fig_bt.add_trace(go.Scatter(x=test_dates, y=strategy_portfolio, name='LSTM Strategy', line=dict(color='#00C853', width=2)), row=1, col=1)
fig_bt.add_trace(go.Scatter(x=test_dates, y=bh_portfolio,        name='Buy & Hold',   line=dict(color='#888',    width=2, dash='dot')), row=1, col=1)
fig_bt.add_hline(y=initial_capital, line_dash='dash', line_color='white', opacity=0.2, row=1, col=1)

fig_bt.add_trace(go.Scatter(x=test_dates, y=y_true_price, name='Price', line=dict(color='#4FC3F7', width=1.5)), row=2, col=1)

buy_idx  = np.where(signals == 1)[0]
sell_idx = np.where(signals == -1)[0]
if len(buy_idx):
    fig_bt.add_trace(go.Scatter(
        x=test_dates[buy_idx], y=y_true_price[buy_idx],
        mode='markers', name='BUY',
        marker=dict(color='#00C853', symbol='triangle-up', size=10)
    ), row=2, col=1)
if len(sell_idx):
    fig_bt.add_trace(go.Scatter(
        x=test_dates[sell_idx], y=y_true_price[sell_idx],
        mode='markers', name='SELL',
        marker=dict(color='#EF5350', symbol='triangle-down', size=10)
    ), row=2, col=1)

fig_bt.update_layout(height=580, template='plotly_dark', hovermode='x unified', margin=dict(t=40, b=20))
st.plotly_chart(fig_bt, use_container_width=True)

# 7. Attention Weights
st.markdown('<p class="section-title">ATTENTION WEIGHTS</p>', unsafe_allow_html=True)

try:
    lstm_out   = model.get_layer('attention').input
    _, attn_t  = model.get_layer('attention')(lstm_out)
    attn_model = Model(inputs=model.input, outputs=attn_t)
    attn_w     = attn_model.predict(X_test[0:1], verbose=0)[0]

    fig_attn = go.Figure(go.Bar(
        x=list(range(seq_len)), y=attn_w,
        marker=dict(color=attn_w, colorscale='Teal', showscale=True)
    ))
    fig_attn.update_layout(
        height=300, template='plotly_dark',
        title='Which Past Timesteps the Model Focuses On (0=oldest)',
        xaxis_title='Timestep', yaxis_title='Attention Weight',
        margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig_attn, use_container_width=True)
except Exception:
    st.info("Attention visualization skipped.")

# 8. Forecast
st.markdown(f'<p class="section-title">{forecast_days}-DAY PRICE FORECAST</p>', unsafe_allow_html=True)

future_prices = forecast_future(model, X_test[-1], close_scaler, n_days=forecast_days)
future_dates  = pd.bdate_range(start=df.index[-1], periods=forecast_days + 1)[1:]

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=df.index[-90:], y=df['Close'].values[-90:],
                             name='Historical', line=dict(color='#4FC3F7', width=2)))
fig_fc.add_trace(go.Scatter(x=future_dates, y=future_prices,
                             name='Forecast', mode='lines+markers',
                             line=dict(color='#FFA726', width=2, dash='dot'),
                             marker=dict(size=5)))
fig_fc.add_vline(x=str(df.index[-1]), line_dash='dash', line_color='white', opacity=0.3)
fig_fc.update_layout(height=380, template='plotly_dark', hovermode='x unified', margin=dict(t=20, b=20))
st.plotly_chart(fig_fc, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Last Close",      f"${df['Close'].iloc[-1]:.2f}")
col2.metric("Forecast (Day 1)", f"${future_prices[0]:.2f}", f"{(future_prices[0]/df['Close'].iloc[-1]-1)*100:.2f}%")
col3.metric(f"Forecast (Day {forecast_days})", f"${future_prices[-1]:.2f}", f"{(future_prices[-1]/df['Close'].iloc[-1]-1)*100:.2f}%")

# 9. Download metrics
st.markdown('<p class="section-title">EXPORT</p>', unsafe_allow_html=True)
metrics_dict = {
    "ticker": ticker, "mae": round(float(mae), 4),
    "rmse": round(float(rmse), 4), "mape": round(float(mape), 4),
    "r2": round(float(r2), 4),
    "strategy_return_pct": round(float(strategy_return), 2),
    "bh_return_pct": round(float(bh_return), 2),
}
st.download_button(
    "⬇️ Download Metrics JSON",
    data=json.dumps(metrics_dict, indent=2),
    file_name=f"{ticker}_metrics.json",
    mime="application/json"
)
st.json(metrics_dict)
