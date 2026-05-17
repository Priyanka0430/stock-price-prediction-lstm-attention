"""
Stock Price Prediction — MLP Neural Network (sklearn)
Streamlit Web App  ·  No TensorFlow required
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock MLP Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem; font-weight: 700;
    color: #00D4AA; letter-spacing: -1px; margin-bottom: 0;
}
.sub-header {
    font-family: 'DM Sans', sans-serif;
    color: #888; font-size: 1rem;
    margin-top: 4px; margin-bottom: 32px;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #00D4AA33;
    border-radius: 12px; padding: 20px 24px; text-align: center;
}
.metric-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem;
    color: #888; letter-spacing: 2px; text-transform: uppercase;
}
.metric-value {
    font-family: 'Space Mono', monospace; font-size: 1.6rem;
    font-weight: 700; color: #00D4AA; margin-top: 4px;
}
.section-title {
    font-family: 'Space Mono', monospace; font-size: 1rem;
    color: #00D4AA; letter-spacing: 1px;
    border-left: 3px solid #00D4AA;
    padding-left: 12px; margin-bottom: 16px;
}
[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #00D4AA22;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
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


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_engineer(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    df['Returns']           = df['Close'].pct_change()
    df['Log_Returns']       = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Ratio']          = (df['High'] - df['Low']) / df['Close']
    df['OC_Ratio']          = (df['Close'] - df['Open']) / df['Open']
    df['SMA_10']            = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_30']            = ta.trend.sma_indicator(df['Close'], window=30)
    df['EMA_12']            = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26']            = ta.trend.ema_indicator(df['Close'], window=26)
    df['Price_SMA10_Ratio'] = df['Close'] / df['SMA_10']
    df['RSI']               = ta.momentum.rsi(df['Close'], window=14)
    macd                    = ta.trend.MACD(df['Close'])
    df['MACD']              = macd.macd()
    df['MACD_Signal']       = macd.macd_signal()
    df['MACD_Diff']         = macd.macd_diff()
    bb                      = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_High']           = bb.bollinger_hband()
    df['BB_Low']            = bb.bollinger_lband()
    df['BB_Width']          = (df['BB_High'] - df['BB_Low']) / df['Close']
    df['BB_Pos']            = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'] + 1e-8)
    df['ATR']               = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['Volume_SMA']        = ta.trend.sma_indicator(df['Volume'], window=20)
    df['Volume_Ratio']      = df['Volume'] / (df['Volume_SMA'] + 1)
    df['OBV']               = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df.dropna(inplace=True)
    return df


def prepare_data(df, seq_len, test_split=0.15, val_split=0.15):
    """
    Build flattened (seq_len × n_features) → Close(t) samples for MLPRegressor.
    Returns flat X arrays instead of 3-D tensors.
    """
    data       = df[FEATURE_COLS].values
    n          = len(data)
    test_size  = int(n * test_split)
    val_size   = int(n * val_split)
    train_size = n - val_size - test_size

    train_data = data[:train_size]
    val_data   = data[train_size : train_size + val_size]
    test_data  = data[train_size + val_size :]

    scaler        = MinMaxScaler()
    train_scaled  = scaler.fit_transform(train_data)
    val_scaled    = scaler.transform(val_data)
    test_scaled   = scaler.transform(test_data)

    close_scaler  = MinMaxScaler()
    close_scaler.fit(train_data[:, [TARGET_COL_IDX]])

    def make_seq(d, sl):
        X, y = [], []
        for i in range(sl, len(d)):
            # flatten window → 1-D vector
            X.append(d[i - sl : i].flatten())
            y.append(d[i, TARGET_COL_IDX])
        return np.array(X), np.array(y)

    X_train, y_train = make_seq(train_scaled, seq_len)
    X_val,   y_val   = make_seq(val_scaled,   seq_len)
    X_test,  y_test  = make_seq(test_scaled,  seq_len)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler, close_scaler, train_size, val_size, train_scaled, val_scaled, test_scaled)


def build_and_train_model(X_train, y_train, X_val, y_val, hidden_layers, max_iter, progress_bar, status_text):
    """Train MLPRegressor with live progress updates."""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1,                # we call partial_fit manually
        warm_start=True,
        random_state=42,
        early_stopping=False,      # we handle it manually
        n_iter_no_change=15,
    )

    train_losses, val_losses = [], []
    best_val_loss   = np.inf
    patience_count  = 0
    patience        = 15
    best_coefs      = None
    best_intercepts = None

    # partial_fit loop  ─ sklearn MLPRegressor supports warm_start + max_iter=1
    for epoch in range(max_iter):
        model.max_iter = epoch + 1
        model.fit(X_train, y_train)

        y_pred_tr  = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        t_loss     = mean_squared_error(y_train, y_pred_tr)
        v_loss     = mean_squared_error(y_val,   y_pred_val)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        pct = int((epoch + 1) / max_iter * 100)
        progress_bar.progress(pct)
        status_text.markdown(
            f"**Epoch {epoch+1}/{max_iter}** — "
            f"train MSE: `{t_loss:.5f}` | val MSE: `{v_loss:.5f}`"
        )

        # early stopping
        if v_loss < best_val_loss - 1e-6:
            best_val_loss   = v_loss
            patience_count  = 0
            best_coefs      = [c.copy() for c in model.coefs_]
            best_intercepts = [i.copy() for i in model.intercepts_]
        else:
            patience_count += 1
            if patience_count >= patience:
                status_text.markdown(f"**Early stop at epoch {epoch+1}** — best val MSE: `{best_val_loss:.5f}`")
                break

    # restore best weights
    if best_coefs is not None:
        model.coefs_      = best_coefs
        model.intercepts_ = best_intercepts

    return model, train_losses, val_losses


def generate_signals(actual, predicted, threshold=0.005):
    signals = []
    for a, p in zip(actual, predicted):
        pct = (p - a) / (a + 1e-8)
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


def forecast_future(model, last_flat_seq, scaler, close_scaler, seq_len, n_feat, n_days=30):
    """
    Autoregressive forecast: at each step predict the next Close,
    update the window, then repeat.
    """
    # Reshape flat → (seq_len, n_feat)
    seq  = last_flat_seq.reshape(seq_len, n_feat).copy()
    preds = []

    for _ in range(n_days):
        p = model.predict(seq.flatten().reshape(1, -1))[0]
        preds.append(p)
        new_row                    = seq[-1].copy()
        new_row[TARGET_COL_IDX]    = p
        seq                        = np.vstack([seq[1:], new_row])

    return close_scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    ticker        = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    start_date    = st.date_input("Start Date",   value=pd.Timestamp("2018-01-01"))
    end_date      = st.date_input("End Date",     value=pd.Timestamp("2024-12-31"))
    seq_len       = st.slider("Lookback Window (days)", 20, 90, 60)
    max_epochs    = st.slider("Max Training Epochs",    20, 200, 80)
    threshold     = st.slider("Signal Threshold (%)", 0.1, 2.0, 0.5) / 100
    forecast_days = st.slider("Forecast Days", 7, 60, 30)

    st.markdown("---")
    st.markdown("**MLP Architecture**")
    layer_options = {
        "Small  (64-32)":        (64, 32),
        "Medium (128-64-32)":    (128, 64, 32),
        "Large  (256-128-64)":   (256, 128, 64),
        "XLarge (256-128-64-32)":(256, 128, 64, 32),
    }
    arch_choice   = st.selectbox("Hidden Layers", list(layer_options.keys()), index=1)
    hidden_layers = layer_options[arch_choice]

    run_btn = st.button("🚀 Train & Predict", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("Built with sklearn · yfinance · Plotly")


# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">📈 Stock Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'MLP Neural Network · 20 Technical Indicators · Backtesting · Forecast'
    '</p>',
    unsafe_allow_html=True
)

if not run_btn:
    st.info("👈 Configure your settings in the sidebar and click **Train & Predict** to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔧 Architecture**")
        st.markdown("- Multi-Layer Perceptron\n- Flattened sequence input\n- ReLU activations\n- Adam optimizer")
    with col2:
        st.markdown("**📊 20 Input Features**")
        st.markdown("- RSI, MACD, Bollinger Bands\n- ATR, OBV, EMA, SMA\n- Volume ratio, Returns\n- HL/OC ratios")
    with col3:
        st.markdown("**📦 Outputs**")
        st.markdown("- Price predictions\n- Buy/Sell/Hold signals\n- Backtest vs benchmark\n- Future price forecast")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Run pipeline
# ══════════════════════════════════════════════════════════════════════════════

# 1. Load & engineer features
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
    ('SMA_10','#FFA500','SMA 10'), ('SMA_30','#4FC3F7','SMA 30'),
    ('BB_High','#66BB6A','BB High'), ('BB_Low','#EF5350','BB Low')
]:
    fig_eda.add_trace(
        go.Scatter(x=df.index, y=df[col_name],
                   line=dict(color=color, width=1), name=name, opacity=0.8),
        row=1, col=1
    )
fig_eda.add_trace(
    go.Scatter(x=df.index, y=df['RSI'],
               line=dict(color='#CE93D8', width=1.5), name='RSI'),
    row=2, col=1
)
fig_eda.add_hline(y=70, line_dash='dash', line_color='red',   opacity=0.5, row=2, col=1)
fig_eda.add_hline(y=30, line_dash='dash', line_color='green', opacity=0.5, row=2, col=1)
fig_eda.add_trace(
    go.Scatter(x=df.index, y=df['MACD'],
               line=dict(color='#4FC3F7', width=1.2), name='MACD'),
    row=3, col=1
)
fig_eda.add_trace(
    go.Scatter(x=df.index, y=df['MACD_Signal'],
               line=dict(color='#FF8A65', width=1.2), name='Signal'),
    row=3, col=1
)
fig_eda.add_trace(
    go.Bar(x=df.index, y=df['MACD_Diff'],
           name='Hist', marker_color='#78909C', opacity=0.5),
    row=3, col=1
)
fig_eda.update_layout(
    height=650, template='plotly_dark', showlegend=True,
    xaxis_rangeslider_visible=False, margin=dict(t=40, b=20)
)
st.plotly_chart(fig_eda, use_container_width=True)

# 2. Prepare data
with st.spinner("Preparing sequences..."):
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, close_scaler, train_size, val_size,
     train_scaled, val_scaled, test_scaled) = prepare_data(df, seq_len)

n_feat = len(FEATURE_COLS)

# 3. Train model
st.markdown('<p class="section-title">MODEL TRAINING</p>', unsafe_allow_html=True)
st.markdown(
    f"Architecture: **{hidden_layers}** hidden units  |  "
    f"Input size: **{seq_len} × {n_feat} = {seq_len * n_feat}** features  |  "
    f"Max epochs: **{max_epochs}**"
)

progress_bar = st.progress(0)
status_text  = st.empty()

model, train_losses, val_losses = build_and_train_model(
    X_train, y_train, X_val, y_val,
    hidden_layers, max_epochs, progress_bar, status_text
)

progress_bar.progress(100)
status_text.success("✅ Training complete!")

# Live loss chart
if len(train_losses) > 1:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=train_losses, name='Train MSE', line=dict(color='#4FC3F7')))
    fig_loss.add_trace(go.Scatter(y=val_losses,   name='Val MSE',   line=dict(color='#EF5350')))
    fig_loss.update_layout(
        height=260, template='plotly_dark',
        title='Training Loss (MSE)', margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig_loss, use_container_width=True)

# 4. Evaluate
y_pred_scaled = model.predict(X_test)
y_pred_price  = close_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
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
fig_pred.add_trace(go.Scatter(
    x=test_dates, y=y_true_price,
    name='Actual',    line=dict(color='#4FC3F7', width=2)
))
fig_pred.add_trace(go.Scatter(
    x=test_dates, y=y_pred_price,
    name='Predicted', line=dict(color='#EF5350', width=2, dash='dot')
))
fig_pred.update_layout(
    height=380, template='plotly_dark',
    hovermode='x unified', margin=dict(t=20, b=20)
)
st.plotly_chart(fig_pred, use_container_width=True)

# 6. Trading signals & backtest
st.markdown('<p class="section-title">TRADING SIGNALS & BACKTEST</p>', unsafe_allow_html=True)

signals            = generate_signals(y_true_price, y_pred_price, threshold)
strategy_portfolio = backtest(y_true_price, signals)

initial_capital   = 100_000
shares_bh         = int(initial_capital / y_true_price[0])
bh_portfolio      = (y_true_price * shares_bh
                     + (initial_capital - shares_bh * y_true_price[0]))

strategy_return   = (strategy_portfolio[-1] - initial_capital) / initial_capital * 100
bh_return         = (bh_portfolio[-1]       - initial_capital) / initial_capital * 100

col_a, col_b, col_c, col_d = st.columns(4)
col_a.markdown(f"""<div class="metric-card">
<div class="metric-label">STRATEGY RETURN</div>
<div class="metric-value" style="color:{'#00C853' if strategy_return>0 else '#EF5350'}">{strategy_return:.1f}%</div>
</div>""", unsafe_allow_html=True)
col_b.markdown(f"""<div class="metric-card">
<div class="metric-label">BUY & HOLD RETURN</div>
<div class="metric-value">{bh_return:.1f}%</div>
</div>""", unsafe_allow_html=True)
col_c.markdown(f"""<div class="metric-card">
<div class="metric-label">BUY SIGNALS</div>
<div class="metric-value" style="color:#00C853">{np.sum(signals==1)}</div>
</div>""", unsafe_allow_html=True)
col_d.markdown(f"""<div class="metric-card">
<div class="metric-label">SELL SIGNALS</div>
<div class="metric-value" style="color:#EF5350">{np.sum(signals==-1)}</div>
</div>""", unsafe_allow_html=True)

fig_bt = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=('Portfolio Value', 'Price with Signals'),
    row_heights=[0.55, 0.45]
)
fig_bt.add_trace(go.Scatter(
    x=test_dates, y=strategy_portfolio,
    name='MLP Strategy', line=dict(color='#00C853', width=2)
), row=1, col=1)
fig_bt.add_trace(go.Scatter(
    x=test_dates, y=bh_portfolio,
    name='Buy & Hold', line=dict(color='#888', width=2, dash='dot')
), row=1, col=1)
fig_bt.add_hline(y=initial_capital, line_dash='dash', line_color='white', opacity=0.2, row=1, col=1)
fig_bt.add_trace(go.Scatter(
    x=test_dates, y=y_true_price,
    name='Price', line=dict(color='#4FC3F7', width=1.5)
), row=2, col=1)

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

fig_bt.update_layout(
    height=580, template='plotly_dark',
    hovermode='x unified', margin=dict(t=40, b=20)
)
st.plotly_chart(fig_bt, use_container_width=True)

# 7. Feature importance (replaces Attention weights)
st.markdown('<p class="section-title">FEATURE IMPORTANCE (Permutation)</p>', unsafe_allow_html=True)
st.caption("How much each feature group contributes to prediction accuracy.")

with st.spinner("Computing feature importance — this may take ~30 s…"):
    # Use a small random subset to keep it fast
    rng      = np.random.RandomState(0)
    n_sample = min(500, len(X_test))
    idx      = rng.choice(len(X_test), n_sample, replace=False)
    result   = permutation_importance(
        model, X_test[idx], y_test[idx],
        n_repeats=5, random_state=0, scoring='neg_mean_squared_error'
    )

# Aggregate importance per feature (averaged across all time-steps in window)
n_feat_cols = len(FEATURE_COLS)
importance_per_feature = np.zeros(n_feat_cols)
for f in range(n_feat_cols):
    # columns f, f+n_feat_cols, f+2*n_feat_cols, ... correspond to same feature at different timesteps
    cols = list(range(f, seq_len * n_feat_cols, n_feat_cols))
    importance_per_feature[f] = result.importances_mean[cols].mean()

# Normalise to [0,1]
imp_norm = importance_per_feature - importance_per_feature.min()
if imp_norm.max() > 0:
    imp_norm = imp_norm / imp_norm.max()
sorted_idx = np.argsort(imp_norm)

fig_imp = go.Figure(go.Bar(
    x=imp_norm[sorted_idx],
    y=[FEATURE_COLS[i] for i in sorted_idx],
    orientation='h',
    marker=dict(
        color=imp_norm[sorted_idx],
        colorscale='Teal', showscale=True
    )
))
fig_imp.update_layout(
    height=480, template='plotly_dark',
    title='Feature Importance (higher = more impact on prediction)',
    xaxis_title='Normalised Importance',
    margin=dict(t=50, b=20, l=130)
)
st.plotly_chart(fig_imp, use_container_width=True)

# 8. Future forecast
st.markdown(f'<p class="section-title">{forecast_days}-DAY PRICE FORECAST</p>', unsafe_allow_html=True)

future_prices = forecast_future(
    model, X_test[-1], scaler, close_scaler,
    seq_len, n_feat, n_days=forecast_days
)
future_dates  = pd.bdate_range(start=df.index[-1], periods=forecast_days + 1)[1:]

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=df.index[-90:], y=df['Close'].values[-90:],
    name='Historical', line=dict(color='#4FC3F7', width=2)
))
fig_fc.add_trace(go.Scatter(
    x=future_dates, y=future_prices,
    name='Forecast', mode='lines+markers',
    line=dict(color='#FFA726', width=2, dash='dot'),
    marker=dict(size=5)
))
fig_fc.add_vline(x=str(df.index[-1]), line_dash='dash', line_color='white', opacity=0.3)
fig_fc.update_layout(
    height=380, template='plotly_dark',
    hovermode='x unified', margin=dict(t=20, b=20)
)
st.plotly_chart(fig_fc, use_container_width=True)

last_close = float(df['Close'].iloc[-1])
col1, col2, col3 = st.columns(3)
col1.metric("Last Close",           f"${last_close:.2f}")
col2.metric("Forecast (Day 1)",     f"${future_prices[0]:.2f}",
            f"{(future_prices[0]/last_close-1)*100:.2f}%")
col3.metric(f"Forecast (Day {forecast_days})", f"${future_prices[-1]:.2f}",
            f"{(future_prices[-1]/last_close-1)*100:.2f}%")

# 9. Export metrics
st.markdown('<p class="section-title">EXPORT</p>', unsafe_allow_html=True)
metrics_dict = {
    "ticker":               ticker,
    "mae":                  round(float(mae),  4),
    "rmse":                 round(float(rmse), 4),
    "mape":                 round(float(mape), 4),
    "r2":                   round(float(r2),   4),
    "strategy_return_pct":  round(float(strategy_return), 2),
    "bh_return_pct":        round(float(bh_return), 2),
}
st.download_button(
    "⬇️ Download Metrics JSON",
    data=json.dumps(metrics_dict, indent=2),
    file_name=f"{ticker}_metrics.json",
    mime="application/json"
)
st.json(metrics_dict)
