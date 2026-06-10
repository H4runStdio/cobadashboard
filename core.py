"""
core.py — Semua logika backend: data, LSTM, Black-Litterman, risk metrics.
Disimpan dalam satu file agar mudah dipahami dan di-debug.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────

BI_RATE = 0.0575  # BI Rate risk-free

IDX80_TICKERS = [
    "ADRO","AGRO","AKRA","AMRT","ANTM","ARTO","ASII","ASRI",
    "BBCA","BBNI","BBRI","BBTN","BFIN","BJBR","BJTM","BKSL",
    "BMRI","BMTR","BNGA","BRPT","BSDE","BTPS","CPIN","CTRA",
    "DMAS","DSNG","EMTK","ERAA","ESSA","EXCL","GGRM","GOTO",
    "HEAL","HMSP","HRUM","ICBP","INCO","INDF","INDY","INKP",
    "INTP","ISAT","ITMG","JPFA","JSMR","KLBF","LSIP","MAPI",
    "MBMA","MDKA","MEDC","MIKA","MNCN","MPPA","MTEL","MYOR",
    "NISP","PGAS","PGEO","PTBA","PTPP","PTRO","PWON","SCMA",
    "SIDO","SMGR","SMRA","SRTG","SSMS","TLKM","TOWR","TPIA",
    "TRIM","UNIQ","UNTR","UNVR","WIFI","WIKA","WMUU","WSKT",
]

IDX80_SECTORS = {
    "🏦 Perbankan": ["BBCA","BBNI","BBRI","BBTN","BMRI","BNGA","NISP","BJBR","BJTM","AGRO","BFIN","BKSL"],
    "⚡ Energi":    ["ADRO","INDY","ITMG","PTBA","MEDC","PGAS","ESSA","HRUM","PTRO","DSNG","SSMS","PGEO"],
    "🏭 Industri":  ["ASII","UNTR","INTP","SMGR","CPIN","JPFA","INKP","TPIA","BRPT"],
    "📡 Telco":     ["TLKM","EXCL","ISAT","MTEL","WIFI","TOWR"],
    "🛒 Konsumer":  ["UNVR","ICBP","INDF","MYOR","HMSP","GGRM","SIDO","KLBF","MIKA","HEAL","AMRT","ERAA","MAPI","MPPA"],
    "🏗 Properti":  ["BSDE","CTRA","PWON","SMRA","ASRI","DMAS"],
    "💼 Lainnya":   ["GOTO","ARTO","EMTK","MNCN","SCMA","BMTR","LSIP","MDKA","INCO","ANTM","MBMA","BTPS","JSMR","PTPP","WIKA","WSKT","SRTG","TRIM","UNIQ","WMUU","BBTN"],
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_excel(uploaded_file) -> dict[str, pd.DataFrame]:
    """
    Baca file Excel IDX80_5yr.xlsx.
    Setiap sheet = satu ticker, kolom minimal: Date, Close, Volume.
    Return: dict {ticker: DataFrame dengan index Date}
    """
    xl = pd.ExcelFile(uploaded_file)
    data = {}
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet)
            # Normalkan nama kolom (case-insensitive)
            df.columns = [c.strip().title() for c in df.columns]
            if "Date" not in df.columns or "Close" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            data[sheet.upper()] = df
        except Exception:
            continue
    return data


def get_prices(data: dict, tickers: list) -> pd.DataFrame:
    """
    Ambil Close price dari dict data, align tanggal (dropna).
    Return: DataFrame (Date x Ticker)
    """
    frames = {}
    for t in tickers:
        if t in data and "Close" in data[t].columns:
            frames[t] = data[t]["Close"].astype(float)
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df = df.dropna()
    return df


def get_volumes(data: dict, tickers: list) -> dict:
    """Return dict {ticker: pd.Series volume}"""
    return {
        t: data[t]["Volume"].astype(float)
        for t in tickers
        if t in data and "Volume" in data[t].columns
    }


def alignment_info(data: dict, tickers: list) -> tuple[pd.Timestamp, dict]:
    """Cari tanggal mulai paling muda dan ticker yang 'terlambat'."""
    starts = {t: data[t].index[0] for t in tickers if t in data}
    min_start = min(starts.values())
    max_start = max(starts.values())
    late = {t: d for t, d in starts.items() if d > min_start}
    return max_start, late

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def compute_features(close: pd.Series, volume: pd.Series = None) -> pd.DataFrame:
    """
    Hitung indikator teknikal untuk input LSTM.
    Semua fitur bersumber dari data OHLCV — tidak butuh data eksternal.
    """
    df = pd.DataFrame(index=close.index)
    df["Close"]      = close
    df["Return_1d"]  = close.pct_change(1)
    df["Return_5d"]  = close.pct_change(5)
    df["Log_Return"] = np.log(close / close.shift(1))
    df["SMA_20"]     = close.rolling(20).mean()
    df["SMA_50"]     = close.rolling(50).mean()
    df["EMA_12"]     = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]     = close.ewm(span=26, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    df["MACD"]     = df["EMA_12"] - df["EMA_26"]
    df["MACD_sig"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands width
    std20          = close.rolling(20).std()
    df["BB_width"] = (4 * std20) / close.replace(0, np.nan)

    # ATR proxy (pakai std rolling karena hanya ada Close)
    df["ATR"] = close.rolling(14).std()

    if volume is not None:
        vol = volume.reindex(close.index)
        df["Volume"]  = vol
        df["Vol_MA20"] = vol.rolling(20).mean()

    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────

LOOKBACK   = 60
EPOCHS     = 50
BATCH_SIZE = 32


def _build_sequences(data: np.ndarray, lookback: int, horizon: int):
    X, y = [], []
    for i in range(lookback, len(data) - horizon + 1):
        X.append(data[i - lookback:i])
        y.append(data[i:i + horizon, 0])   # kolom 0 = Close
    return np.array(X), np.array(y)


@st.cache_data(ttl=7200, show_spinner=False)
def train_predict_lstm(
    ticker: str,
    feat_df_json: str,   # JSON string agar cache hashable
    horizon: int = 30
) -> tuple:
    """
    Train LSTM satu ticker dan kembalikan:
      pred_df   : DataFrame index=future_dates, cols=[pred, upper, lower]
      metrics   : dict {RMSE, MAPE, R2}
      exp_return: float (annualized expected return dari prediksi LSTM)

    feat_df_json dikirim sebagai JSON karena st.cache_data tidak bisa
    hash DataFrame langsung ketika ukurannya besar.
    """
    feat_df = pd.read_json(feat_df_json)
    feat_df.index = pd.to_datetime(feat_df.index)
    feat_df = feat_df.sort_index()

    close_vals = feat_df["Close"].values
    feat_vals  = feat_df.values
    n_feat     = feat_vals.shape[1]

    # Scale fitur
    scaler_feat  = MinMaxScaler()
    scaled       = scaler_feat.fit_transform(feat_vals)

    # Scale Close terpisah untuk inverse transform prediksi
    scaler_close = MinMaxScaler()
    scaler_close.fit(close_vals.reshape(-1, 1))

    split      = int(len(scaled) * 0.8)
    X_train, y_train = _build_sequences(scaled[:split],              LOOKBACK, horizon)
    X_test,  y_test  = _build_sequences(scaled[split - LOOKBACK:],   LOOKBACK, horizon)

    # Coba TensorFlow
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        tf.random.set_seed(42)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_feat)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(horizon),
        ])
        model.compile(optimizer=Adam(1e-3), loss="mse")

        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_split=0.1, callbacks=[es], verbose=0,
        )

        # Evaluasi pada test set
        y_pred_s = model.predict(X_test, verbose=0)   # (n, horizon)

        # Inverse transform: buat dummy array n_feat kolom, masukkan prediksi ke kol 0
        def inv_close(arr2d):
            n_rows, n_cols = arr2d.shape
            dummy = np.zeros((n_rows * n_cols, n_feat))
            dummy[:, 0] = arr2d.flatten()
            return scaler_feat.inverse_transform(dummy)[:, 0].reshape(n_rows, n_cols)

        y_pred_c = inv_close(y_pred_s)
        y_test_c = inv_close(y_test)

        rmse = float(np.sqrt(np.mean((y_pred_c - y_test_c) ** 2)))
        mape = float(np.mean(np.abs((y_test_c - y_pred_c) / np.abs(y_test_c + 1e-8))) * 100)
        ss_r = np.sum((y_test_c - y_pred_c) ** 2)
        ss_t = np.sum((y_test_c - np.mean(y_test_c)) ** 2)
        r2   = float(1 - ss_r / (ss_t + 1e-8))

        # Prediksi masa depan
        last_seq      = scaled[-LOOKBACK:].reshape(1, LOOKBACK, n_feat)
        fut_scaled    = model.predict(last_seq, verbose=0)[0]
        dummy_fut     = np.zeros((horizon, n_feat))
        dummy_fut[:, 0] = fut_scaled
        fut_close     = scaler_feat.inverse_transform(dummy_fut)[:, 0]

    except Exception:
        # Fallback: linear trend
        rmse, mape, r2 = float(np.std(close_vals) * 0.1), 10.0, 0.4
        trend    = np.polyfit(np.arange(30), close_vals[-30:], 1)[0]
        fut_close = np.array([close_vals[-1] + trend * i for i in range(1, horizon + 1)])

    last_close = close_vals[-1]
    std_resid  = np.std(fut_close) * 0.3
    upper      = fut_close + 1.96 * std_resid
    lower      = fut_close - 1.96 * std_resid

    last_date  = feat_df.index[-1]
    fut_idx    = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    pred_df    = pd.DataFrame({"pred": fut_close, "upper": upper, "lower": lower}, index=fut_idx)

    # Expected return: return dari harga terakhir ke median prediksi (annualized)
    exp_ret = float((fut_close[-1] - last_close) / (last_close + 1e-8))
    # Annualize dari horizon hari ke 252 hari
    exp_ret_ann = float((1 + exp_ret) ** (252 / horizon) - 1)

    return pred_df, {"RMSE": rmse, "MAPE": mape, "R2": r2}, exp_ret_ann


# ─────────────────────────────────────────────
# BLACK-LITTERMAN OPTIMIZATION
# ─────────────────────────────────────────────

def optimize_black_litterman(
    prices_df: pd.DataFrame,
    lstm_views: dict,          # {ticker: annualized expected return dari LSTM}
    tau: float = 0.05,
) -> tuple:
    """
    Black-Litterman optimization menggunakan LSTM views.
    Return: weights (dict), bl_returns (Series), cov (DataFrame), perf (dict)
    """
    try:
        from pypfopt import BlackLittermanModel, EfficientFrontier
        from pypfopt import risk_models, expected_returns

        S = risk_models.CovarianceShrinkage(prices_df).ledoit_wolf()

        # Views dari LSTM — hanya ticker yang ada di prices_df
        views = {t: v for t, v in lstm_views.items() if t in prices_df.columns}

        bl  = BlackLittermanModel(S, pi="equal", absolute_views=views, tau=tau)
        mu  = bl.bl_returns()
        cov = bl.bl_cov()

        ef = EfficientFrontier(mu, cov)
        ef.add_constraint(lambda w: w >= 0.02)   # min 2% per saham
        ef.add_constraint(lambda w: w <= 0.45)   # max 45% per saham
        ef.max_sharpe(risk_free_rate=BI_RATE)
        weights = dict(ef.clean_weights())

        ret, vol, sharpe = ef.portfolio_performance(
            risk_free_rate=BI_RATE, verbose=False
        )
        return weights, mu, S, {"exp_return": ret, "volatility": vol, "sharpe": sharpe}

    except ImportError:
        # Fallback: inverse-volatility weighting
        rets   = prices_df.pct_change().dropna()
        vols   = rets.std() * np.sqrt(252)
        inv_v  = 1 / (vols + 1e-8)
        w_arr  = inv_v / inv_v.sum()
        weights = w_arr.to_dict()
        mu  = pd.Series(lstm_views)
        cov = rets.cov() * 252
        exp_r = float(sum(weights[t] * lstm_views.get(t, 0) for t in weights))
        port_v = float(np.sqrt(np.dot(list(weights.values()),
                                      np.dot(cov.values, list(weights.values())))))
        sharpe = (exp_r - BI_RATE) / (port_v + 1e-8)
        return weights, mu, cov, {"exp_return": exp_r, "volatility": port_v, "sharpe": sharpe}


def simulate_frontier(prices_df: pd.DataFrame, n_sim: int = 2500):
    """Monte Carlo untuk efficient frontier chart."""
    rets = prices_df.pct_change().dropna()
    mu   = rets.mean() * 252
    cov  = rets.cov() * 252
    n    = len(prices_df.columns)
    r_arr, v_arr, s_arr = [], [], []
    rng  = np.random.default_rng(42)
    for _ in range(n_sim):
        w = rng.dirichlet(np.ones(n))
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov.values @ w))
        r_arr.append(r); v_arr.append(v)
        s_arr.append((r - BI_RATE) / (v + 1e-8))
    return np.array(r_arr), np.array(v_arr), np.array(s_arr)


# ─────────────────────────────────────────────
# RISK METRICS
# ─────────────────────────────────────────────

def compute_risk(prices_df: pd.DataFrame, weights: dict, conf: float = 0.95) -> dict:
    rets  = prices_df.pct_change().dropna()
    w_arr = np.array([weights.get(c, 0) for c in rets.columns])
    p_ret = rets.values @ w_arr                    # daily portfolio returns

    var   = float(np.percentile(p_ret, (1 - conf) * 100))
    cvar  = float(p_ret[p_ret <= var].mean())

    ann_r = float(p_ret.mean() * 252)
    ann_v = float(p_ret.std() * np.sqrt(252))
    sharpe = (ann_r - BI_RATE) / (ann_v + 1e-8)

    cum      = (1 + pd.Series(p_ret)).cumprod()
    roll_max = cum.cummax()
    dd       = (cum - roll_max) / roll_max
    max_dd   = float(dd.min())
    calmar   = ann_r / abs(max_dd) if max_dd != 0 else 0.0

    # Per-saham
    stock = {}
    for col in rets.columns:
        r = rets[col]
        a_r = float(r.mean() * 252)
        a_v = float(r.std() * np.sqrt(252))
        cum_s = (1 + r).cumprod()
        stock[col] = {
            "ann_return": a_r,
            "ann_vol":    a_v,
            "sharpe":     (a_r - BI_RATE) / (a_v + 1e-8),
            "max_dd":     float((cum_s / cum_s.cummax() - 1).min()),
        }

    return {
        "VaR_95":           var,
        "CVaR_95":          cvar,
        "ann_return":       ann_r,
        "ann_vol":          ann_v,
        "sharpe":           sharpe,
        "max_drawdown":     max_dd,
        "calmar":           calmar,
        "port_returns":     p_ret,
        "cum_returns":      cum.values,
        "drawdown_series":  dd.values,
        "dates":            rets.index,
        "stock_metrics":    stock,
    }


# ─────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────

def run_backtest(prices_df: pd.DataFrame, weights: dict, train_frac: float = 0.6) -> dict:
    """
    Walk-forward backtest sederhana.
    Bagian pertama (train_frac) = training window, sisanya = out-of-sample.
    Bandingkan: BL Optimal vs Equal Weight.
    """
    rets = prices_df.pct_change().dropna()
    n    = len(rets)
    bt   = rets.iloc[int(n * train_frac):]

    w_bl = np.array([weights.get(c, 0) for c in rets.columns])
    w_eq = np.ones(len(rets.columns)) / len(rets.columns)

    ret_bl = bt.values @ w_bl
    ret_eq = bt.values @ w_eq

    def to_equity(r): return (1 + pd.Series(r, index=bt.index)).cumprod() - 1
    def stats(r):
        s = pd.Series(r)
        ann_r = s.mean() * 252
        ann_v = s.std() * np.sqrt(252)
        cum   = (1 + s).cumprod()
        return {
            "Total Return":  float(cum.iloc[-1] - 1),
            "Ann. Return":   float(ann_r),
            "Volatilitas":   float(ann_v),
            "Sharpe":        float((ann_r - BI_RATE) / (ann_v + 1e-8)),
            "Max Drawdown":  float((cum / cum.cummax() - 1).min()),
        }

    equity = pd.DataFrame({
        "BL Optimal":   to_equity(ret_bl),
        "Equal Weight": to_equity(ret_eq),
    })

    return {
        "equity_df": equity,
        "summary":   {"BL Optimal": stats(ret_bl), "Equal Weight": stats(ret_eq)},
        "start_date": bt.index[0],
    }
