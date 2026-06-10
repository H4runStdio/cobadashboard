"""
charts.py — Semua fungsi visualisasi Plotly.
Setiap fungsi menerima data dan mengembalikan go.Figure.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Design tokens ──
BG        = "#0F1117"
CARD_BG   = "#1A1D27"
GRID      = "#1E2130"
TEXT      = "#E2E8F0"
MUTED     = "#64748B"
ACCENT    = "#38BDF8"     # sky blue
GREEN     = "#34D399"
RED       = "#F87171"
ORANGE    = "#FB923C"
PURPLE    = "#A78BFA"

PALETTE   = [ACCENT, GREEN, ORANGE, PURPLE, "#F472B6", "#FBBF24",
             "#60A5FA", "#4ADE80", "#FB7185", "#C084FC"]

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID),
)


def _apply_layout(fig, **kwargs):
    fig.update_layout(**LAYOUT_BASE, **kwargs)
    return fig


# ────────────────────────────────────────────
# 1. Harga historis (multi-ticker)
# ────────────────────────────────────────────

def price_history_chart(prices_df: pd.DataFrame) -> go.Figure:
    """Line chart harga penutupan semua ticker, dinormalisasi ke 100."""
    norm = prices_df / prices_df.iloc[0] * 100
    fig  = go.Figure()
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            hovertemplate=f"<b>{col}</b><br>%{{x|%d %b %Y}}<br>Nilai: %{{y:.1f}}<extra></extra>",
        ))
    return _apply_layout(fig, title="Pergerakan Harga (Dinormalisasi, Awal = 100)")


# ────────────────────────────────────────────
# 2. Correlation heatmap
# ────────────────────────────────────────────

def correlation_heatmap(prices_df: pd.DataFrame) -> go.Figure:
    corr = prices_df.pct_change().dropna().corr().round(2)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, "#312E81"], [0.5, CARD_BG], [1, "#0E7490"]],
        zmid=0, zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Korelasi: %{z:.2f}<extra></extra>",
        colorbar=dict(tickfont=dict(color=TEXT)),
    ))
    return _apply_layout(fig, title="Matriks Korelasi Return Harian")


# ────────────────────────────────────────────
# 3. LSTM prediksi satu ticker
# ────────────────────────────────────────────

def lstm_prediction_chart(
    hist_close: pd.Series,
    pred_df: pd.DataFrame,
    ticker: str,
    lookback_days: int = 120,
) -> go.Figure:
    hist = hist_close.iloc[-lookback_days:]
    fig  = go.Figure()

    # Historis
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name="Historis",
        line=dict(color=ACCENT, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Harga: Rp %{y:,.0f}<extra></extra>",
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(pred_df.index) + list(pred_df.index[::-1]),
        y=list(pred_df["upper"]) + list(pred_df["lower"][::-1]),
        fill="toself",
        fillcolor="rgba(56,189,248,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="CI 95%",
        hoverinfo="skip",
    ))

    # Prediksi
    fig.add_trace(go.Scatter(
        x=pred_df.index, y=pred_df["pred"], name="Prediksi LSTM",
        line=dict(color=GREEN, width=2, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>Prediksi: Rp %{y:,.0f}<extra></extra>",
    ))

    # Garis pemisah sekarang
    fig.add_vline(
        x=hist.index[-1], line_dash="dash",
        line_color=MUTED, opacity=0.6,
        annotation_text="Hari ini", annotation_font_color=MUTED,
    )

    return _apply_layout(fig, title=f"Prediksi Harga {ticker} — LSTM")


# ────────────────────────────────────────────
# 4. Indikator teknikal (RSI, MACD)
# ────────────────────────────────────────────

def technical_chart(feat_df: pd.DataFrame, ticker: str) -> go.Figure:
    """Subplot: Close + Bollinger | RSI | MACD"""
    days = min(180, len(feat_df))
    df   = feat_df.iloc[-days:]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.04,
        subplot_titles=["Harga & Bollinger Bands", "RSI (14)", "MACD"],
    )

    # ── Harga + BB ──
    if "SMA_20" in df:
        sma20 = df["SMA_20"]
        std   = df["Close"].rolling(20).std()
        bb_u  = sma20 + 2 * std
        bb_l  = sma20 - 2 * std

        fig.add_trace(go.Scatter(
            x=list(df.index) + list(df.index[::-1]),
            y=list(bb_u) + list(bb_l[::-1]),
            fill="toself", fillcolor="rgba(56,189,248,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="BB Band", hoverinfo="skip",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=sma20, name="SMA 20",
            line=dict(color=ORANGE, width=1, dash="dot"), hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color=ACCENT, width=2),
        hovertemplate="%{x|%d %b %Y}<br>Rp %{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # ── RSI ──
    if "RSI_14" in df:
        rsi = df["RSI_14"]
        colors = [GREEN if v < 30 else RED if v > 70 else ACCENT for v in rsi]
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, name="RSI",
            line=dict(color=PURPLE, width=1.5),
            hovertemplate="%{y:.1f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=RED,   opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=GREEN, opacity=0.5, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ── MACD ──
    if "MACD" in df and "MACD_sig" in df:
        hist_m = df["MACD"] - df["MACD_sig"]
        bar_c  = [GREEN if v >= 0 else RED for v in hist_m]
        fig.add_trace(go.Bar(
            x=df.index, y=hist_m, name="MACD Hist",
            marker_color=bar_c, opacity=0.6, hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=ACCENT, width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_sig"], name="Signal",
            line=dict(color=ORANGE, width=1.5, dash="dot"),
        ), row=3, col=1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"Analisis Teknikal — {ticker}",
        height=550,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    return fig


# ────────────────────────────────────────────
# 5. Efficient Frontier
# ────────────────────────────────────────────

def efficient_frontier_chart(
    sim_rets: np.ndarray,
    sim_vols: np.ndarray,
    sim_sharpes: np.ndarray,
    opt_ret: float,
    opt_vol: float,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sim_vols * 100, y=sim_rets * 100,
        mode="markers",
        marker=dict(
            color=sim_sharpes, colorscale="Viridis",
            size=4, opacity=0.5,
            colorbar=dict(title="Sharpe", tickfont=dict(color=TEXT)),
        ),
        text=[f"Sharpe: {s:.2f}" for s in sim_sharpes],
        hovertemplate="Volatilitas: %{x:.1f}%<br>Return: %{y:.1f}%<br>%{text}<extra></extra>",
        name="Simulasi Portofolio",
    ))

    fig.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_ret * 100],
        mode="markers",
        marker=dict(color=ACCENT, size=16, symbol="star",
                    line=dict(color="white", width=1.5)),
        name="⭐ BL Optimal",
        hovertemplate=f"<b>Portofolio Optimal</b><br>Volatilitas: {opt_vol*100:.1f}%<br>Return: {opt_ret*100:.1f}%<extra></extra>",
    ))

    return _apply_layout(fig,
        title="Efficient Frontier — Simulasi 2.500 Portofolio",
        xaxis_title="Volatilitas Tahunan (%)",
        yaxis_title="Return Tahunan (%)",
    )


# ────────────────────────────────────────────
# 6. Pie chart bobot portofolio
# ────────────────────────────────────────────

def weights_pie_chart(weights: dict) -> go.Figure:
    filtered = {k: v for k, v in weights.items() if v > 0.001}
    labels   = list(filtered.keys())
    vals     = list(filtered.values())
    colors   = PALETTE[:len(labels)]

    fig = go.Figure(go.Pie(
        labels=labels, values=vals,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color=CARD_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Bobot: %{percent}<br>(%{value:.1%})<extra></extra>",
    ))
    return _apply_layout(fig, title="Alokasi Bobot Portofolio Optimal",
                         showlegend=False)


# ────────────────────────────────────────────
# 7. Return distribution
# ────────────────────────────────────────────

def return_distribution_chart(port_returns: np.ndarray, var: float, cvar: float) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=port_returns * 100,
        nbinsx=60,
        marker_color=ACCENT, opacity=0.7,
        name="Distribusi Return",
        hovertemplate="Return: %{x:.2f}%<br>Frekuensi: %{y}<extra></extra>",
    ))

    fig.add_vline(x=var * 100,  line_dash="dash", line_color=RED,
                  annotation_text=f"VaR 95%: {var*100:.2f}%",
                  annotation_font_color=RED, annotation_position="top right")
    fig.add_vline(x=cvar * 100, line_dash="dash", line_color=ORANGE,
                  annotation_text=f"CVaR 95%: {cvar*100:.2f}%",
                  annotation_font_color=ORANGE, annotation_position="top right")

    return _apply_layout(fig,
        title="Distribusi Return Harian Portofolio",
        xaxis_title="Return Harian (%)",
        yaxis_title="Frekuensi",
    )


# ────────────────────────────────────────────
# 8. Drawdown chart
# ────────────────────────────────────────────

def drawdown_chart(drawdown_series: np.ndarray, dates) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=drawdown_series * 100,
        fill="tozeroy",
        fillcolor="rgba(248,113,113,0.15)",
        line=dict(color=RED, width=1.5),
        name="Drawdown",
        hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    return _apply_layout(fig,
        title="Drawdown Historis Portofolio",
        yaxis_title="Drawdown (%)",
    )


# ────────────────────────────────────────────
# 9. Backtest equity curve
# ────────────────────────────────────────────

def backtest_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    colors = {
        "BL Optimal":   ACCENT,
        "Equal Weight": GREEN,
        "IHSG":         MUTED,
    }
    fig = go.Figure()
    for col in equity_df.columns:
        fig.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df[col] * 100,
            name=col,
            line=dict(color=colors.get(col, ORANGE), width=2),
            hovertemplate=f"<b>{col}</b><br>%{{x|%d %b %Y}}<br>Return: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED, opacity=0.4)
    return _apply_layout(fig,
        title="Equity Curve — Out-of-Sample Backtest",
        yaxis_title="Cumulative Return (%)",
    )


# ────────────────────────────────────────────
# 10. Bar chart metrik per saham
# ────────────────────────────────────────────

def stock_metrics_bar(stock_metrics: dict, metric: str = "sharpe") -> go.Figure:
    tickers = list(stock_metrics.keys())
    values  = [stock_metrics[t].get(metric, 0) for t in tickers]
    colors  = [GREEN if v > 0 else RED for v in values]

    label_map = {
        "sharpe":     "Sharpe Ratio",
        "ann_return": "Return Tahunan (%)",
        "ann_vol":    "Volatilitas Tahunan (%)",
        "max_dd":     "Max Drawdown (%)",
    }
    scale = 100 if metric in ("ann_return", "ann_vol", "max_dd") else 1

    fig = go.Figure(go.Bar(
        x=tickers, y=[v * scale for v in values],
        marker_color=colors,
        hovertemplate=f"<b>%{{x}}</b><br>{label_map.get(metric, metric)}: %{{y:.2f}}<extra></extra>",
    ))
    return _apply_layout(fig, title=f"{label_map.get(metric, metric)} per Saham",
                         yaxis_title=label_map.get(metric, metric))
