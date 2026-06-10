"""
app.py — Entry point dashboard Portfolio Optimization IDX80
Struktur: 1 file utama + core.py + charts.py + requirements.txt

Cara jalankan:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config (HARUS paling atas) ──
st.set_page_config(
    page_title="PortfolioAI — IDX80 Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Global ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background utama */
.stApp { background-color: #0F1117; }
section[data-testid="stSidebar"] { background-color: #13161F; }
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }

/* Sembunyikan header & footer default Streamlit */
header[data-testid="stHeader"] { display: none; }
footer { display: none; }

/* ── KPI Card ── */
.kpi-card {
    background: linear-gradient(135deg, #1A1D27 0%, #1E2130 100%);
    border: 1px solid #2D3448;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 4px;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #38BDF8; }
.kpi-label {
    font-size: 11px;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #E2E8F0;
    line-height: 1;
}
.kpi-sub {
    font-size: 12px;
    color: #64748B;
    margin-top: 6px;
}
.kpi-up   { color: #34D399 !important; }
.kpi-down { color: #F87171 !important; }

/* ── Section header ── */
.section-title {
    font-size: 15px;
    font-weight: 600;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 14px 0 6px 0;
    border-bottom: 1px solid #1E2130;
    margin-bottom: 16px;
}

/* ── Ticker badge ── */
.badge {
    display: inline-block;
    background: #1E2D3D;
    color: #38BDF8;
    border: 1px solid #38BDF8;
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
}

/* ── Status banner ── */
.banner-info    { background:#172554; border-left:3px solid #38BDF8; border-radius:8px; padding:10px 14px; color:#BAE6FD; font-size:13px; margin:8px 0; }
.banner-success { background:#052e16; border-left:3px solid #34D399; border-radius:8px; padding:10px 14px; color:#86EFAC; font-size:13px; margin:8px 0; }
.banner-warning { background:#2d1a00; border-left:3px solid #FB923C; border-radius:8px; padding:10px 14px; color:#FED7AA; font-size:13px; margin:8px 0; }

/* ── Tombol proses ── */
.stButton > button[kind="primary"], div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0EA5E9, #38BDF8) !important;
    border: none !important;
    color: #0F1117 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    width: 100%;
}

/* ── Multiselect ── */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #1E2D3D !important;
    border: 1px solid #38BDF8 !important;
    color: #38BDF8 !important;
    border-radius: 6px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #13161F;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #1E2130;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748B;
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
    padding: 7px 16px;
}
.stTabs [aria-selected="true"] {
    background: #1E2D3D !important;
    color: #38BDF8 !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #1E2130 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background-color: #1A1D27 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}

/* ── Input fields ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background-color: #1A1D27 !important;
    border: 1px solid #2D3448 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════

def kpi(label: str, value: str, sub: str = "", up: bool = None):
    color_class = ""
    if up is True:  color_class = "kpi-up"
    if up is False: color_class = "kpi-down"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
        {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def banner(msg: str, kind: str = "info"):
    st.markdown(f'<div class="banner-{kind}">{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════

def init_state():
    defaults = {
        "page":           "home",
        "data_loaded":    False,
        "processed":      False,
        "excel_data":     None,
        "selected":       [],
        "prices_df":      None,
        "feat_dict":      {},
        "pred_dict":      {},
        "metrics_dict":   {},
        "lstm_views":     {},
        "weights":        {},
        "bl_returns":     None,
        "bl_cov":         None,
        "bl_perf":        {},
        "risk_metrics":   {},
        "backtest_result":{},
        "sim_r":          None,
        "sim_v":          None,
        "sim_s":          None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════

def render_sidebar():
    from core import IDX80_TICKERS

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 16px 0;">
            <div style="font-size:32px">📈</div>
            <div style="font-size:18px; font-weight:700; color:#E2E8F0; line-height:1.2;">PortfolioAI</div>
            <div style="font-size:11px; color:#64748B; margin-top:4px;">IDX80 — Optimizer</div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        nav_items = [
            ("🏠", "Beranda",       "home"),
            ("📊", "Overview",      "overview"),
            ("🤖", "Prediksi LSTM", "lstm"),
            ("⚙️", "Optimasi",      "optimizer"),
            ("⚠️", "Risiko",        "risk"),
            ("🔁", "Backtest",      "backtest"),
        ]

        for icon, label, key in nav_items:
            active = "background:#1E2D3D; color:#38BDF8 !important;" if st.session_state.page == key else ""
            disabled = "" if (st.session_state.processed or key in ("home", "overview")) else "opacity:0.4; pointer-events:none;"
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True,
                         disabled=(not st.session_state.processed and key not in ("home", "overview"))):
                st.session_state.page = key
                st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # Status panel
        st.markdown('<div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Status</div>', unsafe_allow_html=True)

        def dot(on): return "🟢" if on else "⚪"
        st.markdown(f"""
        <div style="font-size:12px; line-height:2;">
            {dot(st.session_state.data_loaded)} Data Excel dimuat<br>
            {dot(bool(st.session_state.selected))} Emiten dipilih ({len(st.session_state.selected)})<br>
            {dot(bool(st.session_state.pred_dict))} LSTM selesai<br>
            {dot(bool(st.session_state.weights))} Optimasi selesai
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.processed:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Emiten Aktif</div>', unsafe_allow_html=True)
            badges = "".join([f'<span class="badge">{t}</span>' for t in st.session_state.selected])
            st.markdown(badges, unsafe_allow_html=True)


render_sidebar()


# ══════════════════════════════════════════
# PAGE: BERANDA
# ══════════════════════════════════════════

def page_home():
    from core import IDX80_TICKERS, IDX80_SECTORS, load_excel, get_prices, alignment_info

    # ── Hero ──
    st.markdown("""
    <div style="padding: 40px 0 32px 0;">
        <div style="font-size:13px; color:#38BDF8; font-weight:500;
                    text-transform:uppercase; letter-spacing:.1em; margin-bottom:10px;">
            Machine Learning × Portfolio Theory
        </div>
        <h1 style="font-size:42px; font-weight:800; color:#E2E8F0;
                   line-height:1.1; margin:0 0 12px 0;">
            Optimasi Portofolio<br>
            <span style="color:#38BDF8;">Saham IDX80</span>
        </h1>
        <p style="font-size:15px; color:#94A3B8; max-width:520px; margin:0;">
            Gabungkan prediksi harga <b style="color:#E2E8F0;">LSTM</b> dengan 
            framework optimasi <b style="color:#E2E8F0;">Black-Litterman</b> 
            untuk membangun portofolio saham Indonesia yang optimal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Step guide ──
    col1, col2, col3 = st.columns(3)
    steps = [
        ("01", "Upload Data", "Upload file IDX80_5yr.xlsx yang sudah diunduh dari Google Colab."),
        ("02", "Pilih Emiten", "Pilih 2–10 saham IDX80 yang ingin dievaluasi dan dioptimasi."),
        ("03", "Proses & Analisis", "Klik Proses — LSTM dan Black-Litterman akan berjalan otomatis."),
    ]
    for col, (num, title, desc) in zip([col1, col2, col3], steps):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-color:#1E2D3D;">
                <div style="font-size:28px;font-weight:800;color:#38BDF8;opacity:.5;line-height:1;">{num}</div>
                <div style="font-size:15px;font-weight:600;color:#E2E8F0;margin:6px 0 4px 0;">{title}</div>
                <div style="font-size:13px;color:#64748B;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Upload File Data")

    uploaded = st.file_uploader(
        "Upload IDX80_5yr.xlsx",
        type=["xlsx"],
        help="File Excel dengan satu sheet per ticker (hasil download dari Colab).",
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Membaca file Excel..."):
            data = load_excel(uploaded)
        st.session_state.excel_data  = data
        st.session_state.data_loaded = True

        n_ok = len(data)
        banner(f"✅ Berhasil memuat <b>{n_ok} emiten</b> dari file Excel.", "success")

        section("Pilih Emiten (2 – 10 saham)")

        # Pilih per sektor
        sector_names = list(IDX80_SECTORS.keys())
        available_all = sorted(data.keys())

        # Tab: semua vs per sektor
        tab_all, *tab_sectors = st.tabs(["Semua"] + sector_names)

        with tab_all:
            sel = st.multiselect(
                "Pilih ticker",
                options=available_all,
                default=st.session_state.selected if st.session_state.selected else [],
                max_selections=10,
                label_visibility="collapsed",
                placeholder="Ketik atau pilih ticker...",
                key="sel_all",
            )

        sector_sel = []
        for tab, sname in zip(tab_sectors, sector_names):
            with tab:
                opts_in = [t for t in IDX80_SECTORS[sname] if t in data]
                s = st.multiselect(
                    sname, options=opts_in,
                    default=[t for t in st.session_state.selected if t in opts_in],
                    max_selections=10, label_visibility="collapsed",
                    key=f"sel_{sname}",
                )
                sector_sel.extend(s)

        # Gabung dan deduplicate
        combined = list(dict.fromkeys(sel + sector_sel))[:10]
        st.session_state.selected = combined

        if combined:
            st.markdown("<br>", unsafe_allow_html=True)
            badges = "".join([f'<span class="badge">{t}</span>' for t in combined])
            st.markdown(f'<div>Dipilih: {badges}</div>', unsafe_allow_html=True)

            # Info alignment
            max_start, late = alignment_info(data, combined)
            if late:
                late_str = ", ".join([f"{t} (IPO {d.strftime('%d %b %Y')})" for t, d in late.items()])
                banner(f"⚠️ Beberapa emiten belum listing sejak 2021: {late_str}.<br>Periode analisis akan dimulai dari <b>{max_start.strftime('%d %b %Y')}</b>.", "warning")

            prices = get_prices(data, combined)
            st.markdown(f'<div class="kpi-sub" style="margin-top:8px;">📅 Periode data: <b>{prices.index[0].strftime("%d %b %Y")}</b> — <b>{prices.index[-1].strftime("%d %b %Y")}</b> &nbsp;|&nbsp; {len(prices):,} hari trading</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if len(combined) < 2:
                banner("Pilih minimal 2 emiten untuk melanjutkan.", "warning")
            else:
                if st.button("🚀 Proses Portofolio", type="primary"):
                    run_pipeline(data, combined)

    else:
        banner("⬆️ Upload file IDX80_5yr.xlsx untuk memulai. File ini dihasilkan dari script Google Colab.", "info")


# ══════════════════════════════════════════
# PIPELINE PROSES
# ══════════════════════════════════════════

def run_pipeline(data: dict, tickers: list):
    from core import (get_prices, get_volumes, compute_features,
                      train_predict_lstm, optimize_black_litterman,
                      simulate_frontier, compute_risk, run_backtest)

    progress = st.progress(0, text="Memulai proses...")
    status   = st.empty()

    try:
        # ── 1. Load prices ──
        status.markdown('<div class="banner-info">📥 Menyiapkan data harga...</div>', unsafe_allow_html=True)
        prices_df = get_prices(data, tickers)
        volumes   = get_volumes(data, tickers)
        st.session_state.prices_df = prices_df
        progress.progress(10, text="Data harga siap")

        # ── 2. Feature engineering ──
        status.markdown('<div class="banner-info">🔧 Menghitung indikator teknikal...</div>', unsafe_allow_html=True)
        feat_dict = {}
        for t in tickers:
            vol = volumes.get(t)
            if vol is not None:
                vol = vol.reindex(prices_df.index)
            feat_dict[t] = compute_features(prices_df[t], vol)
        st.session_state.feat_dict = feat_dict
        progress.progress(25, text="Indikator teknikal selesai")

        # ── 3. LSTM per ticker ──
        pred_dict    = {}
        metrics_dict = {}
        lstm_views   = {}
        n = len(tickers)

        for i, t in enumerate(tickers):
            pct = 25 + int((i / n) * 40)
            status.markdown(f'<div class="banner-info">🤖 Training LSTM: <b>{t}</b> ({i+1}/{n})...</div>', unsafe_allow_html=True)
            progress.progress(pct, text=f"LSTM: {t}")

            feat_json = feat_dict[t].to_json(date_format="iso")
            pred_df, met, exp_ret = train_predict_lstm(t, feat_json, horizon=30)
            pred_dict[t]    = pred_df
            metrics_dict[t] = met
            lstm_views[t]   = exp_ret

        st.session_state.pred_dict    = pred_dict
        st.session_state.metrics_dict = metrics_dict
        st.session_state.lstm_views   = lstm_views
        progress.progress(65, text="LSTM selesai")

        # ── 4. Black-Litterman ──
        status.markdown('<div class="banner-info">⚙️ Menjalankan optimasi Black-Litterman...</div>', unsafe_allow_html=True)
        weights, bl_ret, bl_cov, bl_perf = optimize_black_litterman(prices_df, lstm_views)
        st.session_state.weights    = weights
        st.session_state.bl_returns = bl_ret
        st.session_state.bl_cov     = bl_cov
        st.session_state.bl_perf    = bl_perf
        progress.progress(78, text="Optimasi selesai")

        # ── 5. Efficient frontier sim ──
        status.markdown('<div class="banner-info">📐 Mensimulasikan efficient frontier...</div>', unsafe_allow_html=True)
        sim_r, sim_v, sim_s = simulate_frontier(prices_df)
        st.session_state.sim_r = sim_r
        st.session_state.sim_v = sim_v
        st.session_state.sim_s = sim_s
        progress.progress(87, text="Simulasi frontier selesai")

        # ── 6. Risk ──
        status.markdown('<div class="banner-info">⚠️ Menghitung risk metrics...</div>', unsafe_allow_html=True)
        risk = compute_risk(prices_df, weights)
        st.session_state.risk_metrics = risk
        progress.progress(93, text="Risk metrics selesai")

        # ── 7. Backtest ──
        status.markdown('<div class="banner-info">🔁 Menjalankan backtest...</div>', unsafe_allow_html=True)
        bt = run_backtest(prices_df, weights)
        st.session_state.backtest_result = bt
        progress.progress(100, text="Selesai!")

        st.session_state.processed = True
        status.markdown('<div class="banner-success">✅ Semua proses selesai! Navigasi ke tab hasil di sidebar.</div>', unsafe_allow_html=True)
        st.session_state.page = "overview"
        st.rerun()

    except Exception as e:
        status.markdown(f'<div class="banner-warning">❌ Error: {e}</div>', unsafe_allow_html=True)
        progress.empty()
        raise e


# ══════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════

def page_overview():
    from charts import price_history_chart, correlation_heatmap, stock_metrics_bar

    if not st.session_state.processed:
        banner("Belum ada data. Kembali ke Beranda dan upload file Excel, lalu tekan Proses.", "warning")
        return

    prices  = st.session_state.prices_df
    risk    = st.session_state.risk_metrics
    weights = st.session_state.weights
    perf    = st.session_state.bl_perf

    st.markdown('<h2 style="color:#E2E8F0;font-weight:700;margin:0 0 4px 0;">Overview Portofolio</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#64748B;font-size:13px;margin-bottom:24px;">Periode: {prices.index[0].strftime("%d %b %Y")} — {prices.index[-1].strftime("%d %b %Y")} &nbsp;|&nbsp; {len(prices):,} hari trading &nbsp;|&nbsp; {len(prices.columns)} emiten</div>', unsafe_allow_html=True)

    # ── KPI row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Return Tahunan",  f"{perf.get('exp_return',0)*100:.1f}%", "BL Expected", up=perf.get('exp_return',0)>0)
    with c2: kpi("Volatilitas",     f"{perf.get('volatility',0)*100:.1f}%", "Tahunan")
    with c3: kpi("Sharpe Ratio",    f"{perf.get('sharpe',0):.2f}",          "Risk-adjusted return", up=perf.get('sharpe',0)>1)
    with c4: kpi("VaR 95%",         f"{risk.get('VaR_95',0)*100:.2f}%",     "Kerugian maks harian", up=False)
    with c5: kpi("Max Drawdown",    f"{risk.get('max_drawdown',0)*100:.1f}%","Penurunan terbesar",   up=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──
    col_l, col_r = st.columns([3, 2])
    with col_l:
        section("Pergerakan Harga Historis")
        st.plotly_chart(price_history_chart(prices), use_container_width=True, config={"displayModeBar": False})

    with col_r:
        section("Matriks Korelasi")
        st.plotly_chart(correlation_heatmap(prices), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    section("Metrik per Saham")

    m1, m2 = st.columns(2)
    metric_sel = m1.selectbox("Pilih metrik", ["sharpe", "ann_return", "ann_vol", "max_dd"],
                              format_func=lambda x: {"sharpe":"Sharpe Ratio","ann_return":"Return Tahunan",
                                                     "ann_vol":"Volatilitas","max_dd":"Max Drawdown"}[x])

    sm = risk.get("stock_metrics", {})
    st.plotly_chart(stock_metrics_bar(sm, metric_sel), use_container_width=True, config={"displayModeBar": False})

    # ── Tabel ringkasan ──
    section("Ringkasan Metrik Saham")
    rows = []
    for t, m in sm.items():
        rows.append({
            "Ticker":         t,
            "Return (%)":     f"{m['ann_return']*100:.1f}",
            "Volatilitas (%)":f"{m['ann_vol']*100:.1f}",
            "Sharpe":         f"{m['sharpe']:.2f}",
            "Max DD (%)":     f"{m['max_dd']*100:.1f}",
            "Bobot BL (%)":   f"{weights.get(t,0)*100:.1f}",
        })
    df_sum = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(df_sum, use_container_width=True)


# ══════════════════════════════════════════
# PAGE: LSTM
# ══════════════════════════════════════════

def page_lstm():
    from charts import lstm_prediction_chart, technical_chart

    if not st.session_state.processed:
        banner("Proses data terlebih dahulu di halaman Beranda.", "warning")
        return

    prices       = st.session_state.prices_df
    pred_dict    = st.session_state.pred_dict
    metrics_dict = st.session_state.metrics_dict
    lstm_views   = st.session_state.lstm_views
    feat_dict    = st.session_state.feat_dict
    tickers      = st.session_state.selected

    st.markdown('<h2 style="color:#E2E8F0;font-weight:700;margin:0 0 24px 0;">Prediksi Harga — LSTM</h2>', unsafe_allow_html=True)

    # ── Pilih ticker ──
    ticker = st.selectbox("Pilih saham", tickers, key="lstm_sel")

    pred_df = pred_dict.get(ticker)
    met     = metrics_dict.get(ticker, {})
    exp_ret = lstm_views.get(ticker, 0)

    # ── KPI ──
    c1, c2, c3, c4 = st.columns(4)
    last_close = float(prices[ticker].iloc[-1])
    pred_30    = float(pred_df["pred"].iloc[-1]) if pred_df is not None else 0
    chg        = (pred_30 - last_close) / last_close if last_close else 0

    with c1: kpi("Harga Terakhir", f"Rp {last_close:,.0f}", prices.index[-1].strftime("%d %b %Y"))
    with c2: kpi("Prediksi 30 Hari", f"Rp {pred_30:,.0f}", f"{'▲' if chg>0 else '▼'} {abs(chg)*100:.1f}%", up=chg>0)
    with c3: kpi("Expected Return", f"{exp_ret*100:.1f}%", "Annualized (BL input)", up=exp_ret>0)
    with c4: kpi("RMSE", f"{met.get('RMSE',0):,.1f}", f"MAPE: {met.get('MAPE',0):.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction chart ──
    section("Grafik Prediksi")
    if pred_df is not None:
        st.plotly_chart(
            lstm_prediction_chart(prices[ticker], pred_df, ticker),
            use_container_width=True, config={"displayModeBar": False}
        )
    else:
        banner("Prediksi tidak tersedia untuk ticker ini.", "warning")

    # ── Technical chart ──
    section("Analisis Teknikal")
    feat = feat_dict.get(ticker)
    if feat is not None and len(feat) > 60:
        st.plotly_chart(
            technical_chart(feat, ticker),
            use_container_width=True, config={"displayModeBar": False}
        )

    # ── Tabel ringkasan semua ticker ──
    section("Ringkasan Prediksi Semua Saham")
    rows = []
    for t in tickers:
        p_df = pred_dict.get(t)
        last = float(prices[t].iloc[-1])
        p30  = float(p_df["pred"].iloc[-1]) if p_df is not None else 0
        chg_ = (p30 - last) / last if last else 0
        m_   = metrics_dict.get(t, {})
        rows.append({
            "Ticker":           t,
            "Harga Terakhir":   f"Rp {last:,.0f}",
            "Prediksi 30 Hari": f"Rp {p30:,.0f}",
            "Perubahan (%)":    f"{chg_*100:+.1f}%",
            "Exp. Return Ann.": f"{lstm_views.get(t,0)*100:.1f}%",
            "RMSE":             f"{m_.get('RMSE',0):,.0f}",
            "MAPE (%)":         f"{m_.get('MAPE',0):.1f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)


# ══════════════════════════════════════════
# PAGE: OPTIMIZER
# ══════════════════════════════════════════

def page_optimizer():
    from charts import efficient_frontier_chart, weights_pie_chart, correlation_heatmap

    if not st.session_state.processed:
        banner("Proses data terlebih dahulu di halaman Beranda.", "warning")
        return

    weights  = st.session_state.weights
    bl_ret   = st.session_state.bl_returns
    perf     = st.session_state.bl_perf
    sim_r    = st.session_state.sim_r
    sim_v    = st.session_state.sim_v
    sim_s    = st.session_state.sim_s
    prices   = st.session_state.prices_df
    views    = st.session_state.lstm_views

    st.markdown('<h2 style="color:#E2E8F0;font-weight:700;margin:0 0 24px 0;">Optimasi Portofolio — Black-Litterman</h2>', unsafe_allow_html=True)

    # ── KPI ──
    c1, c2, c3 = st.columns(3)
    with c1: kpi("Expected Return", f"{perf.get('exp_return',0)*100:.1f}%", "Tahunan (BL)", up=perf.get('exp_return',0)>0)
    with c2: kpi("Volatilitas",     f"{perf.get('volatility',0)*100:.1f}%", "Tahunan")
    with c3: kpi("Sharpe Ratio",    f"{perf.get('sharpe',0):.2f}",          "Max Sharpe Portfolio", up=perf.get('sharpe',0)>1)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Efficient frontier + pie ──
    col_l, col_r = st.columns([3, 2])
    with col_l:
        section("Efficient Frontier")
        if sim_r is not None:
            fig_ef = efficient_frontier_chart(
                sim_r, sim_v, sim_s,
                perf.get("exp_return", 0),
                perf.get("volatility", 0),
            )
            st.plotly_chart(fig_ef, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        section("Alokasi Bobot")
        st.plotly_chart(weights_pie_chart(weights), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabel bobot & LSTM views ──
    section("Detail Bobot & LSTM Views")
    w_df = pd.DataFrame({
        "Bobot BL (%)":      [f"{weights.get(t,0)*100:.1f}" for t in prices.columns],
        "LSTM Expected Ret": [f"{views.get(t,0)*100:.1f}%" for t in prices.columns],
        "BL Exp Return":     [f"{float(bl_ret.get(t,0))*100:.1f}%" for t in prices.columns] if bl_ret is not None else ["N/A"]*len(prices.columns),
    }, index=list(prices.columns))
    st.dataframe(w_df, use_container_width=True)

    # ── Metodologi ──
    with st.expander("📖 Metodologi Black-Litterman"):
        st.markdown("""
        **Black-Litterman** menggabungkan dua sumber informasi:

        1. **Prior (Market Equilibrium)** — return implisit dari kapitalisasi pasar (CAPM)
        2. **Investor Views** — prediksi expected return dari model LSTM

        **Formula:**
        ```
        μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
        ```
        - `π` = Market equilibrium returns
        - `Q` = Views dari LSTM (expected return tiap saham)
        - `Σ` = Covariance matrix (Ledoit-Wolf shrinkage)
        - `τ` = Uncertainty parameter (0.05)

        Output μ_BL digunakan sebagai expected return di optimizer **Max Sharpe**.
        """)


# ══════════════════════════════════════════
# PAGE: RISK
# ══════════════════════════════════════════

def page_risk():
    from charts import return_distribution_chart, drawdown_chart, stock_metrics_bar

    if not st.session_state.processed:
        banner("Proses data terlebih dahulu di halaman Beranda.", "warning")
        return

    risk = st.session_state.risk_metrics

    st.markdown('<h2 style="color:#E2E8F0;font-weight:700;margin:0 0 24px 0;">Analisis Risiko Portofolio</h2>', unsafe_allow_html=True)

    # ── KPI ──
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("VaR 95%",      f"{risk['VaR_95']*100:.2f}%",         "Value at Risk harian",     up=False)
    with c2: kpi("CVaR 95%",     f"{risk['CVaR_95']*100:.2f}%",        "Expected Shortfall",       up=False)
    with c3: kpi("Max Drawdown", f"{risk['max_drawdown']*100:.1f}%",    "Penurunan terbesar",       up=False)
    with c4: kpi("Calmar Ratio", f"{risk['calmar']:.2f}",               "Return / Max Drawdown",    up=risk['calmar']>1)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        section("Distribusi Return Harian")
        fig_dist = return_distribution_chart(
            risk["port_returns"], risk["VaR_95"], risk["CVaR_95"]
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        section("Drawdown Historis")
        fig_dd = drawdown_chart(risk["drawdown_series"], risk["dates"])
        st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    section("Risiko per Saham")

    m_sel = st.selectbox("Metrik", ["max_dd", "ann_vol", "sharpe", "ann_return"],
                         format_func=lambda x: {"max_dd":"Max Drawdown","ann_vol":"Volatilitas",
                                                "sharpe":"Sharpe","ann_return":"Return"}[x],
                         key="risk_metric_sel")
    st.plotly_chart(stock_metrics_bar(risk["stock_metrics"], m_sel),
                    use_container_width=True, config={"displayModeBar": False})

    # ── Penjelasan ──
    with st.expander("📖 Penjelasan Metrik Risiko"):
        st.markdown("""
        | Metrik | Penjelasan |
        |--------|-----------|
        | **VaR 95%** | Kerugian maksimal dalam 1 hari dengan keyakinan 95% |
        | **CVaR 95%** | Rata-rata kerugian ketika melampaui batas VaR (Expected Shortfall) |
        | **Max Drawdown** | Penurunan terbesar dari puncak ke lembah secara historis |
        | **Calmar Ratio** | Return tahunan dibagi Max Drawdown — semakin tinggi semakin baik |
        """)


# ══════════════════════════════════════════
# PAGE: BACKTEST
# ══════════════════════════════════════════

def page_backtest():
    from charts import backtest_equity_chart

    if not st.session_state.processed:
        banner("Proses data terlebih dahulu di halaman Beranda.", "warning")
        return

    bt = st.session_state.backtest_result
    if not bt:
        banner("Data backtest tidak tersedia.", "warning")
        return

    equity_df = bt["equity_df"]
    summary   = bt["summary"]
    start_dt  = bt.get("start_date")

    st.markdown('<h2 style="color:#E2E8F0;font-weight:700;margin:0 0 8px 0;">Backtest Out-of-Sample</h2>', unsafe_allow_html=True)
    if start_dt:
        st.markdown(f'<div style="color:#64748B;font-size:13px;margin-bottom:24px;">Periode evaluasi: <b>{start_dt.strftime("%d %b %Y")}</b> — {equity_df.index[-1].strftime("%d %b %Y")}</div>', unsafe_allow_html=True)

    # ── KPI per strategi ──
    for strat, met in summary.items():
        accent = "#38BDF8" if strat == "BL Optimal" else "#34D399"
        c1, c2, c3, c4, c5 = st.columns(5)
        st.markdown(f'<div style="font-size:13px;font-weight:600;color:{accent};margin:8px 0 2px 0;">{strat}</div>', unsafe_allow_html=True)
        with c1: kpi("Total Return",  f"{met['Total Return']*100:.1f}%", "", up=met["Total Return"]>0)
        with c2: kpi("Ann. Return",   f"{met['Ann. Return']*100:.1f}%",  "", up=met["Ann. Return"]>0)
        with c3: kpi("Volatilitas",   f"{met['Volatilitas']*100:.1f}%",  "")
        with c4: kpi("Sharpe",        f"{met['Sharpe']:.2f}",            "", up=met["Sharpe"]>1)
        with c5: kpi("Max Drawdown",  f"{met['Max Drawdown']*100:.1f}%", "", up=False)

    st.markdown("<br>", unsafe_allow_html=True)
    section("Equity Curve")
    st.plotly_chart(backtest_equity_chart(equity_df), use_container_width=True, config={"displayModeBar": False})

    # ── Tabel perbandingan ──
    section("Perbandingan Strategi")
    rows = []
    for strat, met in summary.items():
        rows.append({
            "Strategi":      strat,
            "Total Return":  f"{met['Total Return']*100:.1f}%",
            "Ann. Return":   f"{met['Ann. Return']*100:.1f}%",
            "Volatilitas":   f"{met['Volatilitas']*100:.1f}%",
            "Sharpe":        f"{met['Sharpe']:.2f}",
            "Max Drawdown":  f"{met['Max Drawdown']*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Strategi"), use_container_width=True)

    with st.expander("📖 Metodologi Backtest"):
        st.markdown("""
        **Walk-Forward Backtest:**
        - **Train set** (60%): periode yang digunakan untuk melatih LSTM dan menghitung bobot BL
        - **Test set** (40%): portofolio dievaluasi secara out-of-sample pada data yang tidak pernah dilihat model

        **Strategi pembanding:**
        - **BL Optimal**: bobot hasil Black-Litterman + LSTM, di-rebalance sekali di awal
        - **Equal Weight**: setiap saham mendapat bobot yang sama (1/N)
        """)


# ══════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════

PAGE_MAP = {
    "home":      page_home,
    "overview":  page_overview,
    "lstm":      page_lstm,
    "optimizer": page_optimizer,
    "risk":      page_risk,
    "backtest":  page_backtest,
}

page_fn = PAGE_MAP.get(st.session_state.page, page_home)
page_fn()
