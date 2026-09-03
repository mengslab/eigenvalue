
from __future__ import annotations
import io
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.data_validator import prepare_time_series_dataframe, validation_summary_text
from src.rhythm_methods import METHOD_ORDER, analyze_selected_methods, compare_methods_narrative
from src.method_recommender import recommend_methods
from src.stats_analysis import compare_groups, build_stats_summary
from src.figure_suite import (
    apply_pub_layout,
    figure_download_bundle,
    make_publication_figures,
    make_group_comparison_figure,
    build_demo_signal_gallery,
)

st.set_page_config(page_title="Eigenvalue Analysis v14.4.1.1", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
APP_VERSION = "v14.4.1"

CUSTOM_CSS = '''
<style>
:root {
  --ink:#1f2a44; --muted:#60708a; --card:#ffffff; --border:#dbe4f0;
  --accent:#2f6fed; --accent2:#12a594; --accent3:#7a4ff3; --shadow:0 10px 28px rgba(31,42,68,0.08);
}
.stApp { background: radial-gradient(circle at top left, #f8fbff, #eef7f3 60%, #f9fbff 100%); }
.block-container { max-width: 1560px; padding-top:1rem; padding-bottom:2rem; }
.hero { padding:1.15rem 1.3rem; border-radius:24px; background:linear-gradient(135deg, rgba(47,111,237,.14), rgba(18,165,148,.14), rgba(122,79,243,.10)); border:1px solid var(--border); box-shadow:var(--shadow); margin-bottom:1rem; }
.hero-title { font-size:2.2rem; font-weight:800; color:var(--ink); margin:0 0 .2rem 0; }
.hero-sub { color:var(--muted); margin:0; }
.small-chip { display:inline-block; padding:.2rem .55rem; border-radius:999px; background:#eef4ff; border:1px solid var(--border); color:var(--accent); font-size:.78rem; margin-right:.35rem; }
.callout, .info-card { background:var(--card); border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); padding:.9rem 1rem; }
.section-note { color:var(--muted); font-size:.92rem; margin-top:-.25rem; margin-bottom:.7rem; }
.info-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin:.65rem 0 1rem 0; }
.kpi-label { font-size:.8rem; color:var(--muted); } .kpi-value { font-size:1.05rem; font-weight:700; color:var(--ink); }
div[data-testid="stMetric"], div[data-testid="stDataFrame"], div[data-testid="stPlotlyChart"] { background:var(--card); border:1px solid var(--border); border-radius:18px; box-shadow:var(--shadow); padding:6px; }
div[data-testid="stButton"] > button[kind="primary"] { border-radius:14px; font-weight:700; }
div[data-testid="stButton"] > button { min-height: 2.75rem; }
div[data-testid="stSidebar"] { background:linear-gradient(180deg, #f7f9fc 0%, #f4f8f7 100%); border-right:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { border-radius:999px; padding:.45rem .9rem; background:#f5f8ff; border:1px solid var(--border); }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg, rgba(47,111,237,.10), rgba(18,165,148,.10)) !important; color:var(--ink) !important; }
</style>
'''
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# session state defaults BEFORE any UI access
for key, default in {
    "datasets": {},
    "active_methods": METHOD_ORDER.copy(),
    "runner_results": pd.DataFrame(),
    "runner_fits": {},
    "runner_time": np.array([]),
    "runner_y": np.array([]),
    "runner_signal_name": "",
    "runner_recommendation_df": pd.DataFrame(),
    "runner_profile": {},
    "group_stats_df": pd.DataFrame(),
    "group_stats_results": pd.DataFrame(),
    "runner_source_prefill": "__representative__",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def startup_diagnostics():
    issues = []
    checks = []
    required_paths = [
        APP_ROOT / "src",
        APP_ROOT / "src" / "data_validator.py",
        APP_ROOT / "src" / "stats_analysis.py",
        APP_ROOT / "src" / "figure_suite.py",
        APP_ROOT / "src" / "rhythm_methods.py",
        APP_ROOT / "src" / "method_recommender.py",
        APP_ROOT / "src" / "rain_method.py",
        APP_ROOT / "data" / "representative_sample_timeseries.csv",
        APP_ROOT / "data" / "control_vs_treated_replicates_sample.csv",
    ]
    for p in required_paths:
        ok = p.exists()
        checks.append({"item": str(p.relative_to(APP_ROOT)), "status": "OK" if ok else "Missing"})
        if not ok:
            issues.append(f"Missing required file: {p.name}")
    pkg_checks = {
        "datasets_state": isinstance(st.session_state.get("datasets", None), dict),
        "active_methods_state": isinstance(st.session_state.get("active_methods", None), list),
    }
    for k, ok in pkg_checks.items():
        checks.append({"item": k, "status": "OK" if ok else "Check"})
        if not ok:
            issues.append(f"Session state issue: {k}")
    return pd.DataFrame(checks), issues

def load_representative_sample():
    return pd.read_csv(APP_ROOT / "data" / "representative_sample_timeseries.csv")

def load_control_treated_sample():
    return pd.read_csv(APP_ROOT / "data" / "control_vs_treated_replicates_sample.csv")

def make_demo_signals():
    t = np.arange(0, 49, 4, dtype=float)
    return pd.DataFrame({
        "time": t,
        "sinusoidal": 0.8 * np.sin(2*np.pi*t/24 + 0.4),
        "sawtooth_like": np.mod(t, 24) / 24.0 - 0.5,
        "damped": 0.9*np.sin(2*np.pi*t/12) * np.exp(-0.03*t),
        "multi_frequency": 0.5*np.sin(2*np.pi*t/24) + 0.25*np.sin(2*np.pi*t/12 + 0.7),
    })

def build_group_curves(stats_df: pd.DataFrame, feature_name: str):
    sub = stats_df[stats_df["feature"] == feature_name].copy()
    if sub.empty:
        return pd.DataFrame()
    grp = sub.groupby("condition").agg({"period_h":"mean","phase_deg":"mean","amplitude":"mean"}).reset_index()
    if {"control","treated"} - set(grp["condition"]):
        return pd.DataFrame()
    t = np.linspace(0, 48, 200)
    curves = {"time": t}
    summary_rows = []
    for condition in ["control","treated"]:
        row = grp[grp["condition"] == condition].iloc[0]
        period = max(float(row["period_h"]), 1e-6)
        amp = float(row["amplitude"])
        phase_deg = float(row["phase_deg"])
        phase_rad = np.deg2rad(phase_deg)
        y = amp * np.cos(2*np.pi*t/period + phase_rad)
        curves[condition] = y
        summary_rows.append({"parameter":"period_h","group":condition,"value":period})
        summary_rows.append({"parameter":"amplitude","group":condition,"value":amp})
        summary_rows.append({"parameter":"phase_deg","group":condition,"value":phase_deg})
    cdf = pd.DataFrame(curves)
    summary = pd.DataFrame(summary_rows).pivot(index="parameter", columns="group", values="value").reset_index()
    cdf.attrs["summary_table"] = summary.rename(columns={"control":"control","treated":"treated"})
    return cdf

def run_analysis_for_source(source_name: str, signal_name: str, methods: list[str], interval: float):
    raw_df = load_representative_sample() if source_name == "__representative__" else st.session_state["datasets"][source_name]
    prep = prepare_time_series_dataframe(raw_df, interval=float(interval))
    if not prep["signal_cols"]:
        return False, "No analyzable signal columns were found."
    if signal_name not in prep["signal_cols"]:
        return False, "Selected signal is not available."
    time = prep["clean_df"][prep["time_col"]].to_numpy(dtype=float)
    y = prep["clean_df"][signal_name].to_numpy(dtype=float)
    rec_df, rec_summary, profile = recommend_methods(time, y)
    chosen = methods if methods else (rec_df[rec_df["recommended"]]["method"].tolist() or st.session_state["active_methods"])
    mm_df, mm_fits = analyze_selected_methods(time, y, chosen)
    st.session_state["runner_results"] = mm_df
    st.session_state["runner_fits"] = mm_fits
    st.session_state["runner_time"] = time
    st.session_state["runner_y"] = y
    st.session_state["runner_signal_name"] = signal_name
    st.session_state["runner_recommendation_df"] = rec_df
    st.session_state["runner_profile"] = profile
    st.session_state["runner_source_prefill"] = source_name
    return True, "Analysis completed."

st.markdown(f'''
<div class="hero">
  <div class="hero-title">🧬 Eigenvalue Analysis {APP_VERSION}</div>
  <p class="hero-sub">User-friendly rhythm analysis platform with a standalone top-left Run Analysis panel, publication-style figures, native RAIN support, and decision-aware multi-method inference.</p>
  <div style="margin-top:.75rem;">
    <span class="small-chip">Quick run workflow</span>
    <span class="small-chip">11 rhythm methods</span>
    <span class="small-chip">SVG / PNG / HTML export</span>
    <span class="small-chip">Benchmark galleries</span>
  </div>
</div>
''', unsafe_allow_html=True)
st.markdown('<div class="callout"><b>Design note:</b> v14.4 is rebuilt from the last stable workflow and adds the standalone top-left Run Analysis panel without changing the core analysis logic.</div>', unsafe_allow_html=True)


diag_df, diag_issues = startup_diagnostics()
status_cols = st.columns(4)
status_cols[0].metric("Startup diagnostics", "OK" if not diag_issues else "Check")
status_cols[1].metric("Required modules", int((diag_df["status"] == "OK").sum()))
status_cols[2].metric("Datasets loaded", len(st.session_state["datasets"]))
status_cols[3].metric("Run workflow", "Top-left panel")
if diag_issues:
    st.warning("Startup diagnostics detected issues. Open the Diagnostics tab for details.")
else:
    st.markdown('<div class="section-note">Startup diagnostics passed. This build is intended as a UI-polish and verification release.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Analysis settings")
    interval = st.number_input("Sampling interval I", min_value=1e-6, value=1.0, step=0.1, format="%.6f")
    delimiter = st.selectbox("Text delimiter", [",","tab","space","semicolon"], index=0)
    st.divider()
    st.markdown("### Active models")
    current = st.multiselect("Model set", METHOD_ORDER, default=st.session_state["active_methods"], key="sidebar_methods")
    if current:
        st.session_state["active_methods"] = current

left_top, right_top = st.columns([1.15, 2.2], gap="large")
with left_top:
    st.markdown("### Quick analysis")
    st.markdown('<div class="callout"><b>Start here.</b> Choose a data source and signal, then click the stand-alone run button.</div>', unsafe_allow_html=True)
    quick_options = list(st.session_state["datasets"].keys()) + ["__representative__"]
    quick_source = st.selectbox("Data source", quick_options, format_func=lambda x: "Representative bundled sample" if x=="__representative__" else x, key="quick_source")
    quick_df = load_representative_sample() if quick_source=="__representative__" else st.session_state["datasets"][quick_source]
    quick_prep = prepare_time_series_dataframe(quick_df, interval=float(interval))
    if quick_prep["signal_cols"]:
        quick_signal = st.selectbox("Signal", quick_prep["signal_cols"], key="quick_signal")
        quick_time = quick_prep["clean_df"][quick_prep["time_col"]].to_numpy(dtype=float)
        quick_y = quick_prep["clean_df"][quick_signal].to_numpy(dtype=float)
        quick_rec_df, quick_rec_summary, quick_profile = recommend_methods(quick_time, quick_y)
        quick_defaults = quick_rec_df[quick_rec_df["recommended"]]["method"].tolist() or st.session_state["active_methods"]
        quick_methods = st.multiselect("Models", METHOD_ORDER, default=quick_defaults, key="quick_methods")
        st.caption("Recommended methods are preselected automatically. This is the fastest analysis workflow in the app.")
        if st.button("▶ Run Analysis", use_container_width=True, type="primary", key="standalone_run_analysis"):
            ok, msg = run_analysis_for_source(quick_source, quick_signal, quick_methods, interval)
            if ok:
                st.success(msg + " Review the tabs below.")
            else:
                st.error(msg)
    else:
        st.warning("No analyzable signal columns were found in the selected dataset.")
with right_top:
    st.markdown("### Guided workflow")
    st.markdown('<div class="callout"><b>1.</b> Upload or select data  <b>2.</b> Validate  <b>3.</b> Run analysis from the left panel  <b>4.</b> Inspect publication figures  <b>5.</b> Export SVG/PNG/HTML</div>', unsafe_allow_html=True)
    wc = st.columns(4)
    wc[0].metric("Loaded datasets", len(st.session_state["datasets"]))
    wc[1].metric("Active models", len(st.session_state["active_methods"]))
    wc[2].metric("Representative sample", "Ready")
    wc[3].metric("Figure export", "SVG/PNG/HTML")

tabs = st.tabs(["Upload / Input", "Data validator", "Model runner", "Publication figures", "Control vs treated", "Demo / Benchmarks", "Representative sample", "Diagnostics"])

with tabs[0]:
    up1, up2 = st.columns([1.1, 1.1])
    with up1:
        files = st.file_uploader("Upload CSV or TXT files", type=["csv","txt"], accept_multiple_files=True)
        if files:
            sep_map = {",": ",", "tab": "\t", "space": r"\s+", "semicolon": ";"}
            imported = []
            for f in files:
                df = pd.read_csv(f, sep=sep_map[delimiter], engine="python")
                st.session_state["datasets"][f.name] = df
                imported.append(f.name)
            st.success(f"Loaded {len(imported)} datasets.")
    with up2:
        name = st.text_input("Dataset name", value="pasted_dataset")
        pasted = st.text_area("Paste numeric data", height=180, placeholder="time,signal\n0,1.0\n4,1.3")
        if st.button("Add pasted dataset"):
            if pasted.strip():
                sep_map = {",": ",", "tab": "\t", "space": r"\s+", "semicolon": ";"}
                df = pd.read_csv(io.StringIO(pasted), sep=sep_map[delimiter], engine="python")
                st.session_state["datasets"][name] = df
                st.success(f"Added dataset: {name}")
    if st.session_state["datasets"]:
        registry = [{"dataset": k, "rows": int(v.shape[0]), "columns": int(v.shape[1]), "preview_columns": ", ".join(map(str, v.columns[:6]))} for k,v in st.session_state["datasets"].items()]
        st.dataframe(pd.DataFrame(registry), use_container_width=True, hide_index=True)

with tabs[1]:
    options = list(st.session_state["datasets"].keys()) + ["__representative__"]
    source = st.selectbox("Dataset to validate", options, format_func=lambda x: "Representative bundled sample" if x=="__representative__" else x)
    df = load_representative_sample() if source=="__representative__" else st.session_state["datasets"][source]
    prep = prepare_time_series_dataframe(df, interval=float(interval))
    st.write(validation_summary_text(prep))
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Rows", prep["stats"]["n_rows"]); m2.metric("Signals", prep["stats"]["n_signals"]); m3.metric("Missing cells", prep["stats"]["missing_cells"]); m4.metric("Sampling", "Regular" if prep["stats"]["even_sampling"] else "Irregular / mixed")
    for w in prep["warnings"]:
        st.warning(w)
    for e in prep["issues"]:
        st.error(e)
    st.dataframe(prep["clean_df"].head(30), use_container_width=True)
    if prep["signal_cols"]:
        signal_choice = st.selectbox("Preview signal", prep["signal_cols"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prep["clean_df"][prep["time_col"]], y=prep["clean_df"][signal_choice], mode="lines+markers", name=signal_choice))
        st.plotly_chart(apply_pub_layout(fig, "Time", "Signal", 360), use_container_width=True, key="validator_preview")

with tabs[2]:
    st.markdown("### Model runner")
    st.markdown('<div class="callout">This detailed panel mirrors the stand-alone Run Analysis box. Use either workflow.</div>', unsafe_allow_html=True)
    options = list(st.session_state["datasets"].keys()) + ["__representative__"]
    pref = st.session_state.get("runner_source_prefill", "__representative__")
    idx = options.index(pref) if pref in options else 0
    source = st.selectbox("Signal source", options, index=idx, format_func=lambda x: "Representative bundled sample" if x=="__representative__" else x)
    run_df = load_representative_sample() if source=="__representative__" else st.session_state["datasets"][source]
    prep = prepare_time_series_dataframe(run_df, interval=float(interval))
    run_df = prep["clean_df"]
    st.write(validation_summary_text(prep))
    if prep["signal_cols"]:
        signal_name = st.selectbox("Signal", prep["signal_cols"])
        time = run_df[prep["time_col"]].to_numpy(dtype=float)
        y = run_df[signal_name].to_numpy(dtype=float)
        rec_df, rec_summary, profile = recommend_methods(time, y)
        st.write(rec_summary)
        st.dataframe(rec_df, use_container_width=True, height=240)
        defaults = rec_df[rec_df["recommended"]]["method"].tolist() or st.session_state["active_methods"]
        selected = st.multiselect("Models to run now", METHOD_ORDER, default=defaults)
        if st.button("Run selected models", use_container_width=True):
            ok, msg = run_analysis_for_source(source, signal_name, selected, interval)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        if not st.session_state["runner_results"].empty:
            mm_df = st.session_state["runner_results"]
            st.dataframe(mm_df, use_container_width=True, height=320)
            st.write(compare_methods_narrative(mm_df))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state["runner_time"], y=st.session_state["runner_y"], mode="lines+markers", name="Observed"))
            for m, fdf in st.session_state["runner_fits"].items():
                fig.add_trace(go.Scatter(x=fdf["time"], y=fdf["fitted"], mode="lines", name=m, opacity=0.75))
            st.plotly_chart(apply_pub_layout(fig, "Time", "Signal", 430), use_container_width=True, key="runner_overlay")

with tabs[3]:
    st.markdown("### Publication figures")
    if st.session_state["runner_results"].empty:
        st.info("Run a signal using the top-left panel or Model runner to populate the figure suite.")
    else:
        figs = make_publication_figures(st.session_state["runner_time"], st.session_state["runner_y"], st.session_state["runner_results"], st.session_state["runner_fits"], st.session_state.get("runner_recommendation_df"))
        fig_name = st.selectbox("Figure", list(figs.keys()))
        fig = figs[fig_name]
        st.plotly_chart(fig, use_container_width=True, key=f"publication_main_{fig_name}")
        dl = figure_download_bundle(fig)
        c1, c2, c3 = st.columns(3)
        if dl["svg"] is not None:
            c1.download_button("Download SVG", dl["svg"], file_name=f"{fig_name.replace(' ','_').lower()}.svg", mime="image/svg+xml", use_container_width=True)
        if dl["png"] is not None:
            c2.download_button("Download PNG", dl["png"], file_name=f"{fig_name.replace(' ','_').lower()}.png", mime="image/png", use_container_width=True)
        if dl["html"] is not None:
            c3.download_button("Download HTML", dl["html"], file_name=f"{fig_name.replace(' ','_').lower()}.html", mime="text/html", use_container_width=True)
        st.markdown("#### Figure gallery")
        cols = st.columns(2)
        for i, name in enumerate(figs.keys()):
            with cols[i % 2]:
                st.markdown(f"**{name}**")
                st.plotly_chart(figs[name], use_container_width=True, key=f"publication_gallery_{name}")

with tabs[4]:
    st.markdown("### Control-vs-treated statistics and figure panel")
    uploaded = st.file_uploader("Upload replicate parameter CSV", type=["csv"], key="stats_upload")
    if st.button("Load bundled control-vs-treated sample"):
        st.session_state["group_stats_df"] = load_control_treated_sample()
    if uploaded is not None:
        st.session_state["group_stats_df"] = pd.read_csv(uploaded)
    if not st.session_state["group_stats_df"].empty:
        sdf = st.session_state["group_stats_df"].copy()
        st.dataframe(sdf, use_container_width=True, height=240)
        params = [c for c in ["period_h","phase_deg","peak_time_h","amplitude","eig_abs"] if c in sdf.columns]
        c1,c2,c3 = st.columns(3)
        param = c1.selectbox("Parameter", params)
        test = c2.selectbox("Test", ["permutation","welch","mannwhitney"])
        n_perm = c3.number_input("Permutations", min_value=200, max_value=10000, value=2000, step=200)
        if st.button("Run statistical comparison"):
            st.session_state["group_stats_results"] = compare_groups(sdf, parameter=param, test=test, phase_parameter=(param=="phase_deg"), n_perm=int(n_perm))
            st.session_state["group_stats_parameter"] = param
        if not st.session_state["group_stats_results"].empty:
            res = st.session_state["group_stats_results"]
            st.dataframe(res, use_container_width=True, height=260)
            st.write(build_stats_summary(res, st.session_state.get("group_stats_parameter","parameter")))
            feature_choice = st.selectbox("Feature for fitted comparison panel", sorted(sdf["feature"].unique()))
            curve_df = build_group_curves(sdf, feature_choice)
            if not curve_df.empty:
                fig = make_group_comparison_figure(curve_df, title=f"Control vs treated fits: {feature_choice}")
                st.plotly_chart(fig, use_container_width=True, key=f"group_comparison_{feature_choice}")

with tabs[5]:
    st.markdown("### Demo / benchmarks")
    demo = make_demo_signals()
    st.plotly_chart(build_demo_signal_gallery(demo), use_container_width=True, key="demo_signal_gallery")
    scenario = st.selectbox("Demo scenario", [c for c in demo.columns if c != "time"])
    if st.button("Run benchmark on selected scenario"):
        rows = []
        tt = demo["time"].to_numpy(dtype=float)
        for sc in [c for c in demo.columns if c != "time"]:
            yy = demo[sc].to_numpy(dtype=float)
            res, _ = analyze_selected_methods(tt, yy, METHOD_ORDER)
            res["scenario"] = sc
            rows.append(res)
        st.session_state["benchmark_table"] = pd.concat(rows, ignore_index=True)
    if "benchmark_table" in st.session_state and not st.session_state["benchmark_table"].empty:
        bdf = st.session_state["benchmark_table"].copy()
        st.dataframe(bdf, use_container_width=True, height=320)
        heat = bdf.pivot_table(index="scenario", columns="method", values="period_h", aggfunc="mean")
        fig = px.imshow(heat, text_auto=".2g", aspect="auto", color_continuous_scale="Viridis", labels=dict(x="Method", y="Scenario", color="Period"))
        st.plotly_chart(apply_pub_layout(fig, "", "", 420), use_container_width=True, key="benchmark_period_heatmap")
        heat2 = bdf.pivot_table(index="scenario", columns="method", values="amplitude", aggfunc="mean")
        fig2 = px.imshow(heat2, text_auto=".2g", aspect="auto", color_continuous_scale="Magma", labels=dict(x="Method", y="Scenario", color="Amplitude"))
        st.plotly_chart(apply_pub_layout(fig2, "", "", 420), use_container_width=True, key="benchmark_amplitude_heatmap")

with tabs[6]:
    st.markdown("### Representative sample")
    rep = load_representative_sample()
    st.dataframe(rep, use_container_width=True)
    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        if st.button("Load representative sample into workspace"):
            st.session_state["datasets"]["representative_sample.csv"] = rep.copy()
            st.success("Representative sample loaded.")
    with c2:
        st.markdown('<div class="section-note">After loading, use the stand-alone <b>▶ Run Analysis</b> button in the top-left panel for the fastest workflow, or use Model runner for more detailed step-by-step control.</div>', unsafe_allow_html=True)


with tabs[7]:
    st.markdown("### Startup diagnostics")
    st.markdown('<div class="callout"><b>Purpose:</b> this panel verifies that local packaging, sample data, and session-state initialization are healthy before analysis. It is intended for troubleshooting and verification, not for routine use.</div>', unsafe_allow_html=True)
    diag_df, diag_issues = startup_diagnostics()
    st.dataframe(diag_df, use_container_width=True, hide_index=True)
    if diag_issues:
        for issue in diag_issues:
            st.error(issue)
    else:
        st.success("All startup diagnostics passed.")
    st.markdown("#### Current session summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Datasets", len(st.session_state["datasets"]))
    s2.metric("Active methods", len(st.session_state["active_methods"]))
    s3.metric("Last signal", st.session_state["runner_signal_name"] if st.session_state["runner_signal_name"] else "None")
    s4.metric("Results ready", "Yes" if not st.session_state["runner_results"].empty else "No")
