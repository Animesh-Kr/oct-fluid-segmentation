"""
Attention-Guided TransUNet — Retinal Fluid Segmentation Dashboard
Author: Animesh A. Kumar | Newcastle University MSc Advanced Computer Science 2025-26
Target: OMIA 2026 Workshop + medRxiv preprint
"""

import os, json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCT Fluid Segmentation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040810 0%, #0a1628 100%);
    border-right: 1px solid #1a2744;
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

.main { background: #060c1a; color: #e8f0ff; }
h1, h2, h3 { color: #e8f0ff !important; letter-spacing: -0.02em; }

[data-testid="metric-container"] {
    background: #0d1b33;
    border: 1px solid #1e3058;
    border-radius: 8px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #64b5f6 !important; font-family: 'JetBrains Mono' !important; }
[data-testid="stMetricLabel"] { color: #7a9cc8 !important; }

.badge-urgent  { background:#1a0505; border-left:4px solid #ef5350; padding:12px 16px; border-radius:6px; margin:8px 0; }
.badge-review  { background:#1a1005; border-left:4px solid #ffa726; padding:12px 16px; border-radius:6px; margin:8px 0; }
.badge-monitor { background:#051a0a; border-left:4px solid #66bb6a; padding:12px 16px; border-radius:6px; margin:8px 0; }

.stTabs [data-baseweb="tab-list"] { background: #0d1b33; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #7a9cc8 !important; font-family: 'Syne' !important; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #e8f0ff !important; border-radius:6px; }

.stButton > button {
    background: #1e3a5f; color: #e8f0ff;
    border: 1px solid #2a5080; border-radius: 6px;
    font-family: 'Syne'; transition: all 0.2s;
}
.stButton > button:hover { background: #2a5080; border-color: #64b5f6; }

.result-card {
    background: #0d1b33; border: 1px solid #1e3058;
    border-radius: 10px; padding: 20px; margin: 8px 0;
}
.section-title {
    font-size: 11px; font-family: 'JetBrains Mono';
    color: #4a7aaa; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO    = "animeshakr/oct-fluid-segmentation"
FLUID_CMAP = ListedColormap(['#00000000', '#1565C0', '#E65100', '#2E7D32'])

# ── Load demo data ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_demo_data():
    local = Path("demo_results.json")
    if local.exists():
        with open(local) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename="demo_results.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load demo data: {e}")
        return {"samples": [], "metadata": {}, "main_results": "", "ablation_results": ""}

# ── Load ONNX models ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_onnx_models():
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        providers = ['CPUExecutionProvider']
        hf_hub_download(repo_id=HF_REPO, filename="deployment/slot1_v2l_seed2024.onnx.data")
        hf_hub_download(repo_id=HF_REPO, filename="deployment/slot2_v2l_seed123.onnx.data")
        p1 = hf_hub_download(repo_id=HF_REPO, filename="deployment/slot1_v2l_seed2024.onnx")
        p2 = hf_hub_download(repo_id=HF_REPO, filename="deployment/slot2_v2l_seed123.onnx")
        sess1 = ort.InferenceSession(p1, providers=providers)
        sess2 = ort.InferenceSession(p2, providers=providers)
        return sess1, sess2, None
    except ImportError:
        return None, None, "onnxruntime not installed"
    except Exception as e:
        return None, None, str(e)

# ── Preprocessing ──────────────────────────────────────────────────────────────
def clahe_preprocess(img_gray: np.ndarray) -> np.ndarray:
    mn, mx = img_gray.min(), img_gray.max()
    u8 = ((img_gray - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(u8).astype(np.float32) / 255.0

# ── Live inference ─────────────────────────────────────────────────────────────
def run_inference(img_array, sess1, sess2):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array.copy()
    resized = cv2.resize(gray.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
    processed = clahe_preprocess(resized)
    x = processed[np.newaxis, np.newaxis, :, :].astype(np.float32)
    out1 = sess1.run(None, {sess1.get_inputs()[0].name: x})[0]
    out2 = sess2.run(None, {sess2.get_inputs()[0].name: x})[0]
    mean_logits = (out1 + out2) / 2.0
    pred_mask   = np.argmax(mean_logits, axis=1)[0]
    uncertainty = np.var(np.stack([out1, out2], axis=0), axis=0).mean(axis=1)[0]
    return processed, pred_mask, uncertainty

# ── Overlay helper ─────────────────────────────────────────────────────────────
def make_overlay_fig(img, mask=None, uncertainty=None, title=""):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#060c1a')
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    if mask is not None:
        masked = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked, cmap=FLUID_CMAP, alpha=0.55, vmin=0, vmax=3)
    if uncertainty is not None:
        ax.imshow(uncertainty, cmap='magma', alpha=0.45)
    ax.axis('off')
    ax.set_title(title, color='#7a9cc8', fontsize=9, pad=4)
    fig.tight_layout(pad=0)
    return fig

# ── Parse CSV ──────────────────────────────────────────────────────────────────
def parse_csv(csv_str):
    import io, csv
    return list(csv.DictReader(io.StringIO(csv_str.strip())))

# ── Architecture plot ──────────────────────────────────────────────────────────
def architecture_plot():
    nodes = [
        {"label":"OCT Input\n(1×512×512)",     "x":0,"y":0,"z":0,"color":"#78909C"},
        {"label":"Stage 1\n(32ch)",            "x":1,"y":1,"z":0,"color":"#1565C0"},
        {"label":"Stage 2\n(64ch)",            "x":1,"y":2,"z":0,"color":"#1565C0"},
        {"label":"Stage 3\n(96ch)",            "x":1,"y":3,"z":0,"color":"#1565C0"},
        {"label":"Stage 4\n(192ch)",           "x":1,"y":4,"z":0,"color":"#1565C0"},
        {"label":"Bottleneck\n(640ch)",        "x":1,"y":5,"z":0,"color":"#1565C0"},
        {"label":"Transformer\nBottleneck",    "x":2,"y":5,"z":1,"color":"#6A1B9A"},
        {"label":"MHA Block 1",                "x":3,"y":5,"z":1,"color":"#6A1B9A"},
        {"label":"MHA Block 2",                "x":3,"y":4,"z":1,"color":"#6A1B9A"},
        {"label":"Attention Gate 4",           "x":4,"y":4,"z":0,"color":"#E65100"},
        {"label":"Attention Gate 3",           "x":4,"y":3,"z":0,"color":"#E65100"},
        {"label":"Attention Gate 2",           "x":4,"y":2,"z":0,"color":"#E65100"},
        {"label":"Attention Gate 1",           "x":4,"y":1,"z":0,"color":"#E65100"},
        {"label":"Decoder L4",                 "x":5,"y":4,"z":0,"color":"#2E7D32"},
        {"label":"Decoder L3",                 "x":5,"y":3,"z":0,"color":"#2E7D32"},
        {"label":"Decoder L2",                 "x":5,"y":2,"z":0,"color":"#2E7D32"},
        {"label":"Decoder L1",                 "x":5,"y":1,"z":0,"color":"#2E7D32"},
        {"label":"SA-BatchNorm",               "x":5,"y":0,"z":1,"color":"#00838F"},
        {"label":"MC Dropout\np=0.3",          "x":6,"y":2,"z":0,"color":"#AD1457"},
        {"label":"Output Head\n(4 classes)",   "x":7,"y":2,"z":0,"color":"#558B2F"},
        {"label":"Seg Mask",                   "x":8,"y":3,"z":0,"color":"#558B2F"},
        {"label":"Uncertainty",                "x":8,"y":2,"z":0,"color":"#558B2F"},
        {"label":"UCUS Score",                 "x":8,"y":1,"z":0,"color":"#558B2F"},
    ]
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),
             (8,9),(8,10),(8,11),(8,12),
             (9,13),(10,14),(11,15),(12,16),
             (13,14),(14,15),(15,16),
             (16,17),(16,18),(18,19),(19,20),(19,21),(19,22)]
    fig = go.Figure()
    for e in edges:
        n0, n1 = nodes[e[0]], nodes[e[1]]
        fig.add_trace(go.Scatter3d(
            x=[n0["x"],n1["x"]], y=[n0["y"],n1["y"]], z=[n0["z"],n1["z"]],
            mode='lines', line=dict(color='#2a4060',width=2),
            hoverinfo='none', showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[n["x"] for n in nodes], y=[n["y"] for n in nodes], z=[n["z"] for n in nodes],
        mode='markers+text', text=[n["label"] for n in nodes],
        marker=dict(size=10, color=[n["color"] for n in nodes], opacity=0.9,
                    line=dict(color='white',width=1)),
        textposition="top center", textfont=dict(size=8,color='#c8d8f0'),
        hoverinfo='text'))
    fig.update_layout(
        height=580, margin=dict(l=0,r=0,b=0,t=30),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(bgcolor='rgba(0,0,0,0)',
                   xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)),
        title=dict(text="AttentionTransUNetL — 3D Architecture",font=dict(color='#c8d8f0',size=14)),
        showlegend=False)
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
demo_data = load_demo_data()
samples   = demo_data.get("samples", [])

with st.sidebar:
    st.markdown("## 👁️ OCT Fluid AI")
    st.caption("Newcastle University · MSc Advanced Computer Science 2025–26")
    st.markdown("---")
    st.markdown('<p class="section-title">Sample Selection</p>', unsafe_allow_html=True)

    sel_sample = None
    if samples:
        sources    = sorted(set(s["source"] for s in samples))
        sel_source = st.selectbox("Scanner Source", sources)
        filtered   = [s for s in samples if s["source"] == sel_source]
        sel_id     = st.selectbox("Sample ID", [s["id"] for s in filtered])
        sel_sample = next((s for s in filtered if s["id"] == sel_id), None)
    else:
        st.warning("No demo samples loaded")

    st.markdown("---")
    meta = demo_data.get("metadata", {}).get("model_info", {})
    st.markdown('<p class="section-title">Model Info</p>', unsafe_allow_html=True)
    st.caption(f"V2S: {meta.get('V2S','EfficientNetV2S + TransformerBottleneck (~22M)')}")
    st.caption(f"V2L: {meta.get('V2L','EfficientNetV2L + TransformerBottleneck (~127M)')}")
    st.caption(f"Uncertainty: {meta.get('uncertainty','MC Dropout (20 passes) + TTA')}")

    st.markdown("---")
    st.markdown("**🔗 Links**")
    st.markdown("[📦 HuggingFace](https://huggingface.co/animeshakr/oct-fluid-segmentation)")
    st.markdown("[💻 GitHub](https://github.com/Animesh-Kr/oct-fluid-segmentation)")

# =============================================================================
# MAIN
# =============================================================================
st.markdown("# Retinal Fluid Segmentation")
st.markdown("*Attention-Guided TransUNet · Dual V2L Ensemble · MC Dropout Uncertainty · OMIA 2026*")
st.markdown("---")

tab_demo, tab_live, tab_val, tab_arch = st.tabs([
    "📋  Demo Records",
    "🔬  Live Inference",
    "📊  Validation",
    "🧠  Architecture",
])

# ─── TAB 1 — DEMO ─────────────────────────────────────────────────────────────
with tab_demo:
    if sel_sample is None:
        st.info("Select a sample from the sidebar.")
    else:
        vols   = sel_sample.get("fluid_volumes_mm3", {})
        ucus   = sel_sample.get("ucus", {})
        dice   = sel_sample.get("dice", {})
        band   = ucus.get("urgency_band", ucus.get("urgency", "Monitor"))
        score  = ucus.get("ucus_score", ucus.get("score", 0))
        action = ucus.get("action", "Routine follow-up at next scheduled appointment")
        comp   = ucus.get("components", {})

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("IRF Volume",  f"{vols.get('IRF',0):.4f} mm³")
        m2.metric("SRF Volume",  f"{vols.get('SRF',0):.4f} mm³")
        m3.metric("PED Volume",  f"{vols.get('PED',0):.4f} mm³")
        m4.metric("Mean Dice",   f"{dice.get('mean_fluid',0):.3f}")
        m5.metric("UCUS Score",  f"{score:.3f}")

        css = {"Urgent":"badge-urgent","Review":"badge-review","Monitor":"badge-monitor"}.get(band,"badge-monitor")
        st.markdown(f'<div class="{css}"><b>Clinical Urgency: {band}</b><br><small>{action}</small></div>', unsafe_allow_html=True)

        # OCT image from demo_results
        oct_raw = np.array(sel_sample["oct_image"], dtype=np.float32)
        if oct_raw.max() > 1.0:
            oct_raw = (oct_raw - oct_raw.min()) / (oct_raw.max() - oct_raw.min() + 1e-6)

        has_mask = "pred_mask" in sel_sample
        has_unc  = "uncertainty" in sel_sample

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = make_overlay_fig(oct_raw, title="Raw OCT B-scan")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.caption(f"Source: {sel_sample.get('source','')}  |  Slice: {sel_sample.get('slice_idx','')}")

        with c2:
            mask = np.array(sel_sample["pred_mask"]) if has_mask else None
            fig  = make_overlay_fig(oct_raw, mask, title="Segmentation Overlay")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.caption(f"🟦 IRF={dice.get('IRF',0):.3f}  🟧 SRF={dice.get('SRF',0):.3f}  🟩 PED={dice.get('PED',0):.3f}")

        with c3:
            unc = np.array(sel_sample["uncertainty"]) if has_unc else None
            fig = make_overlay_fig(oct_raw, uncertainty=unc, title="Uncertainty Heatmap")
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.caption(f"Boundary uncertainty: {comp.get('boundary_uncertainty',0):.4f}")

        if comp:
            st.markdown("#### UCUS Components")
            u1, u2, u3, u4 = st.columns(4)
            u1.metric("Volume Score",         f"{comp.get('volume_score',0):.3f}")
            u2.metric("Foveal Multiplier",    f"{comp.get('foveal_multiplier',0):.4f}")
            u3.metric("Boundary Uncertainty", f"{comp.get('boundary_uncertainty',0):.4f}")
            u4.metric("Uncertainty Discount", f"{comp.get('uncertainty_discount',0):.4f}")

# ─── TAB 2 — LIVE INFERENCE ───────────────────────────────────────────────────
with tab_live:
    st.markdown("### Live ONNX Inference")
    st.markdown("Upload any OCT B-scan. Models load from HuggingFace automatically.")
    st.info("First load takes ~2 minutes to download ONNX files from HuggingFace.")

    uploaded = st.file_uploader("Upload OCT scan (PNG/JPG)", type=["png","jpg","jpeg","tif"])

    if uploaded:
        with st.spinner("Loading models from HuggingFace..."):
            sess1, sess2, err = load_onnx_models()

        if err:
            st.error(f"Model load failed: {err}")
            st.code("pip install onnxruntime", language="bash")
        elif sess1 is None:
            st.error("Models not available.")
        else:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Running dual V2L ensemble..."):
                try:
                    proc, mask, unc = run_inference(raw, sess1, sess2)
                    irf_px  = int(np.sum(mask == 1))
                    srf_px  = int(np.sum(mask == 2))
                    ped_px  = int(np.sum(mask == 3))
                    mean_unc = float(unc.mean())
                    band_live = "Review" if mean_unc > 0.30 else "Monitor"

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("IRF Pixels",       irf_px if irf_px > 0 else "None detected")
                    m2.metric("SRF Pixels",       srf_px if srf_px > 0 else "None detected")
                    m3.metric("PED Pixels",       ped_px if ped_px > 0 else "None detected")
                    m4.metric("Mean Uncertainty", f"{mean_unc:.4f}")

                    css_live = "badge-monitor" if band_live == "Monitor" else "badge-review"
                    st.markdown(f'<div class="{css_live}"><b>Clinical Band: {band_live}</b></div>', unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        fig = make_overlay_fig(proc, title="Preprocessed (CLAHE)")
                        st.pyplot(fig, use_container_width=True); plt.close(fig)
                    with c2:
                        fig = make_overlay_fig(proc, mask, title="Segmentation")
                        st.pyplot(fig, use_container_width=True); plt.close(fig)
                    with c3:
                        fig = make_overlay_fig(proc, uncertainty=unc, title="Uncertainty")
                        st.pyplot(fig, use_container_width=True); plt.close(fig)

                    st.caption("🟦 IRF (Intraretinal Fluid)  🟧 SRF (Subretinal Fluid)  🟩 PED (Pigment Epithelial Detachment)")

                except Exception as e:
                    st.error(f"Inference error: {e}")

# ─── TAB 3 — VALIDATION ───────────────────────────────────────────────────────
with tab_val:
    st.markdown("### Validation Results")

    c_res, c_abl = st.columns(2)

    with c_res:
        st.markdown("#### Test Set (503 slices, 4 sources)")
        main_csv = demo_data.get("main_results", "")
        if main_csv:
            for row in parse_csv(main_csv):
                label = row["metric"].replace("dice_","").upper()
                st.metric(label, f"{float(row['mean']):.4f}", f"± {float(row['std']):.4f}")

        st.markdown("---")
        st.markdown("#### Clinical Safety Metrics")
        st.metric("Inter-grader Human Ceiling",      "0.9030")
        st.metric("Uncertainty Ratio",               "1.34×  (p=3.77e-05) ✅")
        st.metric("SRF Volume Correlation",          "r=0.778  p=6.33e-04 ✅")
        st.metric("PED Volume Correlation",          "r=0.841  p=8.64e-05 ✅")
        st.metric("V2L Multi-seed Mean Dice",        "0.7843 ± 0.0058")

    with c_abl:
        st.markdown("#### Ablation Study")
        abl_csv = demo_data.get("ablation_results", "")
        if abl_csv:
            rows = parse_csv(abl_csv)
            fig = go.Figure(go.Bar(
                x=[float(r["dice_mean_fluid"]) for r in rows],
                y=[r["variant"] for r in rows],
                orientation='h',
                error_x=dict(type='data', array=[float(r["std"]) for r in rows],
                             visible=True, color='#4a7aaa'),
                marker_color=['#1e3a5f']*(len(rows)-1)+['#1565C0'],
                marker_line=dict(color='#2a5080', width=1)))
            fig.update_layout(
                height=300, margin=dict(l=0,r=20,t=10,b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Mean Dice (fluid)", color='#7a9cc8', gridcolor='#1a2744', range=[0,0.55]),
                yaxis=dict(color='#7a9cc8', autorange='reversed'),
                font=dict(color='#c8d8f0', size=10))
            st.plotly_chart(fig, use_container_width=True)

    # Per-source heatmap
    st.markdown("#### Per-Source Dice Breakdown")
    src_csv = demo_data.get("per_source_results", "")
    if src_csv:
        rows   = parse_csv(src_csv)
        srcs   = [r["source"] for r in rows]
        cols   = ["dice_IRF","dice_SRF","dice_PED","dice_mean_fluid"]
        labels = ["IRF","SRF","PED","Mean Fluid"]
        z = [[float(r.get(c,0)) for c in cols] for r in rows]
        fig = go.Figure(go.Heatmap(
            z=z, x=labels, y=srcs, colorscale='Blues', zmin=0, zmax=1,
            text=[[f"{v:.3f}" for v in row] for row in z],
            texttemplate="%{text}", textfont=dict(size=13)))
        fig.update_layout(
            height=220, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c8d8f0'))
        st.plotly_chart(fig, use_container_width=True)

    # Multi-seed table
    st.markdown("#### Multi-Seed Stability (seeds 42 / 123 / 2024)")
    ms_csv = demo_data.get("multiseed_results", "")
    if ms_csv:
        rows = parse_csv(ms_csv)
        c1, c2 = st.columns(2)
        for row in rows:
            target = c1 if row["model"] == "V2S" else c2
            label  = f"{row['model']} {row['metric'].replace('dice_','').upper()}"
            target.metric(label, f"{float(row['mean']):.4f}", f"± {float(row['std']):.4f}")

# ─── TAB 4 — ARCHITECTURE ─────────────────────────────────────────────────────
with tab_arch:
    st.markdown("### AttentionTransUNetL — Dual V2L Ensemble Architecture")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(architecture_plot(), use_container_width=True)
    with c2:
        st.markdown("#### Components")
        for comp, detail, note in [
            ("Encoder",      "EfficientNetV2L",       "s1=32 s2=64 s3=96 s4=192 bot=640"),
            ("Transformer",  "2× MHA Bottleneck",      "d_model=512, 16 heads"),
            ("Attention",    "4× Attention Gates",     "spatial filtering"),
            ("Domain",       "SA-BatchNorm",           "5 scanner sources"),
            ("Uncertainty",  "MC Dropout p=0.3",       "20 forward passes"),
            ("Triage",       "UCUS Score",             "Monitor/Review/Urgent"),
            ("Output",       "4-class head",           "BG/IRF/SRF/PED"),
        ]:
            st.markdown(f"""
            <div class="result-card">
                <div class="section-title">{comp}</div>
                <b style="color:#64b5f6">{detail}</b><br>
                <small style="color:#4a7aaa">{note}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Novel Contributions")
    for c in ["UCUS — Uncertainty-Weighted Clinical Urgency Score",
              "Dual uncertainty: MC Dropout + inter-model disagreement",
              "Source-Adaptive BatchNorm for scanner domain adaptation",
              "Multi-source 4-dataset evaluation pipeline"]:
        st.markdown(f"✅ {c}")

    st.markdown("---")
    st.markdown("#### INT8 Quantisation")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("V2L FP32", "510 MB")
    q2.metric("V2L INT8", "132 MB", "−74%")
    q3.metric("V2S FP32",  "91 MB")
    q4.metric("V2S INT8",  "24 MB", "−74%")
