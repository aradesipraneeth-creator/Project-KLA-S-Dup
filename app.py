import os
import sys
import json
import io
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import streamlit as st
from PIL import Image
from scipy.ndimage import zoom

# Absolute path resolution for project root and utils directory
PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_DIR = PROJECT_ROOT / "utils"

for path_obj in [PROJECT_ROOT, UTILS_DIR, Path(os.getcwd())]:
    path_str = str(path_obj)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from configs.config import Config
from models.airnet_v3 import AIRNetV3
from models.airnet_v4 import AIRNetV4
from models.airnet_v5 import AIRNetV5

from utils.checkpoint_manager import CheckpointManager, VerificationResult
from utils.image_normalization import (
    normalize_for_display_and_metrics, normalize_input, normalize_target,
    denormalize_output, prepare_for_metric, prepare_for_display,
    validate_metric_inputs, compute_array_stats
)
from utils.metrics import (
    compute_psnr, compute_ssim, compute_lpips, run_metric_sanity_test,
    compute_edge_error, compute_gradient_error, compute_laplacian_error,
    compute_high_frequency_error, compute_brightness_error, compute_contrast_error,
    compute_all_metrics
)
from utils.edge_analysis import compute_sobel_edge_magnitude, prepare_edge_map_display, compute_edge_statistics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config(MODEL_VERSION="AIR-Net-v3")

st.set_page_config(
    page_title="AIR-Net v3 / v4 / v5 — High-Fidelity Semiconductor Restoration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Contrast UI Styling (Fully Readable in Both Light and Dark Themes)
st.markdown("""
    <style>
    /* High-contrast base text */
    html, body, [class*="st-"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #E6EDF3 !important;
        opacity: 1.0 !important;
    }
    
    /* High-contrast Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #58A6FF !important;
        font-weight: 700 !important;
        opacity: 1.0 !important;
        letter-spacing: 0.2px;
        margin-top: 15px !important;
        margin-bottom: 10px !important;
    }
    
    /* Markdown Text & Captions */
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
        color: #E6EDF3 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        opacity: 1.0 !important;
    }
    
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #8B949E !important;
        font-weight: 500 !important;
        opacity: 1.0 !important;
    }
    
    /* Streamlit Metric Styling */
    [data-testid="stMetricLabel"] p {
        color: #C9D1D9 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        opacity: 1.0 !important;
    }
    [data-testid="stMetricValue"] div {
        color: #58A6FF !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        opacity: 1.0 !important;
    }
    
    /* High-Contrast Info/Explanation Cards */
    .high-contrast-card {
        background-color: #161B22;
        color: #F0F6FC;
        border-left: 5px solid #58A6FF;
        border-top: 1px solid #30363D;
        border-right: 1px solid #30363D;
        border-bottom: 1px solid #30363D;
        padding: 18px;
        border-radius: 6px;
        margin-top: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .card-heading { color: #58A6FF !important; font-weight: 700; font-size: 18px; margin-bottom: 8px; }
    .card-text { color: #F0F6FC !important; font-size: 15px; line-height: 1.6; }
    
    /* Streamlit Table Styling */
    [data-testid="stTable"] table {
        color: #F0F6FC !important;
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        font-size: 15px !important;
    }
    [data-testid="stTable"] th {
        color: #58A6FF !important;
        background-color: #21262D !important;
        font-weight: 700 !important;
    }
    [data-testid="stTable"] td {
        color: #E6EDF3 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0D1117 !important;
        border-right: 1px solid #30363D !important;
    }
    
    /* Expander Header Styling */
    .streamlit-expanderHeader {
        color: #58A6FF !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_verify_models():
    norm_path = PROJECT_ROOT / "outputs" / "v3" / "indexes" / "index_normalization.json"
    norm_params = None
    if norm_path.exists():
        with open(norm_path, "r") as f:
            norm_params = json.load(f)

    # 1. AIR-Net v3 Foundation
    v3_model = AIRNetV3(norm_params=norm_params).to(DEVICE)
    v3_env = os.environ.get("AIRNET_V3_CHECKPOINT", "")
    v3_cand = ([Path(v3_env)] if v3_env else []) + [
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_ema_best_model.pth",
        PROJECT_ROOT / "outputs" / "v3" / "checkpoints" / "airnet_v3_best_model.pth"
    ]
    v3_path = None
    for c in v3_cand:
        if c.exists():
            v3_path = str(c)
            break

    v3_ver = None
    if v3_path:
        v3_ver = CheckpointManager.verify_checkpoint(v3_model, v3_path, architecture_name="AIR-Net v3 Foundation", device=DEVICE)

    # 2. AIR-Net v4 Refinement System
    v4_model = AIRNetV4(norm_params=norm_params).to(DEVICE)
    v4_env = os.environ.get("AIRNET_V4_CHECKPOINT", "")
    v4_cand = ([Path(v4_env)] if v4_env else []) + [
        PROJECT_ROOT / "outputs" / "v4" / "checkpoints" / "best_v4_model.pth",
        PROJECT_ROOT / "outputs" / "v4" / "checkpoints" / "latest_v4_model.pth"
    ]
    v4_path = None
    for c in v4_cand:
        if c.exists():
            v4_path = str(c)
            break

    v4_ver = None
    if v4_path:
        v4_ver = CheckpointManager.verify_checkpoint(v4_model, v4_path, architecture_name="AIR-Net v4 System", device=DEVICE)

    # 3. AIR-Net v5 High-Fidelity System
    v5_model = AIRNetV5(norm_params=norm_params).to(DEVICE)
    v5_env = os.environ.get("AIRNET_V5_CHECKPOINT", "")
    v5_cand = ([Path(v5_env)] if v5_env else []) + [
        PROJECT_ROOT / "outputs" / "v5" / "checkpoints" / "best_v5_model.pth",
        PROJECT_ROOT / "outputs" / "v5" / "checkpoints" / "latest_v5_model.pth"
    ]
    v5_path = None
    for c in v5_cand:
        if c.exists():
            v5_path = str(c)
            break

    v5_ver = None
    if v5_path:
        v5_ver = CheckpointManager.verify_checkpoint(v5_model, v5_path, architecture_name="AIR-Net v5 System", device=DEVICE)

    return {
        "v3_model": v3_model if (v3_ver and v3_ver.is_verified) else None,
        "v3_ver": v3_ver,
        "v4_model": v4_model if (v4_ver and v4_ver.is_verified) else None,
        "v4_ver": v4_ver,
        "v5_model": v5_model if (v5_ver and v5_ver.is_verified) else None,
        "v5_ver": v5_ver,
        "device": DEVICE
    }

def explain_category_routing(raw_indices: dict, dominant_cat: str, routing_probs: dict) -> str:
    sobel = raw_indices.get("sobel_edge_index", 0.0)
    texture = raw_indices.get("texture_index", 0.0)
    noise = raw_indices.get("noise_index", 0.0)
    contrast = raw_indices.get("contrast_index", 0.0)
    density = raw_indices.get("edge_density", 0.0)
    prob_pct = routing_probs.get(dominant_cat, 0.0) * 100

    if dominant_cat == "EDGE_DOMINANT":
        return f"Routed to **EDGE_DOMINANT** ({prob_pct:.1f}%) due to high gradient activity (Sobel Index: `{sobel:.4f}`, Edge Density: `{density*100:.1f}%`)."
    elif dominant_cat == "TEXTURE_DOMINANT":
        return f"Routed to **TEXTURE_DOMINANT** ({prob_pct:.1f}%) due to high micro-texture variation (Texture Index: `{texture:.4f}`)."
    elif dominant_cat == "NOISE_DOMINANT":
        return f"Routed to **NOISE_DOMINANT** ({prob_pct:.1f}%) due to high noise proxy (`{noise:.4f}`)."
    elif dominant_cat == "SMOOTH_LOW_CONTRAST":
        return f"Routed to **SMOOTH_LOW_CONTRAST** ({prob_pct:.1f}%) due to low contrast (`{contrast:.4f}`)."
    else:
        return f"Routed to **SPARSE_FEATURE** ({prob_pct:.1f}%) due to localized sparse gradient spikes."

def display_image(img_input: np.ndarray, caption: str = "", width_mode: str = "stretch"):
    """
    Safe visualization helper function.
    1. Converts array to 2D numpy float32.
    2. Validates shape and finite values.
    3. Displays with st.image without modifying raw prediction arrays or metrics.
    """
    if img_input is None:
        st.info("Image N/A")
        return
    arr = normalize_for_display_and_metrics(img_input)
    st.image(arr, caption=caption if caption else None, width=width_mode, clamp=True)

def compute_percentiles(arr: np.ndarray) -> dict:
    if arr is None:
        return {"Mean": 0.0, "Std": 0.0, "P5": 0.0, "P25": 0.0, "P50": 0.0, "P75": 0.0, "P95": 0.0}
    return {
        "Mean": float(np.mean(arr)),
        "Std": float(np.std(arr)),
        "P5": float(np.percentile(arr, 5)),
        "P25": float(np.percentile(arr, 25)),
        "P50": float(np.percentile(arr, 50)),
        "P75": float(np.percentile(arr, 75)),
        "P95": float(np.percentile(arr, 95))
    }

st.title("🔬 KLA Semiconductor AIR-Net v3 / v4 / v5 System Viewer")
st.caption("Content-Adaptive Semiconductor Image Restoration Evaluation System (128×128 → 256×256)")

try:
    info_dict = load_and_verify_models()
    v3_m = info_dict["v3_model"]
    v3_ver = info_dict["v3_ver"]
    v4_m = info_dict["v4_model"]
    v4_ver = info_dict["v4_ver"]
    v5_m = info_dict["v5_model"]
    v5_ver = info_dict["v5_ver"]
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

st.sidebar.header("📁 Control Panel & Checkpoint Status")

# V3 Status
if v3_ver and v3_ver.is_verified:
    st.sidebar.markdown(f"**AIR-Net v3 Foundation:**\n- ✓ Verified & Loaded\n- File: `{os.path.basename(v3_ver.filepath)}`\n- Params: `{v3_ver.num_parameters:,}`")
else:
    st.sidebar.markdown("**AIR-Net v3 Foundation:**\n- ❌ Checkpoint Unavailable")

# V4 Status
if v4_ver and v4_ver.is_verified:
    st.sidebar.markdown(f"**AIR-Net v4 System:**\n- ✓ Verified & Loaded\n- File: `{os.path.basename(v4_ver.filepath)}`\n- Params: `{v4_ver.num_parameters:,}`")
else:
    st.sidebar.markdown("**AIR-Net v4 System:**\n- ⚠️ Checkpoint Unavailable")

# V5 Status
if v5_ver and v5_ver.is_verified:
    st.sidebar.markdown(f"**AIR-Net v5 System:**\n- ✓ Verified & Loaded\n- File: `{os.path.basename(v5_ver.filepath)}`\n- Params: `{v5_ver.num_parameters:,}`")
else:
    st.sidebar.markdown("**AIR-Net v5 System:**\n- ⚠️ Checkpoint Unavailable (Training in Progress)")

if st.sidebar.button("🔍 Verify Checkpoints"):
    st.sidebar.json({
        "v3_status": v3_ver.status_summary if v3_ver else "NOT_FOUND",
        "v4_status": v4_ver.status_summary if v4_ver else "NOT_FOUND",
        "v5_status": v5_ver.status_summary if v5_ver else "NOT_FOUND"
    })

source_mode = st.sidebar.radio("Select Input Source:", ["Dataset Browser", "Manual 128×128 File Upload"])

lr_raw, gt_raw = None, None
selected_sample_name = "N/A"

train_lr_dir = config.train_lr_dir
train_gt_dir = config.train_gt_dir

if source_mode == "Dataset Browser":
    if os.path.exists(train_lr_dir):
        lr_files = sorted([f for f in os.listdir(train_lr_dir) if f.endswith(".npy")])
        if lr_files:
            selected_file = st.sidebar.selectbox("Select Sample:", lr_files)
            selected_sample_name = selected_file
            lr_raw = np.load(os.path.join(train_lr_dir, selected_file))

            if os.path.exists(train_gt_dir):
                gt_path = os.path.join(train_gt_dir, selected_file)
                if os.path.exists(gt_path):
                    gt_raw = np.load(gt_path)

elif source_mode == "Manual 128×128 File Upload":
    uploaded_lr = st.sidebar.file_uploader("Upload 128×128 Image (.npy, .png, .jpg, .jpeg)", type=["npy", "png", "jpg", "jpeg"])
    uploaded_gt = st.sidebar.file_uploader("Upload Reference 256×256 Ground Truth (Optional)", type=["npy", "png", "jpg", "jpeg"])

    if uploaded_lr:
        selected_sample_name = uploaded_lr.name
        lr_raw = np.load(uploaded_lr) if uploaded_lr.name.endswith(".npy") else Image.open(uploaded_lr)

    if uploaded_gt:
        gt_raw = np.load(uploaded_gt) if uploaded_gt.name.endswith(".npy") else Image.open(uploaded_gt)


# Execution & Central Master Normalization Data Flow
if lr_raw is not None:
    # 1. Master Normalized Input Array (128x128 float32 in [0.0, 1.0])
    display_noisy = normalize_input(lr_raw)
    assert display_noisy.shape == (128, 128), f"Input resolution mismatch: expected (128, 128), got {display_noisy.shape}"

    # 2. Master Normalized Ground Truth Array (256x256 float32 in [0.0, 1.0]) if available
    display_gt = normalize_target(gt_raw) if gt_raw is not None else None
    if display_gt is not None:
        assert display_gt.shape == (256, 256), f"GT resolution mismatch: expected (256, 256), got {display_gt.shape}"

    lr_t = torch.from_numpy(display_noisy).unsqueeze(0).unsqueeze(0).to(DEVICE)
    assert lr_t.shape[-2:] == (128, 128), f"Tensor input resolution mismatch: {lr_t.shape}"

    display_v3, display_v4, display_v5 = None, None, None
    routing_probs = np.ones(5) / 5.0
    raw_indices = {}

    # AIR-Net v3 Inference
    if v3_m is not None and v3_ver and v3_ver.is_verified:
        with torch.no_grad(), torch.inference_mode():
            v3_out = v3_m(lr_t)
            v3_pred_t = v3_out["restored"]
            assert v3_pred_t.shape[-2:] == (256, 256)
            display_v3 = denormalize_output(v3_pred_t)
            routing_probs = v3_out["routing_probs"].squeeze().cpu().numpy()
            raw_indices = v3_m.indexer.compute_indices(display_noisy)

    # AIR-Net v4 Inference
    if v4_m is not None and v4_ver and v4_ver.is_verified:
        with torch.no_grad(), torch.inference_mode():
            v4_out = v4_m(lr_t)
            v4_pred_t = v4_out["restored"]
            assert v4_pred_t.shape[-2:] == (256, 256)
            display_v4 = denormalize_output(v4_pred_t)

    # AIR-Net v5 Inference
    if v5_m is not None and v5_ver and v5_ver.is_verified:
        with torch.no_grad(), torch.inference_mode():
            v5_out = v5_m(lr_t)
            v5_pred_t = v5_out["restored"]
            assert v5_pred_t.shape[-2:] == (256, 256)
            display_v5 = denormalize_output(v5_pred_t)

    # Bicubic 2x Baseline (256x256 float32 in [0.0, 1.0])
    zoom_factors = (256 / display_noisy.shape[0], 256 / display_noisy.shape[1])
    bicubic_raw = zoom(display_noisy, zoom_factors, order=3)
    display_bicubic = prepare_for_display(bicubic_raw)
    assert display_bicubic.shape == (256, 256)

    categories = ["EDGE_DOMINANT", "TEXTURE_DOMINANT", "NOISE_DOMINANT", "SMOOTH_LOW_CONTRAST", "SPARSE_FEATURE"]
    dom_cat = categories[int(np.argmax(routing_probs))]
    rout_dict = {cat: float(p) for cat, p in zip(categories, routing_probs)}


    # =========================================================================
    # SECTION 1: RESTORATION GRID COMPARISON (6 COLUMNS)
    # =========================================================================
    st.markdown("---")
    st.subheader(f"Restoration Grid Comparison — Sample: {selected_sample_name}")

    g1, g2, g3, g4, g5, g6 = st.columns(6)
    with g1:
        st.markdown("**1. NoisyLR**\n128×128")
        display_image(display_noisy)

    with g2:
        st.markdown("**2. Bicubic**\n256×256")
        display_image(display_bicubic)

    with g3:
        st.markdown("**3. AIR-Net v3**\n256×256")
        if display_v3 is not None:
            display_image(display_v3)
        else:
            st.warning("V3 Unavailable")

    with g4:
        st.markdown("**4. AIR-Net v4**\n256×256")
        if display_v4 is not None:
            display_image(display_v4)
        else:
            st.info("V4 Unavailable")

    with g5:
        st.markdown("**5. AIR-Net v5**\n256×256")
        if display_v5 is not None:
            display_image(display_v5)
        else:
            st.info("V5 Unavailable")

    with g6:
        st.markdown("**6. Ground Truth**\n256×256")
        if display_gt is not None:
            display_image(display_gt)
        else:
            st.info("GT N/A")


    # =========================================================================
    # SECTION 2: SOBEL EDGE MAP ANALYSIS (6 COLUMNS)
    # =========================================================================
    st.markdown("---")
    st.subheader("Sobel Edge Map Analysis (Native Resolution)")

    inp_mag = compute_sobel_edge_magnitude(display_noisy)
    bic_mag = compute_sobel_edge_magnitude(display_bicubic)
    v3_mag = compute_sobel_edge_magnitude(display_v3) if display_v3 is not None else None
    v4_mag = compute_sobel_edge_magnitude(display_v4) if display_v4 is not None else None
    v5_mag = compute_sobel_edge_magnitude(display_v5) if display_v5 is not None else None
    gt_mag = compute_sobel_edge_magnitude(display_gt) if display_gt is not None else None

    e1, e2, e3, e4, e5, e6 = st.columns(6)
    e1.markdown("**Input Edge**")
    display_image(prepare_edge_map_display(inp_mag))

    e2.markdown("**Bicubic Edge**")
    display_image(prepare_edge_map_display(bic_mag))

    e3.markdown("**v3 Edge**")
    if v3_mag is not None:
        display_image(prepare_edge_map_display(v3_mag))
    else:
        e3.info("N/A")

    e4.markdown("**v4 Edge**")
    if v4_mag is not None:
        display_image(prepare_edge_map_display(v4_mag))
    else:
        e4.info("N/A")

    e5.markdown("**v5 Edge**")
    if v5_mag is not None:
        display_image(prepare_edge_map_display(v5_mag))
    else:
        e5.info("N/A")

    e6.markdown("**GT Edge**")
    if gt_mag is not None:
        display_image(prepare_edge_map_display(gt_mag))
    else:
        e6.info("N/A")


    # =========================================================================
    # SECTION 3: ABSOLUTE ERROR MAPS (|PREDICTION - GROUND TRUTH|)
    # =========================================================================
    if display_gt is not None:
        st.markdown("---")
        st.subheader("Absolute Error Map Analysis (|Prediction - Ground Truth|)")
        err1, err2, err3 = st.columns(3)

        v3_err = np.abs(display_v3 - display_gt) if display_v3 is not None else None
        v4_err = np.abs(display_v4 - display_gt) if display_v4 is not None else None
        v5_err = np.abs(display_v5 - display_gt) if display_v5 is not None else None
        bic_err = np.abs(display_bicubic - display_gt)

        max_e = max(
            bic_err.max(),
            v3_err.max() if v3_err is not None else 0.0,
            v4_err.max() if v4_err is not None else 0.0,
            v5_err.max() if v5_err is not None else 0.0
        ) + 1e-8

        with err1:
            st.markdown("**AIR-Net v3 Error Map**")
            if v3_err is not None:
                display_image(v3_err / max_e)
                st.caption(f"Max Error: `{v3_err.max():.4f}` | Mean: `{v3_err.mean():.4f}`")
            else:
                st.info("V3 Error N/A")

        with err2:
            st.markdown("**AIR-Net v4 Error Map**")
            if v4_err is not None:
                display_image(v4_err / max_e)
                st.caption(f"Max Error: `{v4_err.max():.4f}` | Mean: `{v4_err.mean():.4f}`")
            else:
                st.info("V4 Error N/A")

        with err3:
            st.markdown("**AIR-Net v5 Error Map**")
            if v5_err is not None:
                display_image(v5_err / max_e)
                st.caption(f"Max Error: `{v5_err.max():.4f}` | Mean: `{v5_err.mean():.4f}`")
            else:
                st.info("V5 Error N/A")


    # =========================================================================
    # SECTION 4: QUANTITATIVE PERFORMANCE COMPARISON (256×256 BASIS)
    # =========================================================================
    if display_gt is not None:
        st.markdown("---")
        st.subheader("📈 Quantitative Performance Comparison (256×256 Basis)")

        try:
            p_bic, g_bic = validate_metric_inputs(display_bicubic, display_gt)
            m_bic = compute_all_metrics(p_bic, g_bic, DEVICE)
            metrics_rows = [{
                "Model": "Bicubic 2x Baseline",
                "PSNR (dB)": f"{m_bic['PSNR (dB)']:.4f}",
                "SSIM": f"{m_bic['SSIM']:.4f}",
                "LPIPS": f"{m_bic['LPIPS']:.4f}"
            }]

            m_v3, m_v4, m_v5 = None, None, None
            if display_v3 is not None:
                p_v3, _ = validate_metric_inputs(display_v3, display_gt)
                m_v3 = compute_all_metrics(p_v3, g_bic, DEVICE)
                metrics_rows.append({"Model": "AIR-Net v3 Foundation", "PSNR (dB)": f"{m_v3['PSNR (dB)']:.4f}", "SSIM": f"{m_v3['SSIM']:.4f}", "LPIPS": f"{m_v3['LPIPS']:.4f}"})

            if display_v4 is not None:
                p_v4, _ = validate_metric_inputs(display_v4, display_gt)
                m_v4 = compute_all_metrics(p_v4, g_bic, DEVICE)
                metrics_rows.append({"Model": "AIR-Net v4 Refined System", "PSNR (dB)": f"{m_v4['PSNR (dB)']:.4f}", "SSIM": f"{m_v4['SSIM']:.4f}", "LPIPS": f"{m_v4['LPIPS']:.4f}"})

            if display_v5 is not None:
                p_v5, _ = validate_metric_inputs(display_v5, display_gt)
                m_v5 = compute_all_metrics(p_v5, g_bic, DEVICE)
                metrics_rows.append({"Model": "AIR-Net v5 High-Fidelity System", "PSNR (dB)": f"{m_v5['PSNR (dB)']:.4f}", "SSIM": f"{m_v5['SSIM']:.4f}", "LPIPS": f"{m_v5['LPIPS']:.4f}"})

            st.table(pd.DataFrame(metrics_rows))

            if m_v3 is not None and m_v5 is not None:
                gain_p = m_v5["PSNR (dB)"] - m_v3["PSNR (dB)"]
                gain_s = m_v5["SSIM"] - m_v3["SSIM"]
                red_l = m_v3["LPIPS"] - m_v5["LPIPS"]
                st.markdown(f"**V5 Improvement over v3:** PSNR: `{gain_p:+.4f} dB` | SSIM: `{gain_s:+.4f}` | LPIPS Reduction: `{red_l:+.4f}`")

        except ValueError as val_err:
            st.error(f"Metric validation error: {val_err}")


    # =========================================================================
    # SECTION 5: INTENSITY DISTRIBUTION ANALYSIS (MEAN, STD, P5, P25, P50, P75, P95)
    # =========================================================================
    st.markdown("---")
    st.subheader("📊 Intensity Distribution & Robust Percentile Analysis")

    p_rows = [
        {"Image": "NoisyLR (Input)", **compute_percentiles(display_noisy)},
        {"Image": "Bicubic 2x", **compute_percentiles(display_bicubic)},
        {"Image": "AIR-Net v3", **compute_percentiles(display_v3)},
        {"Image": "AIR-Net v4", **compute_percentiles(display_v4)},
        {"Image": "AIR-Net v5", **compute_percentiles(display_v5)},
        {"Image": "Ground Truth", **compute_percentiles(display_gt)}
    ]
    st.table(pd.DataFrame(p_rows))


    # =========================================================================
    # SECTION 6: WHY THIS IMAGE WAS ROUTED THIS WAY
    # =========================================================================
    st.markdown("---")
    st.subheader("Why This Image Was Routed This Way")
    exp_text = explain_category_routing(raw_indices, dom_cat, rout_dict)
    st.markdown(f"""
        <div class="high-contrast-card">
            <div class="card-heading">CATEGORY CLASSIFICATION: {dom_cat}</div>
            <div class="card-text">{exp_text}</div>
        </div>
    """, unsafe_allow_html=True)


    # =========================================================================
    # SECTION 7: HIGH-FIDELITY IMAGE DOWNLOADS
    # =========================================================================
    st.markdown("---")
    st.subheader("💾 Download Restored Semiconductor Images (256×256 PNG)")

    d1, d2, d3 = st.columns(3)
    if display_v3 is not None:
        v3_img_uint8 = (display_v3 * 255.0).round().clip(0, 255).astype(np.uint8)
        buf_v3 = io.BytesIO()
        Image.fromarray(v3_img_uint8).save(buf_v3, format="PNG")
        d1.download_button("📥 Download AIR-Net v3 Restored Image", buf_v3.getvalue(), f"airnet_v3_{selected_sample_name.replace('.npy', '')}.png", "image/png")

    if display_v4 is not None:
        v4_img_uint8 = (display_v4 * 255.0).round().clip(0, 255).astype(np.uint8)
        buf_v4 = io.BytesIO()
        Image.fromarray(v4_img_uint8).save(buf_v4, format="PNG")
        d2.download_button("📥 Download AIR-Net v4 Restored Image", buf_v4.getvalue(), f"airnet_v4_{selected_sample_name.replace('.npy', '')}.png", "image/png")

    if display_v5 is not None:
        v5_img_uint8 = (display_v5 * 255.0).round().clip(0, 255).astype(np.uint8)
        buf_v5 = io.BytesIO()
        Image.fromarray(v5_img_uint8).save(buf_v5, format="PNG")
        d3.download_button("📥 Download AIR-Net v5 Restored Image", buf_v5.getvalue(), f"airnet_v5_{selected_sample_name.replace('.npy', '')}.png", "image/png")

else:
    st.info("💡 Select a sample from the sidebar to inspect AIR-Net restoration predictions.")
