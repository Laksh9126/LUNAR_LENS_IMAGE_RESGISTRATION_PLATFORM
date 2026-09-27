import os
import sys
import traceback
from pathlib import Path
from typing import Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from src.preprocessing.phase_congruency import apply_clahe, preprocess_lunar_frame
from src.matching.multi_scale_bridge import LunarScalePyramidMatcher
from src.geometry.tps_magsac import LunarGeometryEngine
from src.metrics.ablation_engine import MetricsAndAblationEngine, generate_dynamic_ablation_df

matcher_engine = LunarScalePyramidMatcher()
geometry_engine = LunarGeometryEngine()
metrics_engine = MetricsAndAblationEngine(abort_threshold_pct=15.0)

def extract_file_path(file_obj: Any) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)

def smart_ingest(img_input: np.ndarray, target_dim: int = 512) -> Tuple[np.ndarray, str]:
    """Smart windowing: cuts a square crop rather than compressing aspect ratios."""
    if img_input is None:
        return None, "No data provided"
    gray = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY) if img_input.ndim == 3 else img_input.copy()
    h, w = gray.shape
    aspect = max(h, w) / max(1, min(h, w))

    if aspect > 1.8:
        crop_size = min(h, w)
        if h > w:
            center_y = int(h * 0.55)
            y1 = max(0, min(h - crop_size, center_y - crop_size // 2))
            patch = gray[y1 : y1 + crop_size, 0:crop_size]
        else:
            center_x = int(w * 0.55)
            x1 = max(0, min(w - crop_size, center_x - crop_size // 2))
            patch = gray[0:crop_size, x1 : x1 + crop_size]
        cropped = cv2.resize(patch, (target_dim, target_dim), interpolation=cv2.INTER_LANCZOS4)
        msg = f"Windowed ({w}x{h}) to {target_dim}x{target_dim}"
    else:
        cropped = cv2.resize(gray, (target_dim, target_dim), interpolation=cv2.INTER_LANCZOS4)
        msg = f"Direct ({w}x{h}) scaled to {target_dim}x{target_dim}"

    return cropped, msg

def render_tie_line_canvas(
    src_img: np.ndarray, tgt_img: np.ndarray, src_pts: np.ndarray, tgt_pts: np.ndarray, inlier_mask: np.ndarray
) -> np.ndarray:
    h, w = src_img.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR) if src_img.ndim == 2 else src_img
    canvas[:, w:] = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR) if tgt_img.ndim == 2 else tgt_img

    if src_pts is None or len(src_pts) == 0:
        return canvas

    mask = inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.ones(len(src_pts), dtype=bool)
    step = max(1, len(src_pts) // 160)
    for i in range(0, len(src_pts), step):
        pt1 = (int(round(src_pts[i][0])), int(round(src_pts[i][1])))
        pt2 = (int(round(tgt_pts[i][0] + w)), int(round(tgt_pts[i][1])))
        if i < len(mask) and mask[i]:
            cv2.line(canvas, pt1, pt2, (45, 212, 191), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 2, (34, 211, 238), -1)
        else:
            cv2.line(canvas, pt1, pt2, (239, 68, 68), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 2, (239, 68, 68), -1)
    return canvas

def create_3d_warp_composite(tgt_img: np.ndarray, warped_src: np.ndarray) -> np.ndarray:
    if warped_src is None or warped_src.size == 0:
        return tgt_img
    t_bgr = cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR) if tgt_img.ndim == 2 else tgt_img.copy()
    w_bgr = cv2.cvtColor(warped_src, cv2.COLOR_GRAY2BGR) if warped_src.ndim == 2 else warped_src.copy()

    valid_mask = warped_src > 0
    if valid_mask.ndim == 3:
        valid_mask = valid_mask[:, :, 0]

    composite = t_bgr.copy()
    alpha = 0.55
    blended = cv2.addWeighted(w_bgr, alpha, t_bgr, 1.0 - alpha, 0)
    composite[valid_mask] = blended[valid_mask]
    return composite

def run_pipeline(src_file, tgt_file, use_pc, threshold_val):
    empty_tie = np.zeros((512, 1024, 3), dtype=np.uint8)
    empty_sq = np.zeros((512, 512, 3), dtype=np.uint8)
    empty_df = pd.DataFrame(columns=["Pipeline Configuration", "Matches", "Inlier Ratio (%)", "RMSE (px)", "Coverage"])

    src_img_path = extract_file_path(src_file)
    tgt_img_path = extract_file_path(tgt_file)

    if not src_img_path or not tgt_img_path:
        return empty_tie, empty_sq, "0", "0", "--", "--", "--", "STANDBY", None, empty_df

    try:
        src_raw = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
        tgt_raw = cv2.imread(tgt_img_path, cv2.IMREAD_UNCHANGED)
        src_gray, _ = smart_ingest(src_raw)
        tgt_gray, _ = smart_ingest(tgt_raw)

        src_proc = preprocess_lunar_frame(src_gray) if use_pc else apply_clahe(src_gray)
        tgt_proc = preprocess_lunar_frame(tgt_gray) if use_pc else apply_clahe(tgt_gray)

        pts_src, pts_tgt, _ = matcher_engine.match_coarse_to_fine(src_proc, tgt_proc)
        num_raw = len(pts_src)
        if num_raw < 4:
            return empty_tie, empty_sq, "0", str(num_raw), "FAIL", "0.0%", "0.00", "ABORTED", None, empty_df

        inliers_src, inliers_tgt, inlier_mask, warped_img, abort = geometry_engine.verify_and_warp(
            src_gray, tgt_gray, pts_src, pts_tgt, threshold_val
        )
        num_inliers = len(inliers_src)
        num_pruned = num_raw - num_inliers
        inlier_ratio = (num_inliers / num_raw) * 100.0 if num_raw > 0 else 0.0

        canvas_result = render_tie_line_canvas(src_gray, tgt_gray, pts_src, pts_tgt, inlier_mask)
        composite_view = create_3d_warp_composite(tgt_gray, warped_img)

        metrics_eval = metrics_engine.evaluate_run(
            regime="Live Inference", raw_count=num_raw, inlier_pts=inliers_src,
            pred_pts=inliers_src, gt_pts=inliers_tgt, img_shape=(512, 512)
        )

        os.makedirs("outputs", exist_ok=True)
        gcp_path = "outputs/live_ground_control_points.csv"
        pd.DataFrame({
            "source_x": inliers_src[:, 0] if num_inliers > 0 else [],
            "source_y": inliers_src[:, 1] if num_inliers > 0 else [],
            "target_x": inliers_tgt[:, 0] if num_inliers > 0 else [],
            "target_y": inliers_tgt[:, 1] if num_inliers > 0 else [],
        }).to_csv(gcp_path, index=False)

        ablation_df = generate_dynamic_ablation_df(
            metrics_eval.rmse_pixels, inlier_ratio, num_inliers, metrics_eval.spatial_coverage_score
        )
        status_text = "ABORT_TRIGGERED (Guardrail)" if abort else "PASSED & CONVERGED"

        return (
            canvas_result, composite_view, str(num_inliers), str(num_pruned),
            f"{metrics_eval.rmse_pixels} px", f"{inlier_ratio:.2f}%",
            str(metrics_eval.spatial_coverage_score), status_text, gcp_path, ablation_df
        )
    except Exception:
        print(f"Pipeline Crash:\n{traceback.format_exc()}")
        return empty_tie, empty_sq, "ERR", "ERR", "ERR", "ERR", "ERR", "CRASH", None, empty_df

# ================= UI & LUNAR CRESCENT BACKGROUND =================
custom_css = """
:root {
    --card-bg: rgba(6, 12, 26, 0.82);
    --border-color: rgba(56, 189, 248, 0.25);
    --primary-cyan: #38bdf8;
    --text-main: #e2e8f0;
}
body, .gradio-container {
    background-image: 
        radial-gradient(circle at 80% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
        linear-gradient(rgba(4, 9, 20, 0.82), rgba(4, 9, 20, 0.88)),
        url('https://images.unsplash.com/photo-1532693322450-2cb5c511067d?auto=format&fit=crop&w=2000&q=80') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    color: var(--text-main) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.custom-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 12px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45) !important;
}
.card-header-bar {
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
    text-transform: uppercase;
}
.init-reg-btn {
    background: linear-gradient(135deg, #0284c7 0%, #2dd4bf 100%) !important;
    border: none !important;
    color: #031024 !important;
    font-weight: 800 !important;
    border-radius: 8px !important;
    letter-spacing: 0.02em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.init-reg-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 16px rgba(45, 212, 191, 0.4) !important;
}
"""

with gr.Blocks(css=custom_css, title="LUNAR LENS v2.4") as demo:
    gr.HTML("""<div style="font-size:1.25rem; font-weight:bold; padding:10px 0; color:#f8fafc;">🛰️ LUNAR LENS <span style="color:#2dd4bf; font-size:0.9rem;">OPTIMIZED REGISTRATION PLATFORM</span></div>""")

    with gr.Row():
        with gr.Column(elem_classes=["custom-card"]):
            gr.HTML('<div class="card-header-bar">SOURCE IMAGE</div>')
            src_in = gr.Image(type="filepath", label="Upload Raster", height=180)
        with gr.Column(elem_classes=["custom-card"]):
            gr.HTML('<div class="card-header-bar">TARGET REFERENCE IMAGE</div>')
            tgt_in = gr.Image(type="filepath", label="Upload Raster", height=180)

    with gr.Row(elem_classes=["custom-card"]):
        pc_toggle = gr.Checkbox(label="Phase Congruency (Strip Shadows)", value=True)
        cutoff_slider = gr.Slider(minimum=1.0, maximum=30.0, value=3.0, step=1.0, label="SAFETY CUTOFF (%)")
        start_btn = gr.Button("🚀 INITIALIZE REGISTRATION", elem_classes=["init-reg-btn"])

    with gr.Row():
        with gr.Column(scale=2, elem_classes=["custom-card"]):
            gr.HTML('<div class="card-header-bar">🔗 TIE-LINE CORRESPONDENCE (Green = Inliers, Red = Pruned)</div>')
            tie_canvas_out = gr.Image(type="numpy", show_label=False, height=310)
        with gr.Column(scale=1, elem_classes=["custom-card"]):
            gr.HTML('<div class="card-header-bar">🧩 3D WARP COMPOSITE</div>')
            warp_out = gr.Image(type="numpy", show_label=False, height=310)

    with gr.Row():
        val_inliers = gr.Textbox(label="Verified Inliers", scale=1)
        val_pruned = gr.Textbox(label="Pruned Outliers", scale=1)
        val_rmse = gr.Textbox(label="Sub-Pixel RMSE", scale=1)
        val_ratio = gr.Textbox(label="Inlier Ratio", scale=1)
        val_coverage = gr.Textbox(label="Spatial Coverage", scale=1)
        val_status = gr.Textbox(label="System Status", scale=1)

    with gr.Row():
        with gr.Column(elem_classes=["custom-card"]):
            gr.HTML('<div class="card-header-bar">🧪 ABLATION STUDY COMPARISON</div>')
            ablation_table = gr.DataFrame(interactive=False)
        with gr.Column(elem_classes=["custom-card"]):
            gr.HTML('<div class="card-header-bar">DELIVERABLES (GCP CSV)</div>')
            gcp_file_out = gr.File(label=None, file_count="single")

    start_btn.click(
        fn=run_pipeline,
        inputs=[src_in, tgt_in, pc_toggle, cutoff_slider],
        outputs=[
            tie_canvas_out, warp_out, val_inliers, val_pruned, val_rmse, val_ratio,
            val_coverage, val_status, gcp_file_out, ablation_table
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)