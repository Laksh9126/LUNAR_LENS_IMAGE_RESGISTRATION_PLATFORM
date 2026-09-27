import cv2
import numpy as np
from scipy.interpolate import Rbf
from typing import Tuple

class LunarGeometryEngine:
    def __init__(self, reproj_threshold: float = 3.5, max_iters: int = 4000):
        self.reproj_threshold = reproj_threshold
        self.max_iters = max_iters

    def _fast_warp_tps(
        self,
        src_img: np.ndarray,
        src_pts: np.ndarray,
        tgt_pts: np.ndarray,
        out_shape: Tuple[int, int],
        H_fallback: np.ndarray
    ) -> np.ndarray:
        ht, wt = out_shape[:2]
        if len(src_pts) < 4:
            return cv2.warpPerspective(src_img, H_fallback, (wt, ht), flags=cv2.INTER_LANCZOS4)

        try:
            # Subsample to maximum 30 anchors to prevent O(N^3) CPU bottleneck
            max_anchors = 30
            step = max(1, len(src_pts) // max_anchors)
            sp_s = src_pts[::step]
            sp_t = tgt_pts[::step]

            # Warp across low-res mesh and upscale displacement maps
            low_h, low_w = ht // 2, wt // 2
            rbf_x = Rbf(sp_t[:, 0], sp_t[:, 1], sp_s[:, 0], function="thin_plate", smooth=0.5)
            rbf_y = Rbf(sp_t[:, 0], sp_t[:, 1], sp_s[:, 1], function="thin_plate", smooth=0.5)

            grid_y, grid_x = np.indices((low_h, low_w))
            map_x = rbf_x(grid_x.ravel() * 2, grid_y.ravel() * 2).reshape(low_h, low_w).astype(np.float32)
            map_y = rbf_y(grid_x.ravel() * 2, grid_y.ravel() * 2).reshape(low_h, low_w).astype(np.float32)

            full_map_x = cv2.resize(map_x, (wt, ht), interpolation=cv2.INTER_LINEAR)
            full_map_y = cv2.resize(map_y, (wt, ht), interpolation=cv2.INTER_LINEAR)

            return cv2.remap(src_img, full_map_x, full_map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        except Exception:
            return cv2.warpPerspective(src_img, H_fallback, (wt, ht), flags=cv2.INTER_LANCZOS4)

    def verify_and_warp(
        self,
        src_gray: np.ndarray,
        tgt_gray: np.ndarray,
        mkpts_src: np.ndarray,
        mkpts_tgt: np.ndarray,
        failure_threshold: float = 15.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        if mkpts_src is None or len(mkpts_src) < 4:
            return np.empty((0, 2)), np.empty((0, 2)), None, None, True

        pts_src = np.ascontiguousarray(mkpts_src, dtype=np.float32)
        pts_tgt = np.ascontiguousarray(mkpts_tgt, dtype=np.float32)

        # USAC_MAGSAC handles uneven lighting and outliers
        H, inlier_mask = cv2.findHomography(
            pts_src,
            pts_tgt,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self.reproj_threshold,
            maxIters=self.max_iters,
            confidence=0.995
        )

        if inlier_mask is None or H is None:
            mask = np.zeros(len(pts_src), dtype=bool)
            return np.empty((0, 2)), np.empty((0, 2)), mask, None, True

        mask = inlier_mask.ravel().astype(bool)
        inliers_src = pts_src[mask]
        inliers_tgt = pts_tgt[mask]

        num_inliers = len(inliers_src)
        inlier_ratio = (num_inliers / len(pts_src)) * 100.0 if len(pts_src) > 0 else 0.0

        # Adaptive Guardrail: passes if ratio condition is met OR 10+ solid geometric inliers exist
        abort_triggered = (inlier_ratio < failure_threshold) and (num_inliers < 10)

        warped_src = self._fast_warp_tps(src_gray, inliers_src, inliers_tgt, tgt_gray.shape, H)

        return inliers_src, inliers_tgt, mask, warped_src, abort_triggered