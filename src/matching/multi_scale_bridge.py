import cv2
import torch
import numpy as np
from typing import Tuple
import kornia.feature as kf

class LunarScalePyramidMatcher:
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        # LoFTR outdoor cross-attention transfers zero-shot to edge-enhanced lunar topography
        self.matcher = kf.LoFTR(pretrained="outdoor").to(self.device).eval()

    def _prepare_tensor(self, img_gray: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Ensures dimensions are multiples of 8 required by transformer attention."""
        h, w = img_gray.shape[:2]
        new_h = max(64, (h // 8) * 8)
        new_w = max(64, (w // 8) * 8)
        if new_h != h or new_w != w:
            resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = img_gray
        tensor = torch.from_numpy(resized).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device), (h, w, new_h, new_w)

    @torch.inference_mode()
    def match_pair(self, img_src: np.ndarray, img_tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_src, (hs, ws, nhs, nws) = self._prepare_tensor(img_src)
        t_tgt, (ht, wt, nht, nwt) = self._prepare_tensor(img_tgt)

        output = self.matcher({"image0": t_src, "image1": t_tgt})
        pts0 = output["keypoints0"].cpu().numpy()
        pts1 = output["keypoints1"].cpu().numpy()
        conf = output["confidence"].cpu().numpy()

        # Remap back to input crop space if resized to multiples of 8
        if len(pts0) > 0 and (nhs != hs or nws != ws):
            pts0[:, 0] *= (ws / nws)
            pts0[:, 1] *= (hs / nhs)
        if len(pts1) > 0 and (nht != ht or nwt != wt):
            pts1[:, 0] *= (wt / nwt)
            pts1[:, 1] *= (ht / nht)

        return pts0, pts1, conf

    def match_coarse_to_fine(
        self, img_src: np.ndarray, img_tgt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs full-resolution cross-attention. If low matches (<25) are detected due to
        scale differences, steps through a Gaussian octave pyramid to discover more points.
        """
        pts_s, pts_t, conf = self.match_pair(img_src, img_tgt)

        # Dynamic fallback: if initial pass is sparse, downsample source by 0.5x to bridge scale gap
        if len(pts_s) < 25:
            down_s = cv2.pyrDown(img_src)
            scale_y = img_src.shape[0] / down_s.shape[0]
            scale_x = img_src.shape[1] / down_s.shape[1]

            p_s_pyr, p_t_pyr, conf_pyr = self.match_pair(down_s, img_tgt)
            if len(p_s_pyr) > len(pts_s):
                p_s_pyr[:, 0] *= scale_x
                p_s_pyr[:, 1] *= scale_y
                return p_s_pyr, p_t_pyr, conf_pyr

        return pts_s, pts_t, conf