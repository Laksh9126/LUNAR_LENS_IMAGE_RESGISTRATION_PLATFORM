import cv2
import numpy as np
from typing import Tuple


def run_classical_sift(
        img_src: np.ndarray,
        img_tgt: np.ndarray,
        ratio_thresh: float = 0.75
) -> Tuple[int, int, float, np.ndarray]:
    """
    Runs SIFT + Lowe's Ratio Test + MAGSAC++ on raw images.
    Demonstrates failure caused by moving shadow gradients[cite: 1, 2].
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_src, None)
    kp2, des2 = sift.detectAndCompute(img_tgt, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, 0, 0.0, np.zeros_like(img_src)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    raw_count = len(good_matches)
    if raw_count < 4:
        return raw_count, 0, 0.0, np.zeros_like(img_src)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    inliers = int(np.sum(mask)) if mask is not None else 0
    inlier_ratio = (inliers / raw_count) * 100.0 if raw_count > 0 else 0.0

    vis = cv2.drawMatches(
        img_src, kp1, img_tgt, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return raw_count, inliers, inlier_ratio, vis


def run_classical_orb(
        img_src: np.ndarray,
        img_tgt: np.ndarray
) -> Tuple[int, int, float, np.ndarray]:
    """Runs ORB feature extraction as a secondary classical baseline[cite: 1, 2]."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img_src, None)
    kp2, des2 = orb.detectAndCompute(img_tgt, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, 0, 0.0, np.zeros_like(img_src)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    raw_count = len(matches)

    if raw_count < 4:
        return raw_count, 0, 0.0, np.zeros_like(img_src)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    inliers = int(np.sum(mask)) if mask is not None else 0
    inlier_ratio = (inliers / raw_count) * 100.0 if raw_count > 0 else 0.0

    vis = cv2.drawMatches(
        img_src, kp1, img_tgt, kp2, matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return raw_count, inliers, inlier_ratio, vis


if __name__ == "__main__":
    t1 = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    t2 = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    raw, inl, ratio, _ = run_classical_sift(t1, t2)
    print(f"SIFT baseline operational: Raw={raw}, Inliers={inl}, Ratio={ratio:.2f}%")