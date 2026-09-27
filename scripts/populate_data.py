import os
import cv2
import numpy as np
import pandas as pd

def create_crater_pair(output_dir, high_shadow=False):
    os.makedirs(output_dir, exist_ok=True)
    h, w = 512, 512

    # 1. Base synthetic crater terrain
    base = np.full((h, w), 120, dtype=np.uint8)
    cv2.circle(base, (256, 256), 70, 45, -1)   # Main crater floor
    cv2.circle(base, (256, 256), 70, 220, 3)   # Main crater rim
    cv2.circle(base, (160, 180), 30, 50, -1)
    cv2.circle(base, (160, 180), 30, 210, 2)
    cv2.circle(base, (360, 340), 40, 35, -1)
    cv2.circle(base, (360, 340), 40, 230, 3)

    # Regolith texture noise
    noise = np.random.normal(0, 8, (h, w)).astype(np.int16)
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Target Reference Image
    target_img = base.copy()

    # Source Image: zoom & rotation
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 12, 1.15)
    source_img = cv2.warpAffine(base, M, (w, h))

    # Cast severe low-sun terminator shadows if requested
    if high_shadow:
        shadow = np.zeros_like(source_img)
        cv2.rectangle(shadow, (100, 0), (320, 512), 255, -1)
        source_img = np.where(shadow == 255, (source_img * 0.15).astype(np.uint8), source_img)

    cv2.imwrite(os.path.join(output_dir, "source.png"), source_img)
    cv2.imwrite(os.path.join(output_dir, "target.png"), target_img)

    # Benchmark Ground Truth GCPs
    mock_gt = pd.DataFrame([
        {"pair_id": "pair_01", "source_x": 256.0, "source_y": 256.0, "target_x": 256.0, "target_y": 256.0, "lat": 0.674, "lon": 23.473, "confidence": 1.0},
        {"pair_id": "pair_02", "source_x": 160.0, "source_y": 180.0, "target_x": 156.0, "target_y": 184.0, "lat": 0.675, "lon": 23.474, "confidence": 1.0},
        {"pair_id": "pair_03", "source_x": 360.0, "source_y": 340.0, "target_x": 358.0, "target_y": 342.0, "lat": 0.673, "lon": 23.471, "confidence": 1.0},
    ])
    mock_gt.to_csv(os.path.join(output_dir, "ground_truth_gcp.csv"), index=False)

# Populate both evaluation folders
create_crater_pair("data/gold_eval_suite/equatorial_apollo", high_shadow=False)
create_crater_pair("data/gold_eval_suite/polar_high_shadow", high_shadow=True)

print("Successfully filled gold_eval_suite/ folders with test tiles and GCPs.")