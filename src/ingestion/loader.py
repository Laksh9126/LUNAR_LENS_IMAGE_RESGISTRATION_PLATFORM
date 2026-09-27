import os
import pvl
import rasterio
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

def parse_pds4_metadata(xml_path: str) -> Dict[str, Any]:
    """
    Parses ISRO PDS4 XML labels to extract Solar Azimuth, Solar Elevation,
    and Ground Sampling Distance (GSD)[cite: 14, 15].
    """
    if not xml_path or not os.path.exists(xml_path):
        return {"solar_azimuth": None, "solar_elevation": None, "gsd": None, "status": "No XML provided"}
    try:
        label = pvl.load(xml_path)
        return {
            "solar_azimuth": float(label.get('SolarAzimuth', 0.0)),
            "solar_elevation": float(label.get('SolarElevation', 0.0)),
            "gsd": float(label.get('PixelResolution', 0.0)),
            "status": "success"
        }
    except Exception as e:
        return {"solar_azimuth": None, "solar_elevation": None, "gsd": None, "status": f"error: {e}"}

def load_lunar_raster(tif_path: str) -> Tuple[np.ndarray, Any, Any]:
    """
    Loads a 16-bit GeoTIFF lunar raster mapped to the IAU 2000 Moon CRS,
    applying NoData masking[cite: 14, 15].
    """
    with rasterio.open(tif_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs
        if src.nodata is not None:
            image = np.where(image == src.nodata, 0, image)
        return image, transform, crs

def initialize_split_regime_ledgers() -> None:
    """
    Initializes the two distinct ground-truth regimes required for shadow-invariance validation:
    1. Equatorial Set (Apollo 11/17)
    2. High-Shadow/Polar Set (OHRC/TMC pair)
    """
    regimes = {
        "equatorial": "data/gold_eval_suite/equatorial_apollo/ground_truth_gcp.csv",
        "polar_high_shadow": "data/gold_eval_suite/polar_high_shadow/ground_truth_gcp.csv"
    }
    columns = [
        "pair_id", "source_x", "source_y", "target_x", "target_y",
        "lat", "lon", "confidence", "regime_type", "site_name"
    ]
    for regime_name, path in regimes.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            print(f"Initialized Ground Truth Ledger ({regime_name}): {path}")

if __name__ == "__main__":
    initialize_split_regime_ledgers()
    print("Module 1 (Data Ingestion & Ground Truth) fully initialized.")