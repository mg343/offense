"""
Fetch a small USGS 3DEP DEM crop for bloop.

Default: Mt Rainier foothills / WA Cascades (~46.90 N, 121.80 W).
Writes dem_raw.tif, dem.npy, dem_meta.json into src/files/.

For higher-resolution WA lidar (1 m class), see README → Data sourcing.
You do NOT need this script if dem.npy already exists.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Tuple

import numpy as np
import requests

try:
    import tifffile as tiff
except ImportError:
    print("ERROR: pip install tifffile")
    sys.exit(1)


# Default: rugged Cascade terrain — good spatial frequency content for phase correlation.
DEFAULT_BBOX_WSEN: Tuple[float, float, float, float] = (-121.85, 46.85, -121.75, 46.95)
DEFAULT_SIZE = 1024
IMAGE_SERVER = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)


def fetch_dem_crop(
    bbox_wsen: Tuple[float, float, float, float] = DEFAULT_BBOX_WSEN,
    size: int = DEFAULT_SIZE,
    out_dir: str | None = None,
) -> str:
    west, south, east, north = bbox_wsen
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
    os.makedirs(out_dir, exist_ok=True)

    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "size": f"{size},{size}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "f": "image",
    }
    print(f"[FETCH] GET 3DEP crop bbox={bbox_wsen} size={size}…")
    r = requests.get(IMAGE_SERVER, params=params, timeout=120)
    r.raise_for_status()

    raw_path = os.path.join(out_dir, "dem_raw.tif")
    with open(raw_path, "wb") as f:
        f.write(r.content)
    print(f"[FETCH] Wrote {raw_path} ({len(r.content)} bytes)")

    arr = tiff.imread(raw_path).astype(np.float32)
    h, w = arr.shape[:2]
    lat0 = 0.5 * (south + north)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * float(np.cos(np.radians(lat0)))
    width_m = (east - west) * m_per_deg_lon
    height_m = (north - south) * m_per_deg_lat

    npy_path = os.path.join(out_dir, "dem.npy")
    np.save(npy_path, arr)

    meta = {
        "source": "USGS 3DEP Elevation ImageServer exportImage",
        "url": IMAGE_SERVER,
        "bbox_wsen": [west, south, east, north],
        "crs": "EPSG:4326",
        "size_px": [int(w), int(h)],
        "elevation_units": "meters",
        "center_lat": float(0.5 * (south + north)),
        "center_lon": float(0.5 * (west + east)),
        "metres_per_pixel_x": float(width_m / w),
        "metres_per_pixel_y": float(height_m / h),
        "notes": "WA Cascades DEM crop for bloop. Prefer WA DNR/PSLC 1m lidar for ops.",
    }
    meta_path = os.path.join(out_dir, "dem_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[FETCH] elev range {arr.min():.1f}–{arr.max():.1f} m")
    print(f"[FETCH] Wrote {npy_path} + {meta_path}")
    return npy_path


def main():
    fetch_dem_crop()


if __name__ == "__main__":
    main()
