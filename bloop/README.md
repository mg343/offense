# Bloop

### Background
Descent and low-level flight depend on a continuous altitude reference. Barometric and radio altimeters are the default, but they fail: icing, RF contest, hardware dropout, or baro reference corruption in GPS-denied airspace. When the altimeter drops out mid-mission, the vehicle still has a nadir camera and — if prepared — a local terrain elevation map. Horizontal position may still be known from a system like Sandwalk, dead reckoning, or a last trusted fix. What is missing is height above ground.

### Overall Issue
Without AGL, descent timing, terrain clearance, and landing logic become guesses. IMU double-integration of vertical acceleration drifts within seconds. Barometric backup (if present) shares failure modes with the primary altimeter. Vision-based depth from monocular cues is scale-ambiguous without a metric prior. The vehicle needs a metric altitude estimate that does not rely on the failed altimeter chain.

### Solution (So-Far)
Standard backups are redundant altimeters, radar, or GPS MSL minus a coarse geoid/terrain lookup. Multi-sensor fusion helps until the whole altitude channel is contested. Optical flow and visual odometry track *relative* climb/sink but do not recover absolute AGL after a gap. TERCOM-style systems correlate radar altimetry profiles against DEMs — powerful, but they assume a working ranging altimeter, which is exactly what just failed.

### Why This Fails
Redundant altimeters share environments and often fail together. GPS MSL minus terrain assumes GPS integrity and a DEM lookup that ignores local slope under a wide footprint. Pure visual odometry cannot re-anchor absolute height after dropout. Classical TERCOM still needs a ranging sensor. None of these answer: *given only a nadir image, a known lat/lon, and an onboard DEM, what is my AGL?*

### Bloop
Bloop is an onboard, monocular vision-based altitude estimation system for **altimeter dropout recovery**. It correlates descent-camera imagery against a local terrain elevation map and recovers AGL via **frequency-domain (FFT) phase correlation**, guided — not gated — by the last altimeter reading before failure.

**How it works:**
1. **Altimeter dropout → soft prior**: The last trusted altimeter reading \(A_0\) and time since dropout define a Gaussian belief \(w(h)\) whose width grows with time. This *guides* scoring; it does **not** hard-clip the search range.
2. **Altitude ↔ scale**: For nadir FOV \(\alpha\), ground footprint side length is \(L(h) = 2\,h\,\tan(\alpha/2)\). Each AGL hypothesis crops a differently sized DEM patch around the known lat/lon.
3. **DEM → appearance**: The elevation patch is rendered as shaded relief (hillshade) and resized to the camera resolution — a metric “what the terrain should look like” at that height when the onboard prior is a DEM (not RGB satellite tiles).
4. **FFT phase correlation**: Camera and DEM appearance are phase-correlated in the frequency domain (normalized cross-power spectrum → IFFT). Peak *response* scores the hypothesis; near-zero translation supports the lat/lon assumption.
5. **Combined score**: \(S(h) = \mathrm{response}(h)\cdot w(h)\cdot\mathrm{shift\_penalty}\). Coarse sweep, then refine. Peak sharpness and secondary maxima set confidence / ambiguity flags.

**Key Innovation**: Bloop turns absolute altitude recovery into a **scale-matching** problem in the frequency domain. Wrong height → wrong terrain scale → phase correlation collapses. Right height → ridge/valley structure lines up. The failed altimeter is used only as a fading soft prior, so a bad last reading can still be overridden by vision.

**Operational Security**: Correlation runs onboard on the camera frame and a **pre-cached local DEM**. No ranging emissions, no imagery uplink for the estimate.

**Relationship to Sandwalk**: Sandwalk answers *where am I?* (lat/lon) from satellite RGB under dead-reckoning search. Bloop answers *how high am I?* (AGL) from DEM scale under altimeter dropout. Horizontal in, vertical out.

---

### Data sourcing (do this)

Bloop needs a **metric elevation raster** under the flight fix. The demo ships with a USGS 3DEP crop over the WA Cascades (near Mt Rainier foothills) fetched by `src/fetch_dem.py`. Prefer higher-resolution bare-earth lidar for operational accuracy.

#### A. Quick demo DEM (already automatable)

```bash
cd bloop/src
pip install -r ../requirements.txt
python fetch_dem.py
# writes files/dem.npy + files/dem_meta.json (+ dem_raw.tif)
```

Source: [USGS 3DEP Elevation ImageServer](https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer) `exportImage` over bbox `(-121.85, 46.85, -121.75, 46.95)`.

#### B. Better WA coverage — what to download yourself

| What | Where | Exactly what to grab |
|------|-------|----------------------|
| **Preferred ops DEM** | [WA DNR Lidar Portal](https://lidarportal.dnr.wa.gov/) | Bare-earth DEM GeoTIFF, **1 m** (or finest available) for your AOI. Export / download a **≥ 2 km × 2 km** tile centered on the descent lat/lon so high-AGL footprints still fit. |
| **PSLC / consortium lidar** | [Puget Sound LiDAR Consortium data](https://pugetsoundlidar.ess.washington.edu/lidardata/restricted/index.html) | Same idea: bare-earth DEM for your project area. Use the project list / viewer; prefer vendor bare-earth products that match PSLC specs. DNR portal often mirrors these. |
| **National fallback** | [USGS 3DEP staged GeoTIFF](https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/13/TIFF/) | 1/3 arc-second (~10 m) tile for your 1° cell (e.g. `n47w122` for much of western WA). Heavier and coarser than DNR 1 m — fine for algorithm bring-up, weak for low-AGL precision. |

**After download**, either:

1. Place a float GeoTIFF at `bloop/src/files/dem_raw.tif` and convert:

```bash
cd bloop/src
python - <<'PY'
import json, numpy as np, tifffile as tiff
# Set these to YOUR GeoTIFF georeferencing (or use rasterio if you have it).
# For fetch_dem.py-style crops, re-run fetch with your bbox instead.
arr = tiff.imread("files/dem_raw.tif").astype("float32")
np.save("files/dem.npy", arr)
print(arr.shape, arr.min(), arr.max())
PY
```

2. Or edit `fetch_dem.py` `DEFAULT_BBOX_WSEN` to your AOI and re-fetch a 3DEP crop.

Update `dem_meta.json` so `bbox_wsen`, `metres_per_pixel_*`, and `center_*` match the raster. Wrong metres-per-pixel breaks the altitude↔scale model.

#### C. Camera frame (optional)

| What | Put it here | Notes |
|------|-------------|-------|
| Real nadir descent frame | `bloop/src/files/camera_frame.png` | Same lat/lon as `LAT`/`LON` in `main.py`. Roughly level nadir. |
| None | — | `main.py` synthesizes a hillshade “camera” at `TRUE_AGL_M` for deterministic QA. |

**You should source for a real-world validation run:**

1. WA DNR (or PSLC) **bare-earth DEM** GeoTIFF ≥ 2 km on a side around a known overlook / flight line.
2. One **nadir photo or video frame** over that same fix with a known AGL (GPS+DEM, laser rangefinder, or surveyed height) for truth.
3. Set `LAT`, `LON`, `LAST_ALTIMETER_M` (intentionally offset from truth to simulate dropout drift), and `T_SINCE_DROPOUT_S` in `main.py`.

---

### Run

```bash
cd bloop/src
pip install -r ../requirements.txt
python fetch_dem.py          # if dem.npy missing
python main.py               # single-cycle test → files/output/bloop_test/
```

Hardcoded mission params live in `main()` (Sandwalk style). No API key required for the DEM already on disk.
