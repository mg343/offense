# Drift

Terrain-Based Visual Velocity Estimation via Monocular Feature Tracking.

Tracks statistically salient **color** patches frame-to-frame and converts pixel motion to metres per second using known altitude and camera field of view.

---

### Background

GPS-denied navigation needs more than a one-shot map fix. Dead reckoning requires **velocity** — how fast and which way the vehicle moves over the ground — so the next search window (e.g. Sandwalk's half-annulus) is centered in the right place. IMU integration drifts in seconds. Baro gives climb rate, not ground track.

### Problem

From a nadir camera alone, at known constant altitude: estimate horizontal ground velocity (m/s) and heading without feature detectors hidden inside OpenCV black boxes.

### Approach

1. **Statistical color features (LAB z-score saliency)** — For each pixel, measure how far its `(L,a,b)` vector deviates from the local neighbourhood mean/std. High score = chromatically distinct patch (field boundary, roof, water). We use **LAB** because it separates lightness from chroma; **local z-scores** because nadir motion tracking needs coloured terrain blobs, not generic grey corners (Harris) or opaque descriptors (SIFT/ORB/`goodFeaturesToTrack`).

2. **Patch NCC tracking** — Same feature cannot be identified by image coordinates alone (everything moves). Each feature carries a LAB patch; the next frame is searched locally for the best normalized cross-correlation match.

3. **Sliding 5-frame window** — Per output frame, analyse up to 5 consecutive frames centered on that frame, clipped at stream start/end. Tracks chain across the window; endpoint pixel displacement → ground velocity via GSD.

4. **Monocular nadir kinematics** — Ground footprint from altitude + FOV → metres/pixel. Apparent patch motion `(Δu, Δv)` over `Δt` implies drone velocity `v_east ≈ -(Δu/Δt)·GSD_x`, `v_north ≈ -(Δv/Δt)·GSD_y`. Median across tracks for robustness.

---

### Run

```bash
cd drift/src
pip install -r ../requirements.txt
python main.py
```

Drop your clip at `drift/src/files/stream.mp4`. If missing, a synthetic scrolling demo is generated automatically.

### Knobs (top of `main.py`)

| Variable | Meaning |
|----------|---------|
| `ALTITUDE_M` | Constant AGL (metres) |
| `FPS` | Video frame rate |
| `FOV_DEG` | Vertical field of view |
| `VIDEO_PATH` | Input mp4 |

### Output

- Terminal log: per-frame speed, heading, window indices
- `files/output/drift_playback.gif` — video with yellow feature tracks, red velocity arrow from frame centre, HUD text (speed, heading, cardinal)
