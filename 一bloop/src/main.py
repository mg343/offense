"""
Bloop — Single Cycle Test
Altimeter dropout → recover AGL via FFT phase correlation of nadir camera vs onboard DEM.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# DEM I/O
# ---------------------------------------------------------------------------

def load_dem(files_dir: str) -> Tuple[np.ndarray, dict]:
    """Load dem.npy + dem_meta.json (produced by fetch_dem.py or README steps)."""
    npy_path = os.path.join(files_dir, "dem.npy")
    meta_path = os.path.join(files_dir, "dem_meta.json")
    if not os.path.exists(npy_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing DEM at {npy_path} / {meta_path}. "
            "Run: python fetch_dem.py   OR follow README data sourcing."
        )
    dem = np.load(npy_path).astype(np.float32)
    with open(meta_path) as f:
        meta = json.load(f)
    return dem, meta


def latlon_to_dem_px(lat: float, lon: float, meta: dict) -> Tuple[float, float]:
    """Map lat/lon → fractional (col, row) in the DEM array. row 0 = north."""
    west, south, east, north = meta["bbox_wsen"]
    w, h = meta["size_px"]
    col = (lon - west) / (east - west) * (w - 1)
    row = (north - lat) / (north - south) * (h - 1)
    return col, row


def ground_elevation_m(dem: np.ndarray, lat: float, lon: float, meta: dict) -> float:
    col, row = latlon_to_dem_px(lat, lon, meta)
    c = int(round(col))
    r = int(round(row))
    c = max(0, min(dem.shape[1] - 1, c))
    r = max(0, min(dem.shape[0] - 1, r))
    return float(dem[r, c])


# ---------------------------------------------------------------------------
# Camera / footprint geometry
# ---------------------------------------------------------------------------

def footprint_side_m(altitude_agl_m: float, fov_deg: float) -> float:
    """Square ground footprint side length for a nadir camera with vertical FOV."""
    if altitude_agl_m <= 0:
        return 0.0
    return 2.0 * altitude_agl_m * math.tan(math.radians(fov_deg) / 2.0)


# ---------------------------------------------------------------------------
# DEM → appearance (hillshade)
# ---------------------------------------------------------------------------

def render_hillshade(
    elev: np.ndarray,
    mpp_x: float,
    mpp_y: float,
    azimuth_deg: float = 315.0,
    altitude_sun_deg: float = 45.0,
) -> np.ndarray:
    """
    Classic GIS hillshade from an elevation patch.
    Returns float32 image in [0, 1], same shape as elev.
    """
    # dz/dx, dz/dy in elevation units per metre
    gy, gx = np.gradient(elev, mpp_y, mpp_x)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)

    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_sun_deg)
    shaded = (
        math.sin(alt) * np.sin(slope)
        + math.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    )
    shaded = np.clip(shaded, 0.0, 1.0).astype(np.float32)
    return shaded


def extract_dem_patch_for_altitude(
    dem: np.ndarray,
    meta: dict,
    lat: float,
    lon: float,
    altitude_agl_m: float,
    fov_deg: float,
    out_px: int,
) -> Optional[np.ndarray]:
    """
    Crop the DEM to the ground footprint implied by (altitude, FOV), render
    hillshade, resize to out_px×out_px. Returns None if crop leaves the DEM.
    """
    side_m = footprint_side_m(altitude_agl_m, fov_deg)
    if side_m <= 0:
        return None

    mpp_x = float(meta["metres_per_pixel_x"])
    mpp_y = float(meta["metres_per_pixel_y"])
    col, row = latlon_to_dem_px(lat, lon, meta)

    half_w = (side_m / 2.0) / mpp_x
    half_h = (side_m / 2.0) / mpp_y
    c0 = int(math.floor(col - half_w))
    c1 = int(math.ceil(col + half_w))
    r0 = int(math.floor(row - half_h))
    r1 = int(math.ceil(row + half_h))

    if c0 < 0 or r0 < 0 or c1 >= dem.shape[1] or r1 >= dem.shape[0]:
        return None
    if (c1 - c0) < 4 or (r1 - r0) < 4:
        return None

    patch = dem[r0:r1, c0:c1]
    shade = render_hillshade(patch, mpp_x, mpp_y)
    resized = cv2.resize(shade, (out_px, out_px), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocess camera → same domain as hillshade
# ---------------------------------------------------------------------------

def preprocess_camera(frame_bgr: np.ndarray, out_px: int) -> np.ndarray:
    """
    Map a camera frame into a float32 [0,1] appearance comparable to hillshade:
    gray → blur → equalize → mild sharpen → resize → normalize.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(gray, -1, kernel)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    resized = cv2.resize(sharp, (out_px, out_px), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)


def synthesize_camera_from_dem(
    dem: np.ndarray,
    meta: dict,
    lat: float,
    lon: float,
    true_agl_m: float,
    fov_deg: float,
    out_px: int,
    noise_std: float = 0.04,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    Build a fake nadir camera frame by rendering hillshade at the true altitude,
    then degrading it slightly. Used for deterministic single-cycle validation
    when no real descent imagery is present.
    """
    shade = extract_dem_patch_for_altitude(
        dem, meta, lat, lon, true_agl_m, fov_deg, out_px,
    )
    if shade is None:
        raise RuntimeError(
            f"TRUE_AGL={true_agl_m}m footprint does not fit inside DEM — "
            "lower altitude or fetch a wider DEM."
        )
    img = shade.copy()
    if blur_ksize and blur_ksize >= 3:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    rng = np.random.default_rng(42)
    img = img + rng.normal(0.0, noise_std, img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    # to BGR uint8 for the same pipeline path as a real camera
    u8 = (img * 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Frequency-domain phase correlation (the quintessential FFT match)
# ---------------------------------------------------------------------------

def fft_phase_correlate(
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Phase correlation via normalized cross-power spectrum.

        Fa, Fb = FFT(a), FFT(b)
        R      = Fa * conj(Fb) / |Fa * conj(Fb)|
        r      = IFFT(R)

    Returns (dx, dy, response, correlation_surface) where (dx, dy) is the
    translation that best aligns b onto a (OpenCV convention via phaseCorrelate
    windowing), and response ∈ [0, 1] is peak strength.

    We apply a Hanning window (same idea as cv2.phaseCorrelate) to suppress
    edge discontinuities before the FFT.
    """
    assert a.shape == b.shape
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)

    # zero-mean
    a64 = a64 - a64.mean()
    b64 = b64 - b64.mean()

    win = cv2.createHanningWindow((a64.shape[1], a64.shape[0]), cv2.CV_64F)
    a64 = a64 * win
    b64 = b64 * win

    Fa = np.fft.fft2(a64)
    Fb = np.fft.fft2(b64)
    cross = Fa * np.conj(Fb)
    cross /= np.abs(cross) + 1e-12
    r = np.fft.ifft2(cross)
    r = np.fft.fftshift(np.abs(r))

    peak_idx = np.unravel_index(np.argmax(r), r.shape)
    cy, cx = r.shape[0] // 2, r.shape[1] // 2
    dy = float(peak_idx[0] - cy)
    dx = float(peak_idx[1] - cx)

    # response: peak relative to surface energy (stable 0–1-ish confidence)
    peak = float(r[peak_idx])
    mean_r = float(r.mean()) + 1e-12
    response = peak / (peak + 8.0 * mean_r)  # soft normalize; sharper peaks → higher

    # also get OpenCV's calibrated response for a second opinion
    shift, cv_resp = cv2.phaseCorrelate(a64, b64)
    # prefer cv response when available (well-scaled); keep our surface for viz
    response = float(max(0.0, min(1.0, cv_resp)))
    dx, dy = float(shift[0]), float(shift[1])
    return dx, dy, response, r.astype(np.float32)


# ---------------------------------------------------------------------------
# Soft altimeter prior (guide, not a hard gate)
# ---------------------------------------------------------------------------

def soft_prior_weight(
    h_m: float,
    last_altimeter_m: float,
    t_since_dropout_s: float,
    sigma0_m: float = 80.0,
    sigma_dot_m_per_s: float = 2.0,
) -> float:
    """
    Gaussian belief centered on the last altimeter reading.
    σ widens with time since dropout — no hard clip of the search range.
    """
    sigma = sigma0_m + sigma_dot_m_per_s * max(0.0, t_since_dropout_s)
    sigma = max(sigma, 1.0)
    z = (h_m - last_altimeter_m) / sigma
    return float(math.exp(-0.5 * z * z))


def peak_sharpness(scores: np.ndarray, peak_idx: int, neighbor: int = 3) -> float:
    """How much the winning score stands above its local neighborhood (0–1)."""
    n = len(scores)
    lo = max(0, peak_idx - neighbor)
    hi = min(n, peak_idx + neighbor + 1)
    peak = float(scores[peak_idx])
    if peak <= 1e-12:
        return 0.0
    mask = np.ones(hi - lo, dtype=bool)
    mask[peak_idx - lo] = False
    if not mask.any():
        return 1.0
    local = scores[lo:hi][mask]
    return float(np.clip((peak - local.mean()) / peak, 0.0, 1.0))


def parabolic_peak_agl(hs: np.ndarray, scores: np.ndarray, peak_idx: int) -> float:
    """Sub-grid AGL from a 3-point parabola on the combined score curve."""
    if peak_idx <= 0 or peak_idx >= len(scores) - 1:
        return float(hs[peak_idx])
    y0, y1, y2 = float(scores[peak_idx - 1]), float(scores[peak_idx]), float(scores[peak_idx + 1])
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return float(hs[peak_idx])
    delta = (y0 - y2) / denom  # offset in index units, expected in [-0.5, 0.5]
    delta = float(np.clip(delta, -0.75, 0.75))
    # assume locally uniform spacing around the peak
    step = 0.5 * ((hs[peak_idx] - hs[peak_idx - 1]) + (hs[peak_idx + 1] - hs[peak_idx]))
    return float(hs[peak_idx] + delta * step)


# ---------------------------------------------------------------------------
# Altitude sweep
# ---------------------------------------------------------------------------

def estimate_altitude(
    camera_f32: np.ndarray,
    dem: np.ndarray,
    meta: dict,
    lat: float,
    lon: float,
    fov_deg: float,
    last_altimeter_m: float,
    t_since_dropout_s: float,
    h_min_m: float = 100.0,
    h_max_m: float = 2500.0,
    coarse_step_m: float = 40.0,
    refine_half_width_m: float = 80.0,
    refine_step_m: float = 5.0,
) -> Dict:
    """
    Sweep AGL hypotheses, score each with phase-corr response × soft prior.
    Coarse then refine. Returns full curve + estimate + diagnostics.
    """
    cam_px = camera_f32.shape[0]
    candidates: List[float] = list(
        np.arange(h_min_m, h_max_m + 1e-6, coarse_step_m)
    )

    def score_one(h: float) -> Optional[dict]:
        appearance = extract_dem_patch_for_altitude(
            dem, meta, lat, lon, h, fov_deg, cam_px,
        )
        if appearance is None:
            return None
        dx, dy, resp, surface = fft_phase_correlate(camera_f32, appearance)
        prior = soft_prior_weight(h, last_altimeter_m, t_since_dropout_s)
        shift_mag = math.hypot(dx, dy)
        # soft penalty if translation is large (lat/lon assumption fraying)
        shift_pen = math.exp(-0.5 * (shift_mag / (0.15 * cam_px)) ** 2)
        combined = resp * prior * shift_pen
        return {
            "h": h,
            "response": resp,
            "prior": prior,
            "shift_pen": shift_pen,
            "dx": dx,
            "dy": dy,
            "combined": combined,
            "appearance": appearance,
            "surface": surface,
        }

    print(f"[SWEEP] Coarse {h_min_m:.0f}–{h_max_m:.0f} m step {coarse_step_m:.0f} m "
          f"({len(candidates)} hyp)…")
    coarse_rows = []
    for h in candidates:
        row = score_one(h)
        if row is not None:
            coarse_rows.append(row)

    if not coarse_rows:
        raise RuntimeError("No altitude hypotheses fit inside the DEM — widen DEM or lower h_max.")

    coarse_best = max(coarse_rows, key=lambda r: r["combined"])
    h0 = coarse_best["h"]
    print(f"[SWEEP] Coarse peak @ {h0:.0f} m  (S={coarse_best['combined']:.4f}, "
          f"resp={coarse_best['response']:.4f}, prior={coarse_best['prior']:.3f})")

    refine_hs = list(np.arange(
        max(h_min_m, h0 - refine_half_width_m),
        min(h_max_m, h0 + refine_half_width_m) + 1e-6,
        refine_step_m,
    ))
    print(f"[SWEEP] Refine around {h0:.0f} m ({len(refine_hs)} hyp)…")
    refine_rows = []
    for h in refine_hs:
        row = score_one(h)
        if row is not None:
            refine_rows.append(row)

    # merge curves (prefer refine values where they overlap)
    by_h: Dict[float, dict] = {round(r["h"], 3): r for r in coarse_rows}
    for r in refine_rows:
        by_h[round(r["h"], 3)] = r
    curve = [by_h[k] for k in sorted(by_h.keys())]

    combined = np.array([r["combined"] for r in curve], dtype=np.float64)
    hs_arr = np.array([r["h"] for r in curve], dtype=np.float64)
    peak_i = int(np.argmax(combined))
    best = curve[peak_i]
    sharp = peak_sharpness(combined, peak_i)
    agl_m = parabolic_peak_agl(hs_arr, combined, peak_i)

    # ambiguity: second peak within 85% of best
    order = np.argsort(combined)[::-1]
    ambiguous = False
    if len(order) > 1:
        second = curve[int(order[1])]
        if second["combined"] > 0.85 * best["combined"] and abs(second["h"] - best["h"]) > 60:
            ambiguous = True

    confidence = float(
        np.clip(best["combined"] * (0.5 + 0.5 * sharp) * (0.7 if ambiguous else 1.0), 0.0, 1.0)
    )

    return {
        "agl_m": agl_m,
        "confidence": confidence,
        "sharpness": sharp,
        "ambiguous": ambiguous,
        "best": best,
        "curve": curve,
        "coarse_best_h": h0,
    }


# ---------------------------------------------------------------------------
# Visualization / QA
# ---------------------------------------------------------------------------

def _to_u8(img_f32: np.ndarray) -> np.ndarray:
    return np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)


def save_score_curve(
    output_dir: str,
    curve: List[dict],
    estimate_m: float,
    last_altimeter_m: float,
    true_agl_m: Optional[float],
) -> None:
    hs = [r["h"] for r in curve]
    resp = [r["response"] for r in curve]
    prior = [r["prior"] for r in curve]
    comb = [r["combined"] for r in curve]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hs, resp, color="#6b7c5c", lw=1.5, label="phase-corr response")
    ax.plot(hs, prior, color="#3d5a80", lw=1.5, ls="--", label="soft altimeter prior")
    ax.plot(hs, comb, color="#1a1c18", lw=2.2, label="combined S(h)")
    ax.axvline(estimate_m, color="#8b3a2a", lw=1.8, label=f"estimate {estimate_m:.0f} m")
    ax.axvline(last_altimeter_m, color="#3d5a80", lw=1.2, ls=":", label=f"last altimeter {last_altimeter_m:.0f} m")
    if true_agl_m is not None:
        ax.axvline(true_agl_m, color="#2f6b3a", lw=1.5, ls="-.", label=f"truth {true_agl_m:.0f} m")
    ax.set_xlabel("AGL altitude hypothesis (m)")
    ax.set_ylabel("score")
    ax.set_title("Bloop — phase correlation × soft prior vs altitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "05_score_curve.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[VIZ] Saved {path}")


def save_metrics_figure(
    output_dir: str,
    estimate_m: float,
    last_altimeter_m: float,
    confidence: float,
    sharpness: float,
    true_agl_m: Optional[float],
    t_since_dropout_s: float,
    ground_elev_m: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    err_line = ""
    if true_agl_m is not None:
        err_line = f"error vs truth      : {estimate_m - true_agl_m:+.1f} m\n"
    text = (
        "BLOOP — ALTIMETER DROPOUT RECOVERY\n"
        "─────────────────────────────────\n"
        f"estimated AGL       : {estimate_m:.1f} m\n"
        f"last altimeter      : {last_altimeter_m:.1f} m\n"
        f"Δ from last alt     : {estimate_m - last_altimeter_m:+.1f} m\n"
        f"time since dropout  : {t_since_dropout_s:.1f} s\n"
        f"{err_line}"
        f"confidence          : {confidence:.3f}\n"
        f"peak sharpness      : {sharpness:.3f}\n"
        f"ground elev (DEM)   : {ground_elev_m:.1f} m MSL\n"
        f"implied MSL         : {ground_elev_m + estimate_m:.1f} m\n"
    )
    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=12, color="#1a1c18",
    )
    fig.patch.set_facecolor("#f3f1ec")
    path = os.path.join(output_dir, "06_mission_metrics.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Saved {path}")


def save_match_panel(
    output_dir: str,
    camera_f32: np.ndarray,
    best_appearance: np.ndarray,
    surface: np.ndarray,
) -> None:
    cam_u8 = _to_u8(camera_f32)
    app_u8 = _to_u8(best_appearance)
    surf = surface.copy()
    surf = (surf - surf.min()) / (surf.max() - surf.min() + 1e-12)
    surf_u8 = _to_u8(surf)
    surf_color = cv2.applyColorMap(surf_u8, cv2.COLORMAP_MAGMA)

    # side-by-side camera | DEM appearance
    pair = np.hstack([cam_u8, app_u8])
    pair_bgr = cv2.cvtColor(pair, cv2.COLOR_GRAY2BGR)
    cv2.putText(pair_bgr, "camera", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    cv2.putText(pair_bgr, "DEM @ estimate", (cam_u8.shape[1] + 12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "03_camera_vs_dem_appearance.jpg"), pair_bgr)

    cv2.imwrite(os.path.join(output_dir, "04_phase_corr_surface.jpg"), surf_color)
    print(f"[VIZ] Saved 03_camera_vs_dem_appearance.jpg, 04_phase_corr_surface.jpg")


def print_visual_evaluation_guide() -> None:
    print("\n" + "-" * 60)
    print("VISUAL QA (key images)")
    print("-" * 60)
    print("00_camera_frame.jpg — Nadir input (real or synthetic from DEM).\n"
          "  OK: terrain structure visible. Bad: blank / pure sky / extreme blur.\n")
    print("01_dem_overview.jpg — Full onboard DEM hillshade with position mark.\n"
          "  OK: position inside map, terrain has ridges/valleys (freq content).\n")
    print("02_camera_preprocessed.jpg — Exact float domain image fed to FFT.\n")
    print("03_camera_vs_dem_appearance.jpg — Camera | DEM hillshade at estimated AGL.\n"
          "  OK: same ridge/valley layout & scale. Bad: stretched/compressed features.\n")
    print("04_phase_corr_surface.jpg — |IFFT| surface (fftshifted). Peak near center ⇒\n"
          "  lat/lon assumption holding; off-center ⇒ horizontal error or wrong h.\n")
    print("05_score_curve.png — response, soft prior, combined S(h).\n"
          "  OK: clear combined peak near truth (if known). Bad: flat / multi-modal tie.\n")
    print("06_mission_metrics.png — AGL estimate, Δ from last altimeter, confidence.\n")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("BLOOP — SINGLE CYCLE TEST (altimeter dropout)")
    print("=" * 60 + "\n")

    # ===== USER INPUTS =======================================================
    # Horizontal position assumed good (Sandwalk / last fix). Altitude is the unknown.
    LAT = 46.90
    LON = -121.80

    # Altimeter died holding this reading; belief widens with time (soft prior only).
    LAST_ALTIMETER_M = 920.0
    T_SINCE_DROPOUT_S = 45.0

    # Camera model
    FOV_DEG = 60.0
    CAM_PX = 256

    # Search range is WIDE on purpose — prior guides, does not clip.
    H_MIN_M = 150.0
    H_MAX_M = 1800.0

    # Synthetic validation truth (ignored if a real camera file is present).
    TRUE_AGL_M = 800.0

    # ===== PATHS =============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_dir = os.path.join(script_dir, "files")
    output_dir = os.path.join(files_dir, "output", "bloop_test")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TEST] Output directory: {output_dir}")

    # ===== LOAD DEM ==========================================================
    dem, meta = load_dem(files_dir)
    print(f"[TEST] DEM {dem.shape[1]}×{dem.shape[0]}  "
          f"elev {dem.min():.0f}–{dem.max():.0f} m  "
          f"center ({meta['center_lat']:.4f}, {meta['center_lon']:.4f})")
    z_gnd = ground_elevation_m(dem, LAT, LON, meta)
    print(f"[TEST] Ground elev at fix: {z_gnd:.1f} m MSL")

    # DEM overview viz
    overview = render_hillshade(
        dem, float(meta["metres_per_pixel_x"]), float(meta["metres_per_pixel_y"]),
    )
    overview_u8 = _to_u8(overview)
    overview_bgr = cv2.cvtColor(overview_u8, cv2.COLOR_GRAY2BGR)
    col, row = latlon_to_dem_px(LAT, LON, meta)
    cv2.drawMarker(
        overview_bgr, (int(col), int(row)), (0, 255, 0),
        markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2,
    )
    cv2.imwrite(os.path.join(output_dir, "01_dem_overview.jpg"), overview_bgr)

    # ===== LOAD OR SYNTHESIZE CAMERA =========================================
    camera_path = os.path.join(files_dir, "camera_frame.png")
    true_for_metrics: Optional[float] = None
    if os.path.exists(camera_path):
        frame = cv2.imread(camera_path)
        if frame is None:
            print(f"ERROR: Could not read {camera_path}")
            raise SystemExit(1)
        print(f"[TEST] Loaded real camera frame: {camera_path} {frame.shape}")
        # unknown truth unless you set it
        true_for_metrics = None
    else:
        print(f"[TEST] No {camera_path} — synthesizing camera at TRUE_AGL={TRUE_AGL_M} m")
        frame = synthesize_camera_from_dem(
            dem, meta, LAT, LON, TRUE_AGL_M, FOV_DEG, CAM_PX,
        )
        cv2.imwrite(camera_path, frame)  # cache for inspection
        true_for_metrics = TRUE_AGL_M
        print(f"[TEST] Wrote synthetic camera → {camera_path}")

    cv2.imwrite(os.path.join(output_dir, "00_camera_frame.jpg"), frame)
    camera_f32 = preprocess_camera(frame, CAM_PX)
    cv2.imwrite(
        os.path.join(output_dir, "02_camera_preprocessed.jpg"),
        _to_u8(camera_f32),
    )

    # ===== ESTIMATE ==========================================================
    result = estimate_altitude(
        camera_f32, dem, meta, LAT, LON, FOV_DEG,
        LAST_ALTIMETER_M, T_SINCE_DROPOUT_S,
        h_min_m=H_MIN_M, h_max_m=H_MAX_M,
    )

    best = result["best"]
    save_match_panel(output_dir, camera_f32, best["appearance"], best["surface"])
    save_score_curve(
        output_dir, result["curve"], result["agl_m"],
        LAST_ALTIMETER_M, true_for_metrics,
    )
    save_metrics_figure(
        output_dir, result["agl_m"], LAST_ALTIMETER_M,
        result["confidence"], result["sharpness"],
        true_for_metrics, T_SINCE_DROPOUT_S, z_gnd,
    )

    # ===== RESULTS ===========================================================
    print("\n" + "=" * 60)
    print("ALTITUDE RESULT")
    print("=" * 60)
    print(f"  Estimated AGL     : {result['agl_m']:.1f} m")
    print(f"  Last altimeter    : {LAST_ALTIMETER_M:.1f} m")
    print(f"  Δ from last alt   : {result['agl_m'] - LAST_ALTIMETER_M:+.1f} m")
    print(f"  Confidence        : {result['confidence']:.3f}")
    print(f"  Peak sharpness    : {result['sharpness']:.3f}")
    print(f"  Ambiguous         : {result['ambiguous']}")
    print(f"  Phase response    : {best['response']:.4f}")
    print(f"  Prior weight      : {best['prior']:.4f}")
    print(f"  Shift (dx, dy)    : ({best['dx']:.2f}, {best['dy']:.2f}) px")
    if true_for_metrics is not None:
        err = result["agl_m"] - true_for_metrics
        print(f"  Truth AGL         : {true_for_metrics:.1f} m")
        print(f"  Error             : {err:+.1f} m")
    print(f"  Output directory  : {output_dir}")
    print("=" * 60 + "\n")
    print_visual_evaluation_guide()


if __name__ == "__main__":
    main()
