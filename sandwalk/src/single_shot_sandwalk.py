"""
Sandwalk Single Cycle Test
Tests one localization cycle with a single image.

Range system: systematic tile grid coverage derived from drone altitude.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Dict
import requests
from io import BytesIO
from PIL import Image
import math

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon


# ---------------------------------------------------------------------------
# Coordinate math primitives  (unchanged)
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in metres between two lat/lon points."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def latlon_to_meters(lat: float, lon: float,
                     origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Convert lat/lon to (x_east_m, y_north_m) relative to origin."""
    R = 6371000
    y = (lat - origin_lat) * (math.pi / 180) * R
    x = (lon - origin_lon) * (math.pi / 180) * R * math.cos(math.radians(origin_lat))
    return x, y


def offset_coordinates(lat: float, lon: float,
                       dx_meters: float, dy_meters: float) -> Tuple[float, float]:
    """Offset lat/lon by (dx_east, dy_north) in metres."""
    R = 6371000
    new_lat = lat + (dy_meters / R) * (180 / math.pi)
    new_lon = lon + (dx_meters / (R * math.cos(math.radians(lat)))) * (180 / math.pi)
    return new_lat, new_lon


# ---------------------------------------------------------------------------
# Zoom / altitude / tile-footprint helpers
# ---------------------------------------------------------------------------

# altitude_m = ALTITUDE_ZOOM_NUMERATOR_M / 2**zoom  =>  zoom = round(log2(num / altitude_m))
ALTITUDE_ZOOM_NUMERATOR_M = 591_657_550.5


def altitude_to_zoom(altitude_m: float) -> int:
    """
    Maps barometric / assumed AGL altitude (metres) to Static Maps zoom integer.

    Uses:  altitude_m = ALTITUDE_ZOOM_NUMERATOR_M / 2**zoom_level
    so:     zoom = round( log2(ALTITUDE_ZOOM_NUMERATOR_M / altitude_m) )

    Example: 2257 m -> zoom 18.  Clamped to [0, 21] for the API.
    """
    if altitude_m <= 0:
        return 21
    zoom = round(math.log(ALTITUDE_ZOOM_NUMERATOR_M / altitude_m) / math.log(2))
    return int(max(0, min(21, zoom)))


def zoom_to_tile_footprint(zoom: int, center_lat: float,
                           tile_px: int = 640) -> Tuple[float, float]:
    """
    Return the ground dimensions (width_m, height_m) covered by a single
    tile_px x tile_px Google Maps Static API image at the given zoom level
    and latitude.

    Standard Web Mercator ground resolution at zoom z, latitude phi:
        metres_per_pixel = 156_543.034 * cos(phi) / 2^z

    Both axes use the same metres_per_pixel value because the Static API
    returns a square image and pixel pitch is identical in both directions
    at the tile's centre latitude (small-tile approximation — valid for all
    tile footprints we encounter at search distances <= ~5 km).
    """
    metres_per_pixel = 156_543.034 * math.cos(math.radians(center_lat)) / (2 ** zoom)
    width_m  = tile_px * metres_per_pixel
    height_m = tile_px * metres_per_pixel   # square tile -> same both axes
    return width_m, height_m


# ---------------------------------------------------------------------------
# Search zone geometry
# ---------------------------------------------------------------------------

def _arc_bbox_meters(r_min: float, r_max: float,
                     bearing_rad: float) -> Tuple[float, float, float, float]:
    """
    Compute the tight axis-aligned bounding box (in metres from launch) of the
    search arc: a half-annulus centred on `bearing_rad` with angular half-width
    pi/2, bounded by radii r_min and r_max.

    Returns (x_min, x_max, y_min, y_max).

    The boundary is sampled densely (inner arc + outer arc + two radial edges)
    and the envelope is taken. O(n) and accurate to well under 1 metre for all
    practical search radii.
    """
    angle_min = bearing_rad - math.pi / 2
    angle_max = bearing_rad + math.pi / 2

    xs, ys = [], []
    n = 720   # dense enough for sub-metre accuracy at any radius
    for i in range(n + 1):
        a = angle_min + (angle_max - angle_min) * i / n
        for r in (r_min, r_max):
            # bearing convention: north = 0, clockwise positive
            # x = east component, y = north component
            xs.append(r * math.sin(a))
            ys.append(r * math.cos(a))

    # include the launch origin when the ring starts at zero
    if r_min == 0:
        xs.append(0.0)
        ys.append(0.0)

    return min(xs), max(xs), min(ys), max(ys)


def _snap_bbox(x_min: float, x_max: float,
               y_min: float, y_max: float,
               tile_w: float, tile_h: float) -> Tuple[float, float, float, float]:
    """
    Expand the bounding box outward so its dimensions are exact multiples of
    (tile_w, tile_h).  Grid is anchored at the launch origin (0, 0).

    We always expand outward (floor lower bounds, ceil upper bounds) — never
    shrink — so coverage is never reduced.
    """
    snapped_x_min = math.floor(x_min / tile_w) * tile_w
    snapped_x_max = math.ceil (x_max / tile_w) * tile_w
    snapped_y_min = math.floor(y_min / tile_h) * tile_h
    snapped_y_max = math.ceil (y_max / tile_h) * tile_h
    return snapped_x_min, snapped_x_max, snapped_y_min, snapped_y_max


def _half_annulus_verts(r_inner: float, r_outer: float, bearing_rad: float, n: int = 160):
    """Closed polygon vertices: half-annulus in (east, north) metres from launch."""
    r_lo = max(r_inner, 1e-3)
    a0, a1 = bearing_rad - math.pi / 2, bearing_rad + math.pi / 2
    angles = [a0 + (a1 - a0) * i / n for i in range(n + 1)]
    verts = [(r_outer * math.sin(a), r_outer * math.cos(a)) for a in angles]
    for i in range(n, -1, -1):
        a = angles[i]
        verts.append((r_lo * math.sin(a), r_lo * math.cos(a)))
    return verts


def _search_arc_path(r_min: float, r_max: float, bearing_rad: float) -> MplPath:
    """Matplotlib path for the filled half-annulus (same geometry as mission viz)."""
    v = _half_annulus_verts(r_min, r_max, bearing_rad)
    return MplPath(np.asarray(v, dtype=float), closed=True)


def _tile_intersects_arc(
    cx: float,
    cy: float,
    tw: float,
    th: float,
    arc_path: MplPath,
) -> bool:
    """
    True iff the axis-aligned tile footprint intersects the half-annulus polygon.
    Corner/centre tests catch common cases; inner grid catches slivers the 5-point
    sampling used to miss.
    """
    hw, hh = tw / 2, th / 2
    left, right = cx - hw, cx + hw
    bottom, top = cy - hh, cy + hh
    for px, py in (
        (left, bottom), (right, bottom), (right, top), (left, top), (cx, cy)
    ):
        if arc_path.contains_point((px, py)):
            return True
    steps = 16
    for ix in range(steps + 1):
        for iy in range(steps + 1):
            px = left + (right - left) * ix / steps
            py = bottom + (top - bottom) * iy / steps
            if arc_path.contains_point((px, py)):
                return True
    return False


def generate_tile_grid(
    launch_lat: float,
    launch_lon: float,
    target_lat: float,
    target_lon: float,
    distance_traveled_m: float,
    altitude_m: float,
    tolerance_percent: float = 0.10,
    tile_px: int = 640,
) -> Tuple[List[Tuple[float, float]], int, float, float, float, float, int, int, Dict, float, float]:
    """
    Generate the minimal set of tile-centre coordinates that completely covers
    the search arc with no gaps and no redundant tiles.

    Algorithm
    ---------
    1. Derive zoom level from drone altitude; compute tile footprint in metres.
    2. Build search ring: r_min / r_max from motor estimate + tolerance.
    3. Compute target bearing.
    4. Find tight axis-aligned bbox of the arc in metre space.
    5. Snap bbox outward to the nearest tile boundary (guarantees integer tile counts).
    6. Generate every grid centre inside the snapped bbox.
    7. Keep only centres whose tile rectangle intersects the actual arc.

    Returns
    -------
    candidates, zoom, tile_w_m, tile_h_m, r_min, r_max, n_cols, n_rows, cell_map, x_min_m, y_min_m
      x_min_m / y_min_m: snapped bbox origin (m east / m north from launch) for mosaic georeferencing.
    """
    # 1. Zoom + footprint
    zoom = altitude_to_zoom(altitude_m)
    tile_w_m, tile_h_m = zoom_to_tile_footprint(zoom, launch_lat, tile_px)

    # 2. Search ring
    min_tolerance_m = 20.0
    tolerance_m     = max(min_tolerance_m, distance_traveled_m * tolerance_percent)
    r_min = max(0.0, distance_traveled_m - tolerance_m)
    r_max = distance_traveled_m + tolerance_m

    # 3. Target bearing
    dx_target, dy_target = latlon_to_meters(target_lat, target_lon, launch_lat, launch_lon)
    bearing = math.atan2(dx_target, dy_target)   # atan2(east, north) = bearing from north

    # 4. Tight bbox
    x_min, x_max, y_min, y_max = _arc_bbox_meters(r_min, r_max, bearing)

    # 5. Snap outward
    x_min, x_max, y_min, y_max = _snap_bbox(x_min, x_max, y_min, y_max, tile_w_m, tile_h_m)

    # 6. Grid dimensions (cover snapped rectangle; avoid round()/float drift)
    w_box, h_box = x_max - x_min, y_max - y_min
    n_cols = max(1, int(math.ceil(w_box / tile_w_m - 1e-9)))
    n_rows = max(1, int(math.ceil(h_box / tile_h_m - 1e-9)))

    arc_path = _search_arc_path(r_min, r_max, bearing)

    candidates = []
    cell_map   = {}   # (col, row) -> index into candidates list

    for row in range(n_rows):
        for col in range(n_cols):
            cx = x_min + (col + 0.5) * tile_w_m
            cy = y_min + (row + 0.5) * tile_h_m

            # 7. Keep only tiles that overlap the search zone polygon
            if not _tile_intersects_arc(cx, cy, tile_w_m, tile_h_m, arc_path):
                continue

            cand_lat, cand_lon = offset_coordinates(launch_lat, launch_lon, cx, cy)
            cell_map[(col, row)] = len(candidates)
            candidates.append((cand_lat, cand_lon))

    print(f"[GRID] Altitude {altitude_m:.0f}m  →  zoom {zoom}")
    print(f"[GRID] Tile footprint  : {tile_w_m:.1f}m × {tile_h_m:.1f}m")
    print(f"[GRID] Search ring     : {r_min:.1f}m – {r_max:.1f}m")
    print(f"[GRID] Target bearing  : {math.degrees(bearing):.1f}°")
    print(f"[GRID] Snapped bbox    : x=[{x_min:.1f}, {x_max:.1f}]m  "
          f"y=[{y_min:.1f}, {y_max:.1f}]m")
    print(f"[GRID] Grid dimensions : {n_cols} cols × {n_rows} rows "
          f"= {n_cols * n_rows} total cells")
    print(f"[GRID] Tiles to fetch (zone intersect): {len(candidates)}")

    return candidates, zoom, tile_w_m, tile_h_m, r_min, r_max, n_cols, n_rows, cell_map, x_min, y_min


# ---------------------------------------------------------------------------
# Satellite tile loading  (unchanged)
# ---------------------------------------------------------------------------

def load_satellite_tile(lat: float, lon: float,
                        zoom: int, api_key: str,
                        size: int = 640) -> np.ndarray:
    """Load satellite tile from Google Maps Static API."""
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={size}x{size}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        tile  = np.array(image)
        if len(tile.shape) == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        return tile
    except Exception as e:
        print(f"[TILE] ERROR loading ({lat:.6f}, {lon:.6f}): {e}")
        return None


# ---------------------------------------------------------------------------
# Image processing  (unchanged)
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Preprocess drone frame for SIFT matching."""
    h, w = frame.shape[:2]
    if w >= target_size and h >= target_size:
        sx = (w - target_size) // 2
        sy = (h - target_size) // 2
        resized = frame[sy:sy + target_size, sx:sx + target_size]
    else:
        resized = cv2.resize(frame, (target_size, target_size),
                             interpolation=cv2.INTER_LANCZOS4)
    gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    denoised  = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(denoised)
    kernel    = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    return sharpened


def mosaic_pixel_to_latlon(
    gu: float,
    gv: float,
    x_min: float,
    y_min: float,
    tile_w_m: float,
    tile_h_m: float,
    tile_px: int,
    launch_lat: float,
    launch_lon: float,
    col0: int,
    row1: int,
) -> Tuple[float, float]:
    """Map pixel in north-up tight mosaic to lat/lon. Top row of image = grid row row1 (north)."""
    mpp_x = tile_w_m / tile_px
    mpp_y = tile_h_m / tile_px
    ic    = int(math.floor(gu / tile_px))
    irow  = int(math.floor(gv / tile_px))
    col      = col0 + ic
    grid_row = row1 - irow
    tp = gu - ic * tile_px
    tq = gv - irow * tile_px
    cx = x_min + (col + 0.5) * tile_w_m
    cy = y_min + (grid_row + 0.5) * tile_h_m
    east_nw   = cx - tile_w_m / 2.0
    north_nw  = cy + tile_h_m / 2.0
    east      = east_nw + tp * mpp_x
    north     = north_nw - tq * mpp_y
    return offset_coordinates(launch_lat, launch_lon, east, north)


def build_preprocessed_mosaic(
    tile_px: int,
    tiles_bgr: Dict[Tuple[int, int], np.ndarray],
    col0: int,
    col1: int,
    row0: int,
    row1: int,
    fill: int = 127,
) -> np.ndarray:
    """
    North-up mosaic over inclusive col/row range. Gray fill for cells with no
    tile data (non-intersecting cells or failed loads).
    """
    n_cols_m = col1 - col0 + 1
    n_rows_m = row1 - row0 + 1
    mosaic = np.full((n_rows_m * tile_px, n_cols_m * tile_px), fill, dtype=np.uint8)
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            ir_tight = row1 - row
            y0, y1 = ir_tight * tile_px, (ir_tight + 1) * tile_px
            ic = col - col0
            x0, x1 = ic * tile_px, (ic + 1) * tile_px
            tile = tiles_bgr.get((col, row))
            if tile is not None:
                mosaic[y0:y1, x0:x1] = preprocess_frame(tile, tile_px)
    return mosaic


def localize_template_on_mosaic(
    mosaic: np.ndarray,
    template: np.ndarray,
    x_min: float,
    y_min: float,
    tile_w_m: float,
    tile_h_m: float,
    tile_px: int,
    launch_lat: float,
    launch_lon: float,
    col0: int,
    row1: int,
) -> Tuple[float, float, float, Tuple[int, int], np.ndarray]:
    """
    cv2.matchTemplate + minMaxLoc → lat/lon at template centre (OpenCV flow:
    https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html).
    Returns (lat, lon, peak_score, peak_top_left, correlation_map).
    """
    th, tw = template.shape[:2]
    mh, mw = mosaic.shape[:2]
    if mh < th or mw < tw:
        print("[MOSAIC] Template larger than mosaic; skip template match.")
        z = np.zeros((1, 1), dtype=np.float32)
        return launch_lat, launch_lon, 0.0, (0, 0), z
    res = cv2.matchTemplate(mosaic, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    u, v = max_loc
    cu = u + 0.5 * tw
    cv = v + 0.5 * th
    lat, lon = mosaic_pixel_to_latlon(
        float(cu), float(cv), x_min, y_min,
        tile_w_m, tile_h_m, tile_px, launch_lat, launch_lon,
        col0, row1,
    )
    return lat, lon, float(max_val), (u, v), res


def save_template_vs_mosaic_crop(
    output_dir: str,
    processed_drone: np.ndarray,
    mosaic: np.ndarray,
    tmpl_loc: Tuple[int, int],
) -> None:
    """Left = template passed to matchTemplate; right = mosaic patch at winning top-left."""
    th, tw = processed_drone.shape[:2]
    u, v = tmpl_loc
    crop = mosaic[v : v + th, u : u + tw]
    left  = cv2.cvtColor(processed_drone, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    sep = np.full((th, 6, 3), 255, dtype=np.uint8)
    combo = np.hstack([left, sep, right])
    cv2.putText(combo, "drone template", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 200, 40), 1, cv2.LINE_AA)
    cv2.putText(combo, "mosaic @ peak", (tw + 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 200, 40), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "06_template_vs_mosaic_crop.jpg"), combo)


def save_mission_context_figure(
    output_dir: str,
    launch_lat: float,
    launch_lon: float,
    target_lat: float,
    target_lon: float,
    tile_w_m: float,
    tile_h_m: float,
    r_min: float,
    r_max: float,
    x_min_m: float,
    y_min_m: float,
    n_cols: int,
    n_rows: int,
    launch_bgr: np.ndarray,
    target_bgr: np.ndarray,
    mosaic_peak_bgr: np.ndarray,
    tmpl_lat: float,
    tmpl_lon: float,
) -> None:
    """
    Metre plane (east X, north Y) from launch. Draws hemisphere-limited search ring,
    Static Map footprints for launch & target, semi-transparent stitched mosaic + peak box,
    and the template-fix point in map space.
    """
    tw, th = tile_w_m, tile_h_m
    tgt_x, tgt_y = latlon_to_meters(target_lat, target_lon, launch_lat, launch_lon)
    bearing        = math.atan2(tgt_x, tgt_y)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f4f4f4")
    fig.patch.set_facecolor("#f4f4f4")

    # ---- search zone (under imagery) -------------------------------------
    zone = Polygon(
        _half_annulus_verts(r_min, r_max, bearing),
        closed=True,
        facecolor="#ff4444",
        edgecolor="#b00000",
        linewidth=2.0,
        alpha=0.22,
        zorder=1,
    )
    ax.add_patch(zone)

    # ---- launch tile (centred on origin) ----------------------------------
    if launch_bgr is not None:
        rgb = cv2.cvtColor(launch_bgr, cv2.COLOR_BGR2RGB)
        ext_l = (-tw / 2, tw / 2, -th / 2, th / 2)
        ax.imshow(rgb, extent=ext_l, origin="upper", aspect="auto", zorder=2, interpolation="bilinear")

    # ---- target tile --------------------------------------------------------
    if target_bgr is not None:
        rgb_t = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        ext_t = (tgt_x - tw / 2, tgt_x + tw / 2, tgt_y - th / 2, tgt_y + th / 2)
        ax.imshow(rgb_t, extent=ext_t, origin="upper", aspect="auto", zorder=2, interpolation="bilinear")

    # ---- mosaic + template peak (full snapped grid, same georef as mosaic) ---
    left_m = x_min_m
    right_m = x_min_m + n_cols * tw
    bottom_m = y_min_m
    top_m = y_min_m + n_rows * th
    mp = cv2.cvtColor(mosaic_peak_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(
        mp,
        extent=(left_m, right_m, bottom_m, top_m),
        origin="upper",
        aspect="auto",
        zorder=3,
        alpha=0.48,
        interpolation="bilinear",
    )

    # ---- template lat/lon fix ---------------------------------------------
    mx, my = latlon_to_meters(tmpl_lat, tmpl_lon, launch_lat, launch_lon)
    ax.plot(mx, my, "+", color="lime", markersize=16, markeredgewidth=2.5, zorder=5, label="Template fix")

    ax.plot(0.0, 0.0, "s", color="darkgreen", markersize=8, zorder=6, label="Launch")
    ax.plot(tgt_x, tgt_y, "*", color="darkred", markersize=12, zorder=6, label="Target centre")

    pad = max(r_max, abs(left_m), abs(right_m), abs(bottom_m), abs(top_m), 1.0) * 0.08
    xs = [-tw / 2, tw / 2, tgt_x - tw / 2, tgt_x + tw / 2, left_m, right_m, mx]
    ys = [-th / 2, th / 2, tgt_y - th / 2, tgt_y + th / 2, bottom_m, top_m, my]
    # include annulus extent
    for a in (bearing - math.pi / 2, bearing + math.pi / 2):
        xs += [r_max * math.sin(a) * 1.02]
        ys += [r_max * math.cos(a) * 1.02]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_xlabel("East from launch (m)")
    ax.set_ylabel("North from launch (m)")
    ax.set_title("Sandwalk — search zone, reference tiles, mosaic + template peak")
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    out = os.path.join(output_dir, "07_mission_context.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[VIZ] Saved mission context: {out}")


def print_visual_evaluation_guide() -> None:
    """What each key output looks like and how to read success vs failure."""
    print("\n" + "-" * 60)
    print("VISUAL QA (key images)")
    print("-" * 60)
    print("00_drone_image.jpg — Raw input. Sanity: right scene, roughly nadir-ish satellite-like.\n"
          "  OK: clear ground features. Bad: heavy motion blur, wrong scale vs map zoom.\n")
    print("01_launch_location.jpg / 02_target_location.jpg — Static-map context at matched zoom.\n")
    print("02b_drone_preprocessed_template.jpg — Exact grayscale template slid over the mosaic.\n"
          "  OK: sharp edges/contrast like mosaic tiles. Bad: black/empty; very different look vs 03.\n")
    print("03_mosaic_preprocessed.jpg — Full snapped search grid; gray = cells outside zone or failed loads.\n"
          "  OK: terrain continuous, mostly real imagery. Bad: large gray where tiles failed to load.\n")
    print("04_mosaic_template_peak.jpg — Mosaic + green box = winning 640x640 window.\n"
          "  OK: box covers terrain that matches the template. Bad: box on gray, or wrong texture.\n")
    print("06_template_vs_mosaic_crop.jpg — Template | mosaic patch at peak (quick eyeball match).\n"
          "  OK: left and right look like the same place (feature alignment). Bad: uncorrelated patterns.\n")
    print("07_mission_context.png — Metres from launch: red = search ring, tiles = launch/target footprints,\n"
          "  faded layer = mosaic+green peak, lime + = template lat/lon fix.\n")
    print("NCC peak is only in the console log (not saved as an image).\n"
          + "-" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("SANDWALK — SINGLE CYCLE TEST")
    print("=" * 60 + "\n")

    # ===== USER INPUTS =======================================================
    LAUNCH_LAT  = 15.4800
    LAUNCH_LON  = 44.2200

    TARGET_LAT  = 15.4878
    TARGET_LON  = 44.2261

    DISTANCE_TRAVELED_M = 500.0   # motor estimate (metres)
    ALTITUDE_M          = 2257.0  # AGL metres; with ALTITUDE_ZOOM_NUMERATOR_M -> zoom 18
    TOLERANCE_PERCENT   = 0.10    # ring half-width = ±10 % of distance
    TILE_PX             = 640     # pixel dimension of each tile

    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: Set GOOGLE_MAPS_API_KEY environment variable")
        exit(1)

    # ===== OUTPUT DIRECTORY ==================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "files", "output", "sandwalk_test")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TEST] Output directory: {output_dir}")

    # ===== LOAD DRONE IMAGE ==================================================
    drone_image_path = os.path.join(script_dir, "files", "drone_image.png")
    if not os.path.exists(drone_image_path):
        print(f"ERROR: Drone image not found at {drone_image_path}")
        print(f"Place your drone image at: {drone_image_path}")
        exit(1)
    drone_frame = cv2.imread(drone_image_path)
    if drone_frame is None:
        print("ERROR: Could not read drone image")
        exit(1)
    print(f"[TEST] Loaded drone image: {drone_frame.shape}")
    cv2.imwrite(os.path.join(output_dir, "00_drone_image.jpg"), drone_frame)

    # ===== GENERATE SYSTEMATIC TILE GRID =====================================
    candidates, zoom, tile_w_m, tile_h_m, r_min, r_max, n_cols, n_rows, cell_map, x_min_m, y_min_m = generate_tile_grid(
        LAUNCH_LAT, LAUNCH_LON,
        TARGET_LAT, TARGET_LON,
        DISTANCE_TRAVELED_M,
        ALTITUDE_M,
        TOLERANCE_PERCENT,
        TILE_PX,
    )

    if not candidates:
        print("[TEST] ERROR: No candidates generated — check inputs.")
        exit(1)

    # ===== LOAD REFERENCE IMAGES =============================================
    print(f"\n[TEST] Loading launch location satellite image (zoom {zoom})…")
    launch_image = load_satellite_tile(LAUNCH_LAT, LAUNCH_LON, zoom,
                                       GOOGLE_MAPS_API_KEY, TILE_PX)
    if launch_image is not None:
        cv2.imwrite(os.path.join(output_dir, "01_launch_location.jpg"), launch_image)

    print(f"[TEST] Loading target location satellite image (zoom {zoom})…")
    target_image = load_satellite_tile(TARGET_LAT, TARGET_LON, zoom,
                                       GOOGLE_MAPS_API_KEY, TILE_PX)
    if target_image is not None:
        cv2.imwrite(os.path.join(output_dir, "02_target_location.jpg"), target_image)

    # ===== PREPROCESS DRONE IMAGE ============================================
    processed_drone = preprocess_frame(drone_frame, TILE_PX)
    print(f"[TEST] Preprocessed drone image: {processed_drone.shape}")
    cv2.imwrite(
        os.path.join(output_dir, "02b_drone_preprocessed_template.jpg"),
        processed_drone,
    )

    # ===== FETCH TILES (stitch + template match only; no per-tile matching) ==
    print(f"\n[TEST] Fetching {len(candidates)} tile(s)…\n")

    idx_to_cell = {idx: (c, r) for (c, r), idx in cell_map.items()}
    tiles_bgr: Dict[Tuple[int, int], np.ndarray] = {}

    for idx, (lat, lon) in enumerate(candidates):
        print(f"[TEST] Tile {idx + 1:>3}/{len(candidates)}: "
              f"({lat:.6f}, {lon:.6f})… ", end="", flush=True)

        tile = load_satellite_tile(lat, lon, zoom, GOOGLE_MAPS_API_KEY, TILE_PX)
        if tile is None:
            print("FAILED (tile load error)")
            continue

        cell = idx_to_cell[idx]
        print("ok")
        tiles_bgr[cell] = tile

    # ===== STITCH MOSAIC + TEMPLATE MATCH ====================================
    print(f"\n[TEST] Building preprocessed mosaic and running template match…")
    print(f"[GRID] Mosaic: full snapped grid {n_cols}×{n_rows} cells "
          f"(API tiles placed: {len(tiles_bgr)})")
    mosaic = build_preprocessed_mosaic(
        TILE_PX, tiles_bgr, 0, n_cols - 1, 0, n_rows - 1,
    )
    cv2.imwrite(os.path.join(output_dir, "03_mosaic_preprocessed.jpg"), mosaic)

    tmpl_lat, tmpl_lon, tmpl_peak, tmpl_loc, _tmpl_res = localize_template_on_mosaic(
        mosaic,
        processed_drone,
        x_min_m, y_min_m,
        tile_w_m, tile_h_m, TILE_PX,
        LAUNCH_LAT, LAUNCH_LON,
        0, n_rows - 1,
    )
    th, tw = processed_drone.shape[:2]

    vis = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, tmpl_loc, (tmpl_loc[0] + tw, tmpl_loc[1] + th), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(output_dir, "04_mosaic_template_peak.jpg"), vis)

    save_template_vs_mosaic_crop(output_dir, processed_drone, mosaic, tmpl_loc)
    print(f"[TEST] Template peak NCC = {tmpl_peak:.3f} at pixel offset {tmpl_loc}")

    print(f"\n[TEST] Saving mission context figure…")
    save_mission_context_figure(
        output_dir,
        LAUNCH_LAT, LAUNCH_LON,
        TARGET_LAT, TARGET_LON,
        tile_w_m, tile_h_m,
        r_min, r_max,
        x_min_m, y_min_m,
        n_cols,
        n_rows,
        launch_image,
        target_image,
        vis,
        tmpl_lat,
        tmpl_lon,
    )

    # ===== RESULTS ===========================================================
    print("\n" + "=" * 60)
    print("LOCALIZATION RESULT")
    print("=" * 60)

    print("Template match on stitched mosaic (sub-tile):")
    print(f"  Position      : ({tmpl_lat:.6f}, {tmpl_lon:.6f})")
    print(f"  NCC peak      : {tmpl_peak:.3f}")

    dx_t, dy_t = latlon_to_meters(TARGET_LAT, TARGET_LON, LAUNCH_LAT, LAUNCH_LON)
    bearing    = math.atan2(dx_t, dy_t)
    dr_lat, dr_lon = offset_coordinates(
        LAUNCH_LAT, LAUNCH_LON,
        DISTANCE_TRAVELED_M * math.sin(bearing),
        DISTANCE_TRAVELED_M * math.cos(bearing),
    )
    dr_err_tmpl = haversine_distance(tmpl_lat, tmpl_lon, dr_lat, dr_lon)
    print(f"\nDead-reckoning vs template fix : {dr_err_tmpl:.1f} m")
    print(f"Distance from launch (template): "
          f"{haversine_distance(LAUNCH_LAT, LAUNCH_LON, tmpl_lat, tmpl_lon):.1f} m")
    print(f"Distance to target (template)   : "
          f"{haversine_distance(tmpl_lat, tmpl_lon, TARGET_LAT, TARGET_LON):.1f} m")

    print(f"\nZoom Level Used  : {zoom}")
    print(f"Tile Footprint   : {tile_w_m:.1f}m × {tile_h_m:.1f}m")
    print(f"Tiles Evaluated  : {len(candidates)}")
    print(f"Output Directory : {output_dir}")
    print("=" * 60 + "\n")

    print_visual_evaluation_guide()


if __name__ == "__main__":
    main()