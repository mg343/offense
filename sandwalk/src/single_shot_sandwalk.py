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
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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

def altitude_to_zoom(altitude_m: float) -> int:
    """
    Convert drone altitude (metres AGL) to the best matching Google Maps zoom
    level for the Static Maps API (V3, max zoom 21).

    Derived from Google Maps JS internals (see convertRangeToZoom):
        range = 35_200_000 / 2^zoom
    We treat 'range' as equivalent to camera-eye altitude in metres, so:
        zoom = round( log2(35_200_000 / altitude_m) )

    Clamped to [0, 21] per V3 spec.
    """
    if altitude_m <= 0:
        return 21
    zoom = round(math.log(35_200_000 / altitude_m) / math.log(2))
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


def _tile_intersects_arc(cx: float, cy: float,
                         tw: float, th: float,
                         r_min: float, r_max: float,
                         bearing_rad: float) -> bool:
    """
    Return True if the tile rectangle [cx-tw/2, cx+tw/2] x [cy-th/2, cy+th/2]
    has any overlap with the search arc.

    Search arc = half-annulus:
        r_min <= sqrt(x^2 + y^2) <= r_max
        AND point bearing within +/- pi/2 of bearing_rad

    Test: the four corners of the tile plus its centre.
    If ANY of those five points satisfies both constraints, the tile is kept.

    This is deliberately conservative: edge tiles that barely clip the arc
    boundary are included.  That guarantees zero missed coverage — the hard
    requirement — at the cost of a small number of extra boundary tiles.
    """
    hw, hh = tw / 2, th / 2
    test_pts = [
        (cx,      cy     ),   # centre
        (cx - hw, cy - hh),   # SW corner
        (cx + hw, cy - hh),   # SE corner
        (cx - hw, cy + hh),   # NW corner
        (cx + hw, cy + hh),   # NE corner
    ]

    for px, py in test_pts:
        dist = math.hypot(px, py)

        if dist < 1e-6:
            # exactly at launch origin — inside arc only when r_min == 0
            if r_min == 0:
                return True
            continue

        if not (r_min <= dist <= r_max):
            continue

        # bearing of this point from origin (atan2(east, north) = clockwise from north)
        point_bearing = math.atan2(px, py)

        # normalise angular difference to [-pi, pi]
        diff = (point_bearing - bearing_rad + math.pi) % (2 * math.pi) - math.pi

        if abs(diff) <= math.pi / 2:
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

    # 6. Grid centres
    n_cols = round((x_max - x_min) / tile_w_m)
    n_rows = round((y_max - y_min) / tile_h_m)

    candidates = []
    cell_map   = {}   # (col, row) -> index into candidates list

    for row in range(n_rows):
        for col in range(n_cols):
            cx = x_min + (col + 0.5) * tile_w_m
            cy = y_min + (row + 0.5) * tile_h_m

            # 7. Keep only tiles that overlap the arc
            if not _tile_intersects_arc(cx, cy, tile_w_m, tile_h_m, r_min, r_max, bearing):
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
    print(f"[GRID] Tiles kept (arc intersect): {len(candidates)}")

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


def tight_cell_bounds(cell_map: Dict) -> Tuple[int, int, int, int]:
    """Minimal inclusive (col0, col1, row0, row1) covering all arc tiles."""
    cols = [c for (c, _) in cell_map.keys()]
    rows = [r for (_, r) in cell_map.keys()]
    return min(cols), max(cols), min(rows), max(rows)


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
    North-up mosaic over inclusive col/row range only. Gray only for cells inside
    the box that have no tile (jagged arc); no outer padding beyond fetched tiles.
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


def print_visual_evaluation_guide() -> None:
    """What each key output looks like and how to read success vs failure."""
    print("\n" + "-" * 60)
    print("VISUAL QA (key images)")
    print("-" * 60)
    print("00_drone_image.jpg — Raw input. Sanity: right scene, roughly nadir-ish satellite-like.\n"
          "  OK: clear ground features. Bad: heavy motion blur, wrong scale vs map zoom.\n")
    print("02b_drone_preprocessed_template.jpg — Exact grayscale template slid over the mosaic.\n"
          "  OK: sharp edges/contrast like mosaic tiles. Bad: black/empty; very different look vs 03.\n")
    print("03_mosaic_preprocessed.jpg — Tight rectangle around fetched tiles only (gray = holes inside that box).\n"
          "  OK: terrain continuous, mostly real imagery. Bad: large gray where tiles failed to load.\n")
    print("04_mosaic_template_peak.jpg — Mosaic + green box = winning 640x640 window.\n"
          "  OK: box covers terrain that matches the template. Bad: box on gray, or wrong texture.\n")
    print("05_match_template_heatmap.jpg — NCC surface (min-max stretched to 0-255).\n"
          "  OK: one brightest blob (clear peak). Bad: flat gray (no winner) or several equal hotspots.\n")
    print("06_template_vs_mosaic_crop.jpg — Template | mosaic patch at peak (quick eyeball match).\n"
          "  OK: left and right look like the same place (feature alignment). Bad: uncorrelated patterns.\n")
    print("Also check printed NCC peak: higher is usually more decisive (scene-dependent baseline).\n"
          + "-" * 60)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def create_range_map_collage(
    candidate_data: List[Dict],
    n_cols: int,
    n_rows: int,
    cell_map: Dict,
    tile_px: int,
    output_dir: str,
) -> np.ndarray:
    px = tile_px

    # --- 1. Build placeholder tile as GRAYSCALE (2D) -----------------------
    # Initializing without the '3' channel dimension
    placeholder = np.full((px, px), 220, dtype=np.uint8)   # light grey (2D)

    # Drawing functions work on 2D arrays using scalar values for color
    hatch_color = 180
    spacing = 32
    for k in range(-(px // spacing) - 1, (px // spacing) + 2):
        offset = k * spacing
        pt1 = (offset,      0)
        pt2 = (offset + px, px)
        cv2.line(placeholder, pt1, pt2, hatch_color, 1, cv2.LINE_AA)
        
    cv2.rectangle(placeholder, (0, 0), (px - 1, px - 1), 150, 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = "NO DATA"
    font_scale = px / 640.0
    thickness = max(1, int(font_scale * 1.5))
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx = (px - tw) // 2
    ty = (px + th) // 2
    cv2.putText(placeholder, label, (tx, ty),
                font, font_scale, 120, thickness, cv2.LINE_AA)

    # --- 2. Build index -----------------------------------------------------
    idx_to_img: Dict[int, np.ndarray] = {}
    for cand in candidate_data:
        img = cand.get("image")
        if img is not None:
            for (col, row), cand_idx in cell_map.items():
                if candidate_data[cand_idx]["position"] == cand["position"]:
                    idx_to_img[cand_idx] = img
                    break

    # --- 3. Assemble rows (Ensuring everything is 2D) -----------------------
    row_strips = []
    for row in range(n_rows - 1, -1, -1):
        tiles_in_row = []
        for col in range(n_cols):
            cand_idx = cell_map.get((col, row))
            
            if cand_idx is not None and cand_idx in idx_to_img:
                tile_img = idx_to_img[cand_idx].copy()
                
                # Resize if necessary
                if tile_img.shape[:2] != (px, px):
                    tile_img = cv2.resize(tile_img, (px, px))
                
                # CRITICAL: Convert color tile (3D) to grayscale (2D)
                # to match the placeholder dimension
                if len(tile_img.shape) == 3:
                    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
                
                tiles_in_row.append(tile_img)
            else:
                # Add the 2D placeholder
                tiles_in_row.append(placeholder.copy())
        
        # Every element in tiles_in_row is now 2D
        row_strips.append(np.hstack(tiles_in_row))

    # All row_strips are now 2D, vstack will succeed
    collage = np.vstack(row_strips)

    path = os.path.join(output_dir, "range_map_collage.png")
    cv2.imwrite(path, collage)
    print(f"[VIZ] Saved grayscale range map collage ({n_cols}×{n_rows} tiles): {path}")
    return collage


def create_satellite_overlay(
    launch_lat: float, launch_lon: float,
    target_lat: float, target_lon: float,
    launch_image: np.ndarray,
    target_image: np.ndarray,
    candidate_data: List[Dict],
    tile_w_m: float, tile_h_m: float,
    r_min: float, r_max: float,
    n_cols: int, n_rows: int,
    cell_map: Dict,
    tile_px: int,
    output_dir: str,
):
    """
    Overlay the stitched satellite range-map onto the search-zone diagram.

    Each tile image is rendered at its exact geographic position and size in
    metre-space, replacing the coloured rectangle fill from search_visualization
    while keeping all annotation layers (ring circles, borders, markers, legend)
    on top.

    Saves to: search_visualization_satellite_overlay.png
    """
    target_x, target_y = latlon_to_meters(target_lat, target_lon,
                                          launch_lat, launch_lon)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # --- place each tile as an image at its geographic footprint -------------
    # Build a reverse lookup: candidate position -> image
    pos_to_img: Dict[Tuple, np.ndarray] = {
        tuple(c["position"]): c["image"]
        for c in candidate_data
        if c.get("image") is not None
    }
    # Also need placeholder for cells in cell_map with no fetched image
    placeholder_rgb = None

    for (col, row), cand_idx in cell_map.items():
        pos   = candidate_data[cand_idx]["position"]
        img   = pos_to_img.get(tuple(pos))

        cx, cy = latlon_to_meters(pos[0], pos[1], launch_lat, launch_lon)
        left   = cx - tile_w_m / 2
        bottom = cy - tile_h_m / 2

        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(
                rgb,
                extent=[left, left + tile_w_m, bottom, bottom + tile_h_m],
                origin="upper",
                aspect="auto",
                zorder=3,
                alpha=0.85,
            )
        else:
            # hatched placeholder via a grey rectangle with cross-hatch pattern
            if placeholder_rgb is None:
                ph = np.full((tile_px, tile_px, 3), 200, dtype=np.uint8)
                spacing = 32
                for k in range(-(tile_px // spacing) - 1, (tile_px // spacing) + 2):
                    offset = k * spacing
                    cv2.line(ph, (offset, 0), (offset + tile_px, tile_px), (160, 160, 160), 1)
                placeholder_rgb = ph
            ax.imshow(
                placeholder_rgb,
                extent=[left, left + tile_w_m, bottom, bottom + tile_h_m],
                origin="upper",
                aspect="auto",
                zorder=3,
                alpha=0.60,
            )

    # --- tile borders with confidence colouring ------------------------------
    for cand in candidate_data:
        lat, lon = cand["position"]
        cx, cy   = latlon_to_meters(lat, lon, launch_lat, launch_lon)
        conf     = cand["confidence"]
        color    = "#ffd700" if conf >= 0.60 else "#3a6fd8"
        rect = patches.Rectangle(
            (cx - tile_w_m / 2, cy - tile_h_m / 2),
            tile_w_m, tile_h_m,
            linewidth=1.4, edgecolor=color,
            facecolor="none", zorder=5,
        )
        ax.add_patch(rect)
        # small confidence label centred on each tile
        ax.text(cx, cy, f"{conf:.0%}",
                ha="center", va="center",
                fontsize=max(5, int(7 * min(tile_w_m, 300) / 300)),
                color=color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0d1117",
                          alpha=0.55, edgecolor="none"))

    # --- best match highlight ------------------------------------------------
    if candidate_data:
        best = max(candidate_data, key=lambda c: c["confidence"])
        bx, by = latlon_to_meters(best["position"][0], best["position"][1],
                                  launch_lat, launch_lon)
        rect_best = patches.Rectangle(
            (bx - tile_w_m / 2, by - tile_h_m / 2),
            tile_w_m, tile_h_m,
            linewidth=3.0, edgecolor="#ff4444",
            facecolor="none", zorder=7,
        )
        ax.add_patch(rect_best)
        ax.plot(bx, by, "r^", markersize=11, label="Best match", zorder=8)

    # --- reference ring circles ----------------------------------------------
    for r in (r_min, r_max):
        circle = plt.Circle((0, 0), r, color="#aaaacc",
                             linestyle="--", fill=False, linewidth=1.0, zorder=9)
        ax.add_patch(circle)

    # --- launch + target markers ---------------------------------------------
    ax.plot(0, 0, "go", markersize=14, label="Launch",  zorder=12)
    ax.plot(target_x, target_y, "r*", markersize=18, label="Target", zorder=12)

    def _overlay(img, pos, edge_color, zoom_f=0.13):
        if img is None:
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ib  = OffsetImage(rgb, zoom=zoom_f)
        ab  = AnnotationBbox(ib, pos, frameon=True, boxcoords="data", pad=0.3,
                             bboxprops=dict(edgecolor=edge_color, linewidth=2))
        ax.add_artist(ab)

    _overlay(launch_image, (0, 0), "green")
    _overlay(target_image, (target_x, target_y), "red")

    # --- labels & style ------------------------------------------------------
    ax.set_xlabel("East–West Distance (m)",   fontsize=11, color="#cccccc")
    ax.set_ylabel("North–South Distance (m)", fontsize=11, color="#cccccc")
    ax.set_title("Sandwalk — Search Zone with Satellite Imagery",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
    ax.grid(True, alpha=0.12, color="#444444")
    ax.set_aspect("equal", "box")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    path = os.path.join(output_dir, "search_visualization_satellite_overlay.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[VIZ] Saved: {path}")
    plt.close()


def create_search_visualization(
    launch_lat: float, launch_lon: float,
    target_lat: float, target_lon: float,
    launch_image: np.ndarray,
    target_image: np.ndarray,
    drone_image:  np.ndarray,
    candidate_data: List[Dict],
    tile_w_m: float, tile_h_m: float,
    r_min: float, r_max: float,
    output_dir: str,
):
    """
    Visualise the systematic tile grid over the search arc.
    Tiles drawn as labelled rectangles; arc ring shown as dashed circles.
    Two versions saved: without and with the drone reference image inset.
    """
    target_x, target_y = latlon_to_meters(target_lat, target_lon,
                                          launch_lat, launch_lon)

    def _build_figure():
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        # reference ring circles
        for r in (r_min, r_max):
            circle = plt.Circle((0, 0), r, color="#4a4a6a",
                                 linestyle="--", fill=False, linewidth=1.0, zorder=2)
            ax.add_patch(circle)

        # tile footprint rectangles
        for cand in candidate_data:
            lat, lon = cand["position"]
            cx, cy   = latlon_to_meters(lat, lon, launch_lat, launch_lon)
            conf     = cand["confidence"]
            color    = "#ffd700" if conf >= 0.60 else "#3a6fd8"
            rect = patches.Rectangle(
                (cx - tile_w_m / 2, cy - tile_h_m / 2),
                tile_w_m, tile_h_m,
                linewidth=1.2, edgecolor=color,
                facecolor=color, alpha=0.18, zorder=3,
            )
            ax.add_patch(rect)
            ax.plot(cx, cy, "o", color=color, markersize=4, alpha=0.8, zorder=4)

        # best match highlight
        if candidate_data:
            best = max(candidate_data, key=lambda c: c["confidence"])
            bx, by = latlon_to_meters(best["position"][0], best["position"][1],
                                      launch_lat, launch_lon)
            rect_best = patches.Rectangle(
                (bx - tile_w_m / 2, by - tile_h_m / 2),
                tile_w_m, tile_h_m,
                linewidth=2.5, edgecolor="#ff4444",
                facecolor="#ff4444", alpha=0.30, zorder=6,
            )
            ax.add_patch(rect_best)
            ax.plot(bx, by, "r^", markersize=10, label="Best match", zorder=7)

        # launch + target markers
        ax.plot(0, 0, "go", markersize=14, label="Launch",  zorder=10)
        ax.plot(target_x, target_y, "r*", markersize=18, label="Target", zorder=10)

        def _overlay(img, pos, edge_color, zoom_f=0.13):
            if img is None:
                return
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ib  = OffsetImage(rgb, zoom=zoom_f)
            ab  = AnnotationBbox(ib, pos, frameon=True, boxcoords="data", pad=0.3,
                                 bboxprops=dict(edgecolor=edge_color, linewidth=2))
            ax.add_artist(ab)

        _overlay(launch_image, (0, 0), "green")
        _overlay(target_image, (target_x, target_y), "red")

        ax.set_xlabel("East–West Distance (m)",   fontsize=11, color="#cccccc")
        ax.set_ylabel("North–South Distance (m)", fontsize=11, color="#cccccc")
        ax.legend(fontsize=9, facecolor="#1a1a2e", labelcolor="white")
        ax.grid(True, alpha=0.15, color="#444444")
        ax.set_aspect("equal", "box")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        return fig, ax, _overlay

    # --- version 1: no drone inset ------------------------------------------
    fig, ax, _overlay = _build_figure()
    ax.set_title("Sandwalk — Systematic Tile Grid",
                 fontsize=13, fontweight="bold", color="white", pad=12)
    path1 = os.path.join(output_dir, "search_visualization.png")
    plt.tight_layout()
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[VIZ] Saved: {path1}")
    plt.close()

    # --- version 2: drone inset ---------------------------------------------
    fig, ax, _overlay = _build_figure()
    ax.set_title("Sandwalk — Systematic Tile Grid (with Drone Reference)",
                 fontsize=13, fontweight="bold", color="white", pad=12)

    if drone_image is not None:
        # auto-detect empty corner (quadrant opposite the target)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx_ = xlim[0] + (xlim[1] - xlim[0]) * (0.85 if target_x < 0 else 0.15)
        cy_ = ylim[0] + (ylim[1] - ylim[0]) * (0.85 if target_y < 0 else 0.15)
        _overlay(drone_image, (cx_, cy_), "mediumpurple", zoom_f=0.18)
        ax.text(cx_, cy_ - (ylim[1] - ylim[0]) * 0.07,
                "Drone View", ha="center", fontsize=9,
                fontweight="bold", color="mediumpurple")

    path2 = os.path.join(output_dir, "search_visualization_with_drone.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[VIZ] Saved: {path2}")
    plt.close()


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
    ALTITUDE_M          = 120.0   # drone AGL altitude (metres)
                                  # replaces the old hardcoded ZOOM_LEVEL
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
    candidate_data: List[Dict] = []

    for idx, (lat, lon) in enumerate(candidates):
        print(f"[TEST] Tile {idx + 1:>3}/{len(candidates)}: "
              f"({lat:.6f}, {lon:.6f})… ", end="", flush=True)

        tile = load_satellite_tile(lat, lon, zoom, GOOGLE_MAPS_API_KEY, TILE_PX)
        if tile is None:
            print("FAILED (tile load error)")
            candidate_data.append({"position": (lat, lon), "image": None, "confidence": 0.35})
            continue

        fname = f"tile_{idx + 1:03d}_lat{lat:.6f}_lon{lon:.6f}.jpg"
        cv2.imwrite(os.path.join(output_dir, fname), tile)

        cell = idx_to_cell[idx]
        tiles_bgr[cell] = tile
        candidate_data.append({
            "position": (lat, lon),
            "image": tile,
            "confidence": 0.35,
        })
        print("ok")

    # ===== STITCH MOSAIC + TEMPLATE MATCH ====================================
    print(f"\n[TEST] Building preprocessed mosaic and running template match…")
    c0, c1, r0, r1 = tight_cell_bounds(cell_map)
    print(f"[GRID] Tight mosaic (match only): cols [{c0},{c1}] × rows [{r0},{r1}] "
          f"= {c1 - c0 + 1}×{r1 - r0 + 1} (snapped search grid was {n_cols}×{n_rows})")
    mosaic = build_preprocessed_mosaic(TILE_PX, tiles_bgr, c0, c1, r0, r1)
    cv2.imwrite(os.path.join(output_dir, "03_mosaic_preprocessed.jpg"), mosaic)

    tmpl_lat, tmpl_lon, tmpl_peak, tmpl_loc, tmpl_res = localize_template_on_mosaic(
        mosaic,
        processed_drone,
        x_min_m, y_min_m,
        tile_w_m, tile_h_m, TILE_PX,
        LAUNCH_LAT, LAUNCH_LON,
        c0, r1,
    )
    th, tw = processed_drone.shape[:2]
    cu = tmpl_loc[0] + 0.5 * tw
    cv = tmpl_loc[1] + 0.5 * th
    col_p = c0 + int(math.floor(cu / TILE_PX))
    irow_p = int(math.floor(cv / TILE_PX))
    peak_cell = (col_p, r1 - irow_p)
    for idx in range(len(candidate_data)):
        c, r = idx_to_cell[idx]
        candidate_data[idx]["confidence"] = 0.95 if (c, r) == peak_cell else 0.35

    vis = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, tmpl_loc, (tmpl_loc[0] + tw, tmpl_loc[1] + th), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(output_dir, "04_mosaic_template_peak.jpg"), vis)

    heat = cv2.normalize(tmpl_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "05_match_template_heatmap.jpg"), heat)
    save_template_vs_mosaic_crop(output_dir, processed_drone, mosaic, tmpl_loc)
    print(f"[TEST] Template peak NCC = {tmpl_peak:.3f} at pixel offset {tmpl_loc}")

    # ===== VISUALISATION =====================================================
    print(f"\n[TEST] Creating tile-grid visualisations…")
    create_search_visualization(
        LAUNCH_LAT, LAUNCH_LON,
        TARGET_LAT, TARGET_LON,
        launch_image, target_image, drone_frame,
        candidate_data,
        tile_w_m, tile_h_m,
        r_min, r_max,
        output_dir,
    )

    print(f"[TEST] Creating range map collage…")
    create_range_map_collage(
        candidate_data,
        n_cols, n_rows,
        cell_map,
        TILE_PX,
        output_dir,
    )

    print(f"[TEST] Creating satellite overlay…")
    create_satellite_overlay(
        LAUNCH_LAT, LAUNCH_LON,
        TARGET_LAT, TARGET_LON,
        launch_image, target_image,
        candidate_data,
        tile_w_m, tile_h_m,
        r_min, r_max,
        n_cols, n_rows,
        cell_map,
        TILE_PX,
        output_dir,
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