"""
Sandwalk Single Cycle Test
Tests one localization cycle with a single image.
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two lat/lon points."""
    R = 6371000  # earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def latlon_to_meters(lat: float, lon: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """
    Convert lat/lon to meters from origin.
    Returns (x_meters, y_meters) where x=east-west, y=north-south.
    """
    R = 6371000
    
    # y: north-south distance
    y = (lat - origin_lat) * (math.pi / 180) * R
    
    # x: east-west distance (accounting for latitude)
    x = (lon - origin_lon) * (math.pi / 180) * R * math.cos(math.radians(origin_lat))
    
    return x, y


def offset_coordinates(lat: float, lon: float, dx_meters: float, dy_meters: float) -> Tuple[float, float]:
    """
    Offset lat/lon by dx, dy in meters.
    dx = east-west, dy = north-south
    """
    R = 6371000
    
    # latitude offset (north-south)
    new_lat = lat + (dy_meters / R) * (180 / math.pi)
    
    # longitude offset (east-west)
    new_lon = lon + (dx_meters / (R * math.cos(math.radians(lat)))) * (180 / math.pi)
    
    return new_lat, new_lon


def generate_search_candidates(
    launch_lat: float,
    launch_lon: float,
    target_lat: float,
    target_lon: float,
    distance_traveled_m: float,
    tolerance_percent: float = 0.05
) -> List[Tuple[float, float]]:
    """
    Generate candidate positions in ring constrained to target hemisphere.
    """
    # tolerance
    min_tolerance_m = 20
    tolerance_m = max(min_tolerance_m, distance_traveled_m * tolerance_percent)
    
    r_min = max(0, distance_traveled_m - tolerance_m)
    r_max = distance_traveled_m + tolerance_m
    
    # target bearing
    target_bearing = math.atan2(
        target_lon - launch_lon,
        target_lat - launch_lat
    )
    
    candidates = []
    
    # sample points in ring
    ring_area = math.pi * (r_max**2 - r_min**2)
    num_samples = int(max(12, min(40, ring_area / 1000)))
    
    # Generate angles ONLY in target hemisphere (±90° from target bearing)
    angle_min = target_bearing - math.pi / 2
    angle_max = target_bearing + math.pi / 2
    
    for i in range(num_samples):
        # Angle within target hemisphere only
        angle = angle_min + (i / num_samples) * (angle_max - angle_min)
        
        # sample radius in ring (uniform distribution in area)
        r = math.sqrt(np.random.uniform(r_min**2, r_max**2))
        
        # convert to offset
        dx = r * math.sin(angle)
        dy = r * math.cos(angle)
        
        # convert to lat/lon
        cand_lat, cand_lon = offset_coordinates(launch_lat, launch_lon, dx, dy)
        candidates.append((cand_lat, cand_lon))
    
    print(f"[TEST] Generated {len(candidates)} search candidates")
    print(f"[TEST] Search ring: {r_min:.1f}m - {r_max:.1f}m from launch")
    print(f"[TEST] Hemisphere: {math.degrees(angle_min):.1f}° to {math.degrees(angle_max):.1f}°")
    
    return candidates


def load_satellite_tile(lat: float, lon: float, zoom: int, api_key: str, size: int = 640) -> np.ndarray:
    """Load satellite tile from Google Maps."""
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={size}x{size}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        tile = np.array(image)
        
        if len(tile.shape) == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        
        return tile
    except Exception as e:
        print(f"[TEST] ERROR loading tile: {e}")
        return None


def preprocess_frame(frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Preprocess frame for matching."""
    height, width = frame.shape[:2]
    
    # resize/crop to target size
    if width >= target_size and height >= target_size:
        start_x = (width - target_size) // 2
        start_y = (height - target_size) // 2
        resized = frame[start_y:start_y + target_size, start_x:start_x + target_size]
    else:
        resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    # grayscale
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
    
    # denoise
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # lighting normalization
    equalized = cv2.equalizeHist(denoised)
    
    # edge enhancement
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    
    return sharpened


def compute_similarity(drone_frame: np.ndarray, satellite_tile: np.ndarray) -> float:
    """Compute SIFT-based similarity score."""
    # SIFT detector
    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # preprocess satellite tile
    if len(satellite_tile.shape) == 3:
        sat_gray = cv2.cvtColor(satellite_tile, cv2.COLOR_BGR2GRAY)
    else:
        sat_gray = satellite_tile
    
    # detect and match
    kp1, desc1 = sift.detectAndCompute(drone_frame, None)
    kp2, desc2 = sift.detectAndCompute(sat_gray, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0
    
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        return 0.0
    
    match_ratio = len(good_matches) / min(len(kp1), len(kp2))
    
    # geometric consistency
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask)
            confidence = (match_ratio * 0.4 + inlier_ratio * 0.6)
        else:
            confidence = match_ratio * 0.5
    else:
        confidence = match_ratio * 0.5
    
    return min(max(confidence, 0.0), 1.0)


def create_search_visualization(
    launch_lat: float,
    launch_lon: float,
    target_lat: float,
    target_lon: float,
    launch_image: np.ndarray,
    target_image: np.ndarray,
    drone_image: np.ndarray,
    candidate_data: List[Dict],
    output_dir: str
):
    """
    Create matplotlib visualization of search area.
    Axes in meters from launch location.
    
    Args:
        launch_lat, launch_lon: Origin coordinates
        target_lat, target_lon: Target coordinates
        launch_image: Image at launch location
        target_image: Image at target location
        drone_image: Drone camera image (for reference)
        candidate_data: List of dicts with 'position', 'image', 'confidence'
        output_dir: Where to save visualization
    """
    
    # Convert all positions to meters from launch
    target_x, target_y = latlon_to_meters(target_lat, target_lon, launch_lat, launch_lon)
    
    # VERSION 1: Without drone image in corner
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # plot launch location (origin at 0,0)
    ax.plot(0, 0, 'go', markersize=15, label='Launch', zorder=10)
    
    # plot target location
    ax.plot(target_x, target_y, 'r*', markersize=20, label='Target', zorder=10)
    
    # overlay launch image
    if launch_image is not None:
        launch_rgb = cv2.cvtColor(launch_image, cv2.COLOR_BGR2RGB)
        imagebox = OffsetImage(launch_rgb, zoom=0.15)
        ab = AnnotationBbox(imagebox, (0, 0), frameon=True, 
                           boxcoords="data", pad=0.3, bboxprops=dict(edgecolor='green', linewidth=2))
        ax.add_artist(ab)
    
    # overlay target image
    if target_image is not None:
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        imagebox = OffsetImage(target_rgb, zoom=0.15)
        ab = AnnotationBbox(imagebox, (target_x, target_y), frameon=True,
                           boxcoords="data", pad=0.3, bboxprops=dict(edgecolor='red', linewidth=2))
        ax.add_artist(ab)
    
    # overlay candidate images
    for idx, cand in enumerate(candidate_data):
        lat, lon = cand['position']
        x, y = latlon_to_meters(lat, lon, launch_lat, launch_lon)
        img = cand['image']
        conf = cand['confidence']
        
        # plot point
        color = 'blue' if conf < 0.6 else 'yellow'
        ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.6, zorder=5)
        
        # overlay image
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagebox = OffsetImage(img_rgb, zoom=0.1)
            ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                               boxcoords="data", pad=0.2, 
                               bboxprops=dict(edgecolor=color, linewidth=1, alpha=0.7))
            ax.add_artist(ab)
    
    ax.set_xlabel('East-West Distance (meters)', fontsize=12)
    ax.set_ylabel('North-South Distance (meters)', fontsize=12)
    ax.set_title('Sandwalk Search Zone Visualization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # save version 1
    viz_path = os.path.join(output_dir, 'search_visualization.png')
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"[TEST] Saved visualization (v1) to: {viz_path}")
    plt.close()
    
    # VERSION 2: With drone image in corner
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # plot launch location (origin at 0,0)
    ax.plot(0, 0, 'go', markersize=15, label='Launch', zorder=10)
    
    # plot target location
    ax.plot(target_x, target_y, 'r*', markersize=20, label='Target', zorder=10)
    
    # overlay launch image
    if launch_image is not None:
        launch_rgb = cv2.cvtColor(launch_image, cv2.COLOR_BGR2RGB)
        imagebox = OffsetImage(launch_rgb, zoom=0.15)
        ab = AnnotationBbox(imagebox, (0, 0), frameon=True, 
                           boxcoords="data", pad=0.3, bboxprops=dict(edgecolor='green', linewidth=2))
        ax.add_artist(ab)
    
    # overlay target image
    if target_image is not None:
        target_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
        imagebox = OffsetImage(target_rgb, zoom=0.15)
        ab = AnnotationBbox(imagebox, (target_x, target_y), frameon=True,
                           boxcoords="data", pad=0.3, bboxprops=dict(edgecolor='red', linewidth=2))
        ax.add_artist(ab)
    
    # overlay candidate images
    for idx, cand in enumerate(candidate_data):
        lat, lon = cand['position']
        x, y = latlon_to_meters(lat, lon, launch_lat, launch_lon)
        img = cand['image']
        conf = cand['confidence']
        
        # plot point
        color = 'blue' if conf < 0.6 else 'yellow'
        ax.plot(x, y, 'o', color=color, markersize=8, alpha=0.6, zorder=5)
        
        # overlay image
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagebox = OffsetImage(img_rgb, zoom=0.1)
            ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                               boxcoords="data", pad=0.2, 
                               bboxprops=dict(edgecolor=color, linewidth=1, alpha=0.7))
            ax.add_artist(ab)
    
    # Add drone image in empty corner (top-left or bottom-right depending on target position)
    # Determine which corner is empty
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Find empty corner (opposite quadrant from target)
    if target_x >= 0 and target_y >= 0:
        # target in top-right, put drone in bottom-left
        corner_x = xlim[0] + (xlim[1] - xlim[0]) * 0.15
        corner_y = ylim[0] + (ylim[1] - ylim[0]) * 0.15
    elif target_x < 0 and target_y >= 0:
        # target in top-left, put drone in bottom-right
        corner_x = xlim[1] - (xlim[1] - xlim[0]) * 0.15
        corner_y = ylim[0] + (ylim[1] - ylim[0]) * 0.15
    elif target_x >= 0 and target_y < 0:
        # target in bottom-right, put drone in top-left
        corner_x = xlim[0] + (xlim[1] - xlim[0]) * 0.15
        corner_y = ylim[1] - (ylim[1] - ylim[0]) * 0.15
    else:
        # target in bottom-left, put drone in top-right
        corner_x = xlim[1] - (xlim[1] - xlim[0]) * 0.15
        corner_y = ylim[1] - (ylim[1] - ylim[0]) * 0.15
    
    if drone_image is not None:
        drone_rgb = cv2.cvtColor(drone_image, cv2.COLOR_BGR2RGB)
        imagebox = OffsetImage(drone_rgb, zoom=0.2)
        ab = AnnotationBbox(imagebox, (corner_x, corner_y), frameon=True,
                           boxcoords="data", pad=0.3, 
                           bboxprops=dict(edgecolor='purple', linewidth=3))
        ax.add_artist(ab)
        ax.text(corner_x, corner_y - (ylim[1] - ylim[0]) * 0.08, 
               'Drone View', ha='center', fontsize=10, fontweight='bold', color='purple')
    
    ax.set_xlabel('East-West Distance (meters)', fontsize=12)
    ax.set_ylabel('North-South Distance (meters)', fontsize=12)
    ax.set_title('Sandwalk Search Zone Visualization (with Drone Reference)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # save version 2
    viz_path_v2 = os.path.join(output_dir, 'search_visualization_with_drone.png')
    plt.tight_layout()
    plt.savefig(viz_path_v2, dpi=150, bbox_inches='tight')
    print(f"[TEST] Saved visualization (v2 with drone) to: {viz_path_v2}")
    plt.close()


def main():
    """
    Single cycle localization test.
    
    Required inputs:
    - Launch coordinates (hardcoded below)
    - Target coordinates (hardcoded below)
    - Distance traveled estimate (hardcoded below)
    - Drone image at: ./files/drone_image.png
    """
    
    print("\n" + "="*60)
    print("SANDWALK - SINGLE CYCLE TEST")
    print("="*60 + "\n")
    
    # ===== USER INPUTS =====
    LAUNCH_LAT = 15.4800
    LAUNCH_LON = 44.2200
    
    TARGET_LAT = 15.4878
    TARGET_LON = 44.2261
    
    DISTANCE_TRAVELED_M = 500.0  # motor estimate in meters
    
    ZOOM_LEVEL = 18
    TOLERANCE_PERCENT = 0.10
    
    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: Set GOOGLE_MAPS_API_KEY environment variable")
        exit(1)
    
    # ===== SETUP OUTPUT DIRECTORY =====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "files", "output", "sandwalk_test")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TEST] Output directory: {output_dir}")
    
    # ===== LOAD DRONE IMAGE =====
    drone_image_path = os.path.join(script_dir, "files", "drone_image.png")
    
    if not os.path.exists(drone_image_path):
        print(f"ERROR: Drone image not found at {drone_image_path}")
        print(f"Please place your drone image at: {drone_image_path}")
        exit(1)
    
    drone_frame = cv2.imread(drone_image_path)
    if drone_frame is None:
        print(f"ERROR: Could not read drone image")
        exit(1)
    
    print(f"[TEST] Loaded drone image: {drone_frame.shape}")
    
    # Save drone image to output
    cv2.imwrite(os.path.join(output_dir, "00_drone_image.jpg"), drone_frame)
    
    # ===== LOAD LAUNCH LOCATION IMAGE =====
    print(f"[TEST] Loading launch location satellite image...")
    launch_image = load_satellite_tile(LAUNCH_LAT, LAUNCH_LON, ZOOM_LEVEL, GOOGLE_MAPS_API_KEY)
    if launch_image is not None:
        cv2.imwrite(os.path.join(output_dir, "01_launch_location.jpg"), launch_image)
    
    # ===== LOAD TARGET LOCATION IMAGE =====
    print(f"[TEST] Loading target location satellite image...")
    target_image = load_satellite_tile(TARGET_LAT, TARGET_LON, ZOOM_LEVEL, GOOGLE_MAPS_API_KEY)
    if target_image is not None:
        cv2.imwrite(os.path.join(output_dir, "02_target_location.jpg"), target_image)
    
    # ===== PREPROCESS DRONE IMAGE =====
    processed_drone = preprocess_frame(drone_frame)
    print(f"[TEST] Preprocessed drone image: {processed_drone.shape}")
    
    # ===== GENERATE SEARCH CANDIDATES =====
    candidates = generate_search_candidates(
        LAUNCH_LAT, LAUNCH_LON,
        TARGET_LAT, TARGET_LON,
        DISTANCE_TRAVELED_M,
        TOLERANCE_PERCENT
    )
    
    if len(candidates) == 0:
        print("[TEST] ERROR: No candidates generated")
        exit(1)
    
    # ===== MATCH AGAINST EACH CANDIDATE =====
    print(f"\n[TEST] Testing {len(candidates)} candidate positions...")
    
    best_position = None
    best_confidence = 0.0
    results = []
    candidate_data = []
    
    for idx, (lat, lon) in enumerate(candidates):
        print(f"[TEST] Candidate {idx+1}/{len(candidates)}: ({lat:.6f}, {lon:.6f})...", end=" ")
        
        # load satellite tile
        tile = load_satellite_tile(lat, lon, ZOOM_LEVEL, GOOGLE_MAPS_API_KEY)
        
        if tile is None:
            print("FAILED (tile load error)")
            continue
        
        # save tile to output
        tile_filename = f"candidate_{idx+1:02d}_lat{lat:.6f}_lon{lon:.6f}.jpg"
        cv2.imwrite(os.path.join(output_dir, tile_filename), tile)
        
        # compute similarity
        confidence = compute_similarity(processed_drone, tile)
        
        print(f"confidence: {confidence:.2%}")
        
        results.append({
            "position": (lat, lon),
            "confidence": confidence
        })
        
        candidate_data.append({
            "position": (lat, lon),
            "image": tile,
            "confidence": confidence
        })
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_position = (lat, lon)
    
    # ===== CREATE VISUALIZATIONS =====
    print(f"\n[TEST] Creating search visualizations...")
    create_search_visualization(
        LAUNCH_LAT, LAUNCH_LON,
        TARGET_LAT, TARGET_LON,
        launch_image,
        target_image,
        drone_frame,
        candidate_data,
        output_dir
    )
    
    # ===== RESULTS =====
    print("\n" + "="*60)
    print("LOCALIZATION RESULT")
    print("="*60)
    
    if best_position is None:
        print("STATUS: FAILED - No valid matches")
    elif best_confidence >= 0.60:
        print("STATUS: LOCALIZED")
        print(f"Estimated Position: ({best_position[0]:.6f}, {best_position[1]:.6f})")
        print(f"Confidence: {best_confidence:.2%}")
        
        # calculate dead reckoning error
        bearing = math.atan2(TARGET_LON - LAUNCH_LON, TARGET_LAT - LAUNCH_LAT)
        dr_lat, dr_lon = offset_coordinates(
            LAUNCH_LAT, LAUNCH_LON,
            DISTANCE_TRAVELED_M * math.sin(bearing),
            DISTANCE_TRAVELED_M * math.cos(bearing)
        )
        dr_error = haversine_distance(best_position[0], best_position[1], dr_lat, dr_lon)
        
        print(f"Dead Reckoning Error: {dr_error:.1f}m")
        print(f"Distance from Launch: {haversine_distance(LAUNCH_LAT, LAUNCH_LON, best_position[0], best_position[1]):.1f}m")
        print(f"Distance to Target: {haversine_distance(best_position[0], best_position[1], TARGET_LAT, TARGET_LON):.1f}m")
    else:
        print("STATUS: UNCERTAIN")
        print(f"Best Match: ({best_position[0]:.6f}, {best_position[1]:.6f})")
        print(f"Confidence: {best_confidence:.2%} (below threshold)")
    
    print(f"\nCandidates Evaluated: {len(candidates)}")
    print(f"Output Directory: {output_dir}")
    print("="*60 + "\n")
    
    # Show top 3 matches
    print("Top 3 Matches:")
    sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)[:3]
    for idx, r in enumerate(sorted_results):
        print(f"  {idx+1}. ({r['position'][0]:.6f}, {r['position'][1]:.6f}) - {r['confidence']:.2%}")


if __name__ == "__main__":
    main()