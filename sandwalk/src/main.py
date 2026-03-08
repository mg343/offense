# sandwalk brain dump.

# essentially the same as glasses, just using vision localization over a range of possible inputs to determine location, instead of using vision localization to determine if location has been reached
# majority of the code should stay the same

# inputs always:
# - launch coords (lat, long)
# - motor usage (interpreted as distance traveled in meters)
# - target coords (lat, long) - assume always given, used to cut search space. when are issions without targets ever set.

# search zone logic:
# whole concept hinges on this idea of using motor data to define a potential search zone, rather than the computationally intensive task of checking with like every tile in a large region
# we'll also use a third variable for target location to narrow search when needed, which further dramatically cuts computation - say for instance drone is at (0,0), target is at (0,1), and you've roughly travelled 0.5 units - only need to check tiles in the search zone in ++ and +- quadrants, as there's no real reason for the drone to ever navigate away from this rough heading
# this improves further when at an angle, say 0,0 vs target of 1,1 - rather reasonable to reduce search range from the entire field to just the ++ region. but this is some measurement we'll have to figure out - might be easy to just draw a vector from (a,b) origin to (x,y) target and an estimated travel dist (via motor interpolation) of d, along with a percent d tolerance of like t (tolerance) = 10%, then limit search range via the following model 

# motor interpolation gives expected travel distance d but assuming noise, wheel slip, wind, etc
# so actual distance travelled lies in range from r_min = d * (1 - t) to r_max = d * (1 + t)
# this reduces possible locations to a ring centered at starting point (a,b), and any candidate tile must satisfy r_min^2 <= (x' - a)^2 + (y' - b)^2 <= r_max^2

# the general direction toward target introduces a second constraint, we can calc vector from start to target: v = (x - a, y - b)
# the search zone must the half of the refined ring centered on a portion of the target vector from r_min from the starting point to r_max from the starting point
# this should effetively eliminate the entire irrelvant search region behind the drone's planned heading
# at a later point, we can further introduce a variable that can limit the extent of this "half" of the ring centered on the relevant portion of the target vector, from 50% of the ring to even less.

# this will allow sandwalk to significantly reduce computational spend

# output every cycle:
# - estimated position (lat, long)
# - confidence score
# - number of tiles checked
# - dead reckoning error (how far off motors were)

# processing:
# grab single frame
# resize to 640x640
# preprocess (same as glasses)
# generate search candidates (lat/long points in constrained ring)
# fetch satellite tiles for each candidate
# run sift matching on each
# pick best match
# return coordinates


# a point of clarification after thought. right now, main_test does essentially what we need the program to do. the project could be done here.
# but, I think this has a lot of value and could be really cool to expand to a more deteministic and thought out model, taking it further could prove valuable.
# here, I plan to outline some goal steps that could make this project really useful - sometihng in me wants to piot to something more physics heavy, but i think this project can be really cool
# 1. the first thing that comes to my head is a more thorough look at the image compare system - this is paired with some way of representing zoom level more effectively and with a more one-size fits all generalizaable model
# curently, when we mess around with trying to image match, somehow near exact image matches come back with only 70% similarity. this can have many factors, we will have to dive into the feature analysis, and these error points could also be in the handling of zoom, but this is something to look into
# 2. two, and more specific to this project, is a reworking of what sattelite image of a location means. when we want to compare if the current lat/long we are at is in some set of lat/longs, we source sattelite images from every item in that set - ever min lat to max lat and within each sub lat, min long to max long, and the step size is determined by capture limits on the api
# however, a more human way of doing this would be to essentially use the bounds of the range that we're looking at, grab the min set of images that completely creates essentially our own full sattelite submap of the area, and then compare each of those images to our drone image
# for instance, say you think the drone is within a rectangular range from 0,0 lat long to 8,4 lat long. seperately, say we also know that the level the drone is at is x meters above ground. from this seperate measure, we know that the corresponding level of google maps sattelite zoom that would return sattelite images from a height closest to the height of the drone is z. using z, we finally also know that the images we source from maps to compare with our drone images will render as a square covering u meters per side length - which converts to roughly v1 units of lat and v2 units of long (the side lengths) per image.
# say for example sake, we find that v1/v2 are 2x2, where each image we source from google maps api covers a horizontal/vertical length equivalent to 2 lat/2 long unit per.
# in our current model, all of this is redundant. with only the information known from the 0,0 to 8,4 measurement, our direct next step would have been to simply sample as many coordinates and get sattelite images from as many coord pairs within this range as possible.
# but with this additional preprocessing, what we can do is identiy a reasonable subdivision and create a comprehensive map, in our neat case, nicely covering 100% of the range, and in other cases covering more (in other cases, defer to covering additional outside of range instead of less inside range), and sample, neatly and humanly, from there.
# in our case, the model should recognize this, and sample images from (via its own analysis based on size and range) - 1,1; 1,3; 3,1; 3,3; 5,1; 5,3; 7,1; 7,3.
# graphing or visualizing this: you can see that this method covers exactly the entire range of possible coords we wanted, and also adheres to the limits we set. this is far far more computationally efficient, and with a proper feature analysis system, could yield much more reasonable results
# for cases where its not as neat, this method still works! notice here the only information we have gave it remains the range (which we are already giving our measly current model), and additional info on the height (which we are ALSO giving our model)! this requires nothing but a more thoughtful implementation of the system.
# in not-so-neatly bound cases, we just defer to covering more space - say it was from 0,0 - 9,4, our model should be able to come to the reasonable conclusion that we would sample (again its its own function taking in the range and the v1/v2 numbers): 0,1; 0,3; 2,1; 2,3; 4,1; 4,3; 6,1; 6,3; 8,1; 8,3. this covers the full range completely.
# another edge case which we may see again is odd shapes - for instance a rectangle with edges has decimal bounds but v1/v2 are integers, or a circle (this issue is prone with round edges)
# in this case, estimate the closest range that CAN be covered that INCLUDES the range. a 0,0 - 8.1,4 range should be first expanded to a more convenient 0,0 - 9,4 range (this range is an even multiple of the v1/v2 constaints we would also give the function), then the points to source should be identified. for a circle of radius 2 centered at the origin and a v1/v2 determined to be 2,2, the model should identify that sourcing images from 1,1; -1,1; -1,-1; 1,-1 will cover the entire zone.
# in tandem with a more comprehensive image comparison engine, this system of completely mapping the range with minimal resources would be far more effective at identifying current position.
# some tools that jump to mind - template matching - we can use template matching to identify where in our "range map" our template image (drone image) is located, and then convert the local pixel coordinates to the exact location metrics.
# this system, in theory, yields a far higher, more logical, and clearly explainable approach to gps-denied localization problems.

import os
import cv2
import numpy as np
from typing import Tuple, List, Dict
import requests
from io import BytesIO
from PIL import Image
import time
import math


class Sandwalk:
    """
    GPS-denied navigation via vision-based localization.
    """
    
    def __init__(
        self,
        launch_lat: float,
        launch_lon: float,
        target_lat: float,
        target_lon: float,
        zoom_level: int,
        google_maps_api_key: str,
        localization_confidence_threshold: float = 0.60,
        cycle_interval: float = 3.0,
        processing_width: int = 640,
        processing_height: int = 640,
        tolerance_percent: float = 0.10
    ):
        """
        Initialize Sandwalk system.
        
        Args:
            launch_lat: Launch latitude
            launch_lon: Launch longitude  
            target_lat: Target latitude
            target_lon: Target longitude
            zoom_level: Google Maps zoom level
            google_maps_api_key: API key
            localization_confidence_threshold: Min confidence for position fix
            cycle_interval: Seconds between localization updates (default 3.0)
            processing_width: Image processing width (default 640)
            processing_height: Image processing height (default 640)
            tolerance_percent: Distance estimate tolerance (default 0.10 = 10%)
        """
        self.launch_lat = launch_lat
        self.launch_lon = launch_lon
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.zoom_level = zoom_level
        self.api_key = google_maps_api_key
        self.confidence_threshold = localization_confidence_threshold
        self.cycle_interval = cycle_interval
        self.tolerance_percent = tolerance_percent
        
        self.processing_width = processing_width
        self.processing_height = processing_height
        
        # output directory
        self.output_dir = self._create_output_directory()
        
        # feature detector and matcher (reuse from glasses)
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # tile cache to avoid redundant API calls
        self.tile_cache: Dict[Tuple[float, float], np.ndarray] = {}
        
        # tracking
        self.last_known_position = (launch_lat, launch_lon)
        self.position_history: List[Dict] = []
        
        print(f"[SANDWALK] Initialized")
        print(f"[SANDWALK] Launch: ({launch_lat:.6f}, {launch_lon:.6f})")
        print(f"[SANDWALK] Target: ({target_lat:.6f}, {target_lon:.6f})")
        print(f"[SANDWALK] Cycle interval: {cycle_interval}s")
        print(f"[SANDWALK] Tolerance: ±{tolerance_percent*100}%")
        print(f"[SANDWALK] Output: {self.output_dir}")
    
    
    def _create_output_directory(self) -> str:
        """Create output directory for logging position fixes."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_output = os.path.join(script_dir, "files", "output")
        os.makedirs(base_output, exist_ok=True)
        
        mission_id = f"sandwalk_{self.launch_lat:.4f}_{self.launch_lon:.4f}"
        mission_dir = os.path.join(base_output, mission_id)
        os.makedirs(mission_dir, exist_ok=True)
        
        positions_dir = os.path.join(mission_dir, "position_fixes")
        os.makedirs(positions_dir, exist_ok=True)
        
        return mission_dir
    
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance in meters between two lat/lon points.
        """
        R = 6371000  # earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    
    def _offset_coordinates(self, lat: float, lon: float, dx_meters: float, dy_meters: float) -> Tuple[float, float]:
        """
        Offset lat/lon by dx, dy in meters.
        dx = east-west, dy = north-south
        """
        R = 6371000
        
        # latitude offset (north-south)
        new_lat = lat + (dy_meters / R) * (180 / math.pi)
        
        # longitude offset (east-west), accounting for latitude
        new_lon = lon + (dx_meters / (R * math.cos(math.radians(lat)))) * (180 / math.pi)
        
        return new_lat, new_lon
    
    
    def _generate_search_candidates(self, motor_distance_m: float) -> List[Tuple[float, float]]:
        """
        Generate candidate positions based on motor distance and target constraint.
        
        Returns ring of positions between r_min and r_max from launch,
        constrained to hemisphere pointing toward target.
        """
        # tolerance
        min_tolerance_m = 20  # minimum 20m tolerance
        tolerance_m = max(min_tolerance_m, motor_distance_m * self.tolerance_percent)
        
        r_min = motor_distance_m - tolerance_m
        r_max = motor_distance_m + tolerance_m
        
        # clamp r_min to avoid negatives on very short distances
        r_min = max(0, r_min)
        
        # target vector for hemisphere constraint
        target_bearing = math.atan2(
            self.target_lon - self.launch_lon,
            self.target_lat - self.launch_lat
        )
        
        candidates = []
        
        # sample points in ring
        # number of samples scales with area
        ring_area = math.pi * (r_max**2 - r_min**2)
        num_samples = int(max(8, min(50, ring_area / 1000)))  # 1 sample per ~1000 m^2, capped
        
        for i in range(num_samples):
            # random angle
            angle = (i / num_samples) * 2 * math.pi
            
            # check if angle is in target hemisphere (within ±90° of target bearing)
            angle_diff = ((angle - target_bearing + math.pi) % (2 * math.pi)) - math.pi
            if abs(angle_diff) > math.pi / 2:
                continue  # outside target hemisphere
            
            # random radius in ring
            r = math.sqrt(np.random.uniform(r_min**2, r_max**2))
            
            # convert to offset
            dx = r * math.sin(angle)
            dy = r * math.cos(angle)
            
            # convert to lat/lon
            cand_lat, cand_lon = self._offset_coordinates(self.launch_lat, self.launch_lon, dx, dy)
            candidates.append((cand_lat, cand_lon))
        
        print(f"[SANDWALK] Generated {len(candidates)} search candidates (distance: {motor_distance_m:.1f}m ±{tolerance_m:.1f}m)")
        
        return candidates
    
    
    def _load_satellite_tile(self, lat: float, lon: float) -> np.ndarray:
        """
        Load satellite tile for given coordinates.
        Uses cache to avoid redundant API calls.
        """
        coords_key = (round(lat, 6), round(lon, 6))
        
        if coords_key in self.tile_cache:
            return self.tile_cache[coords_key]
        
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}"
            f"&zoom={self.zoom_level}"
            f"&size={self.processing_width}x{self.processing_height}"
            f"&maptype=satellite"
            f"&key={self.api_key}"
        )
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            tile = np.array(image)
            
            if len(tile.shape) == 3:
                tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            
            self.tile_cache[coords_key] = tile
            return tile
            
        except Exception as e:
            print(f"[SANDWALK] ERROR loading tile for ({lat:.6f}, {lon:.6f}): {e}")
            return np.zeros((self.processing_height, self.processing_width, 3), dtype=np.uint8)
    
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame (same logic as glasses).
        """
        height, width = frame.shape[:2]
        
        # resize/crop to processing dimensions
        if width >= self.processing_width and height >= self.processing_height:
            start_x = (width - self.processing_width) // 2
            start_y = (height - self.processing_height) // 2
            resized = frame[start_y:start_y + self.processing_height, start_x:start_x + self.processing_width]
        else:
            resized = cv2.resize(frame, (self.processing_width, self.processing_height), interpolation=cv2.INTER_LANCZOS4)
        
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
    
    
    def compute_similarity(self, drone_frame: np.ndarray, reference_tile: np.ndarray) -> float:
        """
        Compute similarity score (reuse glasses matching logic).
        Returns just confidence score, not GO/NO_GO.
        """
        if len(reference_tile.shape) == 3:
            ref_gray = cv2.cvtColor(reference_tile, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_tile
        
        kp1, desc1 = self.feature_detector.detectAndCompute(drone_frame, None)
        kp2, desc2 = self.feature_detector.detectAndCompute(ref_gray, None)
        
        if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0.0
        
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return 0.0
        
        match_ratio = len(good_matches) / min(len(kp1), len(kp2))
        
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
    
    
    def localize(self, frame: np.ndarray, motor_distance_m: float, timestamp: float) -> Dict:
        """
        Localize drone position from single frame and motor reading.
        
        Args:
            frame: Camera frame
            motor_distance_m: Estimated distance traveled from launch
            timestamp: Time since mission start
            
        Returns:
            Position fix dictionary
        """
        # preprocess frame
        processed = self.preprocess_frame(frame)
        
        # generate search candidates
        candidates = self._generate_search_candidates(motor_distance_m)
        
        if len(candidates) == 0:
            print("[SANDWALK] WARNING: No search candidates generated")
            return {
                "status": "NO_CANDIDATES",
                "position": self.last_known_position,
                "confidence": 0.0,
                "timestamp": timestamp
            }
        
        # match against each candidate tile
        best_position = None
        best_confidence = 0.0
        
        for lat, lon in candidates:
            tile = self._load_satellite_tile(lat, lon)
            confidence = self.compute_similarity(processed, tile)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_position = (lat, lon)
        
        # determine status
        if best_confidence >= self.confidence_threshold:
            status = "LOCALIZED"
            self.last_known_position = best_position
        elif best_confidence > 0.3:
            status = "UNCERTAIN"
        else:
            status = "NO_MATCH"
            best_position = self.last_known_position
        
        # calculate dead reckoning error
        dr_error = 0.0
        if best_position and status == "LOCALIZED":
            # dead reckoning estimate (simple: straight line from launch)
            bearing = math.atan2(self.target_lon - self.launch_lon, self.target_lat - self.launch_lat)
            dr_lat, dr_lon = self._offset_coordinates(
                self.launch_lat, 
                self.launch_lon,
                motor_distance_m * math.sin(bearing),
                motor_distance_m * math.cos(bearing)
            )
            dr_error = self._haversine_distance(best_position[0], best_position[1], dr_lat, dr_lon)
        
        result = {
            "status": status,
            "position": best_position,
            "confidence": best_confidence,
            "uncertainty_m": (1.0 - best_confidence) * 100,  # rough uncertainty estimate
            "tiles_checked": len(candidates),
            "dead_reckoning_error_m": dr_error,
            "timestamp": timestamp
        }
        
        self.position_history.append(result)
        
        print(f"[SANDWALK] t={timestamp:.1f}s | {status} | pos=({best_position[0]:.6f}, {best_position[1]:.6f}) | conf={best_confidence:.2%} | tiles={len(candidates)}")
        
        return result
    
    
    def process_mission(self, video_path: str, motor_readings: List[float]) -> Dict:
        """
        Process entire mission video with motor readings.
        
        Args:
            video_path: Path to drone video
            motor_readings: List of distance readings (meters) corresponding to cycle intervals
            
        Returns:
            Mission summary
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"status": "ERROR", "message": f"Could not open video: {video_path}"}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"[SANDWALK] Processing mission video ({duration:.1f}s, {total_frames} frames)")
        
        cycle_count = 0
        
        while cap.isOpened() and cycle_count < len(motor_readings):
            # seek to frame corresponding to cycle time
            target_time = cycle_count * self.cycle_interval
            frame_num = int(target_time * fps)
            
            if frame_num >= total_frames:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # localize
            motor_distance = motor_readings[cycle_count]
            result = self.localize(frame, motor_distance, target_time)
            
            cycle_count += 1
        
        cap.release()
        
        # summary
        successful_fixes = sum(1 for r in self.position_history if r["status"] == "LOCALIZED")
        avg_confidence = np.mean([r["confidence"] for r in self.position_history if r["confidence"] > 0])
        
        summary = {
            "status": "COMPLETE",
            "total_cycles": cycle_count,
            "successful_localizations": successful_fixes,
            "localization_rate": successful_fixes / cycle_count if cycle_count > 0 else 0,
            "average_confidence": avg_confidence,
            "final_position": self.last_known_position,
            "output_directory": self.output_dir
        }
        
        return summary


def main():
    """
    Main execution for Sandwalk.
    """
    print("\n" + "="*60)
    print("SANDWALK - GPS-Denied Navigation System")
    print("="*60 + "\n")
    
    # mission parameters
    LAUNCH_LAT = 15.4800
    LAUNCH_LON = 44.2200
    TARGET_LAT = 15.4878
    TARGET_LON = 44.2261
    ZOOM_LEVEL = 18
    CONFIDENCE_THRESHOLD = 0.60
    
    GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not GOOGLE_MAPS_API_KEY:
        print("ERROR: Set GOOGLE_MAPS_API_KEY environment variable")
        exit(1)
    
    # video path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    VIDEO_PATH = os.path.join(script_dir, "files", "stream.mp4")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        exit(1)
    
    # simulated motor readings (distance in meters at each cycle)
    # in real system this would come from telemetry
    MOTOR_READINGS = [0, 50, 150, 300, 500, 700, 900, 1100, 1200]  # example progression
    
    # initialize
    sandwalk = Sandwalk(
        launch_lat=LAUNCH_LAT,
        launch_lon=LAUNCH_LON,
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        zoom_level=ZOOM_LEVEL,
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
        localization_confidence_threshold=CONFIDENCE_THRESHOLD,
        cycle_interval=3.0,
        processing_width=640,
        processing_height=640,
        tolerance_percent=0.10
    )
    
    # process mission
    results = sandwalk.process_mission(VIDEO_PATH, MOTOR_READINGS)
    
    # display results
    print("\n" + "="*60)
    print("MISSION SUMMARY")
    print("="*60)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    main()