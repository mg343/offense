"""
Eyes - Autonomous Drone Payload Deployment System
Post-mission localization via visual similarity matching
"""

import cv2
import numpy as np
from typing import Tuple
import requests
from io import BytesIO
from PIL import Image
import time
import os


class Eyes:
    """
    Main controller for Eyes payload deployment verification system.
    """
    
    def __init__(
        self,
        target_lat: float,
        target_lon: float,
        zoom_level: int,
        google_maps_api_key: str,
        deployment_confidence_threshold: float = 0.90,
        frame_rate: int = 30,
        processing_width: int = 640,
        processing_height: int = 640
    ):
        """
        Initialize Eyes system.
        
        Args:
            target_lat: Target latitude coordinates
            target_lon: Target longitude coordinates
            zoom_level: Google Maps zoom level (13-21)
            google_maps_api_key: Google Maps API key for Static API
            deployment_confidence_threshold: Minimum confidence to deploy (default 0.90)
            frame_rate: Processing frame rate in fps (default 30)
            processing_width: Target width for processing (default 640)
            processing_height: Target height for processing (default 640)
        """
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.zoom_level = zoom_level
        self.api_key = google_maps_api_key
        self.confidence_threshold = deployment_confidence_threshold
        self.frame_interval = 1.0 / frame_rate
        
        # Processing dimensions (default 640x640 to stay within Google Maps API limits)
        self.processing_width = processing_width
        self.processing_height = processing_height
        
        # Load reference satellite imagery at initialization
        self.reference_image = self._load_satellite_reference()
        
        # Feature detector for robust matching
        self.feature_detector = cv2.SIFT_create()
        
        # Matcher for feature comparison
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        print(f"[EYES] Initialized with target: ({target_lat}, {target_lon})")
        print(f"[EYES] Zoom level: {zoom_level}")
        print(f"[EYES] Processing dimensions: {self.processing_width}x{self.processing_height}")
        print(f"[EYES] Deployment threshold: {deployment_confidence_threshold * 100}%")
    
    
    def _load_satellite_reference(self) -> np.ndarray:
        """
        Load satellite imagery of target drop zone from Google Maps Static API.
        Uses processing dimensions (default 640x640) to stay within API limits.
        
        Returns:
            np.ndarray: Reference satellite image
        """
        # Google Maps Static API URL with processing dimensions
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={self.target_lat},{self.target_lon}"
            f"&zoom={self.zoom_level}"
            f"&size={self.processing_width}x{self.processing_height}"
            f"&maptype=satellite"
            f"&key={self.api_key}"
        )
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Convert to numpy array
            image = Image.open(BytesIO(response.content))
            reference = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(reference.shape) == 3:
                reference = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
            
            print(f"[EYES] Loaded satellite reference: {reference.shape} at zoom {self.zoom_level}")
            return reference
            
        except Exception as e:
            print(f"[EYES] ERROR: Could not load satellite imagery: {e}")
            return np.zeros((self.processing_height, self.processing_width, 3), dtype=np.uint8)
    
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess camera frame for accurate comparison.
        Resizes/crops to processing dimensions, then applies image enhancements.
        
        Strategy:
        1. If frame >= processing dimensions: crop center (preserves resolution)
        2. If frame < processing dimensions: resize (upscale if needed)
        
        Args:
            frame: Raw camera frame
            
        Returns:
            np.ndarray: Processed frame at processing dimensions
        """
        height, width = frame.shape[:2]
        
        # Resize/crop to processing dimensions
        if width >= self.processing_width and height >= self.processing_height:
            # Crop center to preserve resolution
            start_x = (width - self.processing_width) // 2
            start_y = (height - self.processing_height) // 2
            resized = frame[start_y:start_y + self.processing_height, start_x:start_x + self.processing_width]
        else:
            # Resize to processing dimensions
            resized = cv2.resize(frame, (self.processing_width, self.processing_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to grayscale
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        # Denoise
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Normalize lighting
        equalized = cv2.equalizeHist(denoised)
        
        # Edge enhancement
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(equalized, -1, kernel)
        
        return sharpened
    
    
    def compute_similarity(
        self, 
        drone_frame: np.ndarray, 
        reference_frame: np.ndarray
    ) -> Tuple[str, float]:
        """
        Compute similarity between drone camera frame and satellite reference.
        Uses SIFT feature matching for robust comparison.
        
        Args:
            drone_frame: Preprocessed frame from drone camera
            reference_frame: Satellite reference image
            
        Returns:
            Tuple[str, float]: ("GO" or "NO_GO", confidence_score)
        """
        # Preprocess reference
        if len(reference_frame.shape) == 3:
            ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_frame
        
        # Detect keypoints and compute descriptors
        kp1, desc1 = self.feature_detector.detectAndCompute(drone_frame, None)
        kp2, desc2 = self.feature_detector.detectAndCompute(ref_gray, None)
        
        # Handle edge case: no features detected
        if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
            return "NO_GO", 0.0
        
        # Match features using k-NN
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # Calculate confidence
        if len(good_matches) < 10:
            confidence = 0.0
        else:
            match_ratio = len(good_matches) / min(len(kp1), len(kp2))
            
            # Geometric consistency check
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
        
        confidence = min(max(confidence, 0.0), 1.0)
        decision = "GO" if confidence >= self.confidence_threshold else "NO_GO"
        
        return decision, confidence
    
    
    def process_stream(self, video_path: str) -> dict:
        """
        Process video stream from drone camera.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Mission results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                "status": "ERROR",
                "message": f"Could not open video stream: {video_path}",
                "deployed": False
            }
        
        print(f"[EYES] Processing stream: {video_path}")
        
        frame_count = 0
        deployed = False
        deployment_frame = None
        deployment_confidence = 0.0
        deployment_timestamp = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert ms to seconds
            start_time = time.time()
            
            # Preprocess and compare
            processed = self.preprocess_frame(frame)
            decision, confidence = self.compute_similarity(processed, self.reference_image)
            
            # Updated logging: Print time elapsed (s) instead of frame count
            if frame_count % 10 == 0:
                print(f"[EYES] Time: {current_timestamp:.2f}s: {decision} (confidence: {confidence:.2%})")
            
            # Check for target detection
            if decision == "GO" and not deployed:
                deployed = True
                deployment_frame = frame_count
                deployment_confidence = confidence
                # Using the timestamp calculated from the video properties
                deployment_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                print(f"\n{'='*60}")
                print(f"[EYES] 🎯 TARGET DETECTED")
                print(f"[EYES] Time Elapsed: {deployment_timestamp:.2f}s")  # Updated line
                print(f"[EYES] Confidence: {deployment_confidence:.2%}")
                print(f"{'='*60}\n")
                break
            
            # Frame rate timing
            elapsed = time.time() - start_time
            time.sleep(max(0, self.frame_interval - elapsed))
        
        cap.release()
        
        # Results
        results = {
            "status": "SUCCESS" if deployed else "NO_DEPLOYMENT",
            "deployed": deployed,
            "total_frames_processed": frame_count,
            "target_coordinates": (self.target_lat, self.target_lon),
            "zoom_level": self.zoom_level,
            "processing_dimensions": f"{self.processing_width}x{self.processing_height}"
        }
        
        if deployed:
            results.update({
                "deployment_frame": deployment_frame,
                "deployment_timestamp_s": deployment_timestamp,
                "deployment_confidence": deployment_confidence
            })
        
        return results


def main():
    """
    Main execution function for Eyes system.
    """
    print("\n" + "="*60)
    print("EYES - Autonomous Payload Deployment System")
    print("="*60 + "\n")
    
    # Configuration
    TARGET_LAT = 15.4878
    TARGET_LON = 44.2261
    ZOOM_LEVEL = 20
    GOOGLE_MAPS_API_KEY = "YOUR_API_KEY_HERE"
    CONFIDENCE_THRESHOLD = 0.60
    
    # Get script directory and construct video path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    VIDEO_STREAM = os.path.join(script_dir, "files", "stream.mp4")
    
    # Initialize and run
    eyes = Eyes(
        target_lat=TARGET_LAT,
        target_lon=TARGET_LON,
        zoom_level=ZOOM_LEVEL,
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
        deployment_confidence_threshold=CONFIDENCE_THRESHOLD,
        frame_rate=30,
        processing_width=640,  # Can be changed if needed
        processing_height=640   # Can be changed if needed
    )
    
    results = eyes.process_stream(VIDEO_STREAM)
    
    # Display results
    print("\n" + "="*60)
    print("MISSION RESULTS")
    print("="*60)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    main()