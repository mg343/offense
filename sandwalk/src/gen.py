import os
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

# Convert DMS → decimal
lat = 15 + 29/60 + 1.32/3600
lon = 44 + 13/60 + 22.11/3600

ZOOM = 18
SIZE = 640

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise ValueError("Set GOOGLE_MAPS_API_KEY environment variable")

# Fetch image
url = (
    f"https://maps.googleapis.com/maps/api/staticmap?"
    f"center={lat},{lon}"
    f"&zoom={ZOOM}"
    f"&size={SIZE}x{SIZE}"
    f"&maptype=satellite"
    f"&key={API_KEY}"
)

response = requests.get(url)
response.raise_for_status()

img = Image.open(BytesIO(response.content))
img = np.array(img)

# Convert to OpenCV format
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ---- Sandwalk-style preprocessing ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
denoised = cv2.GaussianBlur(gray, (5, 5), 0)
equalized = cv2.equalizeHist(denoised)
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
processed = cv2.filter2D(equalized, -1, kernel)

# Save
cv2.imwrite("satellite_zoom18_processed.png", processed)

print("Saved grayscale processed image")