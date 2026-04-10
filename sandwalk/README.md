# Sandwalk

![Schematic](images/demo.svg)

### Background
Modern autonomous drone operations increasingly face GPS-denied or GPS-contested environments where traditional positioning systems are unreliable, jammed, or spoofed. For extended-range missions—whether covert military operations, search and rescue in remote terrain, or operations in electronically hostile environments—drones must navigate without continuous external positioning data while maintaining situational awareness and mission effectiveness.

### Overall Issue
When GPS is unavailable or untrusted, autonomous drones lose their primary means of localization. Dead reckoning using inertial measurement units (IMUs) and motor telemetry degrades rapidly due to sensor drift, accumulating errors of meters per minute. Within minutes of GPS denial, a drone operating on dead reckoning alone may have positional uncertainty of hundreds of meters, rendering precision navigation and payload deployment impossible. Existing vision-based navigation systems either require pre-mapped 3D environments (impractical for dynamic or unfamiliar terrain) or transmit imagery for ground-station processing (compromising operational security and requiring reliable datalinks).

### Solution (So-Far)
Current GPS-denied navigation relies on sensor fusion combining IMU data, barometric altitude, magnetometers, and motor usage estimates. While functional for short durations, these approaches suffer from unbounded drift. Visual odometry can track relative motion but cannot correct absolute position without external reference points. Some systems use SLAM (Simultaneous Localization and Mapping) but require significant onboard computation and struggle in featureless environments like deserts, oceans, or uniform terrain.

### Why This Fails
Dead reckoning fails because errors compound exponentially—a 1% drift in velocity estimation becomes a 60-meter error after just one minute of flight. IMU gyroscope drift, motor inefficiency variations, wind effects, and terrain obstacles all introduce unmodeled errors. Without absolute position correction, even the most sophisticated sensor fusion eventually loses accuracy. Visual odometry works for relative tracking but cannot answer "where am I?" without a known reference frame. SLAM can build local maps but doesn't solve global localization in unknown environments.

### Sandwalk
Sandwalk is an onboard, vision-based absolute positioning system that enables drones to determine their global coordinates in GPS-denied environments using only: (1) known launch location, (2) motor usage telemetry (rough distance traveled), and (3) live camera imagery. Unlike Glasses, which validates arrival at a pre-specified target, Sandwalk continuously localizes the drone anywhere along its flight path by matching observed terrain against satellite imagery within a dynamically constrained search region.

**How it works:**
1. **Launch + dead reckoning + target bearing → search zone**: With target lat/lon, Sandwalk builds a **half-annulus** in the horizontal plane: radial band = estimated distance ± tolerance, angular span = 180° centered on bearing toward the target (not a full disk).
2. **Systematic tile sourcing**: Barometric altitude maps to a **Static Maps zoom** and tile ground footprint. The zone’s axis-aligned bounds are **snapped outward** to full tiles; **only grid cells that intersect the zone polygon** are requested; they are **stitched into one north-up mosaic** (empty cells are never matched).
3. **Template matching (single pass)**: The drone frame is **preprocessed** to a fixed template size; **`cv2.matchTemplate` (normalized cross-correlation)** is run **once over the full mosaic**. The **global NCC peak** fixes the template footprint on the map; **no per-tile SIFT** or best-of-tiles vote.
4. **Position output**: **Sub-tile** lat/lon from the peak (template center), plus match score as confidence.

**Key Innovation**: Sandwalk transforms unbounded dead-reckoning drift into a bounded search problem. By periodically re-localizing against satellite imagery (every ~3 seconds), position uncertainty never exceeds the search radius tolerance, enabling sustained GPS-denied navigation over extended missions.

**Operational Security**: Matching runs **onboard** on the drone frame and the mosaic. **Tiles** are pulled as needed for the current search box (e.g. Google Static Maps or an equivalent cached tile set for the area)—no vision needs to leave the vehicle for the correlation step.

Sandwalk replaces fragile dead-reckoning navigation with robust vision-based localization, giving autonomous drones the ability to answer "where am I?" without GPS, without transmitting data, and without requiring pre-mapped 3D environments.