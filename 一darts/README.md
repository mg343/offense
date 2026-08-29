A projectile engine for computing when and where a drone can be launched from a moving space platform to hit a fixed target on Earth. Given the platform’s velocity and heading, the drone's initial acceleration, and accounting for Earth’s rotation, the system identifies viable launch zones, timing windows, and drone paramaters (speed, possibly launch angle, etc). Projectiles are modeled as unguided masses with an initial launch acceleration and no onboard propulsion, following a purely ballistic trajectory to the surface.

---

## next steps (brain dump)

main.py is still a hello world — need the actual ballistic core first.

rough order:
1. pick a frame — ECEF or ECEF-ish is fine. rotating earth means coriolis + relative ground track matter; don’t overthink the fancy frame names, just pick one and stick the rotation in.
2. platform state: position (orbit / altitude / latlon), velocity, heading. hardcode a sample LEO-ish pass in main() like sandwalk does with mission params. no config system.
3. drone = unguided mass after one kick. model launch as Δv or short accel pulse, then free fall under gravity (+ earth rotation terms). no thrust after that.
4. integrate traj until surface intersect (simple earth sphere or WGS84 ellipsoid — sphere first, ellipsoid later if miss distances look stupid).
5. invert it: given target lat/lon, sweep launch timing / Δv / maybe elevation angle, find windows that hit within some miss radius. brute force grid is fine for v0.
6. output: viable windows, required params, maybe a crude plot of ground track + impact. print numbers in terminal is enough to start.

don’t build a solver framework. one script, hardcoded scenario, print the hit. tighten physics only when the numbers are obviously wrong.
