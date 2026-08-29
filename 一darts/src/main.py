# darts v0 — one ballistic shot, sphere earth, print the hit. no rotation yet.

import math

GM = 3.986e14          # m^3/s^2
R_EARTH = 6_371_000.0  # m
DT = 1.0               # s, crude euler is fine

def latlon_alt_to_ecef(lat_deg, lon_deg, alt_m):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    r = R_EARTH + alt_m
    return [r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat)]

def ecef_to_latlon(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    lat = math.degrees(math.asin(z / r))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon, r - R_EARTH

def integrate(pos, vel):
    # free-fall under GM/r^2 until we hit the sphere (or give up)
    t = 0.0
    while t < 10_000:
        r = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        if r <= R_EARTH:
            return t, pos
        a_scale = -GM / (r**3)
        for i in range(3):
            vel[i] += a_scale * pos[i] * DT
            pos[i] += vel[i] * DT
        t += DT
    return t, pos

# ===== hardcoded sample LEO-ish pass ========================================
PLATFORM_LAT, PLATFORM_LON, PLATFORM_ALT = 0.0, 0.0, 400_000.0   # equator, 400km
PLATFORM_VEL = [0.0, 7660.0, 0.0]   # ~LEO eastward, m/s
DV_KICK = [0.0, 0.0, -200.0]        # shove "down" / southish, m/s — stand-in for launch accel

pos = latlon_alt_to_ecef(PLATFORM_LAT, PLATFORM_LON, PLATFORM_ALT)
vel = [PLATFORM_VEL[i] + DV_KICK[i] for i in range(3)]

t_hit, hit_pos = integrate(pos, vel)
lat, lon, alt = ecef_to_latlon(*hit_pos)
print(f"impact t={t_hit:.0f}s  lat={lat:.3f}  lon={lon:.3f}  alt={alt:.0f}m")
