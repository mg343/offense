# bloop brain dump.
#
# framing: altimeter drop-off backup.
# we have a barometric / radio altimeter. it works, then it doesn't — jam, icing,
# hardware fault, contested RF, whatever. GPS may also be contested. horizontal
# position we treat as known-enough (Sandwalk, last fix, dead reckoning). what we
# need is AGL altitude from the one sensor that still works: the nadir camera,
# correlated against an onboard local terrain elevation map (DEM).
#
# inputs always:
# - lat/lon (assumed good enough — not searching horizontally)
# - onboard DEM covering the local area (lidar / 3DEP / cached mission DEM)
# - last altimeter reading before dropout + time since dropout (soft prior, NOT a hard gate)
# - camera intrinsics / FOV (so altitude ↔ ground footprint scale is deterministic)
# - live nadir frame
#
# core physics:
# at altitude h (AGL), a camera with vertical FOV α sees a ground square of side
#     L(h) = 2 * h * tan(α / 2)
# so the DEM patch that should match the camera frame *changes scale with h*.
# wrong altitude → wrong scale → terrain frequencies don't line up → correlation dies.
# right altitude → rendered DEM appearance lines up with the camera → correlation peaks.
#
# quintessential FFT move: phase correlation.
# take two same-size images a, b (camera vs DEM-rendered appearance at hypothesis h).
#   Fa = FFT(a), Fb = FFT(b)
#   R  = Fa * conj(Fb) / |Fa * conj(Fb)|     # normalized cross-power spectrum
#   r  = IFFT(R)
# peak of |r| is the translation that best aligns them; peak *height* is the match
# quality. we don't actually care about translation much (lat/lon known → expect
# near-zero shift). we care about response vs h. OpenCV's cv2.phaseCorrelate is
# exactly this with a Hanning window — use it, but keep the FFT story visible.
#
# appearance model (DEM → "what the camera should look like"):
# we only have elevation, not an orthophoto. so render a nadir *shaded relief*
# (hillshade) + optional slope magnitude. that shares ridge/valley frequency
# structure with a real nadir optical frame over the same terrain. not perfect —
# optical texture ≠ hillshade — but enough for a scale-sensitive match, and the
# right abstraction when the onboard prior is a DEM not a mosaic of satellite RGB.
# (later: swap in orthophoto tiles if available; same phase-corr loop.)
#
# soft prior / filtering — NO hard altitude tolerance band:
# last altimeter A0 is a *belief*, not a clip. as time-since-dropout grows, belief
# widens. weight each hypothesis:
#     w(h) = exp( -0.5 * ((h - A0) / σ)^2 )
#     σ    = σ0 + σ_dot * t_since_dropout
# combined score:
#     S(h) = phase_response(h) * w(h)
# this is a Bayesian-ish filter: likelihood from FFT × prior from altimeter memory.
# also sanity checks (not hard rejects — down-weight / flag):
#   1. translation peak should be near image center (lat/lon assumption holding)
#   2. peak sharpness — S(h*) vs neighbors; flat ridge → low confidence
#   3. secondary local maxima — if two altitudes nearly tie, flag ambiguity
#
# search:
# coarse sweep over a wide AGL range (physics-limited by DEM extent / FOV), score
# with S(h), then refine around the argmax. never discard candidates outside ±X%
# of A0 — the prior only gently pulls. if the altimeter was already wrong when it
# died, vision can still pull us back.
#
# output every cycle:
# - estimated AGL (m)
# - confidence (peak S, sharpness, shift residual)
# - score curve vs altitude (for viz / QA)
# - delta from last altimeter reading
#
# validation path:
# 1. synthetic: render hillshade at known TRUE_AGL from the DEM, recover it.
# 2. real: drop a real nadir frame over the same lat/lon DEM, compare to truth if known.
#
# data:
# start with USGS 3DEP crop via ImageServer (fetch_dem.py) — good enough to prove
# the loop. for ops over WA, pull 1m bare-earth from WA DNR lidar portal / PSLC
# (see README). put GeoTIFF or dem.npy under files/.
#
# relationship to Sandwalk:
# Sandwalk answers "where am I?" in lat/lon from satellite RGB under a dead-reckoning
# ring. bloop answers "how high am I?" from DEM scale under an altimeter dropout.
# complementary. lat/lon in → AGL out.

"""
Early sketch — the polished single-cycle lives in main.py.
This file keeps the ideation + a thin class outline if we later want a video loop.
"""


class Bloop:
    """Altimeter-dropout backup: nadir camera × onboard DEM → AGL via phase correlation."""

    def __init__(self, dem, meta, fov_deg: float = 60.0, cam_px: int = 256):
        self.dem = dem
        self.meta = meta
        self.fov_deg = fov_deg
        self.cam_px = cam_px

    def estimate_agl(
        self,
        frame_bgr,
        lat: float,
        lon: float,
        last_altimeter_m: float,
        t_since_dropout_s: float,
    ):
        raise NotImplementedError("use main.py for the single-cycle implementation")


if __name__ == "__main__":
    print("bloop ideation stub — run main.py for the single-cycle test")
