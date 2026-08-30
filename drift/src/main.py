# drift — nadir optical-flow-ish velocity from statistical color features
# scrappy prototype. globals at top, minimal abstraction.

import math
import os

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ===== knobs you touch ========================================================
ALTITUDE_M = 120.0          # constant AGL for whole clip (metres)
FPS = 30.0                  # video frame rate (frames / second)
FOV_DEG = 62.0              # vertical field of view, nadir camera (degrees)

WINDOW = 5                  # consecutive frames in sliding analysis window
N_FEATURES = 24             # notable color features to keep per frame
PATCH_HALF = 8              # half-size of LAB patch for tracking (pixels)
SEARCH_R = 20               # search radius frame-to-frame (pixels)
SEARCH_STEP = 2             # stride in search grid (2 = coarse, refine at end)
STAT_WIN = 15               # odd — local neighbourhood for color stats
MIN_TRACK_FRAMES = 2        # need at least this many sightings in window to vote
MIN_NCC = 0.55              # drop track if patch match falls below this

# path to your footage — drop mp4 here or change this
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "files", "stream.mp4")
OUT_GIF = os.path.join(os.path.dirname(__file__), "files", "output", "drift_playback.gif")


# ---------------------------------------------------------------------------
# statistical color features — why LAB + local z-score, not Harris/SIFT/ORB
# ---------------------------------------------------------------------------
# nadir drone footage: useful motion cues are often *colored* patches on terrain
# (fields, roofs, dirt, water) — not generic grey corners.
# we work in CIELAB: L = lightness, a/b = opponent color axes, roughly perceptual.
# for each pixel, compare its (L,a,b) to the mean/std of a local window → mahalanobis-
# like z-score magnitude. high score = statistically unusual color vs surroundings.
# Harris/Shi-Tomasi = gradient corners on intensity only. SIFT/ORB = hand-wavy descriptors
# hidden inside cv2.goodFeaturesToTrack. we want the color stats explicit in the math.


def bgr_to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def local_mean_std(channel, k):
    # var(x) = E[x^2] - E[x]^2 via box blur — fast sliding-window stats
    mu = cv2.blur(channel, (k, k))
    mu2 = cv2.blur(channel * channel, (k, k))
    var = np.maximum(mu2 - mu * mu, 1e-4)
    return mu, np.sqrt(var)


def color_saliency_map(lab):
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    muL, sL = local_mean_std(L, STAT_WIN)
    mua, sa = local_mean_std(a, STAT_WIN)
    mub, sb = local_mean_std(b, STAT_WIN)
    zL = (L - muL) / sL
    za = (a - mua) / sa
    zb = (b - mub) / sb
    # euclidean z-score across channels = how "color-out-of-place" is this pixel
    return np.sqrt(zL * zL + za * za + zb * zb)


def pick_features(saliency, n, min_dist=18):
    # greedy NMS: take global max, suppress disk, repeat
    h, w = saliency.shape
    work = saliency.copy()
    pts = []
    for _ in range(n):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        if work[y, x] < 0.5:
            break
        pts.append((float(x), float(y)))
        cv2.circle(work, (x, y), min_dist, 0, -1)
    return pts


def extract_patch(lab, x, y, half):
    xi, yi = int(round(x)), int(round(y))
    h, w = lab.shape[:2]
    if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
        return None
    return lab[yi - half : yi + half + 1, xi - half : xi + half + 1].copy()


def patch_ncc(patch_a, patch_b):
    # normalized cross-correlation on flattened LAB patch — explicit, no cv2.matchTemplate
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape:
        return -1.0
    a = patch_a.reshape(-1)
    b = patch_b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / denom)


def track_forward(lab_prev, lab_next, x, y):
    # same feature in frame t vs t+1: can't assume coords match — search locally for
    # best NCC of the LAB patch captured at (x,y) in prev frame
    ref = extract_patch(lab_prev, x, y, PATCH_HALF)
    if ref is None:
        return None, -1.0
    xi, yi = int(round(x)), int(round(y))
    best_s, best_xy = -1.0, (x, y)
    for dy in range(-SEARCH_R, SEARCH_R + 1, SEARCH_STEP):
        for dx in range(-SEARCH_R, SEARCH_R + 1, SEARCH_STEP):
            nx, ny = xi + dx, yi + dy
            cand = extract_patch(lab_next, nx, ny, PATCH_HALF)
            s = patch_ncc(ref, cand)
            if s > best_s:
                best_s, best_xy = s, (float(nx), float(ny))
    # 1px refine around coarse winner
    cx, cy = int(best_xy[0]), int(best_xy[1])
    for dy in range(-SEARCH_STEP, SEARCH_STEP + 1):
        for dx in range(-SEARCH_STEP, SEARCH_STEP + 1):
            nx, ny = cx + dx, cy + dy
            cand = extract_patch(lab_next, nx, ny, PATCH_HALF)
            s = patch_ncc(ref, cand)
            if s > best_s:
                best_s, best_xy = s, (float(nx), float(ny))
    if best_s < MIN_NCC:
        return None, best_s
    return best_xy, best_s


def window_slice(i, n, win=WINDOW):
    # centered sliding window, clipped at stream ends
    half = win // 2
    start = max(0, min(i - half, n - win))
    end = min(n, start + win)
    start = max(0, end - win)
    return list(range(start, end))


def gsd_metres_per_pixel(alt_m, fov_deg, img_w, img_h):
    # nadir pinhole: ground footprint = 2*h*tan(fov/2); divide by pixels for scale
    foot_h = 2.0 * alt_m * math.tan(math.radians(fov_deg / 2.0))
    foot_w = foot_h * (img_w / img_h)
    return foot_w / img_w, foot_h / img_h


def velocity_from_tracks(track_hist, frame_ids, gsd_x, gsd_y, fps):
    """
    track_hist: list of (frame_idx, x_px, y_px) for one feature id inside the window.
    frame_ids: indices in the window (for time base).

    pixel motion over dt → ground velocity:
      ground features appear to slide opposite the drone motion.
      if a patch moves +du pixels/frame in image x, drone moved ~ -du * gsd_x * fps m/s east.
      image x → east, image y → south (standard nadir / top-of-image = forward-ish convention).
      so north component = -(dv) * gsd_y * fps.
    we fit each track with simple endpoint slope (robust enough for v0), then median vote.
    """
    if len(track_hist) < MIN_TRACK_FRAMES:
        return None

    # sort by frame index
    pts = sorted(track_hist, key=lambda p: p[0])
    f0, x0, y0 = pts[0]
    f1, x1, y1 = pts[-1]
    df = f1 - f0
    if df <= 0:
        return None

    dt_total = df / fps
    du = x1 - x0
    dv = y1 - y0

    # --- core monocular nadir kinematics (verbose on purpose) ---
    # du,dv = how far the terrain patch moved in the image over dt_total seconds.
    # patch sliding right (+du) means camera moved left relative to ground → v_east negative of du scale.
    v_east = -(du / dt_total) * gsd_x
    # patch sliding down (+dv) means camera moved up relative to ground → v_north positive of -dv scale
    # (image y grows downward; north is up on ground)
    v_north = -(dv / dt_total) * gsd_y

    speed = math.hypot(v_east, v_north)
    # heading from north, clockwise degrees
    heading = math.degrees(math.atan2(v_east, v_north)) % 360.0
    return v_east, v_north, speed, heading


def cardinal(heading_deg):
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((heading_deg + 22.5) // 45) % 8
    return labels[idx]


def build_tracks_for_window(labs, idxs, saliency_cache=None):
    """
    seed features on first frame of window, chain track_forward across subsequent frames.
    returns dict track_id -> [(frame_idx, x, y), ...]
    """
    tracks = {}
    next_id = 0
    seed_f = idxs[0]
    sal = saliency_cache[seed_f] if saliency_cache is not None else color_saliency_map(labs[seed_f])
    seeds = pick_features(sal, N_FEATURES)

    for x, y in seeds:
        tracks[next_id] = [(seed_f, x, y)]
        next_id += 1

    for fi in range(len(idxs) - 1):
        f_a, f_b = idxs[fi], idxs[fi + 1]
        lab_a, lab_b = labs[f_a], labs[f_b]
        alive = {}
        for tid, hist in tracks.items():
            if hist[-1][0] != f_a:
                continue
            x, y = hist[-1][1], hist[-1][2]
            nxy, score = track_forward(lab_a, lab_b, x, y)
            if nxy is not None:
                alive[tid] = hist + [(f_b, nxy[0], nxy[1])]
        tracks = alive
        if not tracks:
            break

    return tracks


def combine_velocity_votes(tracks, gsd_x, gsd_y, fps):
    votes = []
    for tid, hist in tracks.items():
        v = velocity_from_tracks(hist, None, gsd_x, gsd_y, fps)
        if v is not None:
            votes.append(v)
    if not votes:
        return 0.0, 0.0, 0.0, 0.0
    ve = float(np.median([v[0] for v in votes]))
    vn = float(np.median([v[1] for v in votes]))
    sp = float(np.median([v[2] for v in votes]))
    hd = float(np.median([v[3] for v in votes]))
    return ve, vn, sp, hd


def estimate_frame_velocity(labs, frame_i, n_frames, gsd_x, gsd_y, fps, saliency_cache=None):
    idxs = window_slice(frame_i, n_frames, WINDOW)
    tracks = build_tracks_for_window(labs, idxs, saliency_cache)
    ve, vn, sp, hd = combine_velocity_votes(tracks, gsd_x, gsd_y, fps)
    return ve, vn, sp, hd, tracks, idxs


def make_demo_video(path, n_frames=60, w=640, h=480):
    # synthetic nadir scroll — coloured blobs on noise so color saliency has something to grab
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(7)
    vx, vy = 2.4, 1.1  # px/frame texture shift (camera moves opposite)
    # big canvas, no wrap-around — modulo seams confuse trackers
    bw, bh = w + int(n_frames * vx) + 64, h + int(n_frames * vy) + 64
    base = rng.integers(40, 90, (bh, bw, 3), dtype=np.uint8)
    for _ in range(40):
        cx, cy = rng.integers(0, bw), rng.integers(0, bh)
        color = tuple(int(c) for c in rng.integers(30, 255, 3))
        cv2.circle(base, (cx, cy), int(rng.integers(8, 35)), color, -1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (w, h))
    for t in range(n_frames):
        ox, oy = int(t * vx), int(t * vy)
        crop = base[oy : oy + h, ox : ox + w].copy()
        writer.write(crop)
    writer.release()
    print(f"[DRIFT] wrote synthetic demo → {path}")


def render_gif(frames_bgr, velocities, track_cache, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]

    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    ims = []
    cx, cy = w / 2.0, h / 2.0
    arrow_scale = 12.0

    for i in range(n):
        ax.clear()
        ax.imshow(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

        tr, idxs = track_cache[i]
        # yellow dots + lines for tracks active in this frame's window
        for tid, hist in tr.items():
            pts = [(p[1], p[2]) for p in hist if p[0] <= i]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="yellow", linewidth=1.2, alpha=0.85)
            ax.scatter(xs[-1], ys[-1], s=28, c="yellow", edgecolors="black", linewidths=0.4, zorder=5)

        ve, vn, sp, hd = velocities[i]
        # red velocity arrow from image center — direction = heading, length ~ speed
        if sp > 0.05:
            # heading: 0=north (up on ground = -y in image), 90=east (+x)
            rad = math.radians(hd)
            dx = math.sin(rad) * sp * arrow_scale
            dy = -math.cos(rad) * sp * arrow_scale
            ax.add_patch(
                FancyArrowPatch(
                    (cx, cy),
                    (cx + dx, cy + dy),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=2.5,
                    color="red",
                    zorder=6,
                )
            )

        card = cardinal(hd)
        txt = (
            f"v = {sp:.2f} m/s\n"
            f"heading = {hd:.1f}° ({card})\n"
            f"v_e = {ve:.2f}  v_n = {vn:.2f} m/s\n"
            f"window frames {idxs[0]}–{idxs[-1]} ({len(idxs)})"
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.65, pad=0.4),
            zorder=7,
            family="monospace",
        )

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        ims.append(buf[:, :, :3].copy())

    plt.close(fig)
    imageio.mimsave(out_path, ims, fps=int(FPS), loop=0)
    print(f"[DRIFT] gif → {out_path}")


def main():
    print("\n" + "=" * 60)
    print("DRIFT — nadir feature-track velocity (statistical color)")
    print("=" * 60)
    print(f"  altitude   : {ALTITUDE_M} m")
    print(f"  fps        : {FPS}")
    print(f"  fov        : {FOV_DEG}°")
    print(f"  video      : {VIDEO_PATH}")
    print("=" * 60 + "\n")

    if not os.path.exists(VIDEO_PATH):
        print("[DRIFT] no video found — generating synthetic demo clip")
        make_demo_video(VIDEO_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {VIDEO_PATH}")

    frames_bgr = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    if len(frames_bgr) < 2:
        raise RuntimeError("need at least 2 frames")

    print(f"[DRIFT] loaded {len(frames_bgr)} frames")

    labs = [bgr_to_lab(f) for f in frames_bgr]
    h, w = frames_bgr[0].shape[:2]
    gsd_x, gsd_y = gsd_metres_per_pixel(ALTITUDE_M, FOV_DEG, w, h)
    print(f"[DRIFT] GSD ≈ {gsd_x:.3f} m/px (x), {gsd_y:.3f} m/px (y)")

    saliency_cache = [color_saliency_map(l) for l in labs]

    velocities = []
    track_cache = {}
    n = len(frames_bgr)
    for i in range(n):
        idxs = window_slice(i, n, WINDOW)
        tracks = build_tracks_for_window(labs, idxs, saliency_cache)
        ve, vn, sp, hd = combine_velocity_votes(tracks, gsd_x, gsd_y, FPS)
        velocities.append((ve, vn, sp, hd))
        track_cache[i] = (tracks, idxs)
        if i % 10 == 0 or i == n - 1:
            print(f"  frame {i:3d}  v={sp:6.2f} m/s  hdg={hd:6.1f}° ({cardinal(hd)})  window={idxs}", flush=True)

    render_gif(frames_bgr, velocities, track_cache, OUT_GIF)

    print("\n" + "=" * 60)
    print("done")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
