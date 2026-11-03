"""Case Study 1 experiments. Run: python experiments_cs1.py <data_dir> <out_dir>

Query photographs are simulated: each cover is warped with a pinhole-camera
model onto a textured tabletop, with lighting, sensor noise and JPEG compression.
Because the true homography is known, localisation error can be measured exactly.
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import book_matcher as bm

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

TITLES = {
    "9780131873162": "Cryptography & Network Security",
    "9780132350884": "Clean Code",
    "9780134610993": "AI: A Modern Approach",
    "9780135957059": "The Pragmatic Programmer",
    "9780201633610": "Design Patterns",
    "9780262035613": "Deep Learning",
    "9780596517748": "JavaScript: The Good Parts",
}


def load_covers():
    covers = {}
    for path in sorted(glob.glob(os.path.join(DATA, "cover_*.jpg"))):
        isbn = os.path.basename(path)[6:19]
        covers[TITLES[isbn]] = cv2.imread(path)
    return covers


def tabletop(h, w, rng):
    """Procedural wood-like desk surface."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 30) * 90
    grain = 18 * np.sin((x + noise * 2.2) / 7.0) + 9 * np.sin((x + noise) / 2.3)
    fine = cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 1.0)
    base = 118 + grain + fine
    desk = np.dstack([base * 0.55, base * 0.78, base * 1.02])
    return np.clip(desk, 0, 255).astype(np.uint8)


def capture(cover, yaw=0, pitch=0, roll=0, scale=1.0, blur=0.0, gain=1.0,
            noise=2.0, occlusion=0.0, glare=0.0, seed=0, jpeg=90):
    """Simulate a photograph of the cover. Returns (image, true homography)."""
    rng = np.random.default_rng(seed)
    h, w = cover.shape[:2]
    a, b, c = np.radians([pitch, yaw, roll])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    # Plane points centred on the cover, viewed at distance d by a camera of focal length f
    f = 1.4 * max(h, w)
    d = f / scale
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    plane = np.c_[corners - [w / 2, h / 2], np.zeros(4)]
    cam = (R @ plane.T).T + [0, 0, d]
    proj = f * cam[:, :2] / cam[:, 2:3]
    margin = 70
    proj = proj - proj.min(0) + margin
    W, H = np.ceil(proj.max(0) + margin).astype(int)
    M = cv2.getPerspectiveTransform(corners, proj.astype(np.float32))

    # Optional occluder (a second object lying over part of the cover)
    src = cover.copy()
    if occlusion > 0:
        ow = int(w * occlusion)
        src[:, w - ow:] = (55, 60, 70)
    scene = tabletop(H, W, rng)
    warped = cv2.warpPerspective(src, M, (int(W), int(H)), flags=cv2.INTER_AREA
                                 if scale < 1 else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (int(W), int(H)))
    scene[mask > 127] = warped[mask > 127]

    # Soft drop shadow would add little to matching; lighting gradient is applied instead
    x = np.linspace(0, 1, int(W))[None, :, None]
    img = scene.astype(np.float32) * gain * (0.88 + 0.24 * x)
    if glare > 0:
        yy, xx = np.mgrid[0:int(H), 0:int(W)].astype(np.float32)
        cx, cy = proj.mean(0)
        sx, sy = (np.ptp(proj[:, 0]) * 0.32, np.ptp(proj[:, 1]) * 0.22)
        spot = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
        img += 255 * glare * spot[..., None]
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), M


def corner_error(H_est, H_true, shape):
    if H_est is None:
        return None
    h, w = shape[:2]
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, H_est)
    b = cv2.perspectiveTransform(pts, H_true)
    return float(np.linalg.norm(a - b, axis=2).mean())


covers = load_covers()
names = list(covers)
gray = {n: cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for n, c in covers.items()}
sift = cv2.SIFT_create()
ref_feats = {n: sift.detectAndCompute(g, None) for n, g in gray.items()}
res = {"ref_keypoints": {n: len(ref_feats[n][0]) for n in names}}
print("Reference keypoints:", res["ref_keypoints"])


def evaluate_query(name, img, H_true):
    """Match one query against its own reference."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kq, dq = sift.detectAndCompute(g, None)
    kr, dr = ref_feats[name]
    good = bm.knn_match(dr, dq)
    H, mask = bm.verify_geometry(kr, kq, good)
    inl = int(mask.sum())
    err = corner_error(H, H_true, covers[name].shape)
    return {"good": len(good), "inliers": inl, "err": err, "accepted": inl >= bm.MIN_INLIERS,
            "kp": len(kq)}


def sweep(param, values, **fixed):
    rows = []
    for v in values:
        per = [evaluate_query(n, *capture(covers[n], seed=i * 17 + 5, **{param: v}, **fixed))
               for i, n in enumerate(names)]
        errs = [p["err"] for p in per if p["accepted"] and p["err"] is not None]
        rows.append({
            "value": v,
            "good": float(np.mean([p["good"] for p in per])),
            "inliers": float(np.mean([p["inliers"] for p in per])),
            "min_inliers": int(min(p["inliers"] for p in per)),
            "accepted": int(sum(p["accepted"] for p in per)),
            "located": int(sum(p["accepted"] and p["err"] is not None and p["err"] < 10 for p in per)),
            "corner_err": float(np.median(errs)) if errs else None,
            "per_book": {n: p["inliers"] for n, p in zip(names, per)},
        })
        print(f"{param}={v}: inliers={rows[-1]['inliers']:.0f} accepted={rows[-1]['accepted']}/7 "
              f"located={rows[-1]['located']}/7 err={rows[-1]['corner_err']}")
    return rows


res["tilt"] = sweep("yaw", [0, 15, 30, 45, 55, 65, 70, 75, 80], scale=0.9)
res["scale"] = sweep("scale", [0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
res["roll"] = sweep("roll", [0, 30, 60, 90, 120, 150, 180], scale=0.9)
res["blur"] = sweep("blur", [0, 1, 2, 3, 4, 6, 8], scale=0.9)
res["gain"] = sweep("gain", [1.0, 0.6, 0.3, 0.15, 0.08, 0.04], scale=0.9, noise=4.0)
res["occlusion"] = sweep("occlusion", [0, 0.2, 0.4, 0.6, 0.8, 0.9], scale=0.9)
res["glare"] = sweep("glare", [0, 0.3, 0.6, 1.0, 1.5], scale=0.9)

# Mixed, realistic test set: every query against every reference
rng = np.random.default_rng(42)
N_PER_BOOK = 20
queries = []
for name in names:
    for i in range(N_PER_BOOK):
        p = dict(yaw=float(rng.uniform(-50, 50)), pitch=float(rng.uniform(-35, 35)),
                 roll=float(rng.uniform(-180, 180)), scale=float(rng.uniform(0.35, 1.4)),
                 blur=float(rng.uniform(0, 2)), gain=float(rng.uniform(0.6, 1.2)),
                 noise=float(rng.uniform(1, 5)), seed=int(rng.integers(1 << 30)))
        queries.append((name, p))


def mixed(detector_factory, norm, label, thresholds=(4, 6, 8, 10, 12, 15, 20, 25, 30)):
    det = detector_factory()
    refs = {n: det.detectAndCompute(g, None) for n, g in gray.items()}
    matcher = cv2.BFMatcher(norm)
    genuine, impostor, top1, times, errs = [], [], 0, [], []
    per_book = {n: [] for n in names}
    for name, p in queries:
        img, Ht = capture(covers[name], **p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        kq, dq = det.detectAndCompute(g, None)
        t1 = time.perf_counter()
        scores = {}
        H_own = None
        t_m = 0
        for n in names:
            kr, dr = refs[n]
            t2 = time.perf_counter()
            pairs = matcher.knnMatch(dq, dr, k=2) if dq is not None and len(dq) >= 2 else []
            good = [a[0] for a in pairs if len(a) == 2 and a[0].distance < bm.RATIO * a[1].distance]
            H, mask = bm.verify_geometry(kr, kq, good)
            t_m += time.perf_counter() - t2
            scores[n] = int(mask.sum())
            if n == name:
                H_own = H
        genuine.append(scores[name])
        per_book[name].append(scores[name])
        impostor += [scores[n] for n in names if n != name]
        top1 += max(scores, key=scores.get) == name
        times.append(((t1 - t0) * 1e3, t_m * 1e3))
        if scores[name] >= bm.MIN_INLIERS:
            e = corner_error(H_own, Ht, covers[name].shape)
            if e is not None:
                errs.append(e)
    genuine_a, impostor_a = np.array(genuine), np.array(impostor)
    sweep_rows = []
    for th in thresholds:
        tp = int((genuine_a >= th).sum()); fn = len(genuine_a) - tp
        fp = int((impostor_a >= th).sum()); tn = len(impostor_a) - fp
        sweep_rows.append({"th": th, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                           "precision": tp / max(tp + fp, 1), "recall": tp / (tp + fn),
                           "f1": 2 * tp / max(2 * tp + fp + fn, 1)})
    out = {
        "label": label, "n_queries": len(queries), "top1": top1 / len(queries),
        "genuine_median": float(np.median(genuine_a)), "genuine_min": int(genuine_a.min()),
        "genuine_p5": float(np.percentile(genuine_a, 5)),
        "impostor_median": float(np.median(impostor_a)), "impostor_max": int(impostor_a.max()),
        "impostor_p99": float(np.percentile(impostor_a, 99)),
        "ms_extract": float(np.mean([t[0] for t in times])),
        "ms_match_db": float(np.mean([t[1] for t in times])),
        "corner_err_median": float(np.median(errs)) if errs else None,
        "corner_err_p95": float(np.percentile(errs, 95)) if errs else None,
        "thresholds": sweep_rows, "genuine": genuine, "impostor": impostor,
        "per_book_median": {n: float(np.median(v)) for n, v in per_book.items()},
        "per_book_min": {n: int(min(v)) for n, v in per_book.items()},
    }
    print(label, {k: out[k] for k in ("top1", "genuine_median", "genuine_min", "impostor_max",
                                      "ms_extract", "ms_match_db", "corner_err_median")})
    return out


res["mixed_sift"] = mixed(cv2.SIFT_create, cv2.NORM_L2, "SIFT")
res["mixed_orb"] = mixed(lambda: cv2.ORB_create(5000), cv2.NORM_HAMMING, "ORB")

# Ratio-test sweep: "correct" means the match lies within 5 px of the true location
ratio_rows = []
for ratio in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]:
    tot, correct, inl, imp = 0, 0, [], []
    for name, p in queries[::4]:
        img, Ht = capture(covers[name], **p)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kq, dq = sift.detectAndCompute(g, None)
        kr, dr = ref_feats[name]
        good = bm.knn_match(dr, dq, ratio)
        if good:
            src = np.float32([kr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 2)
            d = np.linalg.norm(cv2.perspectiveTransform(src, Ht).reshape(-1, 2) - dst, axis=1)
            tot += len(good); correct += int((d < 5).sum())
        _, mask = bm.verify_geometry(kr, kq, good)
        inl.append(int(mask.sum()))
        other = names[(names.index(name) + 1) % len(names)]
        ko, do = ref_feats[other]
        good2 = bm.knn_match(do, dq, ratio)
        _, m2 = bm.verify_geometry(ko, kq, good2)
        imp.append(int(m2.sum()))
    ratio_rows.append({"ratio": ratio, "matches_per_query": tot / len(inl),
                       "match_precision": correct / max(tot, 1),
                       "genuine_median": float(np.median(inl)),
                       "impostor_max": int(max(imp))})
    print("ratio", ratio_rows[-1])
res["ratio"] = ratio_rows

json.dump(res, open(os.path.join(OUT, "cs1_results.json"), "w"), indent=1)
print("Saved", os.path.join(OUT, "cs1_results.json"))
