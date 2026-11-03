"""Case Study 2 experiments. Run: python experiments_cs2.py <data_dir> <model_dir> <out_dir>

No annotated ground truth ships with these videos. The reference is built from
YOLOv8x run over overlapping high-resolution tiles (conf 0.35), merged with NMS,
and checked by eye on a sample of frames. Results measure agreement with that
reference on clean frames and are not a published mAP.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

DATA, MODELS, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
VEH = [1, 2, 3, 5, 7]
NAMES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VIDEOS = {"Highway (sparse)": "vehicles.mp4", "Urban (dense)": "vehicles-2.mp4"}
N_FRAMES = 40


def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for k in np.linspace(0, total - 1, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def predict(model, frame, imgsz=640, conf=0.25, device=DEVICE):
    r = model.predict(frame, imgsz=imgsz, conf=conf, classes=VEH, device=device, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(gt, pred, conf, thr=0.5):
    """Return (matched gt flags, matched pred flags, pairs) in descending confidence order."""
    iou = iou_matrix(gt, pred)
    gt_hit = np.zeros(len(gt), bool); pred_hit = np.zeros(len(pred), bool)
    pairs = []
    for j in np.argsort(-conf) if len(pred) else []:
        if len(gt) == 0:
            break
        ious = iou[:, j].copy(); ious[gt_hit] = 0
        i = int(ious.argmax())
        if ious[i] >= thr:
            gt_hit[i] = True; pred_hit[j] = True; pairs.append((i, j))
    return gt_hit, pred_hit, pairs


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "tp": tp, "fp": fp, "fn": fn}


def reference(model, frame, tile=None, conf=0.35):
    """Tiled YOLOv8x: boxes cut by an interior tile edge are dropped, so the
    overlapping neighbouring tile supplies the complete box."""
    H, W = frame.shape[:2]
    tile = tile or (W // 3)
    step = int(tile * 0.6)
    boxes, confs, classes = [], [], []
    ys = sorted({min(y, max(H - tile, 0)) for y in range(0, max(H - tile, 0) + step, step)})
    xs = sorted({min(x, W - tile) for x in range(0, W - tile + step, step)})
    for y0 in ys:
        for x0 in xs:
            crop = frame[y0:y0 + tile, x0:x0 + tile]
            b, c, k = predict(model, crop, imgsz=960, conf=conf)
            for bb, cc, kk in zip(b, c, k):
                edge = ((bb[0] < 3 and x0 > 0) or (bb[1] < 3 and y0 > 0) or
                        (bb[2] > crop.shape[1] - 3 and x0 + tile < W) or
                        (bb[3] > crop.shape[0] - 3 and y0 + tile < H))
                if not edge:
                    boxes.append(bb + [x0, y0, x0, y0]); confs.append(cc); classes.append(kk)
    b, c, k = predict(model, frame, imgsz=1280, conf=conf)
    boxes += list(b); confs += list(c); classes += list(k)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    boxes, confs, classes = np.array(boxes), np.array(confs), np.array(classes)
    keep = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32),
                               torch.tensor(confs, dtype=torch.float32), 0.5).numpy()
    return boxes[keep], confs[keep], classes[keep]


# Simulated conditions -------------------------------------------------------
def low_light(f, rng):
    x = (f.astype(np.float32) / 255) ** 2.2 * 0.30
    x += rng.normal(0, 0.018, x.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def fog(f, rng):
    h = f.shape[0]
    depth = np.linspace(0.78, 0.35, h, dtype=np.float32)[:, None, None]
    return np.clip(f.astype(np.float32) * (1 - depth) + 215 * depth, 0, 255).astype(np.uint8)


def rain(f, rng):
    h, w = f.shape[:2]
    layer = (rng.random((h, w)) > 0.992).astype(np.float32) * 255
    size = max(15, h // 40)
    k = np.zeros((size, size), np.float32)
    cv2.line(k, (size // 2 + size // 6, 0), (size // 2 - size // 6, size - 1), 1, 1)
    layer = cv2.filter2D(layer, -1, k / k.sum() * 2.5)
    x = f.astype(np.float32) * 0.78 + layer[..., None] * 0.55
    return cv2.GaussianBlur(np.clip(x, 0, 255).astype(np.uint8), (0, 0), max(0.6, h / 1800))


def motion_blur(f, rng):
    size = max(9, f.shape[0] // 60) | 1
    k = np.zeros((size, size), np.float32); k[size // 2, :] = 1 / size
    return cv2.filter2D(f, -1, k)


def low_res(f, rng):
    h, w = f.shape[:2]
    small = cv2.resize(f, (320, int(320 * h / w)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return cv2.resize(cv2.imdecode(enc, 1), (w, h), interpolation=cv2.INTER_LINEAR)


CONDITIONS = {"Clean": lambda f, r: f, "Low light": low_light, "Fog": fog, "Rain": rain,
              "Motion blur": motion_blur, "Low resolution": low_res}

nano, small, xlarge = (YOLO(os.path.join(MODELS, m)) for m in ("yolov8n.pt", "yolov8s.pt", "yolov8x.pt"))
res = {"device": DEVICE, "videos": {}, "n_frames": N_FRAMES}
# On one frame, show how the same model's count changes with how it is run
_f = sample_frames(os.path.join(DATA, "vehicles-2.mp4"), N_FRAMES)[12]
res["density_probe"] = {
    "standard_1280": len(predict(xlarge, _f, imgsz=1280)[0]),
    "tta_1280": int(sum(int(x) in VEH for x in xlarge.predict(_f, imgsz=1280, conf=.25, augment=True, device=DEVICE, verbose=False)[0].boxes.cls)),
    "tiled": len(reference(xlarge, _f)[0]),
}
print(res["density_probe"])

for label, fn in VIDEOS.items():
    frames = sample_frames(os.path.join(DATA, fn), N_FRAMES)
    cap = cv2.VideoCapture(os.path.join(DATA, fn))
    v = {"fps": cap.get(5), "w": int(cap.get(3)), "h": int(cap.get(4)), "frames": int(cap.get(7))}
    cap.release()
    print("==", label, v)
    gts = [reference(xlarge, f) for f in frames]
    json.dump([g[0].tolist() for g in gts], open(os.path.join(OUT, f"refs_{fn}.json"), "w"))
    v["gt_per_frame"] = [len(g[0]) for g in gts]
    v["gt_mean"] = float(np.mean(v["gt_per_frame"]))
    v["gt_classes"] = {NAMES[c]: int(sum((g[2] == c).sum() for g in gts)) for c in VEH}
    scale_640 = 640 / max(v["w"], v["h"])

    # Model and input-size comparison
    v["models"] = []
    for mname, m, sizes in (("YOLOv8n", nano, (320, 640, 1280)), ("YOLOv8s", small, (640, 1280))):
        for sz in sizes:
            tp = fp = fn_ = 0
            ms = []
            confs = []
            for f, g in zip(frames, gts):
                t = time.perf_counter()
                b, c, k = predict(m, f, imgsz=sz)
                ms.append((time.perf_counter() - t) * 1e3)
                gh, ph, _ = greedy_match(g[0], b, c)
                tp += int(gh.sum()); fn_ += int((~gh).sum()); fp += int((~ph).sum())
                confs += c.tolist()
            row = {"model": mname, "imgsz": sz, "ms": float(np.median(ms[1:])),
                   "mean_conf": float(np.mean(confs)) if confs else 0, **prf(tp, fp, fn_)}
            v["models"].append(row)
            print(row)

    # Recall by object size, using the box width in 640-px input space
    bins = [("Small (<32 px)", 0, 32), ("Medium (32-96 px)", 32, 96), ("Large (>96 px)", 96, 1e9)]
    hits = {b[0]: [0, 0] for b in bins}
    conf_by = {b[0]: [] for b in bins}
    for f, g in zip(frames, gts):
        b, c, k = predict(nano, f)
        gh, ph, pairs = greedy_match(g[0], b, c)
        sqrt_area = np.sqrt((g[0][:, 2] - g[0][:, 0]) * (g[0][:, 3] - g[0][:, 1])) * scale_640
        for name, lo, hi in bins:
            m = (sqrt_area >= lo) & (sqrt_area < hi)
            hits[name][0] += int(gh[m].sum()); hits[name][1] += int(m.sum())
        for i, j in pairs:
            for name, lo, hi in bins:
                if lo <= sqrt_area[i] < hi:
                    conf_by[name].append(float(c[j]))
    v["by_size"] = [{"bin": b[0], "hit": hits[b[0]][0], "total": hits[b[0]][1],
                     "recall": hits[b[0]][0] / max(hits[b[0]][1], 1),
                     "mean_conf": float(np.mean(conf_by[b[0]])) if conf_by[b[0]] else None} for b in bins]
    print(v["by_size"])

    # Confidence threshold sweep
    v["conf_sweep"] = []
    for thr in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75):
        tp = fp = fn_ = 0
        for f, g in zip(frames, gts):
            b, c, k = predict(nano, f, conf=thr)
            gh, ph, _ = greedy_match(g[0], b, c)
            tp += int(gh.sum()); fn_ += int((~gh).sum()); fp += int((~ph).sum())
        v["conf_sweep"].append({"conf": thr, **prf(tp, fp, fn_)})
    print(v["conf_sweep"])

    # Conditions
    v["conditions"] = []
    for cname, fx in CONDITIONS.items():
        rng = np.random.default_rng(7)
        tp = fp = fn_ = 0; counts = []; confs = []
        for f, g in zip(frames, gts):
            b, c, k = predict(nano, fx(f, rng))
            gh, ph, _ = greedy_match(g[0], b, c)
            tp += int(gh.sum()); fn_ += int((~gh).sum()); fp += int((~ph).sum())
            counts.append(len(b)); confs += c.tolist()
        row = {"condition": cname, "count_mean": float(np.mean(counts)),
               "mean_conf": float(np.mean(confs)) if confs else 0, **prf(tp, fp, fn_)}
        v["conditions"].append(row); print(row)

    # Per-frame counts: temporal stability of the nano model against the reference
    v["counts_nano"] = [len(predict(nano, f)[0]) for f in frames]
    v["count_mae"] = float(np.mean(np.abs(np.array(v["counts_nano"]) - np.array(v["gt_per_frame"]))))
    v["count_bias"] = float(np.mean(np.array(v["counts_nano"]) - np.array(v["gt_per_frame"])))

    # Class confusion on matched boxes
    conf_mat = {}
    for f, g in zip(frames, gts):
        b, c, k = predict(nano, f)
        _, _, pairs = greedy_match(g[0], b, c)
        for i, j in pairs:
            key = f"{NAMES[int(g[2][i])]}->{NAMES[int(k[j])]}"
            conf_mat[key] = conf_mat.get(key, 0) + 1
    v["class_pairs"] = conf_mat
    print(conf_mat)
    res["videos"][label] = v

# CPU timing on the laptop processor
frame = sample_frames(os.path.join(DATA, "vehicles-2.mp4"), 1)[0]
res["timing"] = []
for mname, m in (("YOLOv8n", nano), ("YOLOv8s", small), ("YOLOv8x", xlarge)):
    for dev in ("cpu", DEVICE):
        ms = []
        for i in range(12):
            t = time.perf_counter()
            predict(m, frame, imgsz=640, device=dev)
            ms.append((time.perf_counter() - t) * 1e3)
        res["timing"].append({"model": mname, "device": dev, "ms": float(np.median(ms[2:])),
                              "fps": 1000 / float(np.median(ms[2:]))})
        print(res["timing"][-1])

json.dump(res, open(os.path.join(OUT, "cs2_results.json"), "w"), indent=1)
print("Saved", os.path.join(OUT, "cs2_results.json"))
