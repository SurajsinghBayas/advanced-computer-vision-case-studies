"""Build every report figure. Run: python make_figures.py <cv_dir>  (no extra experiments)."""
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from ultralytics import YOLO

ROOT = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# experiments_cs1 runs on import, so its helpers are copied by exec up to the marker
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_cs1.py")).read()
head = src.split("covers = load_covers()")[0]
sys.argv = ["x", os.path.join(ROOT, "data"), os.path.join(ROOT, "outputs/cs1")]
g = {"__file__": os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_cs1.py"),
     "__name__": "cs1"}
exec(compile(head, "experiments_cs1_head", "exec"), g)
capture, load_covers = g["capture"], g["load_covers"]
import book_matcher as bm
import vehicle_detector as vd

R1 = json.load(open(os.path.join(ROOT, "outputs/cs1/cs1_results.json")))
R2 = json.load(open(os.path.join(ROOT, "outputs/cs2/cs2_results.json")))

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 11, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7,
    "axes.axisbelow": True, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig)


def rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def panel(axs, images, titles):
    for ax, im, t in zip(axs, images, titles):
        ax.imshow(rgb(im)); ax.set_title(t, fontsize=10.5, fontweight="normal"); ax.axis("off"); ax.grid(False)


def flow(name, steps, width=9.2):
    fig, ax = plt.subplots(figsize=(width, 1.75)); ax.axis("off"); ax.grid(False)
    n = len(steps); gap = 1 / n
    for i, (title, sub) in enumerate(steps):
        x = i * gap
        ax.add_patch(FancyBboxPatch((x + .008, .14), gap - .036, .72, boxstyle="round,pad=0.004,rounding_size=.018",
                                    fc="#eef4fc", ec=BLUE, lw=1.2, transform=ax.transAxes))
        ax.text(x + .008 + (gap - .036) / 2, .68, title, ha="center", va="center", fontsize=10.5, fontweight="bold",
                transform=ax.transAxes, color=INK)
        ax.text(x + .008 + (gap - .036) / 2, .37, sub, ha="center", va="center", fontsize=9, transform=ax.transAxes, color=MUTED)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + gap - .026, .5), (x + gap + .007, .5), arrowstyle="-|>", mutation_scale=11,
                                         color=MUTED, lw=1.2, transform=ax.transAxes))
    save(fig, name)


# Pipelines -------------------------------------------------------------------
flow("cs1_pipeline.png", [
    ("Input", "Reference and\nquery images"), ("Grayscale", "Intensity\nconversion"), ("SIFT", "Keypoints,\n128-D vectors"),
    ("KNN (k=2)", "Two nearest\nneighbours"), ("Ratio test", "Keep if\nd1 < 0.75 d2"), ("RANSAC", "Homography,\ninlier count"),
    ("Decision", "Same book if\ninliers ≥ 15")])
flow("cs2_pipeline.png", [
    ("Video", "Read frame\nwith OpenCV"), ("Preprocess", "Letterbox to\n640 × 640"), ("YOLOv8", "Pretrained\nCOCO weights"),
    ("NMS", "Remove\nduplicates"), ("Filter", "Vehicle\nclasses only"), ("Annotate", "Boxes, labels\nand counts"),
    ("Output", "Saved video\nand statistics")])

# Case study 1 images ----------------------------------------------------------
covers = load_covers()
names = list(covers)
fig, axs = plt.subplots(1, 7, figsize=(9.4, 2.4))
for ax, n in zip(axs, names):
    ax.imshow(rgb(covers[n])); ax.set_title(n.replace(" & ", " &\n").replace(": ", ":\n").replace("The Pragmatic ", "The Pragmatic\n") if len(n) > 12 else n + "\n", fontsize=8.8, fontweight="normal"); ax.axis("off"); ax.grid(False)
save(fig, "cs1_references.png")

book = "Deep Learning"
cases = [("Frontal (0°)", dict(scale=.9)), ("30° yaw", dict(yaw=30, scale=.9)), ("Pitch and roll", dict(pitch=-30, roll=25, yaw=15, scale=.8)),
         ("Far (scale 0.4)", dict(scale=.4)), ("Blur (σ = 3)", dict(blur=3, scale=.9)), ("Low light (gain 0.3)", dict(gain=.3, noise=4, scale=.9)),
         ("60% occluded", dict(occlusion=.6, scale=.9)), ("Glare", dict(glare=1.0, scale=.9))]
fig, axs = plt.subplots(2, 4, figsize=(9.0, 5.4))
for ax, (t, p) in zip(axs.ravel(), cases):
    ax.imshow(rgb(capture(covers[book], seed=3, **p)[0])); ax.set_title(t, fontsize=10.5, fontweight="normal"); ax.axis("off"); ax.grid(False)
save(fig, "cs1_query_samples.png")

ref = covers[book]; gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(); kps = sift.detect(gray_ref)
strong = sorted(kps, key=lambda k: -k.response)[:220]
rich = cv2.drawKeypoints(ref, strong, None, color=(0, 240, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
all_pts = cv2.drawKeypoints(ref, kps, None, color=(0, 240, 0))
fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.9))
panel(axs, [ref, all_pts, rich], ["Reference cover", f"All {len(kps)} SIFT keypoints", "Strongest 220, with scale and orientation"])
save(fig, "cs1_keypoints.png")

# Save the exact query used in the terminal demo
demo_query, _ = capture(covers[book], yaw=35, pitch=-15, roll=18, scale=.75, blur=.8, seed=11)
os.makedirs(os.path.join(ROOT, "outputs/cs1/demo"), exist_ok=True)
cv2.imwrite(os.path.join(ROOT, "outputs/cs1/demo/reference.png"), ref)
cv2.imwrite(os.path.join(ROOT, "outputs/cs1/demo/query_same.png"), demo_query)
other_q, _ = capture(covers["Clean Code"], yaw=35, pitch=-15, roll=18, scale=.75, blur=.8, seed=12)
cv2.imwrite(os.path.join(ROOT, "outputs/cs1/demo/query_different.png"), other_q)


def matched(refimg, q, inliers_only=True, limit=None):
    r = bm.match_images(cv2.cvtColor(refimg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(q, cv2.COLOR_BGR2GRAY))
    if limit and len(r["good"]) > limit:
        idx = np.random.default_rng(1).choice(len(r["good"]), limit, replace=False)
        vis = dict(r); vis["good"] = [r["good"][i] for i in idx]; vis["mask"] = r["mask"][idx]
    else:
        vis = r
    return bm.draw_matches(refimg, q, vis, inliers_only), r


# Task 2 display: all good matches, then verified inliers
a, r = matched(ref, demo_query, False, 120)
b, _ = matched(ref, demo_query, True, 120)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 8.2))
panel(axs, [a, b], [f"(a) After the ratio test: {len(r['good'])} good matches (120 drawn for clarity)",
                    f"(b) After RANSAC: {r['inliers']} inliers, with the located cover outlined in green"])
fig.tight_layout(); save(fig, "cs1_matches_same.png")
CS1_DEMO = {"good": len(r["good"]), "inliers": r["inliers"], "kp_ref": len(r["kp_ref"]), "kp_query": len(r["kp_query"])}

pairs = [("(a) Same book, scale 0.4", ref, capture(ref, scale=.4, seed=4)[0]),
         ("(b) Same book, rotated 120°", ref, capture(ref, roll=120, scale=.8, seed=5)[0]),
         ("(c) Different book: Clean Code as query", ref, other_q)]
fig, axs = plt.subplots(3, 1, figsize=(6.6, 9.0))
CS1_PAIRS = []
for ax, (t, rr, q) in zip(axs, pairs):
    im, x = matched(rr, q, True, 100)
    if not x["same_book"]:
        rv, xx = matched(rr, q, False, None); im = rv
    panel([ax], [im], [f"{t}: {len(x['good'])} good, {x['inliers']} inliers → " + ("SAME" if x["same_book"] else "DIFFERENT")])
    CS1_PAIRS.append({"t": t, "good": len(x["good"]), "inliers": x["inliers"], "same": bool(x["same_book"])})
fig.tight_layout(); save(fig, "cs1_matches_cases.png")

fails = [("(a) 70° yaw", capture(ref, yaw=70, scale=.9, seed=6)[0]),
         ("(b) Blur, σ = 8", capture(ref, blur=8, scale=.9, seed=7)[0]),
         ("(c) Gain 0.08 (very dark)", capture(ref, gain=.08, noise=4, scale=.9, seed=8)[0])]
fig, axs = plt.subplots(3, 1, figsize=(6.2, 9.0))
CS1_FAILS = []
for ax, (t, q) in zip(axs, fails):
    im, x = matched(ref, q, False)
    panel([ax], [im], [f"{t}: {len(x['good'])} good, {x['inliers']} inliers → " + ("SAME" if x["same_book"] else "DIFFERENT")])
    CS1_FAILS.append({"t": t, "good": len(x["good"]), "inliers": x["inliers"], "same": bool(x["same_book"])})
fig.tight_layout(); save(fig, "cs1_failures.png")
json.dump({"demo": CS1_DEMO, "pairs": CS1_PAIRS, "fails": CS1_FAILS}, open(os.path.join(ROOT, "outputs/cs1/figure_facts.json"), "w"), indent=1)
print(CS1_DEMO, CS1_PAIRS, CS1_FAILS)


def inlier_line(ax, rows, xlabel, title, xs=None, ticks=None):
    x = xs if xs is not None else [r["value"] for r in rows]
    y = [max(r["inliers"], .5) for r in rows]
    ax.plot(x, y, color=BLUE, lw=2, marker="o", ms=5.5, mec="white", mew=1)
    ax.axhline(15, color=MUTED, lw=1.1, ls=(0, (4, 3)))
    ax.set_yscale("log"); ax.set_ylim(.5, 2200)
    ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    if ticks is not None:
        ax.set_xticks(ticks)


fig, axs = plt.subplots(2, 3, figsize=(9.2, 5.5), sharey=True)
inlier_line(axs[0, 0], R1["tilt"], "Yaw angle (degrees)", "Viewing angle", ticks=[0, 15, 30, 45, 60, 75])
axs[0, 0].text(1, 17.5, "Accept threshold = 15", color=MUTED, fontsize=9.5)
s = R1["scale"]
inlier_line(axs[0, 1], s, "Scale relative to reference (log)", "Scale")
axs[0, 1].set_xscale("log"); axs[0, 1].set_xticks([.2, .5, 1, 2, 3]); axs[0, 1].set_xticklabels([".2", ".5", "1", "2", "3"]); axs[0, 1].minorticks_off(); axs[0, 1].set_yticks([1, 10, 100, 1000])
inlier_line(axs[0, 2], R1["blur"], "Gaussian blur σ (pixels)", "Blur")
inlier_line(axs[1, 0], R1["occlusion"], "Cover hidden (%)", "Occlusion", xs=[r["value"] * 100 for r in R1["occlusion"]])
g_rows = R1["gain"]
inlier_line(axs[1, 1], g_rows, "Brightness gain (log)", "Illumination")
axs[1, 1].set_xscale("log"); axs[1, 1].set_xticks([.05, .1, .3, 1]); axs[1, 1].set_xticklabels([".05", ".1", ".3", "1"]); axs[1, 1].minorticks_off(); axs[1, 1].invert_xaxis(); axs[1, 1].set_yticks([1, 10, 100, 1000])
inlier_line(axs[1, 2], R1["glare"], "Glare peak (× full white)", "Specular glare")
for ax in axs[:, 0]:
    ax.set_ylabel("Mean RANSAC inliers (log)")
fig.tight_layout(); save(fig, "cs1_sweeps.png")

# Genuine and impostor distributions
ms = R1["mixed_sift"]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
bins = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 800, 1500, 3000]
for vals, col, lab in ((ms["impostor"], ORANGE, "Different book (impostor pairs)"), (ms["genuine"], BLUE, "Same book (genuine pairs)")):
    h, e = np.histogram(np.maximum(vals, .9), bins=[.5] + bins[1:])
    ax.stairs(100 * h / len(vals), e, color=col, fill=True, alpha=.72, label=lab, ec="white", lw=.8)
ax.axvline(15, color=MUTED, lw=1.1, ls=(0, (4, 3)))
ax.text(16.5, 27, "Threshold = 15", color=MUTED, fontsize=9.5, va="top")
ax.set_xscale("log"); ax.set_xlim(.5, 2200); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"]); ax.minorticks_off()
ax.set_xlabel("RANSAC inliers (log)"); ax.set_ylabel("Share of pairs (%)"); ax.legend(loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "cs1_distribution.png")

fig, axs = plt.subplots(1, 2, figsize=(8.6, 2.8))
rt = R1["ratio"]; xs = [r["ratio"] for r in rt]
axs[0].plot(xs, [100 * r["match_precision"] for r in rt], color=BLUE, lw=2, marker="o", ms=5.5, mec="white"); axs[0].set_ylim(0, 105)
axs[0].set_title("Correct matches (within 5 px)", loc="left"); axs[0].set_ylabel("Share of kept matches (%)"); axs[0].set_xlabel("Ratio threshold")
axs[1].plot(xs, [r["matches_per_query"] for r in rt], color=BLUE, lw=2, marker="o", ms=5.5, mec="white"); axs[1].set_ylim(0)
axs[1].set_title("Matches kept per query", loc="left"); axs[1].set_ylabel("Mean matches"); axs[1].set_xlabel("Ratio threshold")
for ax in axs:
    ax.axvline(.75, color=MUTED, lw=1, ls=(0, (4, 3)))
axs[0].text(.74, 8, "Chosen: 0.75", ha="right", color=MUTED, fontsize=9.5)
fig.tight_layout(); save(fig, "cs1_ratio.png")

# Case study 2 images ----------------------------------------------------------
sys.argv = ["x"]
nano = YOLO(os.path.join(ROOT, "models/yolov8n.pt")); xl = YOLO(os.path.join(ROOT, "models/yolov8x.pt"))
DEV = R2["device"]


def frame_at(path, idx):
    c = cv2.VideoCapture(os.path.join(ROOT, "data", path)); c.set(1, idx); ok, f = c.read(); c.release(); return f


def annotated(f, model=nano, **kw):
    scale = f.shape[1] / 1920
    d = vd.detect(model, f, device=DEV, **kw)
    return vd.draw(f.copy(), d, thickness=max(2, int(3 * scale)), font_scale=.62 * scale), d


hw = frame_at("vehicles.mp4", 80); hw2 = frame_at("vehicles.mp4", 400)
ur = frame_at("vehicles-2.mp4", 150); ur2 = frame_at("vehicles-2.mp4", 800)
cv2.imwrite(os.path.join(ROOT, "outputs/cs2/frame_urban.jpg"), ur)
ims = [annotated(x)[0] for x in (hw, hw2)]
fig, axs = plt.subplots(2, 1, figsize=(7.0, 8.0)); panel(axs, ims, ["(a) Frame 80", "(b) Frame 400"]); fig.tight_layout(); save(fig, "cs2_highway.png")
ims = [annotated(x)[0] for x in (ur, ur2)]
fig, axs = plt.subplots(2, 1, figsize=(7.0, 8.0)); panel(axs, ims, ["(a) Frame 150", "(b) Frame 800"]); fig.tight_layout(); save(fig, "cs2_urban.png")

# Agreement view on the dense frame
exp_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_cs2.py")).read()
import torch, torchvision
renv = {"np": np, "cv2": cv2, "torch": torch, "torchvision": torchvision, "VEH": list(vd.VEHICLE_CLASSES), "DEVICE": DEV}
exec("def predict" + exp_src.split("def predict")[1].split("def iou_matrix")[0], renv)
exec("def reference" + exp_src.split("def reference")[1].split("# Simulated conditions")[0], renv)
b_ref = renv["reference"](xl, ur)[0]
pred = nano.predict(ur, imgsz=640, conf=.25, classes=list(vd.VEHICLE_CLASSES), device=DEV, verbose=False)[0].boxes
b_n, c_n = pred.xyxy.cpu().numpy(), pred.conf.cpu().numpy()
ex = {"np": np}
exec(compile(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_cs2.py")).read().split("def prf")[0].split("def iou_matrix")[1].join(["def iou_matrix", ""]), "m", "exec"), ex)
gh, ph, _ = ex["greedy_match"](b_ref, b_n, c_n) if "greedy_match" in ex else (None, None, None)
vis = ur.copy()
for i, x in enumerate(b_ref):
    cv2.rectangle(vis, tuple(map(int, x[:2])), tuple(map(int, x[2:])), (60, 190, 60) if gh[i] else (40, 40, 235), 2 if gh[i] else 3)
for j, x in enumerate(b_n):
    if not ph[j]:
        cv2.rectangle(vis, tuple(map(int, x[:2])), tuple(map(int, x[2:])), (235, 130, 40), 3)
crop = vis[300:1080, 200:1700]
fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.imshow(rgb(crop)); ax.axis("off"); ax.grid(False); save(fig, "cs2_agreement.png")
json.dump({"ref": len(b_ref), "matched": int(gh.sum()), "missed": int((~gh).sum()), "extra": int((~ph).sum())}, open(os.path.join(ROOT, "outputs/cs2/agreement_frame.json"), "w"))
print("agreement", len(b_ref), int(gh.sum()), int((~gh).sum()), int((~ph).sum()))

# Conditions grid
head2 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_cs2.py")).read()
fx_src = head2.split("# Simulated conditions")[1].split("nano, small, xlarge")[0]
env = {"np": np, "cv2": cv2}
exec("# " + fx_src, env)
rng = np.random.default_rng(7)
titles = list(env["CONDITIONS"])
fig, axs = plt.subplots(3, 2, figsize=(8.0, 7.2))
for ax, t in zip(axs.ravel(), titles):
    im, d = annotated(env["CONDITIONS"][t](ur, rng))
    panel([ax], [im[:, :]], [f"{t}: {len(d)} vehicles detected"])
fig.tight_layout(); save(fig, "cs2_conditions_grid.png")

vids = R2["videos"]; labels = list(vids)
cond = [r["condition"] for r in vids[labels[0]]["conditions"]]


def grouped(ax, cats, key, rows_a, rows_b, title, ylabel="Agreement recall (%)"):
    x = np.arange(len(cats)); w = .38
    ax.bar(x - w / 2 - .01, [100 * r[key] for r in rows_a], w, color=BLUE, label=labels[0])
    ax.bar(x + w / 2 + .01, [100 * r[key] for r in rows_b], w, color=ORANGE, label=labels[1])
    for xx, r in zip(x - w / 2 - .01, rows_a):
        ax.text(xx, 100 * r[key] + 1.5, f"{100 * r[key]:.0f}", ha="center", fontsize=8.8, color=INK)
    for xx, r in zip(x + w / 2 + .01, rows_b):
        ax.text(xx, 100 * r[key] + 1.5, f"{100 * r[key]:.0f}", ha="center", fontsize=8.8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", length=0)


fig, ax = plt.subplots(figsize=(8.0, 3.3))
grouped(ax, cond, "recall", vids[labels[0]]["conditions"], vids[labels[1]]["conditions"], "")
ax.legend(loc="upper right", ncol=2); save(fig, "cs2_conditions_chart.png")

fig, ax = plt.subplots(figsize=(6.4, 3.0))
bins_ = [r["bin"].replace(" (", "\n(") for r in vids[labels[0]]["by_size"]]
grouped(ax, bins_, "recall", vids[labels[0]]["by_size"], vids[labels[1]]["by_size"], "")
ax.legend(loc="upper left"); save(fig, "cs2_size.png")

fig, axs = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=False)
for ax, l in zip(axs, labels):
    v = vids[l]; t = np.arange(len(v["counts_nano"]))
    ax.plot(t, v["gt_per_frame"], color=ORANGE, lw=2, label="Reference (tiled YOLOv8x)")
    ax.plot(t, v["counts_nano"], color=BLUE, lw=2, label="YOLOv8n, 640 px")
    ax.set_title(l, loc="left"); ax.set_ylabel("Vehicles"); ax.set_ylim(0)
axs[1].set_xlabel("Sampled frame index"); axs[0].legend(loc="upper center", ncol=2, bbox_to_anchor=(.62, 1.22))
fig.tight_layout(); save(fig, "cs2_counts.png")

fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True)
for ax, key, t in ((axs[0], "precision", "Precision"), (axs[1], "recall", "Recall")):
    for l, c in zip(labels, (BLUE, ORANGE)):
        rows = vids[l]["conf_sweep"]
        ax.plot([r["conf"] for r in rows], [100 * r[key] for r in rows], color=c, lw=2, marker="o", ms=5.5, mec="white", label=l)
    ax.set_title(t, loc="left"); ax.set_xlabel("Confidence threshold"); ax.set_ylim(0, 105)
    ax.axvline(.25, color=MUTED, lw=1, ls=(0, (4, 3)))
axs[0].set_ylabel("Agreement with reference (%)"); axs[1].legend(loc="lower left")
fig.tight_layout(); save(fig, "cs2_conf.png")
print("figures done", sorted(os.listdir(FIG)))
