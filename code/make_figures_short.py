"""Compact horizontal figures for the 10-page report. Run: python make_figures_short.py <cv_dir>"""
import json, os, sys
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "experiments_cs1.py")).read().split("covers = load_covers()")[0]
sys.argv = ["x", os.path.join(ROOT, "data"), os.path.join(ROOT, "outputs/cs1")]
g = {"__file__": os.path.join(HERE, "experiments_cs1.py"), "__name__": "cs1"}; exec(compile(src, "h", "exec"), g)
capture, covers = g["capture"], g["load_covers"]()
import book_matcher as bm
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white"})
rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
ref = covers["Deep Learning"]; facts = {}
def run(q):
    return bm.match_images(gray(ref), gray(q))
def thin(r, k=80):
    if len(r["good"]) <= k: return r
    idx = np.random.default_rng(1).choice(len(r["good"]), k, replace=False)
    v = dict(r); v["good"] = [r["good"][i] for i in idx]; v["mask"] = r["mask"][idx]; return v

# Task 2 display: before and after RANSAC, side by side
qd = capture(ref, yaw=35, pitch=-15, roll=18, scale=.75, blur=.8, seed=11)[0]; rd = run(qd); vd = thin(rd, 120)
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
axs[0].imshow(rgb(bm.draw_matches(ref, qd, vd, False))); axs[0].set_title(f"(a) After the ratio test: {len(rd['good'])} good matches (120 drawn)", fontsize=11)
axs[1].imshow(rgb(bm.draw_matches(ref, qd, vd, True))); axs[1].set_title(f"(b) After RANSAC: {rd['inliers']} inliers, cover outlined in green", fontsize=11)
for a in axs: a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/cs1_matches_row.png")); plt.close(fig)

# Same-book scale case and different-book case, side by side
q1 = capture(ref, scale=.4, seed=4)[0]; q2 = capture(covers["Clean Code"], yaw=35, pitch=-15, roll=18, scale=.75, blur=.8, seed=12)[0]
r1, r2 = run(q1), run(q2)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.5))
axs[0].imshow(rgb(bm.draw_matches(ref, q1, thin(r1), True))); axs[0].set_title(f"(a) Same book, scale 0.4: {r1['inliers']} inliers, SAME BOOK", fontsize=11)
axs[1].imshow(rgb(bm.draw_matches(ref, q2, r2, False))); axs[1].set_title(f"(b) Different book: {r2['inliers']} inliers, DIFFERENT BOOK", fontsize=11)
for a in axs: a.axis("off")
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/cs1_cases_row.png")); plt.close(fig)

# Failure queries in a row
cases = [("(a) 70° yaw", capture(ref, yaw=70, scale=.9, seed=6)[0]), ("(b) Blur, σ = 8", capture(ref, blur=8, scale=.9, seed=7)[0]), ("(c) Gain 0.08 (very dark)", capture(ref, gain=.08, noise=4, scale=.9, seed=8)[0])]
fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.5))
for a, (t, q) in zip(axs, cases):
    r = run(q); a.imshow(rgb(q)); a.set_title(f"{t}\n{len(r['good'])} good matches, {r['inliers']} inliers", fontsize=11); a.axis("off"); facts[t] = r["inliers"]
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "figures/cs1_failures_row.png")); plt.close(fig)
print(rd["inliers"], len(rd["good"]), r1["inliers"], len(r2["good"]), r2["inliers"], facts)
