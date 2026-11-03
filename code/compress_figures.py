"""Encode photo figures as JPEG with OpenCV and verify each patch against the PNG. Run before build_report.py."""
import glob, os, sys
import cv2, numpy as np
names = ["cs1_references", "cs1_query_samples", "cs1_keypoints", "cs1_matches_same", "cs1_matches_cases", "cs1_failures", "cs2_highway", "cs2_urban", "cs2_agreement", "cs2_conditions_grid"]
os.makedirs("figures_jpg", exist_ok=True)
ok = True
for n in names:
    png = cv2.imread(f"figures/{n}.png"); out = f"figures_jpg/{n}.jpg"
    cv2.imwrite(out, png, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpg = cv2.imread(out); assert jpg.shape == png.shape
    d = np.abs(jpg.astype(int) - png.astype(int)); h, w = d.shape[:2]
    # per-channel mean error in 16 px patches, including edges and the final corner
    worst = max(d[y:y + 16, x:x + 16].mean() for y in range(0, h, 16) for x in range(0, w, 16))
    corner = d[-16:, -16:].max()
    good = worst < 30 and corner < 60
    ok &= bool(good)
    print(f"{n:24s} worst patch {worst:5.1f}  corner max {corner:3d}  {'ok' if good else 'BAD'}  {os.path.getsize(out) // 1024} KB")
sys.exit(0 if ok else 1)
