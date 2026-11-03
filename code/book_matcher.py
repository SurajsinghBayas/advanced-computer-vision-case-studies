"""
Case Study 1: Image Recognition Using Feature Matching
Identifies a book cover using SIFT features, KNN matching, Lowe's ratio test
and RANSAC geometric verification.

Usage:
    python book_matcher.py reference.jpg query.jpg --out matches.png
"""
import argparse
import time

import cv2
import numpy as np

RATIO = 0.75         # Lowe's ratio test threshold
MIN_INLIERS = 15     # minimum RANSAC inliers to accept "same book"
RANSAC_PX = 5.0      # maximum reprojection error for an inlier, in pixels


def load_gray(path):
    """Read an image and return it in colour and in grayscale."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_features(gray, detector=None):
    """Task 1: detect SIFT keypoints and compute 128-D descriptors."""
    detector = detector or cv2.SIFT_create()
    return detector.detectAndCompute(gray, None)


def knn_match(desc_ref, desc_query, ratio=RATIO):
    """Task 1: 2-nearest-neighbour matching filtered by Lowe's ratio test."""
    if desc_ref is None or desc_query is None or len(desc_query) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(desc_query, desc_ref, k=2)
    return [p[0] for p in pairs
            if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def verify_geometry(kp_ref, kp_query, good):
    """Fit a homography with RANSAC and return (H, inlier mask)."""
    if len(good) < 4:
        return None, np.zeros(len(good), dtype=np.uint8)
    src = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None:
        return None, np.zeros(len(good), dtype=np.uint8)
    return H, mask.ravel().astype(np.uint8)


def match_images(gray_ref, gray_query, ratio=RATIO):
    """Run the full pipeline on two grayscale images and report the result."""
    t0 = time.perf_counter()
    kp_ref, desc_ref = extract_features(gray_ref)
    kp_query, desc_query = extract_features(gray_query)
    t1 = time.perf_counter()
    good = knn_match(desc_ref, desc_query, ratio)
    t2 = time.perf_counter()
    H, mask = verify_geometry(kp_ref, kp_query, good)
    t3 = time.perf_counter()
    inliers = int(mask.sum())
    return {
        "kp_ref": kp_ref, "kp_query": kp_query, "good": good, "H": H,
        "mask": mask, "inliers": inliers,
        "inlier_ratio": inliers / len(good) if good else 0.0,
        "same_book": inliers >= MIN_INLIERS,
        "ms_extract": (t1 - t0) * 1e3, "ms_match": (t2 - t1) * 1e3,
        "ms_ransac": (t3 - t2) * 1e3,
    }


def draw_matches(ref, query, result, inliers_only=True):
    """Task 2: display matched features and outline the located cover."""
    canvas = query.copy()
    if result["H"] is not None and result["same_book"]:
        h, w = ref.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        outline = cv2.perspectiveTransform(corners, result["H"])
        cv2.polylines(canvas, [np.int32(outline)], True, (0, 220, 0), 3)
    # drawMatches expects (img1, kp1, img2, kp2) with queryIdx indexing img1
    flipped = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in result["good"]]
    mask = result["mask"].tolist() if inliers_only else None
    return cv2.drawMatches(
        ref, result["kp_ref"], canvas, result["kp_query"], flipped, None,
        matchColor=(0, 255, 0) if inliers_only else (0, 200, 255),
        singlePointColor=None, matchesMask=mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


def main():
    parser = argparse.ArgumentParser(description="Same-book check using SIFT + KNN")
    parser.add_argument("reference")
    parser.add_argument("query")
    parser.add_argument("--out", default="matches.png")
    args = parser.parse_args()

    ref, gray_ref = load_gray(args.reference)
    query, gray_query = load_gray(args.query)
    r = match_images(gray_ref, gray_query)

    print(f"Reference keypoints : {len(r['kp_ref'])}")
    print(f"Query keypoints     : {len(r['kp_query'])}")
    print(f"Good matches (KNN)  : {len(r['good'])}")
    print(f"RANSAC inliers      : {r['inliers']} ({r['inlier_ratio']:.1%})")
    print(f"Time extract/match/ransac: {r['ms_extract']:.0f} / "
          f"{r['ms_match']:.0f} / {r['ms_ransac']:.1f} ms")
    print("Decision            :", "SAME BOOK" if r["same_book"] else "DIFFERENT BOOK")
    cv2.imwrite(args.out, draw_matches(ref, query, r))
    print(f"Saved visualisation : {args.out}")


if __name__ == "__main__":
    main()
