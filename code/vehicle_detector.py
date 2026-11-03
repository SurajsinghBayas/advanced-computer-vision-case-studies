"""
Case Study 2: Vehicle Detection Using YOLO
Detects vehicles in a traffic video with a COCO-pretrained YOLOv8 model and
writes an annotated video with bounding boxes, class labels and live counts.

Usage:
    python vehicle_detector.py traffic.mp4 --model yolov8n.pt --out output.mp4
"""
import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO

# COCO class ids that count as vehicles
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLOURS = {1: (255, 160, 0), 2: (60, 200, 60), 3: (0, 165, 255),
           5: (200, 60, 200), 7: (40, 40, 230)}


def detect(model, frame, conf=0.25, imgsz=640, device="cpu"):
    """Task 1: return a list of (class_id, confidence, x1, y1, x2, y2)."""
    result = model.predict(frame, conf=conf, imgsz=imgsz, device=device,
                           classes=list(VEHICLE_CLASSES), verbose=False)[0]
    return [(int(b.cls), float(b.conf), *b.xyxy[0].tolist()) for b in result.boxes]


def draw(frame, detections, thickness=2, font_scale=0.6):
    """Task 2: draw bounding boxes, class labels and a live count banner."""
    for cls, conf, x1, y1, x2, y2 in detections:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        colour = COLOURS[cls]
        label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, p1, p2, colour, thickness)
        cv2.rectangle(frame, (p1[0], p1[1] - h - 8), (p1[0] + w + 6, p1[1]), colour, -1)
        cv2.putText(frame, label, (p1[0] + 3, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    counts = Counter(VEHICLE_CLASSES[d[0]] for d in detections)
    banner = f"Vehicles: {len(detections)}  " + "  ".join(
        f"{k}: {v}" for k, v in sorted(counts.items()))
    (w, h), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 2)
    cv2.rectangle(frame, (0, 0), (w + 20, h + 20), (20, 20, 20), -1)
    cv2.putText(frame, banner, (10, h + 10), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def process_video(source, model_path, out_path, conf=0.25, imgsz=640,
                  device="cpu", max_frames=None):
    """Run detection over a video and write the annotated result."""
    model = YOLO(model_path)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise FileNotFoundError(source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(3)), int(capture.get(4)))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frames, infer_time = 0, 0.0
    totals = Counter()
    while True:
        ok, frame = capture.read()
        if not ok or (max_frames and frames >= max_frames):
            break
        start = time.perf_counter()
        detections = detect(model, frame, conf, imgsz, device)
        infer_time += time.perf_counter() - start
        totals.update(VEHICLE_CLASSES[d[0]] for d in detections)
        writer.write(draw(frame, detections))
        frames += 1
    capture.release()
    writer.release()

    print(f"Frames processed    : {frames}")
    print(f"Average inference   : {infer_time / frames * 1e3:.1f} ms/frame "
          f"({frames / infer_time:.1f} FPS on {device})")
    print(f"Detections per class: {dict(totals)}")
    print(f"Mean vehicles/frame : {sum(totals.values()) / frames:.2f}")
    print(f"Saved annotated video: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Vehicle detection with YOLO")
    parser.add_argument("video")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--out", default="output.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    process_video(args.video, args.model, args.out, args.conf, args.imgsz,
                  args.device, args.max_frames)


if __name__ == "__main__":
    main()
