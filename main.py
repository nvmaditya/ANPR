import argparse
import os

from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANPR YOLOv8 demo")
    parser.add_argument("--video", default="./sample.mp4", help="Path to input video")
    parser.add_argument("--vehicle-model", default="yolov8n.pt", help="Vehicle detector model")
    parser.add_argument(
        "--plate-model",
        default="./models/license_plate_detector.pt",
        help="License plate detector model",
    )
    parser.add_argument(
        "--fallback-plate-model",
        default="yolov8n.pt",
        help="Fallback plate model if --plate-model file is missing",
    )
    parser.add_argument("--output", default="./test.csv", help="Path to output CSV")
    parser.add_argument(
        "--output-video",
        default="",
        help="Optional path to save an annotated output video (e.g. out.mp4)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an annotated preview window while processing (press 'q' or ESC to quit)",
    )
    parser.add_argument(
        "--show-width",
        type=int,
        default=1280,
        help="Preview window width in pixels (0 to disable resizing)",
    )
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process (-1 for all)")
    parser.add_argument(
        "--max-plate-detections-per-frame",
        type=int,
        default=6,
        help="Cap number of plate detections processed per frame",
    )
    return parser.parse_args()


args = parse_args()

# This repository uses an older ultralytics stack; this keeps it compatible with newer torch defaults.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO(args.vehicle_model)

plate_model_to_use = args.plate_model
if not os.path.exists(args.plate_model):
    print(
        f"[WARN] Plate model not found at {args.plate_model}. "
        f"Falling back to {args.fallback_plate_model}."
    )
    plate_model_to_use = args.fallback_plate_model

license_plate_detector = YOLO(plate_model_to_use)

# load video
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"Failed to open video: {args.video}")

if args.show:
    cv2.namedWindow("ANPR", cv2.WINDOW_NORMAL)
    if args.show_width and args.show_width > 0:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if src_w > 0 and src_h > 0:
            target_w = min(args.show_width, src_w)
            target_h = max(1, int(src_h * (target_w / src_w)))
            cv2.resizeWindow("ANPR", target_w, target_h)

out = None
if args.output_video:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_video, fourcc, float(fps), (width, height))
    if not out.isOpened():
        raise SystemExit(f"Failed to open video writer: {args.output_video}")

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    if args.max_frames > 0 and frame_nmr >= args.max_frames:
        break

    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        frame_plate_vis = []  # list of dicts: {bbox, score, text?, car_id?}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        plate_boxes = sorted(license_plates.boxes.data.tolist(), key=lambda item: item[4], reverse=True)
        for license_plate in plate_boxes[: args.max_plate_detections_per_frame]:
            x1, y1, x2, y2, score, class_id = license_plate

            plate_vis = {"bbox": [x1, y1, x2, y2], "score": float(score)}

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                plate_vis["car_id"] = int(car_id)

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                if license_plate_crop.size != 0:
                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(
                        license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                    )

                    # read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        plate_vis["text"] = license_plate_text
                        plate_vis["text_score"] = float(license_plate_text_score or 0.0)
                        results[frame_nmr][car_id] = {
                            "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                            "license_plate": {
                                "bbox": [x1, y1, x2, y2],
                                "text": license_plate_text,
                                "bbox_score": score,
                                "text_score": license_plate_text_score,
                            },
                        }

            frame_plate_vis.append(plate_vis)

        if args.show or out is not None:
            vis = frame.copy()

            # draw tracked vehicles
            for xcar1, ycar1, xcar2, ycar2, car_id in track_ids.tolist():
                cv2.rectangle(vis, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"ID {int(car_id)}",
                    (int(xcar1), max(0, int(ycar1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # draw plate detections for this frame (even if OCR fails)
            for plate in frame_plate_vis:
                px1, py1, px2, py2 = plate["bbox"]
                cv2.rectangle(vis, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                if plate.get("text"):
                    label = str(plate["text"])
                else:
                    label = f"plate {plate['score']:.2f}"
                cv2.putText(
                    vis,
                    label,
                    (int(px1), max(0, int(py1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if out is not None:
                out.write(vis)

            if args.show:
                cv2.imshow("ANPR", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

# write results
write_csv(results, args.output)
print(f"Saved detections to: {args.output}")

cap.release()
if out is not None:
    out.release()
if args.show:
    cv2.destroyAllWindows()
if args.output_video:
    print(f"Saved annotated video to: {args.output_video}")