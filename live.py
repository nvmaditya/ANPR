import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Optional, TextIO, Union

import cv2
import numpy as np
from ultralytics import YOLO

from sort.sort import Sort
from util import get_car, read_license_plate


def _open_video_capture(source: str) -> cv2.VideoCapture:
    # OpenCV treats numeric strings as filenames, so we special-case digits for webcam indices.
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


@dataclass
class CsvSink:
    file_path: str
    _fh: Optional[TextIO] = None
    _writer: Optional[csv.writer] = None

    def open(self) -> None:
        file_exists = os.path.exists(self.file_path)
        self._fh = open(self.file_path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if not file_exists:
            self._writer.writerow(
                [
                    "ts_unix",
                    "frame_nmr",
                    "car_id",
                    "car_bbox",
                    "license_plate_bbox",
                    "license_plate_bbox_score",
                    "license_number",
                    "license_number_score",
                ]
            )
            self._fh.flush()

    def write_row(
        self,
        *,
        ts_unix: float,
        frame_nmr: int,
        car_id: Union[int, float],
        car_bbox: list[float],
        plate_bbox: list[float],
        plate_score: float,
        plate_text: str,
        plate_text_score: float,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("CsvSink not opened")

        self._writer.writerow(
            [
                f"{ts_unix:.3f}",
                frame_nmr,
                int(car_id),
                f"[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}]",
                f"[{plate_bbox[0]} {plate_bbox[1]} {plate_bbox[2]} {plate_bbox[3]}]",
                plate_score,
                plate_text,
                plate_text_score,
            ]
        )
        # Keep it reasonably real-time without forcing a flush on every row.
        if frame_nmr % 15 == 0:
            assert self._fh is not None
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        self._fh = None
        self._writer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANPR YOLOv8 live (webcam/RTSP/video)")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0), file path, or URL (rtsp/http)",
    )
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
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live window (press 'q' or ESC to quit)",
    )
    parser.add_argument(
        "--show-width",
        type=int,
        default=1280,
        help="Preview window width in pixels (0 to disable resizing)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV output path (leave empty to disable)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum frames to process (-1 for all/until stream ends)",
    )
    parser.add_argument(
        "--max-plate-detections-per-frame",
        type=int,
        default=6,
        help="Cap number of plate detections processed per frame",
    )
    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=0.25,
        help="Minimum confidence for vehicle detections",
    )
    parser.add_argument(
        "--plate-conf",
        type=float,
        default=0.25,
        help="Minimum confidence for plate detections",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # This repository uses an older ultralytics stack; this keeps it compatible with newer torch defaults.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

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

    mot_tracker = Sort()

    cap = _open_video_capture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video source: {args.source}")

    if args.show:
        cv2.namedWindow("ANPR Live", cv2.WINDOW_NORMAL)
        if args.show_width and args.show_width > 0:
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if src_w > 0 and src_h > 0:
                target_w = min(args.show_width, src_w)
                target_h = max(1, int(src_h * (target_w / src_w)))
                cv2.resizeWindow("ANPR Live", target_w, target_h)

    csv_sink: Optional[CsvSink] = None
    if args.output:
        csv_sink = CsvSink(args.output)
        csv_sink.open()
        print(f"Writing CSV to: {args.output}")

    vehicles = [2, 3, 5, 7]

    frame_nmr = -1
    last_seen_plate_by_car: dict[int, tuple[str, float]] = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_nmr += 1
            if args.max_frames > 0 and frame_nmr >= args.max_frames:
                break

            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist():
                if score < args.vehicle_conf:
                    continue
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            track_ids = mot_tracker.update(np.asarray(detections_))

            # detect plates
            license_plates = license_plate_detector(frame)[0]
            plate_boxes = sorted(license_plates.boxes.data.tolist(), key=lambda item: item[4], reverse=True)

            frame_plate_vis = []  # list of dicts: {bbox, score, text?}

            for plate in plate_boxes[: args.max_plate_detections_per_frame]:
                x1, y1, x2, y2, plate_score, _class_id = plate
                if plate_score < args.plate_conf:
                    continue

                plate_vis = {"bbox": [x1, y1, x2, y2], "score": float(plate_score)}

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate, track_ids)
                if car_id == -1:
                    frame_plate_vis.append(plate_vis)
                    continue

                # crop plate
                plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]
                if plate_crop.size == 0:
                    continue

                plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, plate_crop_thresh = cv2.threshold(plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                plate_text, plate_text_score = read_license_plate(plate_crop_thresh)
                if plate_text is None:
                    frame_plate_vis.append(plate_vis)
                    continue

                last_seen_plate_by_car[int(car_id)] = (plate_text, float(plate_text_score or 0.0))
                plate_vis["text"] = plate_text
                frame_plate_vis.append(plate_vis)

                if csv_sink is not None:
                    csv_sink.write_row(
                        ts_unix=time.time(),
                        frame_nmr=frame_nmr,
                        car_id=car_id,
                        car_bbox=[float(xcar1), float(ycar1), float(xcar2), float(ycar2)],
                        plate_bbox=[float(x1), float(y1), float(x2), float(y2)],
                        plate_score=float(plate_score),
                        plate_text=plate_text,
                        plate_text_score=float(plate_text_score or 0.0),
                    )

            if args.show:
                # Draw tracked cars + last seen plate string
                for xcar1, ycar1, xcar2, ycar2, car_id in track_ids.tolist():
                    car_id_int = int(car_id)
                    cv2.rectangle(
                        frame,
                        (int(xcar1), int(ycar1)),
                        (int(xcar2), int(ycar2)),
                        (0, 255, 0),
                        2,
                    )
                    label = f"ID {car_id_int}"
                    if car_id_int in last_seen_plate_by_car:
                        label = f"{label}: {last_seen_plate_by_car[car_id_int][0]}"
                    cv2.putText(
                        frame,
                        label,
                        (int(xcar1), max(0, int(ycar1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                # Draw plate boxes (even if OCR fails)
                for plate in frame_plate_vis:
                    px1, py1, px2, py2 = plate["bbox"]
                    cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                    label = str(plate.get("text") or f"plate {plate['score']:.2f}")
                    cv2.putText(
                        frame,
                        label,
                        (int(px1), max(0, int(py1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("ANPR Live", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    finally:
        cap.release()
        if csv_sink is not None:
            csv_sink.close()
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
