# Automatic Number Plate Recognition (ANPR) with YOLOv8

This project detects vehicles and license plates from video, tracks vehicles across frames, and reads plate text using OCR.

It includes:

- Batch processing from a video file (`main.py`)
- Live processing from webcam/RTSP/video source (`live.py`)
- CSV interpolation for missing detections (`add_missing_data.py`)
- Annotated visualization video generation (`visualize.py`)

## Features

- Vehicle detection with YOLOv8 (`yolov8n.pt`)
- License plate detection with a custom model (`models/license_plate_detector.pt`)
- Multi-object tracking with SORT (`sort/sort.py`)
- License plate OCR with EasyOCR
- CSV export of per-frame detections
- Optional visualization with bounding boxes and recognized plate text

## Project Structure

- `main.py`: Offline ANPR pipeline for a video file.
- `live.py`: Real-time ANPR pipeline for webcam/RTSP/video source.
- `util.py`: OCR helpers, plate text post-processing, CSV writer, car-plate association.
- `add_missing_data.py`: Interpolates missing bounding boxes in CSV output.
- `visualize.py`: Renders an annotated output video using interpolated CSV.
- `models/`: Plate detector weights.
- `sort/`: SORT tracking implementation and related files.

## Requirements

Python dependencies are pinned in `requirements.txt`:

- ultralytics==8.0.114
- pandas==2.0.2
- opencv-python==4.7.0.72
- numpy==1.24.3
- scipy==1.10.1
- easyocr==1.7.0
- filterpy==1.4.5
- Pillow==9.5.0

Recommended Python version: 3.9-3.11.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

The repository already includes:

- `sample.mp4`
- `yolov8n.pt`
- `models/license_plate_detector.pt`

Run batch ANPR on the sample video:

```bash
python main.py --video sample.mp4 --output test.csv --show
```

This writes detections to `test.csv`.

## Usage

### 1) Batch Video Processing (`main.py`)

```bash
python main.py \
  --video sample.mp4 \
  --vehicle-model yolov8n.pt \
  --plate-model models/license_plate_detector.pt \
  --output test.csv \
  --output-video out.mp4 \
  --show
```

Useful flags:

- `--max-frames`: Limit processed frames (default `300`, use `-1` for all).
- `--max-plate-detections-per-frame`: Cap per-frame plate detections.
- `--show-width`: Preview window width.

### 2) Live / Stream Processing (`live.py`)

Webcam (index `0`):

```bash
python live.py --source 0 --show --output test_run.csv
```

RTSP stream:

```bash
python live.py --source rtsp://username:password@ip:port/path --show --output test_run.csv
```

Useful flags:

- `--vehicle-conf`: Vehicle detection confidence threshold.
- `--plate-conf`: Plate detection confidence threshold.
- `--max-frames`: Limit processing for testing.

### 3) Fill Missing Frames (`add_missing_data.py`)

```bash
python add_missing_data.py
```

Reads `test.csv` and writes `test_interpolated.csv` with linearly interpolated car/plate bounding boxes for missing frame spans.

### 4) Create Annotated Output Video (`visualize.py`)

```bash
python visualize.py
```

Reads `test_interpolated.csv` and `sample.mp4`, then writes `out.mp4` with stylized overlays.

## Output CSV Format

Generated CSV columns:

- `frame_nmr`
- `car_id`
- `car_bbox`
- `license_plate_bbox`
- `license_plate_bbox_score`
- `license_number`
- `license_number_score`

Example header:

```csv
frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score
```

## Notes

- `main.py` and `live.py` set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for compatibility with newer PyTorch defaults.
- If `--plate-model` is missing, the scripts fall back to `--fallback-plate-model`.
- OCR quality depends heavily on plate crop quality, motion blur, viewing angle, and lighting.

## Troubleshooting

- If camera does not open, verify `--source` (`0`, `1`, or valid URL/path).
- If you see no plates, verify `models/license_plate_detector.pt` exists and adjust `--plate-conf`.
- If OCR is weak, improve input quality and consider increasing image resolution.
- If installation fails on Windows, upgrade pip first:

```bash
python -m pip install --upgrade pip
```

## License

This repository includes an AGPL-3.0 license in `LICENSE`.
The bundled SORT implementation has its own license under `sort/LICENSE`.
