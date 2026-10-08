# -*- coding: utf-8 -*-
import os
import json
import subprocess
import shutil
import numpy as np
import cv2
from facenet_pytorch import MTCNN

# ---------------- CONFIG ----------------
INPUT_ROOT  = "./fakeavceleb/Videos"
OUTPUT_ROOT = "./fakeavceleb/json"
CLASSES     = ["real", "fake"]

DEVICE      = "cuda"   
BATCH_SIZE  = 16
# ----------------------------------------

print(f"Running Detection on: {DEVICE}")


def build_detector(device):
    return MTCNN(
        keep_all=True,
        margin=0,
        thresholds=[0.85, 0.95, 0.95],
        device=device,
        post_process=False,
        select_largest=False,
    )


def parse_float_safe(s):
    """Safely parse a float string; returns None if invalid (e.g. 'N/A')."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_fps(fps_str):
    """Parse fps string that may be a fraction like '25/1' or '30000/1001'."""
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            den = float(den)
            return float(num) / den if den != 0 else None
        return float(fps_str)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def ffprobe_info(video_path):
    """
    Get fps, duration, width, height from ffprobe using JSON output.
    Parses by key name to avoid CSV field-order bugs.
    duration: tries stream first, then format section (handles most containers).
    Falls back to OpenCV if ffprobe fails entirely.
    Returns (fps, duration, width, height, fps_str).
      fps_str  -- original fraction string e.g. '25/1' kept for lossless
                  passthrough to ffmpeg writers.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,duration,width,height"
                             ":format=duration",
            "-of", "json",
            video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        data = json.loads(out)

        stream  = data.get("streams", [{}])[0]
        fmt     = data.get("format", {})

        fps_str = stream.get("r_frame_rate", "")
        fps     = parse_fps(fps_str)

        width   = stream.get("width")
        height  = stream.get("height")
        if width  is not None: width  = int(width)
        if height is not None: height = int(height)

        # prefer stream duration, fall back to container/format duration
        duration = parse_float_safe(stream.get("duration"))
        if duration is None:
            duration = parse_float_safe(fmt.get("duration"))

        if fps is None or width is None or height is None:
            raise ValueError(
                f"Missing critical value  fps={fps}, "
                f"width={width}, height={height}"
            )

        return fps, duration, width, height, fps_str

    except Exception as e:
        print(f"  [ffprobe error]: {e}")
        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = total / fps if fps else 0.0
        return fps, duration, width, height, str(fps)


def read_frames_ffmpeg(video_path, width, height):
    """
    Decode EVERY frame using ffmpeg piped as raw BGR bytes.
    Avoids OpenCV cap.read() early-stop and grab/retrieve skip bugs.
    Returns list of (H, W, 3) uint8 numpy arrays in BGR.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = width * height * 3
    frames = []
    while True:
        raw = pipe.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame.copy())

    pipe.stdout.close()
    pipe.wait()
    return frames


def detect_batch(detector, frames_rgb):
    """
    Run MTCNN on a list of full-resolution RGB numpy arrays.
    Expands bounding boxes so some background around the face is included.
    Returns list of box lists, one per frame.
    """

    BOX_EXPAND = 0.20   # 20% larger box (increase to 0.3 if needed)

    try:
        batch_boxes, _ = detector.detect(frames_rgb)
    except Exception as e:
        print(f"  [Batch detect error]: {e}")
        batch_boxes = [None] * len(frames_rgb)

    results = []

    # Process each frame
    for frame, boxes in zip(frames_rgb, batch_boxes):

        h_img, w_img, _ = frame.shape

        if boxes is None:
            results.append([])
            continue

        clean = []

        for box in boxes:
            x1, y1, x2, y2 = box

            # Original width and height
            w = x2 - x1
            h = y2 - y1

            # Expand box on all sides
            x1 = x1 - w * BOX_EXPAND
            y1 = y1 - h * BOX_EXPAND
            x2 = x2 + w * BOX_EXPAND
            y2 = y2 + h * BOX_EXPAND

            # Keep inside image boundaries
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(w_img, x2))
            y2 = int(min(h_img, y2))

            clean.append([x1, y1, x2, y2])

        results.append(clean)

    return results


def process_video(video_path, output_json, detector):
    try:
        fps, duration, width, height, fps_str = ffprobe_info(video_path)

        frames_bgr = read_frames_ffmpeg(video_path, width, height)
        total = len(frames_bgr)

        if total == 0:
            print(f"  [SKIP] No frames decoded: {os.path.basename(video_path)}")
            return

        # derive duration from actual frame count only if still unknown
        if duration is None:
            duration = total / fps

        print(f"  {os.path.basename(video_path)}: "
              f"{total} frames  {duration:.4f}s  @ {fps:.4f}fps  "
              f"{width}x{height}")

        results = {
            "video_info": {
                "fps":          fps,
                "fps_str":      fps_str,   # fraction string for writers
                "duration":     duration,
                "width":        width,
                "height":       height,
                "total_frames": total
            },
            "detections": {}
        }

        for i in range(0, total, BATCH_SIZE):
            batch_bgr = frames_bgr[i : i + BATCH_SIZE]
            batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]
            boxes_list = detect_batch(detector, batch_rgb)
            for j, boxes in enumerate(boxes_list):
                results["detections"][str(i + j)] = boxes

        assert len(results["detections"]) == total, \
            f"BUG: {len(results['detections'])} entries for {total} frames"

        with open(output_json, "w") as f:
            json.dump(results, f)

        print(f"  Saved: {os.path.basename(output_json)}  "
              f"({len(results['detections'])} frame entries)")

    except Exception as e:
        print(f"  [FAILED] {os.path.basename(video_path)}: {e}")


def main():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ERROR: ffmpeg/ffprobe not found.")
        print("Install: conda install -c conda-forge ffmpeg")
        return

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    try:
        detector = build_detector(DEVICE)
    except Exception:
        print("Device failed, falling back to CPU...")
        detector = build_detector("cpu")

    for cls in CLASSES:
        in_dir  = os.path.join(INPUT_ROOT,  cls)
        out_dir = os.path.join(OUTPUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            print(f"[SKIP] Not found: {in_dir}")
            continue

        files = sorted([
            f for f in os.listdir(in_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
            and not f.startswith("._")
        ])
        print(f"\nClass '{cls}': {len(files)} videos")

        for i, video in enumerate(files, 1):
            video_path  = os.path.join(in_dir,  video)
            name        = os.path.splitext(video)[0]
            output_json = os.path.join(out_dir, f"{name}.json")

            if os.path.exists(output_json):
                print(f"[{i}/{len(files)}] Skip (exists): {video}")
                continue

            print(f"[{i}/{len(files)}] {video}")
            process_video(video_path, output_json, detector)

    print("\nDetection complete.")


if __name__ == "__main__":
    main()