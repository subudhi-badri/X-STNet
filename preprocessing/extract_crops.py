# -*- coding: utf-8 -*-
import cv2
import json
import numpy as np
import os
import subprocess
import shutil

# ---------------- CONFIG ----------------
INPUT_ROOT  = "./fakeavceleb/Videos"
DETECTION_ROOT = "./fakeavceleb/json"
OUTPUT_ROOT    = "./fakeavceleb/processed"

CLASSES        = ["real", "fake"]
FACE_W         = 512
FACE_H         = 512

IOU_THRESH     = 0.3
SMOOTH_ALPHA   = 0.7
MAX_MISSING    = 30
# ----------------------------------------

BLACK = np.zeros((FACE_H, FACE_W, 3), dtype=np.uint8)


# ------------------------------------------------------------------
# ffprobe helpers
# ------------------------------------------------------------------

def parse_float_safe(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def ffprobe_info(video_path):
    """
    Get fps, duration, width, height from ffprobe via JSON (field-order safe).
    duration: stream first, then format section fallback.
    Also returns fps_str (fraction string, e.g. '25/1') for lossless writer use.
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
        fps     = None
        if fps_str:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                den = float(den)
                fps = float(num) / den if den != 0 else None
            else:
                fps = parse_float_safe(fps_str)

        width  = stream.get("width")
        height = stream.get("height")
        if width  is not None: width  = int(width)
        if height is not None: height = int(height)

        duration = parse_float_safe(stream.get("duration"))
        if duration is None:
            duration = parse_float_safe(fmt.get("duration"))

        if not fps or not width or not height:
            raise ValueError(f"Missing: fps={fps}, w={width}, h={height}")

        return fps, duration, width, height, fps_str

    except Exception as e:
        print(f"  [ffprobe error]: {e}")
        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, total / fps, width, height, str(fps)


def ffprobe_verify(video_path):
    """Return (frame_count, duration) of an already-written file."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets,duration:format=duration",
            "-of", "json",
            video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        data   = json.loads(out)
        stream = data.get("streams", [{}])[0]
        fmt    = data.get("format", {})

        n   = int(stream.get("nb_read_packets", -1))
        dur = parse_float_safe(stream.get("duration"))
        if dur is None:
            dur = parse_float_safe(fmt.get("duration"))
        return n, dur or -1.0
    except Exception:
        return -1, -1.0


# ------------------------------------------------------------------
# ffmpeg frame I/O
# ------------------------------------------------------------------

def read_frames_ffmpeg(video_path, width, height):
    """Decode every frame via ffmpeg stdout as raw BGR. No skips, no early stop."""
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


def write_frames_ffmpeg(frames, fps_str, src_duration, out_path):
    """
    Write frames to mp4 via ffmpeg stdin.

    Key points for duration accuracy:
      - fps_str  : original fraction string '25/1', '30000/1001', etc.
                   Using the fraction avoids float rounding in the timebase.
      - -t src_duration : hard-clamps the output container duration to exactly
                   match the source, eliminating any residual drift from an
                   extra/missing last frame.
    """
    if not frames:
        return

    h, w = frames[0].shape[:2]

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", fps_str,          # fraction string — no float rounding
        "-i", "-",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        "-t", str(src_duration),  # clamp output to exact source duration
        "-video_track_timescale", "90000",  # fine-grained timebase
        out_path
    ]

    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames:
        pipe.stdin.write(frame.tobytes())
    pipe.stdin.close()
    pipe.wait()


# ------------------------------------------------------------------
# Tracking
# ------------------------------------------------------------------

def iou(a, b):
    xA = max(a[0], b[0]);  yA = max(a[1], b[1])
    xB = min(a[2], b[2]);  yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter)


def box_area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])


def crop_face(frame, smooth_box):
    x1, y1, x2, y2 = smooth_box.astype(int)
    h, w = frame.shape[:2]
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(w, x2);  y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        return cv2.resize(crop, (FACE_W, FACE_H))
    except Exception:
        return None


class Track:
    def __init__(self, tid, bbox):
        self.id        = tid
        self.bbox      = np.array(bbox, dtype=np.float32)
        self.smooth    = self.bbox.copy()
        self.missing   = 0
        self.last_good = None

    def update(self, bbox):
        bbox         = np.array(bbox, dtype=np.float32)
        self.smooth  = SMOOTH_ALPHA * self.smooth + (1 - SMOOTH_ALPHA) * bbox
        self.bbox    = bbox
        self.missing = 0

    def miss(self):
        self.missing += 1

    def dead(self):
        return self.missing > MAX_MISSING

    def get_crop(self, frame):
        crop = crop_face(frame, self.smooth)
        if crop is not None:
            self.last_good = crop
            return crop
        return self.last_good


# ------------------------------------------------------------------
# Per-video
# ------------------------------------------------------------------

def process_video(video_path, detection_json, output_dir, base_name):
    if not os.path.exists(detection_json):
        print(f"  [SKIP] No JSON: {detection_json}")
        return
    if not os.path.exists(video_path):
        print(f"  [SKIP] No video: {video_path}")
        return

    with open(detection_json) as f:
        data = json.load(f)

    # fps_str saved by face_detect.py; fall back to string of float if absent
    fps        = data["video_info"]["fps"]
    fps_str    = data["video_info"].get("fps_str", str(fps))
    width      = data["video_info"]["width"]
    height     = data["video_info"]["height"]
    json_total = data["video_info"].get("total_frames", None)
    detections = data["detections"]

    # always re-probe the source for the authoritative duration
    _, src_duration, _, _, _ = ffprobe_info(video_path)
    if src_duration is None:
        src_duration = json_total / fps if json_total else 0.0

    print(f"  src duration : {src_duration:.4f}s")
    print(f"  JSON frames  : {json_total}  fps: {fps:.4f}  fps_str: {fps_str}")

    # decode all frames (no skips, no early stop)
    src_frames = read_frames_ffmpeg(video_path, width, height)
    n = len(src_frames)
    print(f"  decoded      : {n} frames")

    if n == 0:
        print(f"  [SKIP] No frames decoded.")
        return

    if json_total and abs(json_total - n) > 2:
        print(f"  [WARN] JSON has {json_total} entries but decoded {n} frames — "
              f"re-run face_detect.py for accurate detections.")

    # pre-allocate output buffer (one slot per decoded frame)
    output_buf = [BLACK.copy() for _ in range(n)]

    tracks  = []
    next_id = 0

    for frame_idx, frame in enumerate(src_frames):
        boxes = detections.get(str(frame_idx), [])
        used  = set()

        for tr in tracks:
            best_iou, best_i = 0.0, -1
            for i, box in enumerate(boxes):
                if i in used:
                    continue
                score = iou(tr.bbox, box)
                if score > best_iou:
                    best_iou, best_i = score, i
            if best_iou > IOU_THRESH:
                tr.update(boxes[best_i])
                used.add(best_i)
            else:
                tr.miss()

        for i, box in enumerate(boxes):
            if i not in used:
                tracks.append(Track(next_id, box))
                next_id += 1

        best_track = None
        best_area  = -1
        for tr in tracks:
            if not tr.dead():
                a = box_area(tr.bbox.tolist())
                if a > best_area:
                    best_area  = a
                    best_track = tr

        if best_track is not None:
            crop = best_track.get_crop(frame)
            if crop is not None:
                output_buf[frame_idx] = crop

        tracks = [tr for tr in tracks if not tr.dead()]

    face_frames  = sum(1 for f in output_buf if not np.array_equal(f, BLACK))
    black_frames = n - face_frames
    print(f"  face frames  : {face_frames}/{n} ({100*face_frames/n:.1f}%)")
    if black_frames > 0:
        print(f"  black frames : {black_frames} (no detection)")

    # write with fps fraction + hard duration clamp ? exact duration match
    out_path = os.path.join(output_dir, f"{base_name}.mp4")
    write_frames_ffmpeg(output_buf, fps_str, src_duration, out_path)

    # verify
    out_n, out_dur = ffprobe_verify(out_path)
    diff   = abs(src_duration - out_dur) if out_dur > 0 else float("inf")
    status = "[OK]" if diff < 0.04 else "[MISMATCH]"

    print(f"  src          : {src_duration:.4f}s")
    print(f"  out          : {out_dur:.4f}s  ({out_n} frames)")
    print(f"  diff         : {diff:.4f}s  {status}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ERROR: ffmpeg/ffprobe not found.")
        print("Install: conda install -c conda-forge ffmpeg")
        return

    for cls in CLASSES:
        in_cls  = os.path.join(INPUT_ROOT,     cls)
        det_cls = os.path.join(DETECTION_ROOT, cls)
        out_cls = os.path.join(OUTPUT_ROOT,    cls)

        os.makedirs(out_cls, exist_ok=True)

        if not os.path.exists(in_cls):
            print(f"[SKIP] Not found: {in_cls}")
            continue

        videos = sorted([
            v for v in os.listdir(in_cls)
            if v.endswith(".mp4") and not v.startswith("._")
        ])

        print(f"\n{'='*60}")
        print(f"Class: {cls}  |  {len(videos)} videos")
        print(f"{'='*60}")

        for i, video in enumerate(videos, 1):
            name = os.path.splitext(video)[0]
            print(f"\n[{i}/{len(videos)}] {name}")
            process_video(
                video_path     = os.path.join(in_cls,  video),
                detection_json = os.path.join(det_cls, f"{name}.json"),
                output_dir     = out_cls,
                base_name      = name,
            )

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()