# X-STNet
A robust multimodal deepfake detection architecture integrating parallel spatiotemporal encoding and residual dual-branch attention for manipulation-sensitive audio-visual representations. X-STNet further uses cross-gated residual fusion, multistage contrastive learning, and sparsity-aware optimization for improved cross-dataset generalization.

## Repository structure

```
X-STNet
├── preprocessing/
  ├── detect_faces.py    
  └── extract_crops.py    
├── dataset.py              
├── model.py                
├── losses.py              
├── pruning.py             
├── train.py                
├── requirements.txt
└── README.md
```

## Installation

pip install -r requirements.txt


### Preprocessing layout
```

fakeavceleb/
├── Videos/                       
│   ├── real/*.mp4
│   └── fake/*.mp4
├── json/                        
│   ├── real/<video_name>.json
│   └── fake/<video_name>.json
└── processed/                    
    ├── real/<video_name>.mp4     
    └── fake/<video_name>.mp4
```

### Training layout
```
fakeavceleb/train/
├── Videos/
│   ├── real_videos/<name>.mp4
│   └── fake_videos/<name>.mp4
└── Audios/
    ├── real_audios/<name>.wav
    └── fake_audios/<name>.wav

fakeavceleb/test/
├── Videos/
│   ├── real_videos/<name>.mp4
│   └── fake_videos/<name>.mp4
└── Audios/
    ├── real_audios/<name>.wav
    └── fake_audios/<name>.wav
```


## Usage

### 1. Detect faces

```bash
python preprocessing/detect_faces.py
```

Decodes every frame with ffmpeg, runs MTCNN in batches of 16, expands each box by 20 %, and writes one JSON per video. Videos whose JSON already exists are skipped, so the script can be resumed. Set `DEVICE` in its CONFIG block (it falls back to CPU if CUDA fails to initialise). JSON format (one entry per decoded frame; an empty list means no face was found):

```json
{
  "video_info": {"fps": 25.0, "fps_str": "25/1", "duration": 8.2, "width": 1280, "height": 720, "total_frames": 205},
  "detections": {"0": [[x1, y1, x2, y2]], "1": []}
}
```

### 2. Extract face crops

```bash
python preprocessing/extract_crops.py
```

Tracks faces across frames (IoU matching, box smoothing), takes the largest live track per frame, and writes a 512x512 face-crop video. Frames without a usable face are written as black frames, so the frame count matches the source.

### 3. Train

```bash
python train.py
```

### Inspect tensor shapes

```bash
python model.py
```

To print shapes for a real batch, pass `debug=True` in the model call.
