# -*- coding: utf-8 -*-
"""Dataset: file indexing, DeepfakeDataset, collate, label helpers, split, loaders."""

import os
import glob
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import torchaudio.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)   # avoid OpenCV thread oversubscription inside DataLoader workers

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VIDEOS_BASE_PATH = "/home/server/codes/zareena/Datasets/fakeavceleb/train/Videos"
AUDIO_BASE_PATH  = "/home/server/codes/zareena/Datasets/fakeavceleb/train/Audios"


# ------------------------------------------------------------
# Index files and build labels
# ------------------------------------------------------------
def build_index(videos_base_path=VIDEOS_BASE_PATH, audio_base_path=AUDIO_BASE_PATH):
    """Returns video_paths, audio_paths, labels (4-class)."""
    # Index audio files once: {file_name: path}
    real_audio_index = {os.path.splitext(os.path.basename(p))[0]: p
                        for p in glob.glob(os.path.join(audio_base_path, 'real_audios', '*.wav'))}
    fake_audio_index = {os.path.splitext(os.path.basename(p))[0]: p
                        for p in glob.glob(os.path.join(audio_base_path, 'fake_audios', '*.wav'))}

    video_paths, audio_paths, labels = [], [], []
    n_missing = n_ambiguous = 0

    # sorted() -> deterministic order, so the seeded split is reproducible
    for video_path in sorted(glob.glob(os.path.join(videos_base_path, '*_videos', '*.mp4'))):
        file_name  = os.path.splitext(os.path.basename(video_path))[0]
        parent_dir = os.path.basename(os.path.dirname(video_path))

        if   parent_dir == 'real_videos': video_type = 'real'
        elif parent_dir == 'fake_videos': video_type = 'fake'
        else: continue

        in_real = file_name in real_audio_index
        in_fake = file_name in fake_audio_index

        if in_real and in_fake:
            # Same name in both audio folders: can't tell which one belongs to this video.
            n_ambiguous += 1
            continue
        elif in_real: audio_type, audio_path = 'real', real_audio_index[file_name]
        elif in_fake: audio_type, audio_path = 'fake', fake_audio_index[file_name]
        else:
            n_missing += 1
            continue

        # label mapping:
        #   0 = Real-Real   (video real, audio real)
        #   1 = Fake-Fake   (video fake, audio fake)
        #   2 = Real-Fake   (video real, audio fake)
        #   3 = Fake-Real   (video fake, audio real)
        label = {('real', 'real'): 0, ('fake', 'fake'): 1,
                 ('real', 'fake'): 2, ('fake', 'real'): 3}[(video_type, audio_type)]

        video_paths.append(video_path)
        audio_paths.append(audio_path)
        labels.append(label)

    assert len(video_paths) == len(audio_paths) == len(labels)
    print(f"Loaded {len(video_paths)} pairs | missing audio: {n_missing} | ambiguous (skipped): {n_ambiguous}")
    print(f"Distribution: { {l: labels.count(l) for l in sorted(set(labels))} }")
    return video_paths, audio_paths, labels


# ------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, video_paths, audio_paths, labels,
                 max_audio_length=432, frame_height=224, frame_width=224,
                 max_frames=32,                # frames per clip (one per segment)
                 n_fft=2048, hop_length=1024, n_mels=128,
                 window_fn='hann', train=True):
        self.video_paths = video_paths; self.audio_paths = audio_paths
        self.labels = labels; self.max_audio_length = max_audio_length
        self.frame_height = frame_height; self.frame_width = frame_width
        self.max_frames = max_frames
        self.n_fft = n_fft; self.hop_length = hop_length; self.n_mels = n_mels
        self.train = train; self.window_fn = window_fn
        self.target_sr = 44100

        # ImageNet normalisation
        self.img_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.img_std  = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

        # Audio transforms built once. Stereo input -> (2, n_mels, T)
        self.mel_transform = T.MelSpectrogram(sample_rate=self.target_sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels,
                                              window_fn=torch.hann_window)
        self.to_db = T.AmplitudeToDB(stype='power')
        self.resamplers = {}

    def __len__(self): return len(self.video_paths)

    # ---------------- video (segment-based sampling) ----------------
    def _sample_indices(self, n):
        """
        Split n frames into max_frames equal segments and pick ONE frame per segment.
          train: random frame inside each segment (different frames every epoch)
          val  : middle frame of each segment (deterministic)
        """
        k = self.max_frames
        edges = np.linspace(0, n, k + 1)
        lo = np.floor(edges[:-1]).astype(int)
        hi = np.maximum(np.ceil(edges[1:]).astype(int) - 1, lo)
        if self.train:
            return [random.randint(int(l), int(h)) for l, h in zip(lo, hi)]
        return ((lo + hi) // 2).tolist()

    def _prep(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cv2.resize(frame, (self.frame_width, self.frame_height),
                          interpolation=cv2.INTER_AREA)

    def _read_frames(self, path):
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if n > self.max_frames:
            idx = self._sample_indices(n)
            wanted = set(idx)
            decoded = {}
            for i in range(max(idx) + 1):          # stop after the last sampled frame
                if not cap.grab():
                    break
                if i in wanted:
                    ret, frame = cap.retrieve()
                    if ret:
                        decoded[i] = self._prep(frame)
            frames = [decoded[i] for i in idx if i in decoded]
        else:
            # short video, or frame count unavailable: read everything
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(self._prep(frame))
            if len(frames) > self.max_frames:      # reported count was wrong
                idx = self._sample_indices(len(frames))
                frames = [frames[i] for i in idx]
        cap.release()
        return frames

    def _load_frames(self, path):
        frames = self._read_frames(path)
        if not frames:
            raise RuntimeError(f"No frames could be read from {path}")
        while len(frames) < self.max_frames:       # short clips: repeat last frame
            frames.append(frames[-1])

        # (T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 1]
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)

        # Same flip / rotation for every frame of the clip
        if self.train:
            if random.random() < 0.5:
                x = TF.hflip(x)
            x = TF.rotate(x, random.uniform(-10, 10))

        # ImageNet mean/std
        x = (x - self.img_mean) / self.img_std
        return x.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

    # ---------------- audio (stereo) ----------------
    def _load_mel(self, path):
        data, sr = sf.read(path, always_2d=True)
        audio = torch.tensor(data.T, dtype=torch.float32)    # (channels, samples)

        # force exactly 2 channels: mono -> duplicate, >2 -> keep first two
        if   audio.shape[0] == 1: audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:  audio = audio[:2]

        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = T.Resample(sr, self.target_sr)
            audio = self.resamplers[sr](audio)

        mel = self.to_db(self.mel_transform(audio))          # (2, n_mels, T)
        # keep the REAL length (variable); only cap very long clips for memory
        return mel[:, :, :self.max_audio_length]

    # ---------------- item ----------------
    def __getitem__(self, idx):
        try:
            visual_tensor = self._load_frames(self.video_paths[idx])
            mel           = self._load_mel(self.audio_paths[idx])
        except Exception as e:
            print(f"[WARN] Failed to load idx {idx} "
                  f"({self.video_paths[idx]} / {self.audio_paths[idx]}): {e}")
            return self.__getitem__((idx + 1) % len(self))
        return visual_tensor, mel, torch.tensor(self.labels[idx])


# ------------------------------------------------------------
# Collate + label helpers
# ------------------------------------------------------------
def collate_fn(batch):
    """
    visual: (B, 3, T, H, W)  (T identical for all clips with max_frames set;
            otherwise padded by repeating the last frame)
    audio : (B, 2, n_mels, T_audio_max)  padded with silence to the longest clip
            in the batch; true lengths (in mel frames) are returned separately.
    """
    v, a, l = zip(*batch)

    t_max = max(x.shape[1] for x in v)
    v_pad = []
    for x in v:
        pad = t_max - x.shape[1]
        if pad > 0:
            x = torch.cat([x, x[:, -1:].expand(-1, pad, -1, -1)], dim=1)
        v_pad.append(x)

    a_lens = torch.tensor([x.shape[2] for x in a], dtype=torch.long)
    ta_max = int(a_lens.max())
    a_pad = [F.pad(x, (0, ta_max - x.shape[2]), value=x.min().item()) for x in a]

    return torch.stack(v_pad), torch.stack(a_pad), torch.stack(l), a_lens


def make_binary_labels(batch_lbl: torch.Tensor):
    """
    Derive per-modality binary real/fake labels from the 4-class label.

      4-class: 0=RR, 1=FF, 2=RF, 3=FR

      video_label: 0 = real video, 1 = fake video   -> fake iff label in {FF, FR} = {1,3}
      audio_label: 0 = real audio, 1 = fake audio   -> fake iff label in {FF, RF} = {1,2}
    """
    video_label = ((batch_lbl == 1) | (batch_lbl == 3)).long()
    audio_label = ((batch_lbl == 1) | (batch_lbl == 2)).long()
    return video_label, audio_label


# ------------------------------------------------------------
# Train / val split (stratified, so every class appears in both sets)
# ------------------------------------------------------------
def stratified_split(labels, train_frac=0.7, seed=42):
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for i, l in enumerate(labels):
        by_cls[l].append(i)

    train_idx, val_idx = [], []
    for l in sorted(by_cls):
        idxs = by_cls[l][:]
        rng.shuffle(idxs)
        k = int(round(train_frac * len(idxs)))
        if len(idxs) > 1:
            k = min(max(k, 1), len(idxs) - 1)
        train_idx += idxs[:k]
        val_idx   += idxs[k:]
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx


# ------------------------------------------------------------
# Datasets + loaders
# ------------------------------------------------------------
def build_loaders(videos_base_path=VIDEOS_BASE_PATH, audio_base_path=AUDIO_BASE_PATH,
                  batch_size=32, train_frac=0.7, seed=42, num_workers=4):
    """Index files, split, and return (train_loader, val_loader)."""
    video_paths, audio_paths, labels = build_index(videos_base_path, audio_base_path)

    train_idx, val_idx = stratified_split(labels, train_frac=train_frac, seed=seed)
    print(f"Train: {len(train_idx)} {{ {', '.join(f'{c}:{sum(labels[i]==c for i in train_idx)}' for c in range(4))} }} | "
          f"Val: {len(val_idx)} {{ {', '.join(f'{c}:{sum(labels[i]==c for i in val_idx)}' for c in range(4))} }}")

    def make_dataset(idx, train):
        return DeepfakeDataset(
            [video_paths[i] for i in idx],
            [audio_paths[i] for i in idx],
            [labels[i]      for i in idx],
            train=train,
        )

    train_dataset = make_dataset(train_idx, train=True)
    val_dataset   = make_dataset(val_idx,   train=False)

    # 32 frames per clip (segment-based sampling). Lower batch_size if you hit OOM.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
