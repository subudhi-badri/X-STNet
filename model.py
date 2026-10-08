# -*- coding: utf-8 -*-
"""Model: ResidualAttention, VisualStream, AudioStream, MDSNet (+ optional shape prints)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ------------------------------------------------------------
# Debug helper (prints only when debug=True)
# ------------------------------------------------------------
def log_shape(debug, name, x):
    """Print the shape of a tensor (or the value of a non-tensor) when debug=True."""
    if debug:
        shape = tuple(x.shape) if torch.is_tensor(x) else x
        print(f"  {name:<28}{shape}")


# ------------------------------------------------------------
# Residual attention
# ------------------------------------------------------------
class ResidualAttention(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.attn1 = nn.Linear(dim, dim, bias=False)
        self.attn2 = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1   = torch.sigmoid(self.attn1(x))
        out1 = w1 * x + x

        w2   = torch.sigmoid(self.attn2(x))
        out2 = w2 * x + x

        return x + self.gamma * (out1 + out2)


# ------------------------------------------------------------
# Visual stream (2D ResNet-18)
# ------------------------------------------------------------
class VisualStream(nn.Module):
    def __init__(self, lstm_seq_len=None):
        super().__init__()

        # Standard 2D ResNet-18
        resnet18 = models.resnet18(pretrained=True)

        # Remove average pooling and classification layer
        self.features = nn.Sequential(*list(resnet18.children())[:-2])

        # ResNet-18 output: (B*T, 512, H', W')
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Residual attention over temporal feature sequence
        self.attn = ResidualAttention(512)

        self.dropout = nn.Dropout(p=0.2)

        self.lstm_seq_len = lstm_seq_len

    def forward(self, x, debug=False):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        log_shape(debug, "[V] input (B,C,T,H,W)", x)

        # Convert video into individual frames
        x = x.permute(0, 2, 1, 3, 4)          # (B, T, C, H, W)
        log_shape(debug, "[V] permuted (B,T,C,H,W)", x)
        x = x.reshape(B * T, C, H, W)          # (B*T, C, H, W)
        log_shape(debug, "[V] frames (B*T,C,H,W)", x)

        # 2D ResNet feature extraction
        fmap = self.features(x)                # (B*T, 512, H', W')
        log_shape(debug, "[V] resnet fmap", fmap)

        # Spatial global average pooling
        pooled = self.temporal_pool(fmap)      # (B*T, 512, 1, 1)
        log_shape(debug, "[V] avg-pooled", pooled)
        pooled = pooled.flatten(1)             # (B*T, 512)
        log_shape(debug, "[V] flattened", pooled)

        # Restore temporal dimension
        seq = pooled.view(B, T, 512)           # (B, T, 512)
        log_shape(debug, "[V] seq (B,T,512)", seq)

        # lstm_seq_len=None -> keep ALL frames; otherwise pool to that length
        if self.lstm_seq_len is not None and T != self.lstm_seq_len:
            seq = F.adaptive_avg_pool1d(
                seq.transpose(1, 2),
                self.lstm_seq_len
            ).transpose(1, 2)                  # (B, S, 512)
            log_shape(debug, "[V] seq after seq-pool", seq)

        seq = self.dropout(seq)
        log_shape(debug, "[V] after dropout", seq)

        # Residual attention
        seq = self.attn(seq)                   # (B, S, 512)
        log_shape(debug, "[V] after attention", seq)

        return seq, fmap


# ------------------------------------------------------------
# Audio stream (stereo: 2 input channels)
# ------------------------------------------------------------
class AudioStream(nn.Module):
    def __init__(self, lstm_seq_len=None):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2,   32,  3, 1, 1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32,  64,  3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64,  128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.lstm_seq_len = lstm_seq_len
        self.attn         = ResidualAttention(512)
        self.dropout      = nn.Dropout(p=0.3)

    @staticmethod
    def _down_len(lengths):
        # 4 x MaxPool2d(2,2): each one floors the time length / 2
        for _ in range(4):
            lengths = torch.div(lengths, 2, rounding_mode='floor')
        return lengths.clamp(min=1)

    def forward(self, x, lengths=None, debug=False):
        # x: (B, 2, n_mels, T_max) ; lengths: (B,) true mel-frame counts
        if x.dim() == 3: x = x.unsqueeze(1)  # safety: add channel dim
        log_shape(debug, "[A] input (B,2,mels,T)", x)

        feat = self.conv_layers(x)                   # (B, 512, freq', time')
        log_shape(debug, "[A] conv out", feat)

        feat = feat.mean(dim=2)                      # pool frequency -> (B, 512, time')
        log_shape(debug, "[A] after freq-pool", feat)

        if lengths is None:
            out_lens = torch.full((feat.size(0),), feat.size(2),
                                  dtype=torch.long, device=feat.device)
        else:
            out_lens = self._down_len(lengths.to(feat.device)).clamp(max=feat.size(2))
        log_shape(debug, "[A] out_lens", out_lens)
        if debug:
            print(f"  {'[A] out_lens values':<28}{out_lens.tolist()}")

        if self.lstm_seq_len is not None:            # optional fixed-length pooling
            feat = F.adaptive_avg_pool1d(feat, self.lstm_seq_len)
            out_lens = torch.full_like(out_lens, self.lstm_seq_len)
            log_shape(debug, "[A] feat after seq-pool", feat)

        seq = self.dropout(feat.permute(0, 2, 1))    # (B, S, 512)
        log_shape(debug, "[A] seq (B,S,512)", seq)
        seq = self.attn(seq)                         # (B, S, 512)
        log_shape(debug, "[A] after attention", seq)
        return seq, out_lens


# ------------------------------------------------------------
# MDSNet
# NOTE: visual_aux / audio_aux output 2 classes (real/fake for that
# modality only) instead of 4 — they cannot legitimately resolve the
# 4-way distinction since they only ever see one modality's features.
# ------------------------------------------------------------
class MDSNet(nn.Module):

    def __init__(self, lstm_seq_len=None, hidden_size: int = 256,
                 num_layers: int = 1, intermediate_size: int = 512,
                 num_classes: int = 4, num_modality_classes: int = 2):
        super().__init__()
        self.visual_stream = VisualStream(lstm_seq_len)
        self.audio_stream  = AudioStream(lstm_seq_len)

        lstm_drop = 0.2 if num_layers > 1 else 0.0
        self.visual_lstm = nn.LSTM(512, hidden_size, num_layers,
                                   batch_first=True, dropout=lstm_drop)
        self.audio_lstm  = nn.LSTM(512, hidden_size, num_layers,
                                   batch_first=True, dropout=lstm_drop)

        # Cross-gating: gate comes from the OTHER modality
        self.cross_gate_v = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.cross_gate_a = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())

        # Hadamard + Concat fusion
        self.proj_v = nn.Linear(hidden_size, hidden_size)
        self.proj_a = nn.Linear(hidden_size, hidden_size)

        self.fusion_fc       = nn.Linear(3 * hidden_size, intermediate_size)
        self.intermediate_fc = nn.Linear(intermediate_size, num_classes)
        self.fusion_act      = nn.ReLU()
        self.dropout         = nn.Dropout(0.2)

        # Auxiliary classifiers on the (gated) LSTM hidden states — binary
        self.visual_aux = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64,  num_modality_classes),
        )
        self.audio_aux = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64,  num_modality_classes),
        )

    def forward(self, visual_input, audio_input, audio_lens=None, debug=False):

        # ---- Encoder streams ----
        if debug: print("\n=== Visual stream ===")
        v_seq, fmap = self.visual_stream(visual_input, debug)
        if debug: print("\n=== Audio stream ===")
        a_seq, a_lens = self.audio_stream(audio_input, audio_lens, debug)

        # ---- Encoder-level features (purely unimodal) ----
        if debug: print("\n=== Encoder-level features ===")
        v_enc = v_seq.mean(dim=1)
        # masked mean over the REAL audio steps only (ignore padding)
        a_mask = (torch.arange(a_seq.size(1), device=a_seq.device)[None, :]
                  < a_lens[:, None]).unsqueeze(-1).float()
        a_enc = (a_seq * a_mask).sum(dim=1) / a_mask.sum(dim=1).clamp(min=1.0)
        log_shape(debug, "v_enc", v_enc)
        log_shape(debug, "a_mask", a_mask)
        log_shape(debug, "a_enc", a_enc)

        # ---- LSTMs ----
        if debug: print("\n=== LSTMs ===")
        v_out, _ = self.visual_lstm(v_seq)
        log_shape(debug, "visual lstm out", v_out)
        # packed sequence: LSTM stops at each clip's true end (no padding steps)
        a_packed = nn.utils.rnn.pack_padded_sequence(
            a_seq, a_lens.cpu(), batch_first=True, enforce_sorted=False)
        log_shape(debug, "audio packed data", a_packed.data)
        _, (a_h, _) = self.audio_lstm(a_packed)
        log_shape(debug, "audio lstm h_n", a_h)

        v_feat = v_out[:, -1, :]
        a_feat = a_h[-1]                     # hidden state at last REAL step
        log_shape(debug, "v_feat", v_feat)
        log_shape(debug, "a_feat", a_feat)

        # ---- Cross-gating ----
        if debug: print("\n=== Cross-gating ===")
        gate_v = self.cross_gate_v(a_feat)
        gate_a = self.cross_gate_a(v_feat)
        log_shape(debug, "gate_v", gate_v)
        log_shape(debug, "gate_a", gate_a)

        residual_visual = gate_v * v_feat + v_feat
        residual_audio  = gate_a * a_feat + a_feat
        log_shape(debug, "residual_visual", residual_visual)
        log_shape(debug, "residual_audio", residual_audio)

        # ---- Fusion ----
        if debug: print("\n=== Fusion ===")
        pv = self.proj_v(residual_visual)
        pa = self.proj_a(residual_audio)
        log_shape(debug, "pv", pv)
        log_shape(debug, "pa", pa)

        hadamard = pv * pa
        log_shape(debug, "hadamard", hadamard)
        fused    = torch.cat([pv, pa, hadamard], dim=1)
        log_shape(debug, "fused concat", fused)

        fused   = self.dropout(fused)
        fused   = self.fusion_act(self.fusion_fc(fused))
        log_shape(debug, "fusion_fc out", fused)
        outputs = self.intermediate_fc(fused)
        log_shape(debug, "outputs (4-class logits)", outputs)

        # ---- Auxiliary outputs (binary, per modality) ----
        if debug: print("\n=== Aux heads ===")
        visual_out = self.visual_aux(pv)
        audio_out  = self.audio_aux(pa)
        log_shape(debug, "visual_out", visual_out)
        log_shape(debug, "audio_out", audio_out)
        if debug: print()

        return outputs, visual_out, audio_out, v_enc, a_enc, v_feat, a_feat, fmap, pv, pa


# ------------------------------------------------------------
# Quick shape check (dummy data, no dataset needed):  python model.py
# ------------------------------------------------------------
def shape_check():
    model = MDSNet().eval()
    with torch.no_grad():
        model(torch.randn(2, 3, 32, 224, 224),   # (B, C, T, H, W)
              torch.randn(2, 2, 128, 432),       # (B, 2, n_mels, T_audio)
              torch.tensor([432, 300]),          # true mel lengths
              debug=True)


if __name__ == "__main__":
    shape_check()
