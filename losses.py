# -*- coding: utf-8 -*-
"""Losses: cross-modal InfoNCE, intra-modality max-margin contrastive loss, combined loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

criterion_ce = nn.CrossEntropyLoss()


def cross_modal_infonce_loss(
        visual_features,
        audio_features,
        labels,
        temperature=0.07,
        disagreement_weight=0.5):

    """
    Class-aware cross-modal InfoNCE for FakeAVCeleb.

    Labels:
        0 = Real-Real
        1 = Fake-Fake
        2 = Real-Fake
        3 = Fake-Real

    RR / FF:
        Cross-modal features should be aligned.

    RF / FR:
        Cross-modal features should remain inconsistent/separated.
    """

    # 1. L2 normalization
    visual_features = F.normalize(visual_features, p=2, dim=1, eps=1e-8)
    audio_features  = F.normalize(audio_features,  p=2, dim=1, eps=1e-8)

    # 2. Agreement samples: RR + FF
    agree_mask = (labels == 0) | (labels == 1)
    loss_agree = torch.zeros((), device=visual_features.device)

    if agree_mask.sum() >= 2:
        v = visual_features[agree_mask]
        a = audio_features[agree_mask]

        # Cross-modal similarity matrix
        similarity = torch.matmul(v, a.T) / temperature
        target = torch.arange(v.size(0), device=v.device)

        loss_v2a = F.cross_entropy(similarity, target)     # Visual -> Audio
        loss_a2v = F.cross_entropy(similarity.T, target)   # Audio -> Visual

        # Symmetric InfoNCE
        loss_agree = 0.5 * (loss_v2a + loss_a2v)

    # 3. Disagreement samples: RF + FR
    disagree_mask = (labels == 2) | (labels == 3)
    loss_disagree = torch.zeros((), device=visual_features.device)

    if disagree_mask.any():
        v_dis = visual_features[disagree_mask]
        a_dis = audio_features[disagree_mask]

        cosine = F.cosine_similarity(v_dis, a_dis, dim=1)

        # Penalize positive cross-modal similarity
        loss_disagree = torch.mean(F.relu(cosine) ** 2)

    # 4. Final cross-modal objective
    return loss_agree + disagreement_weight * loss_disagree


def immcl_loss(features, labels, margin=0.99):
    """
    Intra-modality Max-Margin Contrastive Loss (paper version).

    features: (B, D) features for one modality
    labels:   (B,)   modality-specific binary real/fake labels
    margin:   gamma, applied to negative pairs
    """
    B = features.size(0)

    f_hat = F.normalize(features, p=2, dim=1)
    sim = torch.matmul(f_hat, f_hat.t())                   # (B, B) cosine similarities

    labels = labels.view(-1, 1)
    eye = torch.eye(B, device=sim.device)
    Y = (labels == labels.t()).float() * (1.0 - eye)       # positive pairs, no self-pairs
    not_Y = (1.0 - Y) * (1.0 - eye)                        # negative pairs, no self-pairs

    pos_term = Y * (1.0 - sim) ** 2
    neg_term = not_Y * F.relu(sim - margin) ** 2

    # (1/B) * sum_k sum_{l != k} [ ... ]
    return (pos_term + neg_term).sum() / B


def compute_loss(outputs, visual_out, audio_out, v_enc, a_enc, v_feat, a_feat,
                 batch_lbl, video_label, audio_label):
    """
    Shared loss computation used by both train and val steps.

      - outputs (fused, 4-class)     supervised by batch_lbl (4-class)
      - v_enc / a_enc (unimodal)     supervised by binary video/audio label
                                       via immcl_loss
      - v_feat / a_feat (cross-modal alignment) supervised by batch_lbl,
                                       conditioned on agreement (RR/FF pull,
                                       RF/FR push)
      - visual_out / audio_out (aux heads, binary) supervised by binary
                                       video/audio label
    """
    loss = (
        criterion_ce(outputs, batch_lbl)
        + 0.2 * immcl_loss(v_enc, video_label, margin=0.2)
        + 0.2 * immcl_loss(a_enc, audio_label, margin=0.2)
        + 0.4 * cross_modal_infonce_loss(v_feat, a_feat, batch_lbl, temperature=0.2, disagreement_weight=1)
        + 0.1 * criterion_ce(visual_out, video_label)
        + 0.1 * criterion_ce(audio_out, audio_label)
    )
    return loss
