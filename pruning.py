# -*- coding: utf-8 -*-
"""Pruning utilities: global L1 unstructured pruning with persistent masks."""

import torch.nn as nn
import torch.nn.utils.prune as prune

PRUNABLE = (nn.Conv2d, nn.Conv3d, nn.Linear)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_parameters(model, title=""):
    total_w  = sum(m.weight.nelement()
                   for m in model.modules() if isinstance(m, PRUNABLE))
    zero_w   = sum((m.weight == 0).sum().item()
                   for m in model.modules() if isinstance(m, PRUNABLE))
    nonzero_w = total_w - zero_w
    sparsity  = 100.0 * zero_w / total_w if total_w > 0 else 0.0
    print(f"\n{title}")
    print(f"  Total prunable weights  : {total_w:,}")
    print(f"  Nonzero prunable weights: {nonzero_w:,}")
    print(f"  Zero prunable weights   : {zero_w:,}")
    print(f"  Sparsity                : {sparsity:.2f}%")
    return total_w, nonzero_w, sparsity


def build_pruning_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, PRUNABLE):
            masks[name] = (module.weight.data != 0).float()
    return masks


def apply_masks(model, masks):
    for name, module in model.named_modules():
        if isinstance(module, PRUNABLE):
            if name in masks:
                module.weight.data.mul_(masks[name])


def apply_pruning_progressive(model, masks, round_idx, prune_per_round=0.05):
    params_to_prune = [
        (m, 'weight')
        for m in model.modules()
        if isinstance(m, PRUNABLE)
    ]

    total_weights  = sum(m.weight.nelement()          for m, _ in params_to_prune)
    zero_before    = sum((m.weight == 0).sum().item() for m, _ in params_to_prune)
    nonzero_before = total_weights - zero_before

    new_zeros_target = int(prune_per_round * nonzero_before)
    effective_amount = (zero_before + new_zeros_target) / total_weights

    print(f"  Pruning round         : {round_idx + 1}")
    print(f"  Round prune rate      : {prune_per_round*100:.1f}% of nonzero weights")
    print(f"  Nonzero before        : {nonzero_before:,}")
    print(f"  New zeros target      : {new_zeros_target:,}")
    print(f"  Effective global amt  : {effective_amount:.4f}")

    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=effective_amount,
    )
    for module, _ in params_to_prune:
        prune.remove(module, 'weight')

    masks = build_pruning_masks(model)
    return model, masks


def print_pruning_stats(model):
    total_p = pruned_p = 0
    for name, module in model.named_modules():
        if isinstance(module, PRUNABLE):
            total  = module.weight.nelement()
            zeros  = (module.weight == 0).sum().item()
            print(f"  {name}: {100.*zeros/total:.2f}% sparsity ({zeros}/{total})")
            total_p  += total
            pruned_p += zeros
    print(f"  Global sparsity: {100.*pruned_p/total_p:.2f}%")
