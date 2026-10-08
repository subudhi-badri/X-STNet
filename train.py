# -*- coding: utf-8 -*-
"""Training loop + entry point.   Run:  python train.py"""

import warnings

import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import build_loaders, make_binary_labels
from model import MDSNet
from losses import compute_loss
from pruning import (count_parameters, print_model_parameters,
                     apply_pruning_progressive, apply_masks, print_pruning_stats)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def train(model, train_loader, val_loader,
          num_epochs=50, save_path="mdsnet_withprune.pth",
          prune_per_round=0.05):

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=6e-6, weight_decay=8e-4)

    best_val_loss  = float('inf')
    best_accuracy  = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    pruning_epochs    = set(range(10, num_epochs, 10))  # {10,20,30,40}
    pruning_round     = 0
    masks             = {}  # empty until first pruning round

    print_model_parameters(model, "Initial Model Parameters")

    for epoch in range(num_epochs):

        # ---- LR Schedule ----
        if epoch == 10:
            optimizer.param_groups[0]['lr'] = 6e-6
        elif epoch == 20:
            optimizer.param_groups[0]['lr'] = 6e-6
        elif epoch == 30:
            optimizer.param_groups[0]['lr'] = 6e-6
        elif epoch >= 40:
            optimizer.param_groups[0]['lr'] = 6e-6

        # ---- Pruning every 10 epochs ----
        if epoch in pruning_epochs:
            print(f"\n--- Pruning at Epoch {epoch+1} "
                  f"(Round {pruning_round+1}) | {prune_per_round*100:.1f}% of remaining nonzero ---")
            print_model_parameters(model, "Before Pruning")

            model, masks = apply_pruning_progressive(
                model, masks,
                round_idx       = pruning_round,
                prune_per_round = prune_per_round,
            )
            print_pruning_stats(model)
            print_model_parameters(model, "After Pruning")
            print("Fine-tuning continues...\n")
            pruning_round += 1

        # ---- Train ----
        model.train()
        t_loss = t_correct = t_total = 0

        for visuals, specs, batch_lbl, a_lens in tqdm(train_loader,
                                              desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            visuals, specs, batch_lbl = (
                visuals.to(device), specs.to(device), batch_lbl.to(device)
            )
            video_label, audio_label = make_binary_labels(batch_lbl)

            optimizer.zero_grad()

            outputs, visual_out, audio_out, v_enc, a_enc, \
                v_feat, a_feat, fmap, pv, pa = model(visuals, specs, a_lens)

            loss = compute_loss(outputs, visual_out, audio_out, v_enc, a_enc,
                                v_feat, a_feat, batch_lbl, video_label, audio_label)

            loss.backward()
            optimizer.step()

            if masks:
                apply_masks(model, masks)

            t_loss    += loss.item()
            preds      = outputs.argmax(dim=1)
            t_correct += (preds == batch_lbl).sum().item()
            t_total   += batch_lbl.size(0)

        avg_train_loss = t_loss / len(train_loader)
        train_acc      = t_correct / t_total

        # ---- Validation ----
        model.eval()
        v_loss = v_correct = v_total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for visuals, specs, batch_lbl, a_lens in tqdm(val_loader,
                                                  desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                visuals, specs, batch_lbl = (
                    visuals.to(device), specs.to(device), batch_lbl.to(device)
                )
                video_label, audio_label = make_binary_labels(batch_lbl)

                outputs, visual_out, audio_out, v_enc, a_enc, \
                    v_feat, a_feat, fmap, pv, pa = model(visuals, specs, a_lens)

                loss = compute_loss(outputs, visual_out, audio_out, v_enc, a_enc,
                                    v_feat, a_feat, batch_lbl, video_label, audio_label)

                v_loss    += loss.item()
                preds      = outputs.argmax(dim=1)
                v_correct += (preds == batch_lbl).sum().item()
                v_total   += batch_lbl.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_lbl.cpu().numpy())

        avg_val_loss = v_loss / len(val_loader)
        val_acc      = v_correct / v_total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} "
              f"| LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # ---- Confusion Matrix ----
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
        ConfusionMatrixDisplay(
            cm, display_labels=["Real-Real", "Fake-Fake", "Real-Fake", "Fake-Real"]
        ).plot(cmap="Blues")
        plt.title(f"Confusion Matrix Epoch {epoch+1}")
        plt.savefig(f"confusion_matrix_epoch_{epoch+1}.png", dpi=150)
        plt.close()

        # ---- Save Best Models ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path.replace(".pth", "_bestloss.pth"))

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), save_path.replace(".pth", "_bestacc.pth"))

    print_model_parameters(model, "Final Model Parameters")
    print("Training complete.")
    return train_losses, val_losses, train_accs, val_accs


def plot_curves(train_losses, val_losses, train_accs, val_accs, path="training_curves1.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses,   'r-x', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss over Epochs'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs,   'r-x', label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


if __name__ == "__main__":
    torch.cuda.empty_cache()

    train_loader, val_loader = build_loaders(batch_size=32, num_workers=4)

    model      = MDSNet(lstm_seq_len=None, hidden_size=256, num_layers=1)
    num_epochs = 50
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,} ({total/1e6:.2f}M)")
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.2f}M)")

    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader,
        num_epochs       = num_epochs,
        save_path        = "corrected_3D.pth",
        prune_per_round  = 0.10,
    )

    plot_curves(train_losses, val_losses, train_accs, val_accs)
