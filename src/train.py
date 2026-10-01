"""
Training loops for BaselineCNN, CAMCNN, and Mask-CAM.

Two training procedures are provided:
    - train_baseline_or_cam: classification-only training with optional MixUp
    - train_masked:          composite loss with sparsity + smoothness + warm-up

Both save the best-validation-accuracy checkpoint and a final checkpoint,
and write the per-epoch history to a CSV for later plotting.
"""

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .losses import composite_loss, lambda_warmup
from .training_utils import mixup_data, mixup_criterion


def train_epoch(model, loader, optimizer, criterion, device, use_mixup=False):
    """One training epoch for BaselineCNN / CAMCNN."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(labels_a)
                        + (1 - lam) * predicted.eq(labels_b)).sum().item()
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader.dataset), 100.0 * correct / total


def validate(model, loader, criterion, device):
    """Validation loop (no MixUp)."""
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return val_loss / len(loader.dataset), 100.0 * correct / total


def _save_history(results_dir, model_name, epochs, train_losses, val_losses,
                  train_accs, val_accs):
    """Write per-epoch training history to a CSV."""
    pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
    }).to_csv(os.path.join(results_dir, f'{model_name}_history.csv'), index=False)


def train_baseline_or_cam(model, train_loader, val_loader, test_loader,
                          num_epochs, lr=0.001, device='cuda',
                          model_name='model', use_mixup=True,
                          checkpoint_dir='./checkpoints',
                          results_dir='./results'):
    """Full training for BaselineCNN and CAMCNN (thesis Section 3.3).

    Returns:
        train_losses, val_losses, train_accs, val_accs — lists per epoch.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, use_mixup)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'{model_name}_best.pth'))
            print(f'  --> New best model saved (Val Acc: {val_acc:.2f}%)')

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    _save_history(results_dir, model_name, num_epochs,
                  train_losses, val_losses, train_accs, val_accs)

    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, f'{model_name}_final.pth'))

    return train_losses, val_losses, train_accs, val_accs


def train_masked(model, train_loader, val_loader, test_loader, num_epochs,
                 lr=0.001, device='cuda', model_name='masked',
                 lambda_warmup_epochs=10, beta_smoothness=0.0005,
                 checkpoint_dir='./checkpoints', results_dir='./results'):
    """Full training for Mask-CAM (thesis Section 3.3.4).

    Uses the composite loss from src.losses, with a λ warm-up that linearly
    ramps from 0 to model.lambda_sparsity over the first `lambda_warmup_epochs`.

    Returns:
        train_losses, val_losses, train_accs, val_accs — lists per epoch.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    # Criterion is only used inside `validate` — composite_loss handles CE itself
    val_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_cls, correct, total = 0.0, 0.0, 0, 0

        # λ warm-up (thesis Equation 5)
        current_lambda = lambda_warmup(epoch, model.lambda_sparsity,
                                        lambda_warmup_epochs)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, mask = model(images, return_mask=True)

            loss, parts = composite_loss(
                logits, labels, mask,
                lambda_sparsity=current_lambda,
                beta_smoothness=beta_smoothness,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_cls += parts['cls'] * images.size(0)

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_cls = running_cls / len(train_loader.dataset)
        train_acc = 100.0 * correct / total

        val_loss, val_acc = validate(model, val_loader, val_criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f'{model_name}_best.pth'))
            print(f'  --> New best saved (Val Acc: {val_acc:.2f}%)')

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:2d}/{num_epochs} | λ={current_lambda:.5f} | "
              f"Total: {train_loss:.4f} | Cls: {train_cls:.4f} | "
              f"TrainAcc: {train_acc:.2f}% | ValAcc: {val_acc:.2f}%")

    _save_history(results_dir, model_name, num_epochs,
                  train_losses, val_losses, train_accs, val_accs)

    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, f'{model_name}_final.pth'))

    return train_losses, val_losses, train_accs, val_accs
