import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import psutil
import numpy as np
from scipy.ndimage import label
from models.model_config import setup_model_cfg
from models.solar_segformer import SolarSegFormer, weighted_segmentation_loss
from dataset.solar_dataset import SolarDataset, split_solar_dataset

def compute_instance_dice(pred_mask, gt_mask):
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    return (2. * intersection + 1) / (union + 1)

def validate(model, val_loader, device):
    model.eval()
    total_dice = 0
    total_frag = 0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            targets = batch['mask'].to(device).squeeze(1).cpu().numpy()

            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(len(targets)):
                dice = compute_instance_dice(preds[i], targets[i])
                _, num_pred = label(preds[i])
                _, num_gt = label(targets[i])
                frag = num_pred / (num_gt + 1e-6)

                total_dice += dice
                total_frag += frag
                count += 1

    return total_dice / count, total_frag / count

def get_system_metrics():
    metrics = {}
    metrics['cpu_usage'] = psutil.cpu_percent()
    metrics['ram_usage'] = psutil.virtual_memory().percent
    if torch.cuda.is_available():
        metrics['gpu_vram_allocated'] = torch.cuda.memory_allocated() / 1024**2
        metrics['gpu_vram_reserved'] = torch.cuda.memory_reserved() / 1024**2
    return metrics

def save_checkpoint(state, filename="checkpoint.pth"):
    try:
        torch.save(state, filename)
        print(f"Checkpoint successfully saved to {filename}")
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")

def load_checkpoint(device="cuda"):
    # 1. Try to load the full training checkpoint first
    ckpt_path = "checkpoint.pth"
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'epoch', 'best_dice']
            if all(k in checkpoint for k in required_keys):
                print(f"Full checkpoint loaded from {ckpt_path}. Resuming training...")
                return checkpoint
            else:
                print(f"Checkpoint at {ckpt_path} is incomplete. Checking for best model...")
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    # 2. Fallback: Try to load only the best model weights
    best_path = "best_model.pth"
    if os.path.exists(best_path):
        try:
            weights = torch.load(best_path, map_location=device, weights_only=True)
            print(f"Full checkpoint not found. Loading best weights from {best_path} and starting from Epoch 0...")
            return {"model_state_dict": weights}
        except Exception as e:
            print(f"Error loading {best_path}: {e}")

    print("No checkpoints or best models found. Starting training from scratch.")
    return None

def main():
    DATA_DIR = '/media/nightking/WD-SN570/deep_learning_projects/solar_filament_segmentation_challenge_2026/dataset/MAGFiLO_1.0_Kaggle_2026'
    JSON_PATH = os.path.join(DATA_DIR, 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    IMG_DIR = os.path.join(DATA_DIR, 'train/train_images')

    cfg = setup_model_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    train_ids, val_ids = split_solar_dataset(data)

    train_dataset = SolarDataset(JSON_PATH, IMG_DIR, image_ids=train_ids)
    val_dataset = SolarDataset(JSON_PATH, IMG_DIR, image_ids=val_ids)

    train_loader = DataLoader(train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=4)

    # Initialize Model, Optimizer, and Scheduler first
    model = SolarSegFormer(model_name=cfg.MODEL.BACKBONE, num_labels=cfg.MODEL.NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # --- RESUME LOGIC ---
    start_epoch = 0
    best_dice = 0
    checkpoint = load_checkpoint(device=device)

    if checkpoint:
        # Load model weights (Always present if checkpoint is returned)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state restored.")

        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state restored.")

        # Load epoch and best dice if available
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_dice = checkpoint.get('best_dice', 0)
        print(f"Resuming training from epoch {start_epoch}")

    # MLflow Setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Solar_SegFormer_ROCm")

    # Use existing run_id if resuming to keep logs in one place
    run_id = checkpoint['run_id'] if checkpoint and 'run_id' in checkpoint else None
    mlflow_run = mlflow.start_run(run_id=run_id)
    mlflow.log_params(vars(cfg))

    for epoch in range(start_epoch, 100):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            targets = batch['mask'].to(device).squeeze(1)
            disk_masks = batch['disk_mask'].to(device).squeeze(1)

            optimizer.zero_grad()
            logits = model(images, disk_mask=disk_masks)
            loss = weighted_segmentation_loss(logits, targets, disk_masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Log Metrics every 10 batches
            if i % 10 == 0:
                # Extract current learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']
                sys_metrics = get_system_metrics()

                mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_loader) + i)
                mlflow.log_metric("learning_rate", current_lr, step=epoch * len(train_loader) + i)
                for k, v in sys_metrics.items():
                    mlflow.log_metric(k, v, step=epoch * len(train_loader) + i)

        avg_loss = epoch_loss / len(train_loader)
        val_dice, val_frag = validate(model, val_loader, device)

        # Step scheduler based on validation Dice
        scheduler.step(val_dice)

        # Log epoch results
        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_dice", val_dice, step=epoch)
        mlflow.log_metric("val_frag", val_frag, step=epoch)

        # Log final LR for the epoch
        final_lr = optimizer.param_groups[0]['lr']
        mlflow.log_metric("epoch_lr", final_lr, step=epoch)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f} | Val Frag: {val_frag:.4f} | LR: {final_lr:.6f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")

        # Foolproof checkpointing at the end of every epoch
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': best_dice,
            'run_id': mlflow_run.info.run_id
        })

    mlflow.end_run()

if __name__ == "__main__":
    main()
