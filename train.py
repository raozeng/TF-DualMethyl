import csv
import json
import os
import shutil
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from torch.utils.data import DataLoader, Subset
from model import get_model_by_name, Config


EPOCH_CSV_COLUMNS = [
    "fold",
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "train_precision",
    "train_sn",
    "train_sp",
    "train_f1",
    "train_mcc",
    "train_auc",
    "train_auprc",
    "train_tp",
    "train_tn",
    "train_fp",
    "train_fn",
    "val_loss",
    "val_acc",
    "val_precision",
    "val_sn",
    "val_sp",
    "val_f1",
    "val_mcc",
    "val_auc",
    "val_auprc",
    "val_tp",
    "val_tn",
    "val_fp",
    "val_fn",
    "best_val_acc",
    "best_epoch",
    "is_best",
    "epochs_no_improve",
    "epoch_time_sec",
]


def calculate_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    else:
        auc = float('nan')
        auprc = float('nan')
        fpr, tpr, pr_precision, pr_recall = [], [], [], []

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'auc': auc,
        'auprc': auprc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'fpr': fpr.tolist() if isinstance(fpr, np.ndarray) else fpr,
        'tpr': tpr.tolist() if isinstance(tpr, np.ndarray) else tpr,
        'pr_precision': pr_precision.tolist() if isinstance(pr_precision, np.ndarray) else pr_precision,
        'pr_recall': pr_recall.tolist() if isinstance(pr_recall, np.ndarray) else pr_recall
    }


def atomic_torch_save(obj, target_path):
    """
    Atomically saves a PyTorch object to target_path via a temporary file + os.replace
    so unexpected power outages or process kills never leave a corrupted checkpoint.
    Automatically removes partial .tmp files if disk-full or I/O errors occur.
    """
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    tmp_path = target_path + ".tmp"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def load_checkpoint_safely(latest_path, prev_path, device):
    """
    Attempts to load the latest checkpoint; falls back to the previous epoch's
    checkpoint if the latest file was somehow damaged.
    """
    for candidate in (latest_path, prev_path):
        if candidate and os.path.exists(candidate):
            try:
                ckpt = torch.load(candidate, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and "epoch" in ckpt:
                    return ckpt, candidate
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint {candidate}: {e}")
    return None, None


def truncate_fold_logs_UpTo_epoch(fold_csv_path, fold_jsonl_path, max_epoch):
    """
    Ensures per-fold CSV and JSONL logs only retain completed epochs <= max_epoch
    when resuming from an interrupted epoch.
    """
    if os.path.exists(fold_csv_path):
        kept_rows = []
        try:
            with open(fold_csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row.get("epoch", 0)) <= max_epoch:
                        kept_rows.append(row)
        except Exception as e:
            print(f"[WARNING] Could not parse existing {fold_csv_path} during resume cleanup: {e}")

        tmp_csv = fold_csv_path + ".tmp"
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
            writer.writeheader()
            for r in kept_rows:
                writer.writerow({k: r.get(k, "") for k in EPOCH_CSV_COLUMNS})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_csv, fold_csv_path)

    if os.path.exists(fold_jsonl_path):
        kept_lines = []
        try:
            with open(fold_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        obj = json.loads(line_s)
                        if int(obj.get("epoch", 0)) <= max_epoch:
                            kept_lines.append(line_s)
                    except Exception:
                        continue
        except Exception as e:
            print(f"[WARNING] Could not parse existing {fold_jsonl_path} during resume cleanup: {e}")

        tmp_jsonl = fold_jsonl_path + ".tmp"
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for l in kept_lines:
                f.write(l + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_jsonl, fold_jsonl_path)


def sync_unified_epoch_logs(save_dir, n_splits=5):
    """
    Synchronizes `epoch_metrics_all_folds.csv` and `train_log.txt` from the per-fold
    `epoch_metrics_fold_{1..5}.csv` files so multi-fold logs remain clean and complete
    even across power-outage resumes.
    """
    all_rows = []
    for fold_num in range(1, n_splits + 1):
        fold_csv = os.path.join(save_dir, f"epoch_metrics_fold_{fold_num}.csv")
        if os.path.exists(fold_csv):
            try:
                with open(fold_csv, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_rows.append(row)
            except Exception:
                pass

    all_csv_path = os.path.join(save_dir, "epoch_metrics_all_folds.csv")
    tmp_all_csv = all_csv_path + ".tmp"
    with open(tmp_all_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in EPOCH_CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_all_csv, all_csv_path)

    unified_log_path = os.path.join(save_dir, "train_log.txt")
    tmp_log = unified_log_path + ".tmp"
    with open(tmp_log, "w", encoding="utf-8") as f:
        header_cols = [
            "Fold", "Epoch", "LR",
            "Train_Loss", "Train_Acc", "Train_Precision", "Train_SN", "Train_SP", "Train_F1", "Train_MCC", "Train_AUC", "Train_AUPRC",
            "Val_Loss", "Val_Acc", "Val_Precision", "Val_SN", "Val_SP", "Val_F1", "Val_MCC", "Val_AUC", "Val_AUPRC",
            "Best_Val_Acc", "Best_Epoch", "Is_Best", "Time_s"
        ]
        f.write("\t".join(header_cols) + "\n")
        for r in all_rows:
            vals = [
                str(r.get("fold", "")),
                str(r.get("epoch", "")),
                str(r.get("lr", "")),
                str(r.get("train_loss", "")),
                str(r.get("train_acc", "")),
                str(r.get("train_precision", "")),
                str(r.get("train_sn", "")),
                str(r.get("train_sp", "")),
                str(r.get("train_f1", "")),
                str(r.get("train_mcc", "")),
                str(r.get("train_auc", "")),
                str(r.get("train_auprc", "")),
                str(r.get("val_loss", "")),
                str(r.get("val_acc", "")),
                str(r.get("val_precision", "")),
                str(r.get("val_sn", "")),
                str(r.get("val_sp", "")),
                str(r.get("val_f1", "")),
                str(r.get("val_mcc", "")),
                str(r.get("val_auc", "")),
                str(r.get("val_auprc", "")),
                str(r.get("best_val_acc", "")),
                str(r.get("best_epoch", "")),
                str(r.get("is_best", "")),
                str(r.get("epoch_time_sec", "")),
            ]
            f.write("\t".join(vals) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_log, unified_log_path)


def write_fold_best_metrics(fold_txt_path, fold_num, best_metrics):
    if not best_metrics:
        return
    tmp_path = fold_txt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(f"--- Fold {fold_num} Best Results ---\n")
        f.write(f"Accuracy: {best_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {best_metrics['precision']:.4f}\n")
        f.write(f"Recall: {best_metrics['recall']:.4f}\n")
        f.write(f"Specificity: {best_metrics['specificity']:.4f}\n")
        f.write(f"F1 Score: {best_metrics['f1']:.4f}\n")
        f.write(f"MCC: {best_metrics['mcc']:.4f}\n")
        f.write(f"AUC: {best_metrics['auc']:.4f}\n")
        f.write(f"AUPRC: {best_metrics.get('auprc', 0.0):.4f}\n\n")

        f.write("--- ROC Data ---\n")
        f.write("FPR: " + ",".join(map(str, best_metrics.get('fpr', []))) + "\n")
        f.write("TPR: " + ",".join(map(str, best_metrics.get('tpr', []))) + "\n\n")

        f.write("--- PR Data ---\n")
        f.write("Precision_Curve: " + ",".join(map(str, best_metrics.get('pr_precision', []))) + "\n")
        f.write("Recall_Curve: " + ",".join(map(str, best_metrics.get('pr_recall', []))) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, fold_txt_path)


def train_kfold(full_dataset, device, config, args):
    """
    5-Fold Stratified Cross Validation Training with:
      1. Per-epoch full metrics logging (Train & Val ACC, Precision, SN, SP, F1, MCC, AUC, AUPRC, TP/TN/FP/FN, Loss, LR)
      2. Atomic per-epoch model & checkpoint saving (model_latest_fold_k.pth, checkpoint_latest_fold_k.pt, model_fold_k.pth)
      3. Automatic power-outage / interruption resume from the exact latest fold and epoch
    """
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("Extracting labels for Stratified K-Fold...")
    labels = [full_dataset[i][1].item() for i in range(len(full_dataset))]

    save_dir = os.path.join(args.save_dir, args.model_name, args.dataset_category, args.species_name)
    os.makedirs(save_dir, exist_ok=True)

    fold_metrics_summary = []

    # Respect args.patience; if <= 0, default to 15 epochs so ablation/main models do not prematurely stop at epoch 2
    effective_patience = args.patience if getattr(args, 'patience', 0) > 0 else 15
    force_retrain = getattr(args, 'force_retrain', False)
    save_all_epochs = getattr(args, 'save_all_epochs', False)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_num = fold + 1
        print(f"\n{'-'*20} Fold {fold_num}/{n_splits} (patience={effective_patience}) {'-'*20}")

        best_model_path = os.path.join(save_dir, f'model_fold_{fold_num}.pth')
        latest_model_path = os.path.join(save_dir, f'model_latest_fold_{fold_num}.pth')
        epoch1_save_path = os.path.join(save_dir, f'model_fold_{fold_num}_epoch_1.pth')
        ckpt_latest_path = os.path.join(save_dir, f'checkpoint_latest_fold_{fold_num}.pt')
        ckpt_prev_path = os.path.join(save_dir, f'checkpoint_prev_fold_{fold_num}.pt')
        fold_csv_path = os.path.join(save_dir, f'epoch_metrics_fold_{fold_num}.csv')
        fold_jsonl_path = os.path.join(save_dir, f'epoch_metrics_fold_{fold_num}.jsonl')
        fold_txt_path = os.path.join(save_dir, f'{fold_num}_val_metrics.txt')

        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)

        model = get_model_by_name(args.model_name, config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        start_epoch = 0
        best_val_acc = -1.0
        best_epoch = 0
        best_metrics = {}
        epochs_no_improve = 0

        if not force_retrain:
            ckpt, loaded_from = load_checkpoint_safely(ckpt_latest_path, ckpt_prev_path, device)
            if ckpt is not None:
                if ckpt.get("completed", False) and os.path.exists(best_model_path):
                    best_val_acc = float(ckpt.get("best_val_acc", -1.0))
                    best_epoch = int(ckpt.get("best_epoch", 0))
                    best_metrics = ckpt.get("best_metrics", {})
                    print(
                        f"[RESUME] Fold {fold_num}/{n_splits} already completed at Epoch {ckpt.get('epoch')} "
                        f"(Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}). Skipping fold training."
                    )
                    if not os.path.exists(fold_txt_path) and best_metrics:
                        write_fold_best_metrics(fold_txt_path, fold_num, best_metrics)
                    sync_unified_epoch_logs(save_dir, n_splits=n_splits)
                    if best_metrics:
                        fold_metrics_summary.append({k: v for k, v in best_metrics.items() if not isinstance(v, list)})
                    continue
                elif int(ckpt.get("epoch", 0)) > 0:
                    start_epoch = int(ckpt["epoch"])
                    model.load_state_dict(ckpt["model_state_dict"])
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    best_val_acc = float(ckpt.get("best_val_acc", -1.0))
                    best_epoch = int(ckpt.get("best_epoch", 0))
                    best_metrics = ckpt.get("best_metrics", {})
                    epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
                    truncate_fold_logs_UpTo_epoch(fold_csv_path, fold_jsonl_path, start_epoch)
                    sync_unified_epoch_logs(save_dir, n_splits=n_splits)
                    print(
                        f"[RESUME] Restored Fold {fold_num}/{n_splits} from {os.path.basename(loaded_from)} "
                        f"(Completed Epoch {start_epoch}/{args.epochs}, Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}, "
                        f"No-improve count: {epochs_no_improve}/{effective_patience}). Continuing at Epoch {start_epoch + 1}..."
                    )

        if start_epoch == 0:
            # Initialize fresh per-fold CSV and JSONL files
            with open(fold_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())
            with open(fold_jsonl_path, "w", encoding="utf-8") as f:
                pass

        for epoch in range(start_epoch, args.epochs):
            epoch_num = epoch + 1
            epoch_t0 = time.time()
            current_lr = float(optimizer.param_groups[0]['lr'])

            model.train()
            train_loss = 0.0
            train_preds, train_targets, train_probs = [], [], []

            for seqs, batch_labels in train_loader:
                seqs, batch_labels = seqs.to(device), batch_labels.to(device)
                optimizer.zero_grad()

                logits = model(seqs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                with torch.no_grad():
                    probs = torch.softmax(logits.detach(), dim=1)[:, 1]
                    preds = torch.argmax(logits.detach(), dim=1)
                    train_probs.extend(probs.cpu().numpy())
                    train_preds.extend(preds.cpu().numpy())
                    train_targets.extend(batch_labels.cpu().numpy())

            scheduler.step()

            train_loss_avg = train_loss / max(len(train_loader), 1)
            train_metrics = calculate_metrics(train_targets, train_preds, train_probs)

            model.eval()
            val_loss = 0.0
            val_preds, val_targets, val_probs = [], [], []

            with torch.no_grad():
                for seqs, batch_labels in val_loader:
                    seqs, batch_labels = seqs.to(device), batch_labels.to(device)
                    logits = model(seqs)
                    loss = criterion(logits, batch_labels)

                    val_loss += loss.item()
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    preds = torch.argmax(logits, dim=1)

                    val_probs.extend(probs.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(batch_labels.cpu().numpy())

            val_loss_avg = val_loss / max(len(val_loader), 1)
            val_metrics = calculate_metrics(val_targets, val_preds, val_probs)
            val_acc = val_metrics['accuracy']
            epoch_elapsed = time.time() - epoch_t0

            if epoch == 0 and fold_num == 1 and args.model_name == "Ultimate_Fusion":
                atomic_torch_save(model.state_dict(), epoch1_save_path)
                print(f"   Saved Epoch 1 weights to {epoch1_save_path}")

            is_best = 0
            if val_acc > best_val_acc:
                is_best = 1
                best_val_acc = val_acc
                best_epoch = epoch_num
                best_metrics = val_metrics
                epochs_no_improve = 0
                atomic_torch_save(model.state_dict(), best_model_path)
                write_fold_best_metrics(fold_txt_path, fold_num, best_metrics)
            else:
                epochs_no_improve += 1

            will_stop = (epoch_num >= args.epochs) or (epochs_no_improve >= effective_patience)

            # 1. Record complete per-epoch metrics to CSV and JSONL
            row_dict = {
                "fold": fold_num,
                "epoch": epoch_num,
                "lr": f"{current_lr:.8f}",
                "train_loss": f"{train_loss_avg:.6f}",
                "train_acc": f"{train_metrics['accuracy']:.6f}",
                "train_precision": f"{train_metrics['precision']:.6f}",
                "train_sn": f"{train_metrics['recall']:.6f}",
                "train_sp": f"{train_metrics['specificity']:.6f}",
                "train_f1": f"{train_metrics['f1']:.6f}",
                "train_mcc": f"{train_metrics['mcc']:.6f}",
                "train_auc": f"{train_metrics['auc']:.6f}",
                "train_auprc": f"{train_metrics['auprc']:.6f}",
                "train_tp": train_metrics['tp'],
                "train_tn": train_metrics['tn'],
                "train_fp": train_metrics['fp'],
                "train_fn": train_metrics['fn'],
                "val_loss": f"{val_loss_avg:.6f}",
                "val_acc": f"{val_metrics['accuracy']:.6f}",
                "val_precision": f"{val_metrics['precision']:.6f}",
                "val_sn": f"{val_metrics['recall']:.6f}",
                "val_sp": f"{val_metrics['specificity']:.6f}",
                "val_f1": f"{val_metrics['f1']:.6f}",
                "val_mcc": f"{val_metrics['mcc']:.6f}",
                "val_auc": f"{val_metrics['auc']:.6f}",
                "val_auprc": f"{val_metrics['auprc']:.6f}",
                "val_tp": val_metrics['tp'],
                "val_tn": val_metrics['tn'],
                "val_fp": val_metrics['fp'],
                "val_fn": val_metrics['fn'],
                "best_val_acc": f"{best_val_acc:.6f}",
                "best_epoch": best_epoch,
                "is_best": is_best,
                "epochs_no_improve": epochs_no_improve,
                "epoch_time_sec": f"{epoch_elapsed:.3f}",
            }

            with open(fold_csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=EPOCH_CSV_COLUMNS)
                writer.writerow(row_dict)
                f.flush()
                os.fsync(f.fileno())

            jsonl_record = {
                "fold": fold_num,
                "epoch": epoch_num,
                "lr": current_lr,
                "train": {
                    "loss": train_loss_avg,
                    "accuracy": train_metrics['accuracy'],
                    "precision": train_metrics['precision'],
                    "sn_recall": train_metrics['recall'],
                    "sp_specificity": train_metrics['specificity'],
                    "f1": train_metrics['f1'],
                    "mcc": train_metrics['mcc'],
                    "auc": train_metrics['auc'],
                    "auprc": train_metrics['auprc'],
                    "tp": train_metrics['tp'],
                    "tn": train_metrics['tn'],
                    "fp": train_metrics['fp'],
                    "fn": train_metrics['fn'],
                },
                "val": {
                    "loss": val_loss_avg,
                    "accuracy": val_metrics['accuracy'],
                    "precision": val_metrics['precision'],
                    "sn_recall": val_metrics['recall'],
                    "sp_specificity": val_metrics['specificity'],
                    "f1": val_metrics['f1'],
                    "mcc": val_metrics['mcc'],
                    "auc": val_metrics['auc'],
                    "auprc": val_metrics['auprc'],
                    "tp": val_metrics['tp'],
                    "tn": val_metrics['tn'],
                    "fp": val_metrics['fp'],
                    "fn": val_metrics['fn'],
                },
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "is_best": bool(is_best),
                "epochs_no_improve": epochs_no_improve,
                "epoch_time_sec": round(epoch_elapsed, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(fold_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            sync_unified_epoch_logs(save_dir, n_splits=n_splits)

            # 2. Atomically save per-epoch trained model & resumable checkpoint
            best_metrics_scalar = {k: v for k, v in best_metrics.items() if not isinstance(v, list)}
            if save_all_epochs:
                epoch_model_dir = os.path.join(save_dir, "epoch_models")
                atomic_torch_save(
                    model.state_dict(),
                    os.path.join(epoch_model_dir, f"model_fold_{fold_num}_epoch_{epoch_num:03d}.pth")
                )

            if not will_stop:
                atomic_torch_save(model.state_dict(), latest_model_path)
                ckpt_payload = {
                    "fold": fold_num,
                    "epoch": epoch_num,
                    "completed": False,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics_scalar,
                    "epochs_no_improve": epochs_no_improve,
                    "args": vars(args) if hasattr(args, "__dict__") else {},
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                if os.path.exists(ckpt_latest_path):
                    try:
                        shutil.copy2(ckpt_latest_path, ckpt_prev_path)
                    except Exception:
                        pass
                atomic_torch_save(ckpt_payload, ckpt_latest_path)

            best_tag = " [*BEST*]" if is_best else ""
            print(
                f"Fold {fold_num} | Epoch {epoch_num:02d}/{args.epochs} | LR: {current_lr:.2e} | "
                f"Train [Loss: {train_loss_avg:.4f}, ACC: {train_metrics['accuracy']:.4f}, "
                f"SN: {train_metrics['recall']:.4f}, SP: {train_metrics['specificity']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}, MCC: {train_metrics['mcc']:.4f}, "
                f"AUC: {train_metrics['auc']:.4f}, AUPRC: {train_metrics['auprc']:.4f}] | "
                f"Val [Loss: {val_loss_avg:.4f}, ACC: {val_acc:.4f}, "
                f"SN: {val_metrics['recall']:.4f}, SP: {val_metrics['specificity']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}, MCC: {val_metrics['mcc']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f}, AUPRC: {val_metrics['auprc']:.4f}]{best_tag}",
                flush=True
            )

            if epochs_no_improve >= effective_patience:
                print(f"Early stopping triggered after {epoch_num} epochs! (Best Epoch: {best_epoch}, Best Val Acc: {best_val_acc:.4f})", flush=True)
                break

        # Fold completed: save lightweight completion marker (~1 KB) and free temporary optimizer/latest weights
        best_metrics_scalar = {k: v for k, v in best_metrics.items() if not isinstance(v, list)}
        completed_ckpt = {
            "fold": fold_num,
            "epoch": epoch_num,
            "completed": True,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics_scalar,
            "epochs_no_improve": epochs_no_improve,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_torch_save(completed_ckpt, ckpt_latest_path)
        for stale_file in (ckpt_prev_path, latest_model_path):
            if os.path.exists(stale_file):
                try:
                    os.remove(stale_file)
                except Exception:
                    pass

        print(f"Fold {fold_num} Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})", flush=True)

        if best_metrics_scalar:
            fold_metrics_summary.append(best_metrics_scalar)
        if best_metrics and any(isinstance(v, list) for v in best_metrics.values()):
            write_fold_best_metrics(fold_txt_path, fold_num, best_metrics)

    if fold_metrics_summary:
        avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics_summary])) for k in fold_metrics_summary[0].keys()}

        print(f"\n{'-'*20} 5-Fold Cross Validation Results {'-'*20}")
        cv_avg_path = os.path.join(save_dir, 'cv_average_metrics.txt')
        tmp_cv = cv_avg_path + ".tmp"
        with open(tmp_cv, 'w', encoding="utf-8") as f:
            f.write("--- 5-Fold Cross Validation Average Results ---\n")
            for k, v in avg_metrics.items():
                print(f"Average {k.capitalize()}: {v:.4f}")
                f.write(f"Average {k.capitalize()}: {v:.4f}\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_cv, cv_avg_path)

    print(f"Models, per-epoch checkpoints, and full metrics logs saved to {save_dir}")
    return save_dir
