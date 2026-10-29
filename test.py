import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from model import get_model_by_name, Config


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


def test_ensemble(model_dir, test_loader, device, args):
    """
    Independent test sets using Ensemble of 5-Fold models
    """
    config = Config()
    models = []

    for i in range(1, 6):
        path = os.path.join(model_dir, f'model_fold_{i}.pth')
        if os.path.exists(path):
            model = get_model_by_name(args.model_name, config).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)

    if not models:
        for i in range(1, 6):
            path = os.path.join(model_dir, f'fold_{i}.pth')
            if os.path.exists(path):
                model = get_model_by_name(args.model_name, config).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                models.append(model)

    if not models:
        raise ValueError(f"No fold models found in {model_dir}")

    test_targets = []
    all_probs = []

    print(f"Running Ensemble Inference using {len(models)} models...")

    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            test_targets.extend(labels.cpu().numpy())

            batch_probs = []
            for model in models:
                logits = model(seqs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                batch_probs.append(probs.cpu().numpy())

            avg_batch_prob = np.mean(batch_probs, axis=0)
            all_probs.extend(avg_batch_prob)

    test_targets = np.array(test_targets)
    all_probs = np.array(all_probs)
    test_preds = (all_probs > 0.5).astype(int)

    metrics = calculate_metrics(test_targets, test_preds, all_probs)
    new_acc = metrics['accuracy']

    save_dir = model_dir
    os.makedirs(save_dir, exist_ok=True)

    test_txt_path = os.path.join(save_dir, 'test_metrics.txt')
    old_acc = -1.0
    if os.path.exists(test_txt_path):
        try:
            with open(test_txt_path, 'r', encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Accuracy:"):
                        old_acc = float(line.split(":")[1].strip())
                        break
        except Exception as e:
            print(f"Warning: Could not read old accuracy from {test_txt_path}: {e}")

    print("\n" + "=" * 40)
    print("Independent Test Results (Ensemble):")
    for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'mcc', 'auc', 'auprc']:
        print(f"{k.capitalize()}: {metrics[k]:.4f}")

    if old_acc >= 0:
        print(f"Previous Best Accuracy: {old_acc:.4f}")
    print("=" * 40)

    if new_acc >= old_acc:
        if old_acc >= 0:
            print(f"Improved! (New: {new_acc:.4f} >= Old: {old_acc:.4f}). Updating records...")
        else:
            print("First run results saved.")

        tmp_txt = test_txt_path + ".tmp"
        with open(tmp_txt, 'w', encoding="utf-8") as f:
            f.write("--- Independent Test Set Best Results (Ensemble) ---\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"Specificity: {metrics['specificity']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"MCC: {metrics['mcc']:.4f}\n")
            f.write(f"AUC: {metrics['auc']:.4f}\n")
            f.write(f"AUPRC: {metrics['auprc']:.4f}\n\n")

            f.write("--- ROC Data ---\n")
            f.write("FPR: " + ",".join(map(str, metrics['fpr'])) + "\n")
            f.write("TPR: " + ",".join(map(str, metrics['tpr'])) + "\n\n")

            f.write("--- PR Data ---\n")
            f.write("Precision_Curve: " + ",".join(map(str, metrics['pr_precision'])) + "\n")
            f.write("Recall_Curve: " + ",".join(map(str, metrics['pr_recall'])) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_txt, test_txt_path)

        # Save structured JSON summary of independent test metrics
        test_json_path = os.path.join(save_dir, 'test_metrics_full.json')
        tmp_json = test_json_path + ".tmp"
        with open(tmp_json, 'w', encoding="utf-8") as f_json:
            json.dump({
                k: metrics[k] for k in [
                    'accuracy', 'precision', 'recall', 'specificity',
                    'f1', 'mcc', 'auc', 'auprc', 'tp', 'tn', 'fp', 'fn'
                ]
            }, f_json, indent=2)
            f_json.flush()
            os.fsync(f_json.fileno())
        os.replace(tmp_json, test_json_path)

        # Save probability.txt atomically for downstream plotting & table calculation
        prob_path = os.path.join(save_dir, 'probability.txt')
        tmp_prob = prob_path + ".tmp"
        with open(tmp_prob, 'w', encoding="utf-8") as f_prob:
            f_prob.write("True_Label\tProbability\tPrediction\n")
            for t, p, pred in zip(test_targets, all_probs, test_preds):
                f_prob.write(f"{int(t)}\t{float(p):.6f}\t{int(pred)}\n")
            f_prob.flush()
            os.fsync(f_prob.fileno())
        os.replace(tmp_prob, prob_path)

        print(f"Test metrics and probability.txt updated at {save_dir}")
        metrics['is_improved'] = True
    else:
        print(f"New result ({new_acc:.4f}) is WORSE than previous ({old_acc:.4f}). Keeping old records.")
        metrics['is_improved'] = False

    return metrics
