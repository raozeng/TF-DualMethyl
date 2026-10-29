import argparse
import json
import os
import shutil
import time
import torch
from torch.utils.data import DataLoader
from dataset import DNADataset, ensure_tsv_from_pos_neg
from model import Ultimate_Fusion, Config
from train import train_kfold
from test import test_ensemble


def get_args():
    parser = argparse.ArgumentParser(description="TF-DualMethyl (Ultimate_Fusion) for DNA Methylation (5-Fold CV)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train.tsv and test.tsv")
    parser.add_argument("--dataset_category", type=str, default="UnknownDataset", help="Dataset Category Name (e.g. 4mC)")
    parser.add_argument("--species_name", type=str, required=True, help="Species Name")
    parser.add_argument("--model_name", type=str, default="Ultimate_Fusion", help="Model Name Identifier")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=41, help="Max length of sequences")
    parser.add_argument("--save_dir", type=str, default="./result", help="Base Dir to save models/metrics")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Run mode (train or test)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--force_retrain", action="store_true", help="Ignore existing checkpoints and retrain from Fold 1 Epoch 1")
    parser.add_argument("--save_all_epochs", action="store_true", help="Also archive every epoch's state_dict under epoch_models/")
    return parser.parse_args()


def write_run_status(model_dir, args, status):
    os.makedirs(model_dir, exist_ok=True)
    status_path = os.path.join(model_dir, "run_status.json")
    tmp_path = status_path + ".tmp"
    payload = {
        "pipeline": "model_plus",
        "model_name": args.model_name,
        "dataset_category": args.dataset_category,
        "species_name": args.species_name,
        "status": status,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, status_path)


def has_active_checkpoints(model_dir):
    if not os.path.exists(model_dir):
        return False
    for k in range(1, 6):
        if os.path.exists(os.path.join(model_dir, f"checkpoint_latest_fold_{k}.pt")):
            return True
    return False


def clear_checkpoints_for_fresh_run(model_dir):
    if not os.path.exists(model_dir):
        return
    for fname in os.listdir(model_dir):
        if (
            fname.startswith("checkpoint_latest_fold_")
            or fname.startswith("checkpoint_prev_fold_")
            or fname.startswith("model_latest_fold_")
            or fname.startswith("epoch_metrics_")
            or fname == "run_status.json"
        ):
            fpath = os.path.join(model_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)


def cleanup_completed_run_files(model_dir, model_name):
    """Remove temporary resume checkpoints and partial .tmp files once all 5 folds & test finish."""
    if not os.path.exists(model_dir):
        return
    for fname in os.listdir(model_dir):
        fpath = os.path.join(model_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if (
            fname.endswith(".tmp")
            or fname.startswith("checkpoint_prev_fold_")
            or fname.startswith("model_latest_fold_")
        ):
            try:
                os.remove(fpath)
            except OSError:
                pass
        elif fname.endswith("_epoch_1.pth") and not (
            model_name == "Ultimate_Fusion" and fname == "model_fold_1_epoch_1.pth"
        ):
            try:
                os.remove(fpath)
            except OSError:
                pass


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    ensure_tsv_from_pos_neg(args.data_dir)
    train_file = os.path.join(args.data_dir, "train.tsv")
    test_file = os.path.join(args.data_dir, "test.tsv")

    if args.mode == "train":
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Missing {train_file}")

        print(f"Loading training dataset from {train_file}...")
        full_train_dataset = DNADataset(tsv_file=train_file, max_len=args.max_len)

        config = Config()

        model_dir = os.path.join(args.save_dir, args.model_name, args.dataset_category, args.species_name)
        backup_dir = model_dir + "_backup"

        resuming_interrupted = (not args.force_retrain) and has_active_checkpoints(model_dir)

        if args.force_retrain and os.path.exists(model_dir):
            if os.path.exists(os.path.join(model_dir, "test_metrics.txt")):
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(model_dir, backup_dir)
                print(f"[FORCE RETRAIN] Backed up previous completed results to {backup_dir}")
            clear_checkpoints_for_fresh_run(model_dir)
        elif resuming_interrupted:
            print(f"[RESUME] Detected existing per-epoch checkpoints in {model_dir}. Will resume training automatically.")
        elif os.path.exists(model_dir):
            if os.path.exists(os.path.join(model_dir, "test_metrics.txt")):
                if os.path.exists(backup_dir):
                    shutil.rmtree(backup_dir)
                shutil.copytree(model_dir, backup_dir)
                print(f"Existing completed results found. Created backup at {backup_dir}")

        write_run_status(model_dir, args, status="in_progress")

        trained_dir = train_kfold(
            full_dataset=full_train_dataset,
            device=device,
            config=config,
            args=args
        )

        is_improved = True
        if os.path.exists(test_file):
            print(f"Evaluating ensemble model on separate test set {test_file}...")
            test_dataset = DNADataset(tsv_file=test_file, max_len=args.max_len)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            metrics = test_ensemble(model_dir=trained_dir, test_loader=test_loader, device=device, args=args)
            is_improved = metrics.get('is_improved', True)

        if not is_improved and os.path.exists(backup_dir):
            print("Result did not improve over previous backup. Restoring previous best test_metrics.txt & model_fold_*.pth while keeping new epoch logs...")
            for fname in os.listdir(backup_dir):
                if fname.startswith("epoch_metrics_") or fname.startswith("checkpoint_") or fname == "run_status.json":
                    continue
                src_f = os.path.join(backup_dir, fname)
                dst_f = os.path.join(model_dir, fname)
                if os.path.isfile(src_f):
                    shutil.copy2(src_f, dst_f)
            shutil.rmtree(backup_dir)
            print("Restoration complete. Previous better models and new per-epoch logs are both preserved.")
        else:
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
                print("New result recorded! Backup removed.")

        cleanup_completed_run_files(model_dir, args.model_name)
        write_run_status(model_dir, args, status="completed")

    elif args.mode == "test":
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Missing {test_file}")

        print(f"Loading test dataset from {test_file}...")
        test_dataset = DNADataset(tsv_file=test_file, max_len=args.max_len)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model_dir = os.path.join(args.save_dir, args.model_name, args.dataset_category, args.species_name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} not found. Please train first!")

        test_ensemble(model_dir=model_dir, test_loader=test_loader, device=device, args=args)


if __name__ == "__main__":
    main()
