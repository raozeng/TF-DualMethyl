import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import torch
from dataset import ensure_tsv_from_pos_neg


def resolve_dirs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check local ./data first, then ../data
    if os.path.exists(os.path.join(script_dir, "data")):
        raw_data_root = os.path.join(script_dir, "data")
    elif os.path.exists(os.path.join(os.path.dirname(script_dir), "data")):
        raw_data_root = os.path.join(os.path.dirname(script_dir), "data")
    else:
        raw_data_root = os.path.join(script_dir, "data")

    if os.path.exists(os.path.join(script_dir, "result")):
        save_dir = os.path.join(script_dir, "result")
    elif os.path.exists(os.path.join(os.path.dirname(script_dir), "result")):
        save_dir = os.path.join(os.path.dirname(script_dir), "result")
    else:
        save_dir = os.path.join(script_dir, "result")

    return script_dir, raw_data_root, save_dir


def reclaim_disk_space(save_dir):
    """
    Reclaims disk space across save_dir without losing any completed models, test metrics,
    epoch CSV logs, or active interrupted fold checkpoints:
      1. Deletes partial *.tmp files and *_backup folders.
      2. Deletes redundant checkpoint_prev_fold_*.pt and model_latest_fold_*.pth.
      3. Deletes unneeded model_fold_*_epoch_1.pth (keeping only Ultimate_Fusion fold 1 epoch 1 for t-SNE/UMAP).
      4. Slims completed checkpoint_latest_fold_*.pt (where completed=True) by stripping heavy
         model_state_dict/optimizer_state_dict while keeping lightweight fold completion metadata.
    """
    if not os.path.exists(save_dir):
        return

    reclaimed_bytes = 0

    # Pass 1: Delete pure temporary/redundant files first so disk has breathing room
    for root, dirs, files in os.walk(save_dir, topdown=True):
        for d in list(dirs):
            if d.endswith("_backup"):
                bpath = os.path.join(root, d)
                try:
                    for r2, _, f2 in os.walk(bpath):
                        for fn in f2:
                            reclaimed_bytes += os.path.getsize(os.path.join(r2, fn))
                    shutil.rmtree(bpath)
                    dirs.remove(d)
                except OSError:
                    pass

        is_ultimate = "Ultimate_Fusion" in root.replace("\\", "/").split("/")
        for fname in files:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            should_delete = False
            if (
                fname.endswith(".tmp")
                or fname.startswith("checkpoint_prev_fold_")
                or fname.startswith("model_latest_fold_")
            ):
                should_delete = True
            elif fname.endswith("_epoch_1.pth") and not (
                is_ultimate and fname == "model_fold_1_epoch_1.pth"
            ):
                should_delete = True

            if should_delete:
                try:
                    sz = os.path.getsize(fpath)
                    os.remove(fpath)
                    reclaimed_bytes += sz
                except OSError:
                    pass

    # Pass 2: Slim down completed checkpoint_latest_fold_*.pt (>50 KB) into 1 KB completion markers
    for root, _, files in os.walk(save_dir):
        for fname in files:
            if not (fname.startswith("checkpoint_latest_fold_") and fname.endswith(".pt")):
                continue
            fpath = os.path.join(root, fname)
            try:
                old_sz = os.path.getsize(fpath)
                if old_sz < 50 * 1024:
                    continue
                ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and ckpt.get("completed", False):
                    slim_ckpt = {
                        "fold": ckpt.get("fold"),
                        "epoch": ckpt.get("epoch"),
                        "completed": True,
                        "best_val_acc": ckpt.get("best_val_acc", 0.0),
                        "best_epoch": ckpt.get("best_epoch", 1),
                        "best_metrics": {
                            k: v
                            for k, v in (ckpt.get("best_metrics") or {}).items()
                            if k not in ("fpr", "tpr")
                        },
                        "early_Stop_counter": ckpt.get("early_Stop_counter", 0),
                    }
                    tmp_path = fpath + ".tmp"
                    torch.save(slim_ckpt, tmp_path)
                    os.replace(tmp_path, fpath)
                    new_sz = os.path.getsize(fpath)
                    if old_sz > new_sz:
                        reclaimed_bytes += old_sz - new_sz
            except Exception:
                # If a checkpoint was corrupted when disk filled up, remove it if model_fold_k.pth isn't finished
                pass

    try:
        usage = shutil.disk_usage(save_dir)
        free_gb = usage.free / (1024 ** 3)
        reclaimed_mb = reclaimed_bytes / (1024 ** 2)
        if reclaimed_mb > 1.0:
            print(f"[DISK CLEANUP] Reclaimed {reclaimed_mb:.1f} MB of redundant checkpoints. Free disk space: {free_gb:.2f} GB")
        else:
            print(f"[DISK CHECK] Free disk space on {save_dir}: {free_gb:.2f} GB")
    except OSError:
        pass


def prepare_all_tsv(raw_data_root, categories=("4mC", "6mA")):
    """
    Ensures data/tsv/{category}/{species}/{train.tsv, test.tsv} exist by converting
    from data/{category}/{species}/{train_pos.txt, train_neg.txt, test_pos.txt, test_neg.txt}
    if needed.
    """
    tsv_root = os.path.join(raw_data_root, "tsv")
    for category in categories:
        raw_cat_dir = os.path.join(raw_data_root, category)
        tsv_cat_dir = os.path.join(tsv_root, category)
        if os.path.exists(raw_cat_dir):
            for species in sorted(os.listdir(raw_cat_dir)):
                raw_sp_dir = os.path.join(raw_cat_dir, species)
                if not os.path.isdir(raw_sp_dir) or species.startswith("."):
                    continue
                target_sp_dir = os.path.join(tsv_cat_dir, species)
                os.makedirs(target_sp_dir, exist_ok=True)
                train_tsv = os.path.join(target_sp_dir, "train.tsv")
                test_tsv = os.path.join(target_sp_dir, "test.tsv")

                if not (os.path.exists(train_tsv) and os.path.exists(test_tsv)):
                    ensure_tsv_from_pos_neg(raw_sp_dir)
                    src_train = os.path.join(raw_sp_dir, "train.tsv")
                    src_test = os.path.join(raw_sp_dir, "test.tsv")
                    if os.path.exists(src_train) and not os.path.exists(train_tsv):
                        shutil.copy2(src_train, train_tsv)
                    if os.path.exists(src_test) and not os.path.exists(test_tsv):
                        shutil.copy2(src_test, test_tsv)
    return tsv_root


def get_run_state(model_save_dir):
    """
    Returns one of:
      - 'completed_v2': Completed under model_plus with full per-epoch metrics & test results
      - 'interrupted_v2': Started under model_plus and has per-epoch checkpoints to resume from
      - 'legacy': Has old pre-model_plus test_metrics.txt but no run_status.json / epoch_metrics_all_folds.csv
      - 'empty': Not yet run
    """
    if not os.path.exists(model_save_dir):
        return "empty"

    status_path = os.path.join(model_save_dir, "run_status.json")
    result_path = os.path.join(model_save_dir, "test_metrics.txt")
    epoch_all_csv = os.path.join(model_save_dir, "epoch_metrics_all_folds.csv")
    has_ckpt = any(
        os.path.exists(os.path.join(model_save_dir, f"checkpoint_latest_fold_{k}.pt"))
        for k in range(1, 6)
    )

    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            if st.get("status") == "completed" and os.path.exists(result_path) and os.path.exists(epoch_all_csv):
                return "completed_v2"
        except Exception:
            pass
        return "interrupted_v2"

    if has_ckpt:
        return "interrupted_v2"

    if os.path.exists(result_path):
        return "legacy"

    return "empty"


def main():
    parser = argparse.ArgumentParser(description="Batch Runner for TF-DualMethyl All Experiments (with Auto-Resume & Epoch Logging)")
    parser.add_argument("--only-ablation", "--only_ablation", dest="only_ablation", action="store_true",
                        help="Only run the 4 Ablation_* models (skip Baseline_* and Ultimate_Fusion)")
    parser.add_argument("--include_baselines", action="store_true", default=True,
                        help="Include Baseline_* and Ultimate_Fusion (enabled by default for full re-run)")
    parser.add_argument("--keep-legacy", "--keep_legacy", dest="keep_legacy", action="store_true",
                        help="Skip legacy Baseline_*/Ultimate_Fusion folders if test_metrics.txt already exists")
    parser.add_argument("--force_ablation", action="store_true", default=True,
                        help="Clean legacy Ablation_* folders before training with fixed architecture")
    parser.add_argument("--reset-all", "--reset_all", dest="reset_all", action="store_true",
                        help="Force re-training from Fold 1 Epoch 1 even if model_plus checkpoints exist")
    parser.add_argument("--save-all-epochs", "--save_all_epochs", dest="save_all_epochs", action="store_true",
                        help="Archive every single epoch's model_fold_k_epoch_e.pth under epoch_models/")
    parser.add_argument("--include_5hmc", action="store_true",
                        help="Also include 5hmC category in addition to 4mC and 6mA")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES GPU ID (e.g. 0)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs per fold (default: 30)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training/validation batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[GPU] Set CUDA_VISIBLE_DEVICES={args.gpu}")

    script_dir, raw_data_root, save_dir = resolve_dirs()
    reclaim_disk_space(save_dir)
    categories = ["4mC", "6mA"] + (["5hmC"] if args.include_5hmc else [])
    base_data_dir = prepare_all_tsv(raw_data_root, categories=categories)

    models_to_run = []
    if not args.only_ablation:
        models_to_run.extend([
            "Baseline_TextCNN",
            "Baseline_BiLSTM",
            "Baseline_CRNN",
            "Baseline_Transformer",
            "Ultimate_Fusion",
        ])
    models_to_run.extend([
        "Ablation_No_BiLSTM",
        "Ablation_No_Transformer",
        "Ablation_No_Freq",
        "Ablation_No_Time",
    ])

    datasets = []
    if os.path.exists(base_data_dir):
        for category in sorted(os.listdir(base_data_dir)):
            if category not in categories:
                continue
            category_path = os.path.join(base_data_dir, category)
            if os.path.isdir(category_path) and not category.startswith("."):
                for species in sorted(os.listdir(category_path)):
                    species_path = os.path.join(category_path, species)
                    if os.path.isdir(species_path) and not species.startswith("."):
                        if os.path.exists(os.path.join(species_path, "train.tsv")):
                            datasets.append({
                                "data_dir": species_path,
                                "category": category,
                                "species": species
                            })

    if not datasets:
        print(f"No valid datasets found in {base_data_dir}! Please check data directory structure.")
        return

    print(f"Found {len(datasets)} valid datasets and {len(models_to_run)} target models.")
    for d in datasets:
        print(f" - [{d['category']}] -> {d['species']}")

    run_py_path = os.path.join(script_dir, "run.py")

    for dataset in datasets:
        print(f"\n\n{'#'*70}")
        print(f"#### Processing Dataset: [{dataset['category']} / {dataset['species']}] ####")
        print(f"{'#'*70}")

        for model_name in models_to_run:
            model_save_dir = os.path.join(save_dir, model_name, dataset["category"], dataset["species"])
            state = get_run_state(model_save_dir)

            force_retrain_flag = False
            if args.reset_all:
                force_retrain_flag = True
                if "Ablation" in model_name and os.path.exists(model_save_dir):
                    shutil.rmtree(model_save_dir)
            elif state == "completed_v2":
                print(f"\n[SKIP - COMPLETED] {model_name} on {dataset['species']} already completed with full epoch metrics.")
                continue
            elif state == "interrupted_v2":
                print(f"\n[RESUME] Detected interrupted training for {model_name} on {dataset['species']}. Resuming from latest checkpoint...")
            elif state == "legacy":
                if "Ablation" in model_name:
                    print(f"\n[CLEAN LEGACY ABLATION] Removing legacy ablation folder {model_save_dir} to run fixed architecture...")
                    shutil.rmtree(model_save_dir)
                elif args.keep_legacy:
                    print(f"\n[SKIP - LEGACY] Keeping existing legacy results for {model_name} on {dataset['species']}.")
                    continue
                else:
                    print(f"\n[FULL RE-RUN] Upgrading legacy {model_name} on {dataset['species']} with per-epoch metrics logging & checkpoints...")

            print(f"\n{'='*60}")
            print(f"--- Target Model: {model_name} ({dataset['category']} / {dataset['species']}) ---")
            print(f"{'='*60}")

            # Unified patience: 15 for Ablation (prevents epoch-2 premature stop), 25 for Main/Baselines
            patience = 15 if "Ablation" in model_name else 25

            cmd = [
                sys.executable, "-u", run_py_path,
                "--data_dir", dataset["data_dir"],
                "--dataset_category", dataset["category"],
                "--species_name", dataset["species"],
                "--model_name", model_name,
                "--save_dir", save_dir,
                "--mode", "train",
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--patience", str(patience),
            ]
            if force_retrain_flag:
                cmd.append("--force_retrain")
            if args.save_all_epochs:
                cmd.append("--save_all_epochs")

            print(f"Executing: {' '.join(cmd)}\n", flush=True)
            try:
                child_env = os.environ.copy()
                child_env["PYTHONUNBUFFERED"] = "1"
                return_code = subprocess.call(cmd, env=child_env)
                time.sleep(1)
                if return_code == 0:
                    print(f"\n[SUCCESS] Completed {model_name} on {dataset['species']}.", flush=True)
                else:
                    print(f"\n[WARNING] {model_name} on {dataset['species']} exited with code {return_code}.", flush=True)
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] Training interrupted by user. Progress has been saved at the latest epoch checkpoint.")
                sys.exit(130)
            except Exception as e:
                print(f"\n[ERROR] Exception while running {model_name}: {e}")


if __name__ == "__main__":
    main()
