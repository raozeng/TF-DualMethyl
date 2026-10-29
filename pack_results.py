#!/usr/bin/env python3
"""
Packages all TF-DualMethyl experimental outputs (metrics, per-epoch CSV logs,
prediction probabilities, cross-species transfer matrices, summary tables,
LaTeX tables, publication PDF figures, page figures, and optional model weights)
into downloadable .zip archives.

Usage:
  python3 pack_results.py                     # Refresh tables/plots if needed & create zip archives
  python3 pack_results.py --only-pack         # Pack existing results immediately without re-running plot scripts
  python3 pack_results.py --light-only        # Only create the lightweight archive (no .pth weights, ~15-30 MB)
  python3 pack_results.py --all-weights       # Include .pth weights for all 9 models (Baseline + Ablation + Main)
"""

import argparse
import os
import subprocess
import sys
import time
import zipfile
from run_all import resolve_dirs, reclaim_disk_space


def run_step(cmd, desc, cwd):
    print(f"\n[POST-PROCESS] {desc}...")
    print(f"  -> Running: {' '.join(cmd)}", flush=True)
    ret = subprocess.call(cmd, cwd=cwd)
    if ret != 0:
        print(f"  [WARNING] Step exited with code {ret}: {desc}", flush=True)
    return ret == 0


def ensure_postprocessing(script_dir, result_dir, project_root, force_post=False):
    """
    Ensures tables and figures are generated before packaging.
    """
    py = sys.executable
    table_file = os.path.join(result_dir, "table.txt")
    latex_file = os.path.join(result_dir, "latex_tables.tex")
    epoch_master = os.path.join(result_dir, "all_epochs_training_history.csv")
    paper_fig_dir = os.path.join(project_root, "paper_new", "figures")

    if force_post or not os.path.exists(table_file) or not os.path.exists(latex_file) or not os.path.exists(epoch_master):
        run_step([py, "-u", "run_probability.py"], "Extracting ensemble prediction probabilities", script_dir)
        run_step([py, "-u", "run_cross_transfer.py"], "Evaluating cross-species zero-shot transferability", script_dir)
        run_step([py, "-u", "generate_table.py"], "Generating summary tables, LaTeX rows & master epoch CSV", script_dir)

    if force_post or not os.path.exists(paper_fig_dir) or len(os.listdir(paper_fig_dir)) < 10:
        plot_scripts = [
            "plot/plot_com_bar.py",
            "plot/plot_ablation_curve.py",
            "plot/plot_curve.py",
            "plot/plot_confidence.py",
            "plot/plot_heatmap.py",
            "plot/plot_attn_heatmap.py",
            "plot/plot_tsne.py",
            "plot/plot_umap.py",
            "plot/plot_motif.py",
            "plot/plot_radar.py",
        ]
        for ps in plot_scripts:
            ps_path = os.path.join(script_dir, ps)
            if os.path.exists(ps_path):
                run_step([py, "-u", ps], f"Running {ps}", script_dir)

        # Refresh composite manifold/motif/architecture PDFs if prepare_paper_figures.py exists
        prep_code = (
            "import sys; sys.path.insert(0, 'plot'); "
            "import prepare_paper_figures as p; "
            "p.generate_architecture_pdf(); "
            "p.generate_manifold_and_motif_pdfs()"
        )
        run_step([py, "-u", "-c", prep_code], "Refreshing composite PDF figures in paper_new/figures", script_dir)


def build_completion_report(result_dir):
    """
    Scans result_dir to summarize completed (model, category, species) runs.
    """
    models = [
        "Baseline_TextCNN",
        "Baseline_BiLSTM",
        "Baseline_CRNN",
        "Baseline_Transformer",
        "Ultimate_Fusion",
        "Ablation_No_BiLSTM",
        "Ablation_No_Transformer",
        "Ablation_No_Freq",
        "Ablation_No_Time",
    ]
    lines = []
    lines.append("=" * 72)
    lines.append(f"TF-DualMethyl Experiment Packaging Report ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append("=" * 72)

    total_completed = 0
    for m in models:
        m_dir = os.path.join(result_dir, m)
        if not os.path.isdir(m_dir):
            lines.append(f"  - {m.ljust(25)}: 0 datasets found")
            continue
        completed_sp = []
        for cat in sorted(os.listdir(m_dir)):
            cat_dir = os.path.join(m_dir, cat)
            if not os.path.isdir(cat_dir) or cat.startswith("."):
                continue
            for sp in sorted(os.listdir(cat_dir)):
                sp_dir = os.path.join(cat_dir, sp)
                if not os.path.isdir(sp_dir) or sp.startswith("."):
                    continue
                if os.path.exists(os.path.join(sp_dir, "test_metrics.txt")):
                    completed_sp.append(f"{cat}/{sp}")
        total_completed += len(completed_sp)
        lines.append(f"  - {m.ljust(25)}: {len(completed_sp):2d} datasets completed")

    lines.append("-" * 72)
    lines.append(f"Total Completed (Model x Dataset) Runs: {total_completed}")
    lines.append("=" * 72)
    return "\n".join(lines)


def should_include_result_file(rel_path, weight_mode):
    """
    Determines whether a file inside `result/` should be included in the zip archive.
      weight_mode:
        - 'none': exclude all .pth and large .pt files (keep 1KB completion markers)
        - 'ultimate': include .pth files only under Ultimate_Fusion/
        - 'all': include all model_fold_*.pth files across all models
    """
    fname = os.path.basename(rel_path)
    if fname.startswith(".") or fname.endswith(".tmp") or "_backup" in rel_path:
        return False
    if fname.startswith("checkpoint_prev_fold_") or fname.startswith("model_latest_fold_"):
        return False

    if fname.endswith(".pth"):
        if weight_mode == "none":
            return False
        elif weight_mode == "ultimate":
            parts = rel_path.replace("\\", "/").split("/")
            return "Ultimate_Fusion" in parts
        elif weight_mode == "all":
            return True

    if fname.endswith(".pt"):
        # Only include lightweight completion markers (< 50 KB)
        return False

    return True


def create_archive(zip_path, project_root, script_dir, result_dir, weight_mode, report_text):
    """
    Creates a zip archive with clean project-relative paths:
      - result/...
      - paper_new/figures/...
      - page/...
      - model_plus/tf_dualmethyl_nohup.log (if present)
      - EXPERIMENT_REPORT.txt
    """
    tmp_zip = zip_path + ".tmp"
    if os.path.exists(tmp_zip):
        os.remove(tmp_zip)

    added_count = 0
    total_raw_bytes = 0

    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.writestr("EXPERIMENT_REPORT.txt", report_text + "\n")

        # 1. Package result/
        if os.path.exists(result_dir):
            for root, dirs, files in os.walk(result_dir):
                dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and not d.endswith("_backup")]
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    if not os.path.isfile(fpath):
                        continue
                    rel_to_res = os.path.relpath(fpath, result_dir)
                    if not should_include_result_file(rel_to_res, weight_mode):
                        continue
                    arcname = os.path.join("result", rel_to_res).replace("\\", "/")
                    zf.write(fpath, arcname)
                    added_count += 1
                    total_raw_bytes += os.path.getsize(fpath)

        # 2. Package paper_new/figures/ (and paper_new/*.tex, *.bib if present)
        paper_dir = os.path.join(project_root, "paper_new")
        if os.path.exists(paper_dir):
            for root, dirs, files in os.walk(paper_dir):
                dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
                for fname in sorted(files):
                    if fname.startswith(".") or fname.endswith((".aux", ".log", ".out", ".bbl", ".blg", ".synctex.gz")):
                        continue
                    fpath = os.path.join(root, fname)
                    if not os.path.isfile(fpath):
                        continue
                    rel_to_proj = os.path.relpath(fpath, project_root).replace("\\", "/")
                    zf.write(fpath, rel_to_proj)
                    added_count += 1
                    total_raw_bytes += os.path.getsize(fpath)

        # 3. Package page/ (all individual high-res PNG/PDF figures)
        page_dir = os.path.join(project_root, "page")
        if os.path.exists(page_dir):
            for root, dirs, files in os.walk(page_dir):
                dirs[:] = [d for d in sorted(dirs) if not d.startswith(".")]
                for fname in sorted(files):
                    if fname.startswith("."):
                        continue
                    fpath = os.path.join(root, fname)
                    if not os.path.isfile(fpath):
                        continue
                    rel_to_proj = os.path.relpath(fpath, project_root).replace("\\", "/")
                    zf.write(fpath, rel_to_proj)
                    added_count += 1
                    total_raw_bytes += os.path.getsize(fpath)

        # 4. Package training log files inside model_plus/
        for log_name in ("tf_dualmethyl_nohup.log", "training.log"):
            log_path = os.path.join(script_dir, log_name)
            if os.path.exists(log_path) and os.path.isfile(log_path):
                zf.write(log_path, f"model_plus/{log_name}")
                added_count += 1
                total_raw_bytes += os.path.getsize(log_path)

    os.replace(tmp_zip, zip_path)
    zip_mb = os.path.getsize(zip_path) / (1024 ** 2)
    raw_mb = total_raw_bytes / (1024 ** 2)
    print(f"  [OK] Created: {zip_path}")
    print(f"       Files  : {added_count} files (raw {raw_mb:.1f} MB -> compressed {zip_mb:.1f} MB)")
    return zip_path, zip_mb


def main():
    parser = argparse.ArgumentParser(description="Package all TF-DualMethyl experimental results & figures into a downloadable .zip")
    parser.add_argument("--only-pack", "--only_pack", dest="only_pack", action="store_true",
                        help="Skip running post-processing scripts and immediately package existing files")
    parser.add_argument("--force-post", "--force_post", dest="force_post", action="store_true",
                        help="Force re-running all table and plotting scripts before packaging")
    parser.add_argument("--light-only", "--light_only", dest="light_only", action="store_true",
                        help="Only build the lightweight archive (tables, metrics, epoch CSVs, probabilities, figures; no .pth weights)")
    parser.add_argument("--all-weights", "--all_weights", dest="all_weights", action="store_true",
                        help="Include .pth weights for all 9 models (instead of only Ultimate_Fusion)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the generated .zip file(s) (default: project root)")
    args = parser.parse_args()

    script_dir, _, result_dir = resolve_dirs()
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
    out_dir = args.output_dir or project_root
    os.makedirs(out_dir, exist_ok=True)

    # First clean up any leftover temporary/redundant checkpoint buffers
    reclaim_disk_space(result_dir)

    if not args.only_pack:
        ensure_postprocessing(script_dir, result_dir, project_root, force_post=args.force_post)

    report_text = build_completion_report(result_dir)
    print("\n" + report_text + "\n")

    # Always create the lightweight archive (fast download, has all tables, CSVs, probabilities, and figures)
    light_zip = os.path.join(out_dir, "TF-DualMethyl_results_light.zip")
    print("[1] Packaging lightweight results archive (tables + epoch CSVs + probabilities + PDF/PNG figures, no .pth weights)...")
    create_archive(light_zip, project_root, script_dir, result_dir, weight_mode="none", report_text=report_text)

    if not args.light_only:
        w_mode = "all" if args.all_weights else "ultimate"
        full_zip = os.path.join(out_dir, "TF-DualMethyl_results.zip")
        label = "all 9 models' .pth weights" if args.all_weights else "Ultimate_Fusion 5-fold .pth weights"
        print(f"\n[2] Packaging complete results archive (everything above + {label})...")
        create_archive(full_zip, project_root, script_dir, result_dir, weight_mode=w_mode, report_text=report_text)

    print("\n" + "=" * 72)
    print("  DOWNLOAD READY! You can now download the zip file(s) to your computer:")
    print(f"  1. Lightweight (Tables + Epoch CSVs + All Figures) : {light_zip}")
    if not args.light_only:
        print(f"  2. Complete (+ Trained Model Weights)              : {os.path.join(out_dir, 'TF-DualMethyl_results.zip')}")
    print("=" * 72)


if __name__ == "__main__":
    main()

