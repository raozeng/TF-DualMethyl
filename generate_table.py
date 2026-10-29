import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from run_all import resolve_dirs


_, _, result_dir = resolve_dirs()
output_file = os.path.join(result_dir, "table.txt")
latex_output_file = os.path.join(result_dir, "latex_tables.tex")

model_mapping = {
    "Baseline_BiLSTM": "BiLSTM",
    "Baseline_CRNN": "CRNN",
    "Baseline_TextCNN": "TextCNN",
    "Baseline_Transformer": "Transformer",
    "Ablation_No_BiLSTM": "w/o BiLSTM",
    "Ablation_No_Freq": "w/o Freq",
    "Ablation_No_Time": "w/o Time",
    "Ablation_No_Transformer": "w/o Transformer",
    "Ultimate_Fusion": "TF-DualMethyl",
}


def compute_metrics_from_prob(prob_path):
    """Computes exact ACC, SN, SP, MCC, AUC, F1 from probability.txt if available."""
    try:
        df = pd.read_csv(prob_path, sep="\t")
        y_true = df["True_Label"].values
        y_prob = df["Probability"].values
        y_pred = df["Prediction"].values

        acc = accuracy_score(y_true, y_pred)
        sn = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {"ACC": acc, "SN": sn, "SP": sp, "MCC": mcc, "AUC": auc, "F1": f1}
    except Exception:
        return None


def extract_metrics(dataset_dir):
    """Extracts metrics from probability.txt (preferred for exact SP) or test_metrics.txt."""
    prob_path = os.path.join(dataset_dir, "probability.txt")
    if os.path.exists(prob_path):
        m = compute_metrics_from_prob(prob_path)
        if m is not None:
            return m

    test_metrics_path = os.path.join(dataset_dir, "test_metrics.txt")
    if not os.path.exists(test_metrics_path):
        return None

    with open(test_metrics_path, "r", encoding="utf-8") as f:
        content = f.read()

    metrics = {}
    acc_match = re.search(r"Accuracy:\s*([0-9.]+)", content)
    if acc_match:
        metrics["ACC"] = float(acc_match.group(1))

    recall_match = re.search(r"Recall:\s*([0-9.]+)", content)
    if recall_match:
        metrics["SN"] = float(recall_match.group(1))

    sp_match = re.search(r"Specificity:\s*([0-9.]+)", content)
    if sp_match:
        metrics["SP"] = float(sp_match.group(1))
    elif "ACC" in metrics and "SN" in metrics:
        metrics["SP"] = max(0.0, min(1.0, (metrics["ACC"] * 2.0) - metrics["SN"]))
    else:
        metrics["SP"] = 0.0

    f1_match = re.search(r"F1 Score:\s*([0-9.]+)", content)
    if f1_match:
        metrics["F1"] = float(f1_match.group(1))

    mcc_match = re.search(r"MCC:\s*([0-9.]+)", content)
    if mcc_match:
        metrics["MCC"] = float(mcc_match.group(1))

    auc_match = re.search(r"AUC:\s*([0-9.]+)", content)
    if auc_match:
        metrics["AUC"] = float(auc_match.group(1))

    return metrics


def format_value(value, width=6):
    return f"{value * 100:.2f}".rjust(width)


def generate_latex_snippets(all_results):
    """Generates ready-to-paste LaTeX table rows for both Baseline Comparison and Ablation Study."""
    lines = ["% Auto-generated LaTeX Table Rows for TF-DualMethyl (paper_new)\n"]

    for mode_title, allowed_models in [
        ("Baseline Comparison", ["BiLSTM", "CRNN", "TextCNN", "Transformer", "TF-DualMethyl"]),
        ("Ablation Study", ["w/o BiLSTM", "w/o Freq", "w/o Time", "w/o Transformer", "TF-DualMethyl"]),
    ]:
        lines.append(f"% ==================== {mode_title} ====================")
        subset = [r for r in all_results if r["Model"] in allowed_models]
        by_ds = {}
        for r in subset:
            by_ds.setdefault((r["Class"], r["Dataset"]), []).append(r)

        for (cls, ds), rows in sorted(by_ds.items()):
            lines.append(f"% --- {cls} / {ds} ---")
            ds_tex = ds.replace("_", "\\_")
            # Sort with TF-DualMethyl last
            rows.sort(key=lambda x: (x["Model"] == "TF-DualMethyl", x["Model"]))
            for idx, r in enumerate(rows):
                is_ours = r["Model"] == "TF-DualMethyl"
                m_str = f"\\textbf{{{r['Model']}}}" if is_ours else r["Model"]
                vals = [f"{r[k]*100:.2f}" for k in ["ACC", "SN", "SP", "MCC", "AUC", "F1"]]
                if is_ours:
                    vals = [f"\\textbf{{{v}}}" for v in vals]
                prefix = f"\\multirow{{{len(rows)}}}{{*}}{{{cls}}} & \\multirow{{{len(rows)}}}{{*}}{{\\texttt{{{ds_tex}}}}} & " if idx == 0 else " & & "
                lines.append(f"{prefix}{m_str} & " + " & ".join(vals) + " \\\\")
            lines.append("\\midrule")
        lines.append("\n")

    with open(latex_output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table snippets saved to: {latex_output_file}")


def main():
    if not os.path.exists(result_dir):
        print(f"Result directory not found: {result_dir}")
        return

    datasets = []
    for model_dir in os.listdir(result_dir):
        if model_dir not in model_mapping:
            continue
        model_path = os.path.join(result_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for class_name in os.listdir(model_path):
            class_path = os.path.join(model_path, class_name)
            if not os.path.isdir(class_path) or class_name.startswith("."):
                continue
            for dataset_name in os.listdir(class_path):
                dataset_path = os.path.join(class_path, dataset_name)
                if not os.path.isdir(dataset_path) or dataset_name.startswith("."):
                    continue
                if os.path.exists(os.path.join(dataset_path, "test_metrics.txt")) or os.path.exists(os.path.join(dataset_path, "probability.txt")):
                    dt = (class_name, dataset_name)
                    if dt not in datasets:
                        datasets.append(dt)

    datasets.sort()
    model_results = {}

    for model_dir, model_name in model_mapping.items():
        model_path = os.path.join(result_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for class_name, dataset_name in datasets:
            ds_dir = os.path.join(model_path, class_name, dataset_name)
            if os.path.exists(ds_dir):
                metrics = extract_metrics(ds_dir)
                if metrics:
                    key = f"{class_name}_{dataset_name}_{model_name}"
                    model_results[key] = {
                        "Class": class_name,
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "ACC": metrics.get("ACC", 0.0),
                        "SN": metrics.get("SN", 0.0),
                        "SP": metrics.get("SP", 0.0),
                        "MCC": metrics.get("MCC", 0.0),
                        "AUC": metrics.get("AUC", 0.0),
                        "F1": metrics.get("F1", 0.0),
                    }

    all_results = list(model_results.values())
    if not all_results:
        print("No model results found to tabulate.")
        return

    all_results.sort(key=lambda x: (x["Class"], x["Dataset"], x["Model"] == "TF-DualMethyl", x["Model"]))

    dataset_best_acc = {}
    for result in all_results:
        key = f"{result['Class']}_{result['Dataset']}"
        if key not in dataset_best_acc or result["ACC"] > dataset_best_acc[key]:
            dataset_best_acc[key] = result["ACC"]

    class_width = max(len("Class"), max(len(r["Class"]) for r in all_results))
    dataset_width = max(len("Dataset"), max(len(r["Dataset"]) for r in all_results))
    model_width = max(len("Model"), max(len(r["Model"]) for r in all_results)) + 2
    metric_width = 6

    with open(output_file, "w", encoding="utf-8") as f:
        header = (
            f"{'Class'.ljust(class_width)} | {'Dataset'.ljust(dataset_width)} | "
            f"{'Model'.ljust(model_width)} | {'ACC'.rjust(metric_width)} | "
            f"{'SN'.rjust(metric_width)} | {'SP'.rjust(metric_width)} | "
            f"{'MCC'.rjust(metric_width)} | {'AUC'.rjust(metric_width)} | {'F1'.rjust(metric_width)}"
        )
        f.write(header + "\n")
        separator = (
            f"{'-' * class_width}-+-{'-' * dataset_width}-+-{'-' * model_width}-+-"
            f"{'-' * metric_width}-+-{'-' * metric_width}-+-{'-' * metric_width}-+-"
            f"{'-' * metric_width}-+-{'-' * metric_width}-+-{'-' * metric_width}"
        )
        f.write(separator + "\n")

        for result in all_results:
            key = f"{result['Class']}_{result['Dataset']}"
            is_best = result["ACC"] >= dataset_best_acc[key] - 1e-6
            m_name = ("* " + result["Model"]) if is_best else result["Model"]
            line = (
                f"{result['Class'].ljust(class_width)} | {result['Dataset'].ljust(dataset_width)} | "
                f"{m_name.ljust(model_width)} | "
                f"{format_value(result['ACC'], metric_width)} | {format_value(result['SN'], metric_width)} | "
                f"{format_value(result['SP'], metric_width)} | {format_value(result['MCC'], metric_width)} | "
                f"{format_value(result['AUC'], metric_width)} | {format_value(result['F1'], metric_width)}"
            )
            f.write(line + "\n")

    print(f"Summary table saved to: {output_file}")
    generate_latex_snippets(all_results)
    export_master_epoch_history(datasets)


def export_master_epoch_history(datasets):
    """
    Aggregates all `epoch_metrics_all_folds.csv` files across all models and datasets
    into a single master file `result/all_epochs_training_history.csv`.
    """
    master_frames = []
    for model_dir, model_name in model_mapping.items():
        model_path = os.path.join(result_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        for class_name, dataset_name in datasets:
            epoch_csv = os.path.join(model_path, class_name, dataset_name, "epoch_metrics_all_folds.csv")
            if os.path.exists(epoch_csv):
                try:
                    df = pd.read_csv(epoch_csv)
                    if not df.empty:
                        df.insert(0, "model_id", model_dir)
                        df.insert(1, "model_name", model_name)
                        df.insert(2, "category", class_name)
                        df.insert(3, "species", dataset_name)
                        master_frames.append(df)
                except Exception:
                    pass

    if master_frames:
        master_df = pd.concat(master_frames, ignore_index=True)
        master_csv_path = os.path.join(result_dir, "all_epochs_training_history.csv")
        master_df.to_csv(master_csv_path, index=False)
        print(f"Master per-epoch training history saved to: {master_csv_path}")


if __name__ == "__main__":
    main()

