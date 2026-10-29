import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_all import resolve_dirs

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8

script_dir, _, result_dir = resolve_dirs()
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
page_radar_dir = os.path.join(project_root, "page", "page_radar", "page_radar")
os.makedirs(page_radar_dir, exist_ok=True)

model_mapping = {
    "Baseline_BiLSTM": "BiLSTM",
    "Baseline_TextCNN": "TextCNN",
    "Baseline_CRNN": "CRNN",
    "Baseline_Transformer": "Transformer",
    "Ultimate_Fusion": "TF-DualMethyl",
}

model_colors = {
    "BiLSTM": "#ff7f0e",
    "TextCNN": "#2ca02c",
    "CRNN": "#1f77b4",
    "Transformer": "#9467bd",
    "TF-DualMethyl": "#E64B35",
}

_SKIP_DIRS = {"page_compare_bar", "page_confidence", "page_heatmap", "page_ablation_curve", "page_curve", "page_radar", "Cross_Transfer"}


def scan_datasets():
    datasets = []
    if not os.path.exists(result_dir):
        return datasets
    for model_dir_name in os.listdir(result_dir):
        if model_dir_name in _SKIP_DIRS:
            continue
        model_path = os.path.join(result_dir, model_dir_name)
        if not os.path.isdir(model_path):
            continue
        for class_name in os.listdir(model_path):
            if class_name not in ("4mC", "6mA"):
                continue
            class_path = os.path.join(model_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for dataset_name in os.listdir(class_path):
                dataset_path = os.path.join(class_path, dataset_name)
                if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "probability.txt")):
                    t = (class_name, dataset_name)
                    if t not in datasets:
                        datasets.append(t)
    datasets.sort()
    return datasets


def read_probability(path):
    if not os.path.exists(path):
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(path, sep='\t')
        if 'True_Label' in df.columns and 'Probability' in df.columns:
            df = df.dropna(subset=['True_Label', 'Probability'])
            return df['True_Label'].astype(int).values, df['Probability'].astype(float).values
    except Exception:
        pass
    return None, None


def compute_metrics(y_true, y_prob):
    if y_true is None or y_prob is None or len(np.unique(y_true)) < 2:
        return None
    auc = roc_auc_score(y_true, y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"AUC": auc, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


def plot_comparison_radar(models, dataset, output_base):
    class_name, dataset_name = dataset
    title = f"Model Comparison - {class_name} - {dataset_name}"
    categories = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)

    for model in models:
        model_path = os.path.join(result_dir, model, class_name, dataset_name, "probability.txt")
        y_true, y_prob = read_probability(model_path)
        metrics = compute_metrics(y_true, y_prob)
        if metrics is None:
            continue
        values = [metrics[cat] for cat in categories] + [metrics[categories[0]]]
        display_name = model_mapping.get(model, model)
        color = model_colors.get(display_name, None)
        ax.plot(angles, values, 'o-', linewidth=1.5, label=display_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), frameon=True, fancybox=True)
    plt.tight_layout()
    fig.savefig(output_base + ".pdf", bbox_inches='tight')
    fig.savefig(output_base + ".png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    default_models = [
        "Baseline_BiLSTM",
        "Baseline_TextCNN",
        "Baseline_CRNN",
        "Baseline_Transformer",
        "Ultimate_Fusion",
    ]
    for ds in scan_datasets():
        plot_comparison_radar(default_models, ds, os.path.join(page_radar_dir, f"radar_{ds[0]}_{ds[1]}"))

