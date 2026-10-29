import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_all import resolve_dirs

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7

script_dir, _, result_dir = resolve_dirs()
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
output_dir = os.path.join(project_root, "page", "page_compare_bar")
com_bar_dir = os.path.join(project_root, "page", "com_bar")
paper_fig_dir = os.path.join(project_root, "paper_new", "figures")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(com_bar_dir, exist_ok=True)
os.makedirs(paper_fig_dir, exist_ok=True)

models = [
    "Baseline_BiLSTM",
    "Baseline_CRNN",
    "Baseline_TextCNN",
    "Baseline_Transformer",
    "Ultimate_Fusion"
]

model_display_names = [
    "BiLSTM",
    "CRNN",
    "TextCNN",
    "Transformer",
    "TF-DualMethyl"
]

model_colors = [
    "#F9B6A0",  # Nature Peach
    "#77CFDD",  # Nature Light Blue
    "#4DC1B0",  # Nature Teal
    "#657FB2",  # Nature Dark Blue
    "#E64B35",  # Nature Crimson Red (TF-DualMethyl)
]


def apply_nature_style(ax, title=None, panel_label=None):
    if hasattr(ax, 'spines'):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=8)
    if panel_label:
        ax.text(-0.10, 1.12, panel_label, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top', ha='right')


def read_metrics(file_path):
    if not os.path.exists(file_path):
        return None

    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    true_labels = []
    probabilities = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            true_label = int(parts[0])
            probability = float(parts[1])
            prediction = int(parts[2])

            true_labels.append(true_label)
            probabilities.append(probability)

            if true_label == prediction:
                correct += 1
                if true_label == 1:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if true_label == 1:
                    false_negative += 1
                else:
                    false_positive += 1
            total += 1

    if total > 0:
        acc = (correct / total) * 100.0
        sn = (true_positive / (true_positive + false_negative)) * 100.0 if (true_positive + false_negative) > 0 else 0.0
        sp = (true_negative / (true_negative + false_positive)) * 100.0 if (true_negative + false_positive) > 0 else 0.0

        if true_positive + false_positive + false_negative > 0:
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
            recall = sn / 100.0
            f1 = (2 * precision * recall / (precision + recall)) * 100.0 if (precision + recall) > 0 else 0.0
        else:
            f1 = 0.0

        denominator = (
            (true_positive + false_positive) * (true_positive + false_negative) *
            (true_negative + false_positive) * (true_negative + false_negative)
        ) ** 0.5
        mcc = ((true_positive * true_negative) - (false_positive * false_negative)) / denominator * 100.0 if denominator > 0 else 0.0

        try:
            auc = roc_auc_score(true_labels, probabilities) * 100.0 if len(set(true_labels)) > 1 else 0.0
        except Exception:
            auc = 0.0

        return {
            "ACC": round(acc, 2),
            "SN": round(sn, 2),
            "SP": round(sp, 2),
            "F1": round(f1, 2),
            "MCC": round(mcc, 2),
            "AUC": round(auc, 2)
        }
    return None


def find_datasets():
    model_datasets = {}
    for model in models:
        model_path = os.path.join(result_dir, model)
        if not os.path.isdir(model_path):
            continue
        datasets = []
        for class_name in sorted(os.listdir(model_path)):
            class_path = os.path.join(model_path, class_name)
            if not os.path.isdir(class_path) or class_name not in ("4mC", "6mA"):
                continue
            for dataset_name in sorted(os.listdir(class_path)):
                dataset_path = os.path.join(class_path, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue
                if os.path.exists(os.path.join(dataset_path, "probability.txt")):
                    datasets.append((class_name, dataset_name))
        model_datasets[model] = set(datasets)

    if not model_datasets:
        return []
    common_datasets = sorted(list(set.intersection(*model_datasets.values())))
    return common_datasets


def plot_comparision_bar(datasets):
    class_datasets = {}
    for class_name, dataset_name in datasets:
        class_datasets.setdefault(class_name, []).append(dataset_name)

    metrics = ["ACC", "SN", "SP", "F1", "MCC", "AUC"]
    metric_labels = {
        "ACC": "Accuracy (%)",
        "SN": "Sensitivity (%)",
        "SP": "Specificity (%)",
        "F1": "F1 Score (%)",
        "MCC": "MCC (%)",
        "AUC": "AUC (%)"
    }
    panel_letters = ['a', 'b', 'c', 'd', 'e', 'f']

    for class_name, ds_list in class_datasets.items():
        ds_list = sorted(ds_list)
        n_datasets = len(ds_list)
        n_models = len(models)

        all_metrics_data = {m: np.zeros((n_models, n_datasets)) for m in metrics}

        for i, model in enumerate(models):
            for j, dataset_name in enumerate(ds_list):
                metrics_file = os.path.join(result_dir, model, class_name, dataset_name, "probability.txt")
                metrics_data = read_metrics(metrics_file)
                if metrics_data is not None:
                    for metric in metrics:
                        all_metrics_data[metric][i, j] = metrics_data[metric]

        # 1. Save individual metric plots (PDF + PNG)
        for metric in metrics:
            data = all_metrics_data[metric]
            max_values = np.max(data, axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
            group_width = 0.8
            bar_width = group_width / n_models
            gap = 0.25
            x = np.arange(n_datasets) * (group_width + gap)

            for i, (model, color, display_name) in enumerate(zip(models, model_colors, model_display_names)):
                model_x = x + i * bar_width
                bars = ax.bar(model_x, data[i, :], width=bar_width, color=color, label=display_name, edgecolor='white', linewidth=0.3)
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    weight = 'bold' if abs(height - max_values[j]) < 1e-4 else 'normal'
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 1.2, f'{height:.1f}',
                            ha='center', va='bottom', fontsize=5.2, fontweight=weight)

            ax.set_title(f'Comparison Study - {metric} ({class_name})', fontsize=10, fontweight='bold', loc='left', y=1.02)
            ax.set_ylabel(metric_labels[metric], fontsize=8)
            ax.set_ylim(0 if metric == "MCC" else 30, 106)
            ax.set_xticks(x + group_width / 2 - bar_width / 2)
            ax.set_xticklabels([ds.split('_')[-1] for ds in ds_list], rotation=35, ha='right', fontsize=7)
            apply_nature_style(ax)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=5, fontsize=7, frameon=False)

            plt.tight_layout(rect=[0, 0, 1, 0.90])
            fig.savefig(os.path.join(output_dir, f"{class_name}_{metric}_comparison.pdf"), bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, f"{class_name}_{metric}_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # 2. Directly render 2x3 multi-panel vector PDF figure for the paper
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
        axes = axes.flatten()
        group_width = 0.82
        bar_width = group_width / n_models
        gap = 0.26
        x = np.arange(n_datasets) * (group_width + gap)

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = all_metrics_data[metric]
            max_values = np.max(data, axis=0)

            for i, (model, color, display_name) in enumerate(zip(models, model_colors, model_display_names)):
                model_x = x + i * bar_width
                bars = ax.bar(model_x, data[i, :], width=bar_width, color=color, label=display_name, edgecolor='white', linewidth=0.3)
                if n_datasets <= 6:
                    for j, bar in enumerate(bars):
                        height = bar.get_height()
                        weight = 'bold' if abs(height - max_values[j]) < 1e-4 else 'normal'
                        ax.text(bar.get_x() + bar.get_width() / 2., height + 1.2, f'{height:.1f}',
                                ha='center', va='bottom', fontsize=5.0, fontweight=weight)

            ax.set_ylabel(metric_labels[metric], fontsize=8)
            ax.set_ylim(0 if metric == "MCC" else 30, 105)
            ax.set_xticks(x + group_width / 2 - bar_width / 2)
            ax.set_xticklabels([ds.split('_')[-1] for ds in ds_list], rotation=35, ha='right', fontsize=7)
            apply_nature_style(ax, title=f"{metric} ({class_name})", panel_label=f"({panel_letters[idx]})")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=5, fontsize=9, frameon=False)
        plt.tight_layout(rect=[0, 0.01, 1, 0.94])

        merged_pdf = os.path.join(com_bar_dir, f"{class_name}_merged.pdf")
        merged_png = os.path.join(com_bar_dir, f"{class_name}_merged.png")
        paper_pdf = os.path.join(paper_fig_dir, f"fig_{class_name}_comparison.pdf")
        fig.savefig(merged_pdf, bbox_inches='tight')
        fig.savefig(merged_png, dpi=300, bbox_inches='tight')
        fig.savefig(paper_pdf, bbox_inches='tight')
        plt.close(fig)
        print(f"-> Vector multi-panel comparison saved to {merged_pdf} and {paper_pdf}")


def main():
    datasets = find_datasets()
    if datasets:
        plot_comparision_bar(datasets)
    else:
        print("No common datasets found.")


if __name__ == "__main__":
    main()

