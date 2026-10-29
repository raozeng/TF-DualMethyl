import os
import sys
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix
)

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
plt.rcParams['figure.titlesize'] = 9
plt.rcParams['lines.linewidth'] = 1.1
plt.rcParams['lines.markersize'] = 4
plt.rcParams['axes.linewidth'] = 0.8

script_dir, _, result_dir = resolve_dirs()
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
page_ablation_curve_dir = os.path.join(project_root, "page", "page_ablation_curve")
paper_fig_dir = os.path.join(project_root, "paper_new", "figures")
os.makedirs(page_ablation_curve_dir, exist_ok=True)
os.makedirs(paper_fig_dir, exist_ok=True)

model_mapping = {
    "Ultimate_Fusion": "TF-DualMethyl",
    "Ablation_No_BiLSTM": "w/o BiLSTM",
    "Ablation_No_Transformer": "w/o Transformer",
    "Ablation_No_Freq": "w/o Freq",
    "Ablation_No_Time": "w/o Time",
}

model_colors = {
    "TF-DualMethyl": "#E64B35",
    "w/o BiLSTM": "#ff7f0e",
    "w/o Transformer": "#2ca02c",
    "w/o Freq": "#9467bd",
    "w/o Time": "#1f77b4",
}

_SKIP_DIRS = {"page_compare_bar", "page_confidence", "page_heatmap", "page_ablation_curve", "Cross_Transfer"}


def apply_nature_style(ax, title=None, panel_label=None):
    if hasattr(ax, 'spines'):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold', pad=8)
    if panel_label:
        ax.text(-0.14, 1.10, panel_label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='right')


def scan_datasets():
    datasets = []
    if not os.path.exists(result_dir):
        return datasets
    for model_dir_name in os.listdir(result_dir):
        if model_dir_name in _SKIP_DIRS or model_dir_name not in model_mapping:
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
                if not os.path.isdir(dataset_path):
                    continue
                if os.path.exists(os.path.join(dataset_path, "probability.txt")):
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
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None, None


def extract_metrics(path):
    y_true, y_prob = read_probability(path)
    if y_true is None:
        return None

    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Fixed: True Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')

    return {
        'ACC': acc,
        'SN': recall,
        'SP': sp,
        'F1': f1,
        'MCC': mcc,
        'AUC': auc
    }


def extract_roc_pr_data(path):
    y_true, y_prob = read_probability(path)
    if y_true is None:
        return None
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': pr_precision.tolist(),
            'recall': pr_recall.tolist(),
        }
    return None


def group_by_class(datasets):
    groups = {}
    for class_name, dataset_name in datasets:
        groups.setdefault(class_name, []).append(dataset_name)
    return groups


def collect_model_metrics(datasets):
    result = {}
    for mdir, mname in model_mapping.items():
        mpath = os.path.join(result_dir, mdir)
        if not os.path.isdir(mpath):
            continue
        result[mname] = {}
        for class_name, dataset_name in datasets:
            p = os.path.join(mpath, class_name, dataset_name, "probability.txt")
            m = extract_metrics(p)
            if m:
                result[mname][f"{class_name}_{dataset_name}"] = m
    return result


def collect_model_roc_pr(datasets):
    result = {}
    for mdir, mname in model_mapping.items():
        mpath = os.path.join(result_dir, mdir)
        if not os.path.isdir(mpath):
            continue
        result[mname] = {}
        for class_name, dataset_name in datasets:
            p = os.path.join(mpath, class_name, dataset_name, "probability.txt")
            d = extract_roc_pr_data(p)
            if d:
                result[mname][f"{class_name}_{dataset_name}"] = d
    return result


def plot_metric_curves(datasets):
    model_metrics = collect_model_metrics(datasets)
    groups = group_by_class(datasets)
    metrics = ['ACC', 'SN', 'SP', 'MCC', 'AUC', 'F1']

    for class_name, ds_list in groups.items():
        models = list(model_metrics.keys())
        x_labels = [ds.split('_')[-1] for ds in ds_list]

        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        axes = axes.flatten()
        panel_letters = 'abcdef'

        for i, metric in enumerate(metrics):
            ax = axes[i]
            panel_label = f"({panel_letters[i]})"

            for model_name in models:
                y_values = [model_metrics[model_name].get(f"{class_name}_{ds}", {}).get(metric, np.nan) for ds in ds_list]
                color = model_colors.get(model_name, '#999999')
                lw = 1.8 if model_name == "TF-DualMethyl" else 1.1
                ax.plot(range(len(ds_list)), y_values, marker='o', label=model_name,
                        color=color, linewidth=lw, markersize=4, markeredgewidth=0.5, markeredgecolor='white')

            ax.set_xlabel('Species', fontsize=7)
            ax.set_ylabel(metric, fontsize=7)
            ax.set_xticks(range(len(ds_list)))
            ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=6.5)
            ax.tick_params(axis='both', labelsize=6.5)
            if metric in ['ACC', 'SN', 'SP', 'F1', 'AUC']:
                ax.set_ylim(0.4, 1.02)
            elif metric == 'MCC':
                ax.set_ylim(0.0, 1.02)
            apply_nature_style(ax, title=metric, panel_label=panel_label)

        fig.text(0.02, 0.98, f"{class_name} Ablation Metrics", fontsize=11, fontweight='bold', va='top', ha='left', transform=fig.transFigure)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(models), bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=8)

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        for out_p in [
            os.path.join(page_ablation_curve_dir, f"{class_name}_metrics.pdf"),
            os.path.join(page_ablation_curve_dir, f"{class_name}_metrics.png"),
            os.path.join(paper_fig_dir, f"fig_{class_name}_ablation_metrics.pdf"),
        ]:
            fig.savefig(out_p, dpi=300, bbox_inches='tight')
        print(f"-> {class_name} ablation metric curves generated.")
        plt.close(fig)


def plot_roc_curves(datasets):
    model_roc_pr = collect_model_roc_pr(datasets)
    groups = group_by_class(datasets)

    for class_name, ds_list in groups.items():
        models = list(model_roc_pr.keys())
        n_datasets = len(ds_list)
        rows, cols = (2, 2) if class_name == '4mC' else (3, 4) if class_name == '6mA' else ((n_datasets + 3) // 4, min(4, n_datasets))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.5))
        axes = np.array([axes]).flatten() if n_datasets > 1 else [axes]

        for i, ds in enumerate(ds_list):
            if i >= len(axes):
                break
            ax = axes[i]
            ax.plot([0, 1], [0, 1], 'k--', linewidth=0.6, alpha=0.5)

            for model_name in models:
                roc_data = model_roc_pr[model_name].get(f"{class_name}_{ds}", {})
                if roc_data and len(roc_data.get('fpr', [])) > 1:
                    color = model_colors.get(model_name, '#999999')
                    lw = 1.5 if model_name == "TF-DualMethyl" else 1.0
                    ax.plot(roc_data['fpr'], roc_data['tpr'], label=model_name, color=color, linewidth=lw)

            species_name = ds.split('_')[-1]
            ax.set_xlabel('False Positive Rate', fontsize=6.5)
            ax.set_ylabel('True Positive Rate', fontsize=6.5)
            ax.tick_params(axis='both', labelsize=5.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0.0, 1.02)
            apply_nature_style(ax, title=species_name)

        for k in range(len(ds_list), len(axes)):
            axes[k].set_visible(False)

        fig.text(0.02, 0.98, f"{class_name} Ablation ROC Curves", fontsize=11, fontweight='bold', va='top', ha='left', transform=fig.transFigure)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(models), 5), bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=7.5)

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        for out_p in [
            os.path.join(page_ablation_curve_dir, f"{class_name}_roc.pdf"),
            os.path.join(page_ablation_curve_dir, f"{class_name}_roc.png"),
            os.path.join(paper_fig_dir, f"fig_{class_name}_ablation_roc.pdf"),
        ]:
            fig.savefig(out_p, dpi=300, bbox_inches='tight')
        print(f"-> {class_name} ablation ROC curves generated.")
        plt.close(fig)
        gc.collect()


def plot_pr_curves(datasets):
    model_roc_pr = collect_model_roc_pr(datasets)
    groups = group_by_class(datasets)

    for class_name, ds_list in groups.items():
        models = list(model_roc_pr.keys())
        n_datasets = len(ds_list)
        rows, cols = (2, 2) if class_name == '4mC' else (3, 4) if class_name == '6mA' else ((n_datasets + 3) // 4, min(4, n_datasets))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.5))
        axes = np.array([axes]).flatten() if n_datasets > 1 else [axes]

        for i, ds in enumerate(ds_list):
            if i >= len(axes):
                break
            ax = axes[i]

            for model_name in models:
                pr_data = model_roc_pr[model_name].get(f"{class_name}_{ds}", {})
                if pr_data and len(pr_data.get('precision', [])) > 1:
                    precision = np.array(pr_data['precision'])
                    recall = np.array(pr_data['recall'])
                    if not np.all(np.isnan(precision)):
                        color = model_colors.get(model_name, '#999999')
                        lw = 1.5 if model_name == "TF-DualMethyl" else 1.0
                        ax.plot(recall, precision, label=model_name, color=color, linewidth=lw)

            species_name = ds.split('_')[-1]
            ax.set_xlabel('Recall', fontsize=6.5)
            ax.set_ylabel('Precision', fontsize=6.5)
            ax.set_xlim(0, 1.01)
            ax.set_ylim(0.0, 1.01)
            ax.tick_params(axis='both', labelsize=5.5)
            apply_nature_style(ax, title=species_name)

        for k in range(len(ds_list), len(axes)):
            axes[k].set_visible(False)

        fig.text(0.02, 0.98, f"{class_name} Ablation PR Curves", fontsize=11, fontweight='bold', va='top', ha='left', transform=fig.transFigure)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(models), 5), bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=7.5)

        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        for out_p in [
            os.path.join(page_ablation_curve_dir, f"{class_name}_pr.pdf"),
            os.path.join(page_ablation_curve_dir, f"{class_name}_pr.png"),
            os.path.join(paper_fig_dir, f"fig_{class_name}_ablation_pr.pdf"),
        ]:
            fig.savefig(out_p, dpi=300, bbox_inches='tight')
        print(f"-> {class_name} ablation PR curves generated.")
        plt.close(fig)
        gc.collect()


def main():
    datasets = scan_datasets()
    print(f"Ablation Curve output: {page_ablation_curve_dir}")
    if datasets:
        plot_metric_curves(datasets)
        plot_roc_curves(datasets)
        plot_pr_curves(datasets)
        print("All ablation curve tasks completed!")
    else:
        print("No ablation datasets found.")


if __name__ == "__main__":
    main()

