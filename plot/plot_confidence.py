import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
page_confidence_dir = os.path.join(project_root, "page", "page_confidence", "confidence")
paper_fig_dir = os.path.join(project_root, "paper_new", "figures")
os.makedirs(page_confidence_dir, exist_ok=True)
os.makedirs(paper_fig_dir, exist_ok=True)

model_mapping = {
    "Baseline_BiLSTM": "BiLSTM",
    "Baseline_TextCNN": "TextCNN",
    "Baseline_CRNN": "CRNN",
    "Baseline_Transformer": "Transformer",
    "Ultimate_Fusion": "TF-DualMethyl",
    "Ablation_No_BiLSTM": "w/o BiLSTM",
    "Ablation_No_Freq": "w/o Freq",
    "Ablation_No_Time": "w/o Time",
    "Ablation_No_Transformer": "w/o Transformer",
}

_SKIP_DIRS = {"page_compare_bar", "page_confidence", "page_heatmap", "page_ablation_curve", "Cross_Transfer"}


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


def group_by_class(datasets):
    groups = {}
    for class_name, dataset_name in datasets:
        groups.setdefault(class_name, []).append(dataset_name)
    return groups


def apply_nature_style(ax, title=None, panel_label=None):
    if hasattr(ax, 'spines'):
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.grid(True, linestyle='--', alpha=0.3)
    if title:
        ax.set_title(title, fontsize=8.5, fontweight='bold', pad=8)
    if panel_label:
        ax.text(-0.12, 1.08, panel_label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='right')


def plot_confidence_distributions(datasets):
    groups = group_by_class(datasets)
    models = ["Ultimate_Fusion"]

    for class_name, dnames in groups.items():
        dnames = sorted(dnames)
        n_species = len(dnames)

        for mdir in models:
            if not os.path.exists(os.path.join(result_dir, mdir)):
                continue
            mname = model_mapping.get(mdir, mdir)
            plot_data = []

            for dname in dnames:
                prob_path = os.path.join(result_dir, mdir, class_name, dname, "probability.txt")
                y_true, y_prob = read_probability(prob_path)
                if y_true is None or len(y_true) == 0:
                    continue
                valid_indices = ~np.isnan(y_prob)
                y_true = y_true[valid_indices]
                y_prob = y_prob[valid_indices]
                pos_probs = y_prob[y_true == 1]
                neg_probs = y_prob[y_true == 0]
                if len(pos_probs) == 0 or len(neg_probs) == 0:
                    continue

                species_parts = dname.split('_')
                species_short = f"{species_parts[0]}-{species_parts[1]}" if len(species_parts) >= 2 else dname
                plot_data.append({
                    'dname': dname,
                    'species_short': species_short,
                    'pos_probs': pos_probs,
                    'neg_probs': neg_probs
                })

            if not plot_data:
                continue

            cols = n_species if n_species <= 4 else 4
            rows = 1 if n_species <= 4 else (n_species + 3) // 4

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.9))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            if len(axes.shape) == 1:
                axes = axes.reshape(1, -1)

            panel_labels = [chr(ord('a') + i) for i in range(len(plot_data))]
            plotted_count = 0

            for i in range(rows):
                for j in range(cols):
                    idx = i * cols + j
                    if idx >= len(plot_data):
                        axes[i, j].set_visible(False)
                        continue
                    data = plot_data[idx]
                    ax = axes[i, j]

                    sns.kdeplot(data['pos_probs'], color='#d62728', alpha=0.45, label='Positive', ax=ax, fill=True)
                    sns.kdeplot(data['neg_probs'], color='#1f77b4', alpha=0.45, label='Negative', ax=ax, fill=True)

                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Predicted Probability", fontsize=7.5)
                    ax.set_ylabel("Density", fontsize=7.5)
                    ax.legend(frameon=False, fontsize=6.5)
                    apply_nature_style(ax, title=data['species_short'], panel_label=f"({panel_labels[idx]})")
                    plotted_count += 1

            if plotted_count > 0:
                fig.suptitle(f'Prediction Confidence Density Distribution — {class_name} ({mname})',
                             fontweight='bold', y=1.02, fontsize=11)
                plt.tight_layout()
                for out_p in [
                    os.path.join(page_confidence_dir, f"{class_name}_{mdir}_confidence.pdf"),
                    os.path.join(page_confidence_dir, f"{class_name}_{mdir}_confidence.png"),
                    os.path.join(paper_fig_dir, f"fig_{class_name}_confidence.pdf"),
                ]:
                    fig.savefig(out_p, dpi=300, bbox_inches='tight')
                print(f"-> {class_name} confidence KDE distribution generated.")
            plt.close(fig)


def main():
    datasets = scan_datasets()
    if datasets:
        plot_confidence_distributions(datasets)


if __name__ == "__main__":
    main()

