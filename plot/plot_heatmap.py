import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_all import resolve_dirs

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8

script_dir, _, base_result_dir = resolve_dirs()
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
result_root = os.path.join(base_result_dir, "Cross_Transfer")
page_dir = os.path.join(project_root, "page", "page_heatmap")
paper_fig_dir = os.path.join(project_root, "paper_new", "figures")
os.makedirs(page_dir, exist_ok=True)
os.makedirs(paper_fig_dir, exist_ok=True)


def read_metrics(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep='\t')
        if 'True_Label' not in df.columns or 'Probability' not in df.columns:
            return None
        y_true = df['True_Label'].values
        y_prob = df['Probability'].values
        if len(np.unique(y_true)) > 1:
            return roc_auc_score(y_true, y_prob)
        return np.nan
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def get_species_list(category_path):
    train_species = set()
    test_species = set()
    for folder in os.listdir(category_path):
        if folder.startswith("Train_") and "_Test_" in folder:
            parts = folder.split("_Test_")
            s_train = parts[0].replace("Train_", "")
            s_test = parts[1]
            train_species.add(s_train)
            test_species.add(s_test)
    return sorted(list(train_species | test_species))


def plot_cross_species_heatmaps():
    categories = ["4mC", "6mA", "5hmC"]

    for cat in categories:
        cat_path = os.path.join(result_root, cat)
        if not os.path.exists(cat_path):
            print(f"Category {cat} not found in {result_root}, skipping...")
            continue

        species = get_species_list(cat_path)
        n = len(species)
        if n == 0:
            continue
        matrix = np.full((n, n), np.nan)

        for r, s_train in enumerate(species):
            for c, s_test in enumerate(species):
                folder_name = f"Train_{s_train}_Test_{s_test}"
                prob_path = os.path.join(cat_path, folder_name, "probability.txt")
                val = read_metrics(prob_path)
                if val is not None:
                    matrix[r, c] = val

        fig, ax = plt.subplots(figsize=(7.5, 6.0) if n > 5 else (6.0, 4.8))
        sns.heatmap(
            matrix, annot=True, fmt=".3f" if n > 5 else ".4f",
            cmap="YlGnBu", cbar=True, square=True,
            xticklabels=species, yticklabels=species,
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 6.5 if n > 5 else 8},
            cbar_kws={"shrink": 0.82, "label": "Cross-Species ROC-AUC"}, ax=ax
        )

        ax.set_title(f"Cross-Species Zero-Shot Transferability ({cat} — TF-DualMethyl)", fontsize=10, fontweight='bold', pad=10)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel("Source (Trained) Species", fontsize=8.5)
        ax.set_xlabel("Target (Testing) Species", fontsize=8.5)

        plt.tight_layout()
        for out_p in [
            os.path.join(page_dir, f"{cat}.pdf"),
            os.path.join(page_dir, f"{cat}.png"),
            os.path.join(paper_fig_dir, f"fig_{cat}_cross_heatmap.pdf"),
        ]:
            fig.savefig(out_p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"-> Cross-species transferability heatmap generated for {cat}")


if __name__ == "__main__":
    plot_cross_species_heatmaps()

