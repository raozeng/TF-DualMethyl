import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from model import Ultimate_Fusion, Config
from dataset import DNADataset
from run_all import resolve_dirs, prepare_all_tsv

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8


def apply_nature_style(ax, title=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)


def extract_features(model, dataloader, device):
    model.eval()
    all_labels = []
    features_list = []

    def hook_fn(module, input, output):
        features_list.append(output.detach().cpu())

    handle = model.classifier[3].register_forward_hook(hook_fn)
    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs = seqs.to(device)
            _ = model(seqs)
            all_labels.extend(labels.numpy())
    handle.remove()
    all_features = torch.cat(features_list, dim=0).numpy()
    return all_features, np.array(all_labels)


def plot_single_tsne(features_raw, labels_raw, save_paths, title_prefix=""):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw)
    n_samples = len(labels_raw)

    max_samples = 3000
    if n_samples > max_samples:
        pos_idx = np.where(labels_raw == 1)[0]
        neg_idx = np.where(labels_raw == 0)[0]
        n_pos = n_neg = max_samples // 2
        np.random.seed(42)
        sel_pos = np.random.choice(pos_idx, min(len(pos_idx), n_pos), replace=False)
        sel_neg = np.random.choice(neg_idx, min(len(neg_idx), n_neg), replace=False)
        sel_idx = np.concatenate([sel_pos, sel_neg])
        features_scaled = features_scaled[sel_idx]
        labels_raw = labels_raw[sel_idx]
        n_samples = len(labels_raw)

    pca_components = min(50, features_scaled.shape[1], n_samples)
    pca = PCA(n_components=pca_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)

    current_perplexity = min(30, max(5, n_samples // 3))
    print(f"  - Calculating t-SNE for {title_prefix} (n={n_samples}, perplexity={current_perplexity})...")

    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        perplexity=current_perplexity,
        random_state=42,
        n_jobs=1
    )
    feat_2d = tsne.fit_transform(features_pca)

    plot_idx = np.arange(len(labels_raw))
    np.random.seed(42)
    np.random.shuffle(plot_idx)
    feat_2d = feat_2d[plot_idx]
    labels_shuffled = labels_raw[plot_idx]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors_map = {0: '#9400D3', 1: '#FF0000'}
    color_list = [colors_map[l] for l in labels_shuffled]

    ax.scatter(feat_2d[:, 0], feat_2d[:, 1], c=color_list, alpha=0.6, s=8, edgecolors='none')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='#9400D3', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='#FF0000', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=False, fontsize=8)

    apply_nature_style(ax, title="Feature Distribution (t-SNE — TF-DualMethyl)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE dimension 1", fontsize=9)
    ax.set_ylabel("t-SNE dimension 2", fontsize=9)

    plt.suptitle(f"Embedding t-SNE Visualization\n{title_prefix}", fontsize=11, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for sp in save_paths:
        fig.savefig(sp, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    script_dir, raw_data_root, base_result_dir = resolve_dirs()
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
    data_root = prepare_all_tsv(raw_data_root, categories=("4mC", "6mA"))
    result_dir = os.path.join(base_result_dir, "Ultimate_Fusion")
    page_tsne_dir = os.path.join(project_root, "page", "page_tsne")
    os.makedirs(page_tsne_dir, exist_ok=True)

    if not os.path.exists(result_dir):
        print(f"Directory {result_dir} not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = Config()
    target_fold = 3

    for category in sorted(os.listdir(result_dir)):
        cat_path = os.path.join(result_dir, category)
        if not os.path.isdir(cat_path) or category.startswith("."):
            continue

        for species in sorted(os.listdir(cat_path)):
            spec_path = os.path.join(cat_path, species)
            if not os.path.isdir(spec_path) or species.startswith("."):
                continue

            best_weight_path = os.path.join(spec_path, f"model_fold_{target_fold}.pth")
            if not os.path.exists(best_weight_path):
                continue

            test_tsv = os.path.join(data_root, category, species, "test.tsv")
            if not os.path.exists(test_tsv):
                continue

            print(f"Processing t-SNE: {category} - {species}")
            dataset = DNADataset(test_tsv)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

            model = Ultimate_Fusion(config).to(device)
            model.load_state_dict(torch.load(best_weight_path, map_location=device))

            feat_best, labels_best = extract_features(model, dataloader, device)
            save_paths = [
                os.path.join(page_tsne_dir, f"{category}_{species}_tsne_nature.pdf"),
                os.path.join(page_tsne_dir, f"{category}_{species}_tsne_nature.png"),
            ]
            plot_single_tsne(feat_best, labels_best, save_paths, title_prefix=f"{category} | {species}")

            del model, dataset, dataloader, feat_best, labels_best
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

