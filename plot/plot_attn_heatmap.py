import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import DNADataset
from model import get_model_by_name, Config
from run_all import resolve_dirs, prepare_all_tsv

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 8


def extract_species_attention(models, test_loader, device, seq_len=41):
    """
    Extracts average 41-bp positional attention weights (alpha_1 ... alpha_41)
    across all positive methylation sequences in the test set using the 5-fold ensemble.
    """
    attn_accum = []
    with torch.no_grad():
        for seqs, labels in test_loader:
            pos_mask = (labels == 1)
            if not torch.any(pos_mask):
                continue
            pos_seqs = seqs[pos_mask].to(device)
            batch_fold_attns = []
            for m in models:
                _ = m(pos_seqs)
                if hasattr(m, "attn_pool") and m.attn_pool.last_attn_weights is not None:
                    # shape: [B_pos, L]
                    batch_fold_attns.append(m.attn_pool.last_attn_weights.numpy())
            if batch_fold_attns:
                avg_batch_attn = np.mean(batch_fold_attns, axis=0)  # [B_pos, L]
                attn_accum.append(avg_batch_attn[:, :seq_len])

    if not attn_accum:
        return None
    all_pos_attn = np.concatenate(attn_accum, axis=0)  # [N_pos, 41]
    mean_attn = np.mean(all_pos_attn, axis=0)          # [41]
    return mean_attn


def plot_attention_heatmaps():
    script_dir, raw_data_root, base_result_dir = resolve_dirs()
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
    base_data_dir = prepare_all_tsv(raw_data_root, categories=("4mC", "6mA"))

    page_dir = os.path.join(project_root, "page", "page_heatmap")
    paper_fig_dir = os.path.join(project_root, "paper_new", "figures")
    os.makedirs(page_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = Config()
    model_name = "Ultimate_Fusion"

    for cat in ["4mC", "6mA"]:
        cat_data_path = os.path.join(base_data_dir, cat)
        cat_model_path = os.path.join(base_result_dir, model_name, cat)
        if not os.path.exists(cat_data_path) or not os.path.exists(cat_model_path):
            print(f"Skipping {cat} attention heatmap (missing data or model weights).")
            continue

        species_list = sorted([
            s for s in os.listdir(cat_model_path)
            if os.path.isdir(os.path.join(cat_model_path, s)) and not s.startswith(".")
        ])

        species_names = []
        attn_rows = []

        for sp in species_list:
            test_tsv = os.path.join(cat_data_path, sp, "test.tsv")
            model_dir = os.path.join(cat_model_path, sp)
            if not os.path.exists(test_tsv):
                continue

            models = []
            for i in range(1, 6):
                p = os.path.join(model_dir, f"model_fold_{i}.pth")
                if not os.path.exists(p):
                    p = os.path.join(model_dir, f"fold_{i}.pth")
                if os.path.exists(p):
                    m = get_model_by_name(model_name, config).to(device)
                    m.load_state_dict(torch.load(p, map_location=device))
                    m.eval()
                    models.append(m)

            if not models:
                continue

            ds = DNADataset(tsv_file=test_tsv, max_len=41)
            loader = DataLoader(ds, batch_size=128, shuffle=False)
            mean_attn = extract_species_attention(models, loader, device, seq_len=41)
            if mean_attn is not None:
                clean_sp = sp.replace(f"{cat}_", "")
                species_names.append(clean_sp)
                attn_rows.append(mean_attn)
                print(f"Extracted 41-bp attention profile for {cat} / {clean_sp}")

        if not attn_rows:
            print(f"No attention profiles extracted for {cat}.")
            continue

        attn_matrix = np.vstack(attn_rows)  # [N_species, 41]
        # Min-max normalize per row for clear visual comparison across species, and keep raw average for top curve
        mean_profile = np.mean(attn_matrix, axis=0)
        norm_matrix = (attn_matrix - attn_matrix.min(axis=1, keepdims=True)) / (
            np.ptp(attn_matrix, axis=1, keepdims=True) + 1e-8
        )

        fig_height = 4.2 if len(species_names) <= 5 else 6.2
        fig = plt.figure(figsize=(11, fig_height))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 2.4], hspace=0.18)

        # Top panel: Mean Attention Profile across 41-bp window
        ax_top = fig.add_subplot(gs[0])
        positions = np.arange(1, 42)
        ax_top.plot(positions, mean_profile, color='#E64B35', linewidth=1.8, marker='o', markersize=3.5, label='Mean Attention Weight ($\\alpha_t$)')
        ax_top.axvspan(18, 25, color='#F39B7F', alpha=0.25, label='Core cis-element window (pos 18–25)')
        ax_top.axvline(21, color='#B2182B', linestyle='--', linewidth=1.2, label='Central Methylation Site (pos 21)')
        ax_top.set_xlim(0.5, 41.5)
        ax_top.set_ylabel("Mean $\\alpha_t$", fontsize=8.5)
        ax_top.set_xticks([])
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.legend(loc='upper right', frameon=False, fontsize=7.5, ncol=3)
        ax_top.set_title(f"41-bp Sequence Positional Attention Weight Distribution ({cat} — TF-DualMethyl)", fontsize=10.5, fontweight='bold', pad=8)

        # Bottom panel: Species x 41-bp Heatmap
        ax_bot = fig.add_subplot(gs[1])
        sns.heatmap(
            norm_matrix, cmap="YlOrRd", cbar=True,
            xticklabels=[str(p) if (p % 2 == 1 or p == 21) else "" for p in positions],
            yticklabels=species_names,
            linewidths=0.3, linecolor='white',
            cbar_kws={" orientation": "vertical", "pad": 0.02, "label": "Normalized Attention"},
            ax=ax_bot
        )
        ax_bot.axvline(20.5, color='#B2182B', linestyle='--', linewidth=1.2)
        ax_bot.set_xlabel("Nucleotide Position along 41-bp Window (Center = Position 21)", fontsize=8.5)
        ax_bot.set_ylabel("Species", fontsize=8.5)
        plt.setp(ax_bot.get_yticklabels(), rotation=0)

        for out_p in [
            os.path.join(page_dir, f"{cat}_attn_heatmap.pdf"),
            os.path.join(page_dir, f"{cat}_attn_heatmap.png"),
            os.path.join(paper_fig_dir, f"fig_{cat}_attn_heatmap.pdf"),
        ]:
            fig.savefig(out_p, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"-> Saved 41-bp sequence attention heatmap for {cat}")


if __name__ == "__main__":
    plot_attention_heatmaps()

