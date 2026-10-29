import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_all import resolve_dirs
import plot_com_bar
import plot_confidence

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

script_dir, _, _ = resolve_dirs()
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
paper_fig_dir = os.path.join(project_root, "paper_new", "figures")
os.makedirs(paper_fig_dir, exist_ok=True)


def draw_box(ax, x, y, w, h, text, subtext=None, fc="#F8F9FA", ec="#343A40", lw=1.3, fontsize=8.5, text_color="#111111"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    if subtext:
        ax.text(x + w / 2, y + h * 0.64, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, subtext, ha='center', va='center', fontsize=fontsize - 1.3, color="#333333", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="#343A40", lw=1.3):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>,head_length=5,head_width=3.2",
        color=color, linewidth=lw, zorder=2
    )
    ax.add_patch(arr)


def generate_architecture_pdf():
    """Generates a crisp vector PDF architecture diagram for TF-DualMethyl."""
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Stage background containers
    stages = [
        (0.01, 0.10, 0.16, 0.80, "Input & Embedding", "#EDF2F7", "#A0AEC0"),
        (0.20, 0.08, 0.31, 0.84, "Stage I: Multi-Resolution Time-Frequency Front-End", "#EBF8FF", "#4299E1"),
        (0.54, 0.08, 0.23, 0.84, "Stage II: Dual-Path Context Backbone", "#FAF5FF", "#805AD5"),
        (0.80, 0.10, 0.19, 0.80, "Stage III: Fusion & Classifier", "#FFF5F5", "#E53E3E"),
    ]
    for x, y, w, h, title, fc, ec in stages:
        bg = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=fc, edgecolor=ec, linewidth=1.2, linestyle='--', alpha=0.65, zorder=1)
        ax.add_patch(bg)
        ax.text(x + w / 2, y + h - 0.035, title, ha='center', va='center', fontsize=8.0, fontweight='bold', color=ec, zorder=2)

    # Title
    ax.text(0.5, 0.965, "TF-DualMethyl: Multi-Resolution Time-Frequency & Dual-Path Deep Learning Architecture",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color="#1A202C")

    # Input Stage
    draw_box(ax, 0.025, 0.62, 0.13, 0.16, "41-bp DNA Sequence", "A, C, G, T, N (Center = pos 21)", fc="#FFFFFF", ec="#4A5568", fontsize=8.0)
    draw_box(ax, 0.025, 0.24, 0.13, 0.22, "Token + Sinusoidal PE", "d=128 + PosEncoding\nX0 in R^(B x 41 x 128)", fc="#E2E8F0", ec="#2D3748", fontsize=8.0)
    draw_arrow(ax, 0.09, 0.62, 0.09, 0.46)

    # Stage I: Serial-Parallel Time-Frequency Front-End
    # Top branch: Temporal Path (TCN -> MSCNN)
    draw_box(ax, 0.22, 0.58, 0.11, 0.20, "Dilated TCN", "Causal Conv1d\nd = 1, 2, 4", fc="#BEE3F8", ec="#2B6CB0")
    draw_box(ax, 0.35, 0.58, 0.11, 0.20, "Multi-Scale CNN", "Parallel Kernels\nk = 3, 5, 7 -> F_ts", fc="#90CDF4", ec="#2C5282")
    draw_arrow(ax, 0.33, 0.68, 0.35, 0.68, color="#2B6CB0")

    # Bottom branch: Frequency-Wavelet Path (VMD -> Wavelet)
    draw_box(ax, 0.22, 0.20, 0.11, 0.20, "VMD Spectral Block", "rFFT -> K=4 Modes\n-> irFFT", fc="#C6F6D5", ec="#2F855A")
    draw_box(ax, 0.35, 0.20, 0.11, 0.20, "Multi-Wavelet CNN", "Haar / Db2 / Symlet\ns = 2, 4, 8 -> F_fw", fc="#9AE6B4", ec="#276749")
    draw_arrow(ax, 0.33, 0.30, 0.35, 0.30, color="#2F855A")

    # Arrows from X0 to TCN and VMD
    draw_arrow(ax, 0.155, 0.35, 0.22, 0.68, color="#2B6CB0")
    draw_arrow(ax, 0.155, 0.35, 0.22, 0.30, color="#2F855A")

    # Residual LayerNorm Node
    draw_box(ax, 0.47, 0.41, 0.032, 0.16, "LN\n(+)", None, fc="#FEFCBF", ec="#B7791F", fontsize=7.5)
    draw_arrow(ax, 0.46, 0.68, 0.486, 0.57, color="#2C5282")
    draw_arrow(ax, 0.46, 0.30, 0.486, 0.41, color="#276749")
    draw_arrow(ax, 0.155, 0.35, 0.47, 0.49, color="#718096")

    # Stage II: Dual-Path Context Backbone
    draw_box(ax, 0.56, 0.56, 0.19, 0.22, "2-Layer BiLSTM Path", "Sequential Directional Context\nO_lstm in R^(B x 41 x 256)", fc="#E9D8FD", ec="#6B46C1")
    draw_box(ax, 0.56, 0.20, 0.19, 0.22, "Transformer Encoder", "2 Layers, 8 Heads Self-Attn\nO_trans in R^(B x 41 x 128)", fc="#D6BCFA", ec="#553C9A")
    draw_arrow(ax, 0.502, 0.49, 0.56, 0.67, color="#6B46C1")
    draw_arrow(ax, 0.502, 0.49, 0.56, 0.31, color="#553C9A")

    # Stage III: Attention Pooling & Classifier
    draw_box(ax, 0.815, 0.60, 0.16, 0.19, "Concat & Attn Pool", "Z = [O_lstm || O_trans] (d=384)\nalpha_t = Softmax(W2 tanh(W1 Z_t))", fc="#FED7D7", ec="#C53030")
    draw_box(ax, 0.815, 0.36, 0.16, 0.17, "MLP Classifier", "Linear(384->128) + BN\n+ GELU + Dropout + Linear", fc="#FEB2B2", ec="#9B2C2C")
    draw_box(ax, 0.815, 0.14, 0.16, 0.15, "5-Fold Ensemble", "Methylation Site Prob.\nP(y = 1 | S) in [0, 1]", fc="#E53E3E", ec="#742A2A", text_color="#FFFFFF")

    draw_arrow(ax, 0.75, 0.67, 0.815, 0.695, color="#C53030")
    draw_arrow(ax, 0.75, 0.31, 0.815, 0.695, color="#C53030")
    draw_arrow(ax, 0.895, 0.60, 0.895, 0.53, color="#9B2C2C")
    draw_arrow(ax, 0.895, 0.36, 0.895, 0.29, color="#742A2A")

    out_pdf = os.path.join(paper_fig_dir, "fig_architecture.pdf")
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Generated vector architecture diagram: {out_pdf}")


def generate_vector_cross_heatmaps():
    """Renders crisp vector PDF cross-species transferability heatmaps from the evaluated AUC matrices."""
    sp_4mc = ["C.equisetifolia", "F.vesca", "S.cerevisiae", "Tolypocladium"]
    mat_4mc = np.array([
        [0.9042, 0.7458, 0.6285, 0.5849],
        [0.8528, 0.9345, 0.6526, 0.6790],
        [0.7854, 0.6796, 0.7846, 0.6299],
        [0.8103, 0.7784, 0.7416, 0.8353],
    ])

    sp_6ma = [
        "A.thaliana", "C.elegans", "C.equisetifolia", "D.melanogaster",
        "F.vesca", "H.sapiens", "R.chinensis", "S.cerevisiae",
        "T.thermophile", "Tolypocladium", "Xoc BLS256"
    ]
    mat_6ma = np.array([
        [0.939, 0.804, 0.818, 0.961, 0.973, 0.951, 0.940, 0.870, 0.626, 0.808, 0.664],
        [0.838, 0.971, 0.760, 0.909, 0.814, 0.871, 0.774, 0.858, 0.561, 0.766, 0.653],
        [0.892, 0.850, 0.824, 0.923, 0.958, 0.931, 0.919, 0.844, 0.625, 0.817, 0.705],
        [0.897, 0.708, 0.762, 0.973, 0.953, 0.908, 0.917, 0.856, 0.521, 0.778, 0.703],
        [0.858, 0.597, 0.753, 0.892, 0.977, 0.888, 0.938, 0.771, 0.509, 0.751, 0.605],
        [0.910, 0.799, 0.795, 0.930, 0.950, 0.969, 0.926, 0.829, 0.613, 0.820, 0.642],
        [0.744, 0.630, 0.684, 0.816, 0.862, 0.810, 0.905, 0.721, 0.562, 0.753, 0.660],
        [0.834, 0.810, 0.758, 0.915, 0.877, 0.833, 0.842, 0.905, 0.575, 0.768, 0.673],
        [0.668, 0.546, 0.625, 0.733, 0.847, 0.736, 0.849, 0.624, 0.948, 0.677, 0.490],
        [0.816, 0.749, 0.750, 0.903, 0.918, 0.862, 0.909, 0.817, 0.538, 0.860, 0.719],
        [0.632, 0.532, 0.593, 0.788, 0.602, 0.681, 0.643, 0.654, 0.379, 0.670, 0.957],
    ])

    for cat, species, matrix, fmt_str, figsize in [
        ("4mC", sp_4mc, mat_4mc, ".4f", (6.0, 4.8)),
        ("6mA", sp_6ma, mat_6ma, ".3f", (7.8, 6.3)),
    ]:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            matrix, annot=True, fmt=fmt_str,
            cmap="YlGnBu", cbar=True, square=True,
            xticklabels=species, yticklabels=species,
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 6.5 if len(species) > 5 else 8.5},
            cbar_kws={"shrink": 0.82, "label": "Cross-Species ROC-AUC"}, ax=ax
        )
        ax.set_title(f"Cross-Species Zero-Shot Transferability ({cat} — TF-DualMethyl)", fontsize=10, fontweight='bold', pad=10)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel("Source (Trained) Species", fontsize=8.5)
        ax.set_xlabel("Target (Testing) Species", fontsize=8.5)
        plt.tight_layout()
        out_pdf = os.path.join(paper_fig_dir, f"fig_{cat}_cross_heatmap.pdf")
        fig.savefig(out_pdf, bbox_inches='tight')
        plt.close(fig)
        print(f"-> Generated vector cross-species heatmap: {out_pdf}")


def generate_manifold_and_motif_pdfs():
    """Combines representative t-SNE and UMAP plots into a multi-panel PDF and exports the motif saliency PDF."""
    tsne_dir = os.path.join(project_root, "page", "page_tsne")
    umap_dir = os.path.join(project_root, "page", "page_umap")
    motif_png = os.path.join(project_root, "page", "page_motif", "motif_saliency_heatmap_6mA_F.vesca.png")

    panels = [
        ("4mC", "4mC_F.vesca", "4mC — F. vesca"),
        ("4mC", "4mC_Tolypocladium", "4mC — Tolypocladium"),
        ("6mA", "6mA_D.melanogaster", "6mA — D. melanogaster"),
        ("6mA", "6mA_F.vesca", "6mA — F. vesca"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4))
    letters = "abcdefgh"
    for col, (cat, sp_dir, label) in enumerate(panels):
        tsne_p = os.path.join(tsne_dir, f"{cat}_{sp_dir}_tsne_nature.png")
        umap_p = os.path.join(umap_dir, f"{cat}_{sp_dir}_umap_nature.png")

        for row, (img_p, method_name) in enumerate([(tsne_p, "t-SNE"), (umap_p, "UMAP")]):
            ax = axes[row, col]
            idx = row * 4 + col
            if os.path.exists(img_p):
                img = Image.open(img_p)
                ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"({letters[idx]}) {label} ({method_name})", fontsize=8.5, fontweight='bold', pad=4)

    plt.suptitle("TF-DualMethyl Penultimate-Layer Feature Manifold Projections (t-SNE & UMAP)", fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_manifold = os.path.join(paper_fig_dir, "fig_tsne_umap_manifold.pdf")
    fig.savefig(out_manifold, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"-> Generated manifold PDF: {out_manifold}")

    if os.path.exists(motif_png):
        img = Image.open(motif_png)
        # Crop out old top title bar (~6% top) and add updated TF-DualMethyl title
        w, h = img.size
        cropped = img.crop((0, int(h * 0.045), w, h))
        fig, ax = plt.subplots(figsize=(13, 11.5))
        ax.imshow(cropped)
        ax.axis('off')
        ax.set_title("TF-DualMethyl Nucleotide-Level Motif Saliency Heatmap (Layer Integrated Gradients) — 6mA (F. vesca)",
                     fontsize=12.5, fontweight='bold', pad=10)
        plt.tight_layout()
        out_motif = os.path.join(paper_fig_dir, "fig_motif_saliency_6mA_F.vesca.pdf")
        fig.savefig(out_motif, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"-> Generated motif saliency PDF: {out_motif}")


def generate_pending_server_placeholders():
    """
    Creates clean vector PDF placeholder figures for the experiments that require re-running
    via model_plus/run_all.sh on the GPU server (Ablation Study & 41-bp Sequence Attention Heatmaps).
    When run_all.sh finishes on the server, these exact PDF files are automatically overwritten.
    """
    placeholders = [
        ("fig_4mC_ablation_metrics.pdf", "4mC Ablation Study — Multi-Metric Comparison (ACC, SN, SP, MCC, AUC, F1)",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_ablation_curve.py)\nFixes Serial-Parallel Front-End Inheritance & Early-Stopping Patience (patience=15)", (12, 5.5)),
        ("fig_4mC_ablation_roc.pdf", "4mC Ablation Study — Receiver Operating Characteristic (ROC) Curves",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_ablation_curve.py)", (9.5, 4.8)),
        ("fig_4mC_ablation_pr.pdf", "4mC Ablation Study — Precision-Recall (PR) Curves",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_ablation_curve.py)", (9.5, 4.8)),
        ("fig_6mA_ablation_metrics.pdf", "6mA Ablation Study — Multi-Metric Comparison Across 11 Species",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_ablation_curve.py)\nFixes Serial-Parallel Front-End Inheritance & Early-Stopping Patience (patience=15)", (12, 5.5)),
        ("fig_6mA_ablation_roc.pdf", "6mA Ablation Study — Receiver Operating Characteristic (ROC) Curves",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_ablation_curve.py)", (11, 5.2)),
        ("fig_6mA_ablation_pr.pdf", "6mA Ablation Study — Precision-Recall (PR) Curves",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_ablation_curve.py)", (11, 5.2)),
        ("fig_4mC_attn_heatmap.pdf", "4mC 41-bp Sequence Positional Attention Weight Heatmap (alpha_1 ... alpha_41)",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_attn_heatmap.py)\nExtracts True AttentionPooling Weights Centered on Methylated Cytosine (Position 21)", (10.5, 3.8)),
        ("fig_6mA_attn_heatmap.pdf", "6mA 41-bp Sequence Positional Attention Weight Heatmap (alpha_1 ... alpha_41)",
         "Pending Server Execution of `model_plus/run_all.sh` (plot/plot_attn_heatmap.py)\nExtracts True AttentionPooling Weights Centered on Methylated Adenine (Position 21)", (10.5, 4.5)),
    ]

    for fname, title, subtitle, figsize in placeholders:
        out_p = os.path.join(paper_fig_dir, fname)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        box = FancyBboxPatch((0.03, 0.08), 0.94, 0.84, boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.5, linestyle='--')
        ax.add_patch(box)
        ax.text(0.5, 0.64, f"[Figure Placeholder: {fname}]", ha='center', va='center', fontsize=12, fontweight='bold', color="#1E293B")
        ax.text(0.5, 0.50, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color="#0F172A")
        ax.text(0.5, 0.32, subtitle, ha='center', va='center', fontsize=9, color="#475569")
        plt.tight_layout()
        fig.savefig(out_p, bbox_inches='tight')
        plt.close(fig)
        print(f"-> Created clean vector placeholder PDF: {out_p}")


if __name__ == "__main__":
    generate_architecture_pdf()
    plot_com_bar.main()
    plot_confidence.main()
    generate_vector_cross_heatmaps()
    generate_manifold_and_motif_pdfs()
    generate_pending_server_placeholders()
    print("All paper_new/figures/*.pdf files prepared!")
