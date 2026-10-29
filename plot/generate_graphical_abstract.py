import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

fig, ax = plt.subplots(figsize=(18.5, 10.4), dpi=300)
ax.set_xlim(0, 185)
ax.set_ylim(0, 104)
ax.axis('off')
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')


def draw_card(ax, x, y, w, h, facecolor='#FFFFFF', edgecolor='#D0D7DE',
              lw=1.2, rounding=1.4, zorder=2, alpha=1.0, linestyle='-'):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, linestyle=linestyle, zorder=zorder, alpha=alpha
    )
    ax.add_patch(patch)
    return patch


def draw_panel(ax, x, y, w, h, num, title, bg='#FAFBFD', border='#CBD5E1',
               hdr_bg='#EBF2FA', hdr_tc='#1E3A5F'):
    draw_card(ax, x, y, w, h, facecolor=bg, edgecolor=border, lw=1.5, rounding=2.0, zorder=1)
    hdr_h = 6.2
    draw_card(ax, x + 0.3, y + h - hdr_h - 0.2, w - 0.6, hdr_h,
              facecolor=hdr_bg, edgecolor='none', lw=0, rounding=1.7, zorder=2)
    ax.plot([x + 0.5, x + w - 0.5], [y + h - hdr_h - 0.2, y + h - hdr_h - 0.2],
            color=border, lw=1.1, zorder=3)
    ax.text(x + w / 2, y + h - hdr_h / 2 - 0.2,
            f"{num}.  {title}",
            ha='center', va='center', fontsize=12.6, fontweight='bold',
            color=hdr_tc, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', lw=1.6, zorder=10):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>,head_length=5.0,head_width=3.2',
        color=color, linewidth=lw, zorder=zorder
    )
    ax.add_patch(arr)


# ==============================================================================
# TOP BANNER
# ==============================================================================
draw_card(ax, 1.8, 96.2, 181.4, 6.4, facecolor='#1E3A5F', edgecolor='#1E3A5F', lw=1.2, rounding=1.6, zorder=2)
ax.text(92.5, 99.4,
        "TF-DualMethyl: Multi-Resolution Time-Frequency & Dual-Path Deep Learning for Cross-Species DNA Methylation Prediction",
        ha='center', va='center', fontsize=13.5, fontweight='bold', color='#FFFFFF', zorder=4)

# ==============================================================================
# PANEL 1 (TOP-LEFT): Cross-Species Epigenomic Challenge (x=1.8..66.8, y=50.2..94.6)
# ==============================================================================
draw_panel(ax, 1.8, 50.2, 65.0, 44.4, "1",
           "Cross-Species 4mC & 6mA Benchmarks (15 Species)",
           bg='#FAFBFD', border='#D5DDE6', hdr_bg='#EBF2FA', hdr_tc='#2C4A7E')

# Left sub-card: 41-bp Non-Stationary DNA Signal
draw_card(ax, 3.6, 68.8, 61.4, 18.2, facecolor='#FFFDF9', edgecolor='#EED2B5', lw=1.2, zorder=3)
ax.text(34.3, 84.4, "41-bp Window: Local Consensus Motif + Helical Periodicity (~10.5 bp)",
        ha='center', va='center', fontsize=9.5, fontweight='bold', color='#8E4B1E', zorder=4)

# Draw DNA double helix + periodic envelope
hx = np.linspace(5.8, 37.8, 120)
hy1 = 76.2 + 3.1 * np.sin((hx - 5.8) * 0.55)
hy2 = 76.2 - 3.1 * np.sin((hx - 5.8) * 0.55)
ax.plot(hx, hy1, color='#E59866', lw=2.0, zorder=5)
ax.plot(hx, hy2, color='#5B9BD5', lw=2.0, zorder=5)
for bx_pos in np.linspace(6.6, 37.0, 15):
    by1 = 76.2 + 3.1 * np.sin((bx_pos - 5.8) * 0.55)
    by2 = 76.2 - 3.1 * np.sin((bx_pos - 5.8) * 0.55)
    is_mid = abs(bx_pos - 21.8) < 1.2
    ax.plot([bx_pos, bx_pos], [by1, by2],
            color=('#C0392B' if is_mid else '#BDC3C7'),
            lw=(2.4 if is_mid else 1.0), zorder=4)
ax.scatter([21.8], [76.2], s=95, color='#F1948A', alpha=0.5, zorder=6)
ax.scatter([21.8], [76.2], s=28, color='#C0392B', zorder=7)
ax.text(21.8, 70.6, "Position 21 Target: C21 (4mC) / A21 (6mA)",
        ha='center', va='center', fontsize=9.2, fontweight='bold', color='#C0392B', zorder=6)

# Right box inside top sub-card: Non-stationary decomposition concept
draw_card(ax, 40.0, 70.4, 23.4, 11.8, facecolor='#FFFFFF', edgecolor='#E8B88A', lw=1.0, zorder=4)
ax.text(51.7, 79.8, "Non-Stationary DNA Signal", ha='center', va='center', fontsize=9.1, fontweight='bold', color='#8E4B1E', zorder=5)
ax.text(51.7, 76.4, "• Short cis-motifs (3–7 bp)", ha='center', va='center', fontsize=8.5, color='#334155', zorder=5)
ax.text(51.7, 73.8, "• Helical pitch (~10.5 bp)", ha='center', va='center', fontsize=8.5, color='#334155', zorder=5)
ax.text(51.7, 71.4, "• Background mutation noise", ha='center', va='center', fontsize=8.5, color='#334155', zorder=5)

# Bottom sub-card: 15 Cross-Species Datasets (489,152 sequences)
draw_card(ax, 3.6, 52.0, 61.4, 15.4, facecolor='#F7FBFE', edgecolor='#B5D6EC', lw=1.2, zorder=3)
ax.text(34.3, 64.8, "15 iDNA-MS Benchmarks (489,152 Sequences Across 5 Taxonomic Clades)",
        ha='center', va='center', fontsize=9.4, fontweight='bold', color='#24557A', zorder=4)

clades = [
    ("4mC (4 Species)", "Plants & Fungi\n66,938 seqs", '#EFF9F5', '#8FD3C1', '#1E6B52'),
    ("Plants (4)", "A. thaliana, F. vesca\nR. chinensis, C. eq.", '#FFFDF5', '#E6BE8A', '#8E4B1E'),
    ("Metazoans (3)", "H. sapiens, C. elegans\nD. melanogaster", '#FDF2F4', '#E6A8B2', '#8A2B3E'),
    ("Fungi / Prot. / Bact.", "S. cerev., Tolyp.,\nT. therm., Xoc BLS256", '#EDF6FC', '#8EC2E6', '#24557A'),
]
for idx, (c_title, c_sub, c_bg, c_ec, c_tc) in enumerate(clades):
    cx = 4.8 + idx * 14.9
    draw_card(ax, cx, 53.2, 14.1, 9.6, facecolor=c_bg, edgecolor=c_ec, lw=1.1, rounding=1.0, zorder=4)
    ax.text(cx + 7.05, 60.4, c_title, ha='center', va='center', fontsize=8.7, fontweight='bold', color=c_tc, zorder=5)
    ax.text(cx + 7.05, 56.0, c_sub, ha='center', va='center', fontsize=7.8, color='#334155', zorder=5)


# ==============================================================================
# PANEL 2 (TOP-RIGHT): TF-DualMethyl Architecture (x=69.2..183.2, y=50.2..94.6)
# ==============================================================================
draw_panel(ax, 69.2, 50.2, 114.0, 44.4, "2",
           "TF-DualMethyl: Multi-Resolution Time-Frequency & Dual-Path Architecture",
           bg='#FAFBFD', border='#D5DDE6', hdr_bg='#FAF1E2', hdr_tc='#7E5119')

draw_arrow(ax, 66.8, 72.4, 69.2, 72.4, color='#3B82F6', lw=2.0)

# Sub-box 2A: Input Encoding
draw_card(ax, 71.0, 52.2, 20.5, 34.8, facecolor='#FFFFFF', edgecolor='#CBE3F5', lw=1.2, zorder=3)
ax.text(81.25, 84.2, "Input Encoding", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#24557A', zorder=4)
draw_card(ax, 72.4, 71.0, 17.7, 10.4, facecolor='#FFFDF9', edgecolor='#E8B88A', lw=1.0, zorder=4)
ax.text(81.25, 77.8, "Token Embedding", ha='center', va='center', fontsize=9.4, fontweight='bold', color='#8E4B1E', zorder=5)
ax.text(81.25, 73.8, "√d · E(S), d = 128", ha='center', va='center', fontsize=8.8, color='#546E7A', zorder=5)

draw_card(ax, 72.4, 58.2, 17.7, 10.4, facecolor='#EDF6FC', edgecolor='#8EC2E6', lw=1.0, zorder=4)
ax.text(81.25, 65.0, "Sinusoidal PE", ha='center', va='center', fontsize=9.4, fontweight='bold', color='#24557A', zorder=5)
ax.text(81.25, 61.0, "Phase Relative to t=21", ha='center', va='center', fontsize=8.5, color='#546E7A', zorder=5)
ax.text(81.25, 54.6, "X₀ ∈ ℝ⁴¹ˣ¹²⁸", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#334155', zorder=5)

draw_arrow(ax, 91.5, 76.2, 94.2, 76.2, color='#D38B56', lw=1.6)
draw_arrow(ax, 91.5, 62.4, 94.2, 62.4, color='#45937A', lw=1.6)

# Sub-box 2B: Multi-Resolution Time-Frequency Front-End
draw_card(ax, 94.2, 52.2, 34.0, 34.8, facecolor='#FFFDF7', edgecolor='#E6BE8A', lw=1.2, zorder=3)
ax.text(111.2, 84.2, "Time-Frequency Front-End", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#9C640C', zorder=4)

# Temporal-Spatial Branch
draw_card(ax, 95.8, 71.8, 30.8, 9.8, facecolor='#FFFFFF', edgecolor='#E59866', lw=1.1, zorder=4)
ax.text(111.2, 79.0, "Temporal-Spatial Branch", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#8E4B1E', zorder=5)
ax.text(111.2, 75.8, "Causal Dilated TCN (d=1,2,4)", ha='center', va='center', fontsize=8.6, color='#334155', zorder=5)
ax.text(111.2, 73.3, "+ Multi-Scale CNN (k=3,5,7)", ha='center', va='center', fontsize=8.6, color='#334155', zorder=5)

# Gated Fusion pill
draw_card(ax, 95.8, 66.4, 30.8, 4.0, facecolor='#FDF2F4', edgecolor='#E69A9A', lw=1.1, linestyle='--', zorder=4)
ax.text(111.2, 68.4, "3-Branch Adaptive Gated Fusion", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#A93226', zorder=5)

# Frequency-Wavelet Branch
draw_card(ax, 95.8, 54.0, 30.8, 11.0, facecolor='#F2FBF9', edgecolor='#7CC6B6', lw=1.1, zorder=4)
ax.text(111.2, 62.4, "Frequency-Wavelet Branch", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E6B52', zorder=5)
ax.text(111.2, 59.0, "Learnable rFFT VMD (K=4 Modes)", ha='center', va='center', fontsize=8.6, color='#334155', zorder=5)
ax.text(111.2, 56.2, "+ Haar/Db2/Gauss QMF Shrinkage", ha='center', va='center', fontsize=8.6, color='#334155', zorder=5)

draw_arrow(ax, 128.2, 76.2, 131.0, 76.2, color='#C88A4B', lw=1.6)
draw_arrow(ax, 128.2, 62.4, 131.0, 62.4, color='#5B9BD5', lw=1.6)

# Sub-box 2C: Dual-Path Context Backbone
draw_card(ax, 131.0, 52.2, 27.6, 34.8, facecolor='#F8FAFC', edgecolor='#B8C4D4', lw=1.2, zorder=3)
ax.text(144.8, 84.2, "Dual-Path Backbone", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#2E4057', zorder=4)

draw_card(ax, 132.4, 71.0, 24.8, 10.6, facecolor='#FFFDF9', edgecolor='#E6BE8A', lw=1.1, zorder=4)
ax.text(144.8, 78.4, "2-Layer BiLSTM", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#8E4B1E', zorder=5)
ax.text(144.8, 75.2, "5'↔3' Reading-Frame Order", ha='center', va='center', fontsize=8.5, color='#334155', zorder=5)
ax.text(144.8, 72.6, "O_lstm ∈ ℝ⁴¹ˣ²⁵⁶", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#8E4B1E', zorder=5)

draw_card(ax, 132.4, 58.6, 24.8, 10.6, facecolor='#F2F8FD', edgecolor='#9BC2E6', lw=1.1, zorder=4)
ax.text(144.8, 66.0, "8-Head MHSA Transformer", ha='center', va='center', fontsize=9.3, fontweight='bold', color='#24557A', zorder=5)
ax.text(144.8, 62.8, "Global All-Pairs Epistasis", ha='center', va='center', fontsize=8.5, color='#334155', zorder=5)
ax.text(144.8, 60.2, "O_trans ∈ ℝ⁴¹ˣ¹²⁸", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#24557A', zorder=5)

draw_card(ax, 132.4, 53.4, 24.8, 3.8, facecolor='#FAF1E2', edgecolor='#E5A970', lw=1.0, zorder=4)
ax.text(144.8, 55.3, "SE Channel Recalibration (384-d)", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#8E4B1E', zorder=5)

draw_arrow(ax, 158.6, 68.4, 161.2, 68.4, color='#8A2B3E', lw=1.6)

# Sub-box 2D: Target-Anchored Readout & 5-Fold Ensemble
draw_card(ax, 161.2, 52.2, 20.4, 34.8, facecolor='#FDF6F7', edgecolor='#E6A8B2', lw=1.2, zorder=3)
ax.text(171.4, 84.2, "Target Readout", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#8A2B3E', zorder=4)

draw_card(ax, 162.5, 70.6, 17.8, 10.8, facecolor='#FFFFFF', edgecolor='#E098A4', lw=1.0, zorder=4)
ax.text(171.4, 78.2, "Sequence Attention", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#8A2B3E', zorder=5)
ax.text(171.4, 75.0, "c = ∑ α_t z_t", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#C0392B', zorder=5)
ax.text(171.4, 72.2, "256-d Latent Space", ha='center', va='center', fontsize=8.4, color='#334155', zorder=5)

draw_card(ax, 162.5, 55.0, 17.8, 13.4, facecolor='#FAF1E2', edgecolor='#E6BE8A', lw=1.1, zorder=4)
ax.text(171.4, 64.8, "5-Fold Soft-Voting", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#8E4B1E', zorder=5)
ax.text(171.4, 61.2, "Ensemble Classifier", ha='center', va='center', fontsize=8.8, color='#7E5109', zorder=5)
ax.text(171.4, 57.4, "P(y = 1 | S)", ha='center', va='center', fontsize=9.6, fontweight='bold', color='#1E6B52', zorder=5)


# ==============================================================================
# PANEL 3 (BOTTOM-LEFT): SOTA Performance & Cross-Species Transfer (x=1.8..90.8, y=2.0..47.4)
# ==============================================================================
draw_panel(ax, 1.8, 2.0, 89.0, 45.4, "3",
           "State-of-the-Art Predictive Performance & Cross-Species Generalization",
           bg='#FAFBFD', border='#D5DDE6', hdr_bg='#FCEEF0', hdr_tc='#8A2B3E')

# Left sub-card: Macro-Average Benchmark Comparison (Bar chart)
draw_card(ax, 3.6, 4.0, 44.5, 35.4, facecolor='#FFFFFF', edgecolor='#E5E8EC', lw=1.2, zorder=3)
ax.text(25.85, 36.8, "Macro-Average ROC-AUC (%) Across 15 Species",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#2C4A7E', zorder=4)
ax.text(25.85, 34.0, "Outperforms 8 baselines on 15/15 datasets (p = 3.05 × 10⁻⁵)",
        ha='center', va='center', fontsize=8.8, color='#8A2B3E', fontweight='bold', zorder=4)

models_cmp = [
    ("TF-DualMethyl", 86.68, 93.04, '#D9534F', '#C0392B', True),
    ("HD-6mAPred",    85.05, 91.89, '#6CA6CD', '#286690', False),
    ("PSAC-6mA",      84.08, 91.22, '#85C1E9', '#286690', False),
    ("DRSN4mCPred",   83.22, 90.51, '#A3E4D7', '#1E6B52', False),
    ("iDNA-ABT",      83.05, 90.50, '#F5CBA7', '#8E4B1E', False),
    ("Transformer",   81.26, 89.97, '#D2B4DE', '#5B2C6F', False),
]

for idx, (m_name, auc_4mc, auc_6ma, b_col, t_col, is_ours) in enumerate(models_cmp):
    by = 28.4 - idx * 4.1
    fw = 'bold' if is_ours else 'normal'
    ax.text(16.2, by + 0.8, m_name, ha='right', va='center', fontsize=8.6, fontweight=fw, color=('#C0392B' if is_ours else '#334155'), zorder=5)
    # 6mA bar
    w_6ma = (auc_6ma - 75.0) * 1.25
    draw_card(ax, 17.0, by + 0.8, w_6ma, 1.45, facecolor=b_col, edgecolor='none', rounding=0.3, zorder=5)
    ax.text(17.4 + w_6ma, by + 1.5, f"6mA: {auc_6ma:.2f}%", ha='left', va='center', fontsize=7.8, fontweight=fw, color=t_col, zorder=6)
    # 4mC bar
    w_4mc = (auc_4mc - 75.0) * 1.25
    draw_card(ax, 17.0, by - 0.8, w_4mc, 1.45, facecolor=b_col, edgecolor='none', rounding=0.3, alpha=0.65, zorder=5)
    ax.text(17.4 + w_4mc, by - 0.1, f"4mC: {auc_4mc:.2f}%", ha='left', va='center', fontsize=7.6, fontweight=fw, color='#475569', zorder=6)

# Right sub-card: Cross-Species Generalization & Cold-Start Utility
draw_card(ax, 49.6, 4.0, 39.4, 35.4, facecolor='#FFFFFF', edgecolor='#E5E8EC', lw=1.2, zorder=3)
ax.text(69.3, 36.8, "Cross-Species Generalization Analysis",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#1E6B52', zorder=4)

# Mini 5x5 conceptual cross-species heatmap
sp_labels = ["A.th", "F.ve", "D.me", "H.sa", "T.th"]
mat_vals = np.array([
    [0.96, 0.95, 0.90, 0.91, 0.56],
    [0.95, 0.98, 0.92, 0.94, 0.55],
    [0.89, 0.92, 0.97, 0.96, 0.54],
    [0.90, 0.94, 0.96, 0.97, 0.53],
    [0.54, 0.53, 0.52, 0.52, 0.95],
])
hm_x0, hm_y0, cell_s = 54.4, 17.2, 3.25
for r in range(5):
    ax.text(hm_x0 - 0.6, hm_y0 + (4 - r + 0.5) * cell_s, sp_labels[r],
            ha='right', va='center', fontsize=7.6, fontweight='bold', color='#334155', zorder=5)
    ax.text(hm_x0 + (r + 0.5) * cell_s, hm_y0 + 5 * cell_s + 0.85, sp_labels[r],
            ha='center', va='center', fontsize=7.4, fontweight='bold', color='#334155', zorder=5)
    for c in range(5):
        v = mat_vals[r, c]
        col = '#D9534F' if v > 0.88 else '#AED6F1'
        alp = 0.85 if v > 0.88 else 0.55
        rect = Rectangle((hm_x0 + c * cell_s, hm_y0 + (4 - r) * cell_s), cell_s - 0.15, cell_s - 0.15,
                         facecolor=col, edgecolor='#FFFFFF', lw=0.6, alpha=alp, zorder=4)
        ax.add_patch(rect)
        ax.text(hm_x0 + (c + 0.5) * cell_s, hm_y0 + (4 - r + 0.5) * cell_s, f"{v:.2f}",
                ha='center', va='center', fontsize=6.9, fontweight='bold',
                color=('#FFFFFF' if v > 0.88 else '#1E3A5F'), zorder=5)

ax.text(79.8, 29.5, "Plant ↔ Metazoan", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#C0392B', zorder=5)
ax.text(79.8, 26.8, "AUC: 0.933–0.975", ha='center', va='center', fontsize=8.4, color='#8A2B3E', zorder=5)
ax.text(79.8, 22.0, "Ciliate / Bacteria", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#24557A', zorder=5)
ax.text(79.8, 19.3, "Lineage Boundary", ha='center', va='center', fontsize=8.2, color='#24557A', zorder=5)

draw_card(ax, 51.2, 5.4, 36.2, 10.2, facecolor='#EFF9F5', edgecolor='#8FD3C1', lw=1.0, zorder=4)
ax.text(69.3, 13.0, "• Conserved eukaryotic motif grammar vs. noise", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E6B52', zorder=5)
ax.text(69.3, 10.2, "• Evolutionary boundary in T. thermophila & Xoc", ha='center', va='center', fontsize=8.5, color='#334155', zorder=5)
ax.text(69.3, 7.4, "• Cold-start transfer to small-sample genomes", ha='center', va='center', fontsize=8.5, color='#334155', zorder=5)


# ==============================================================================
# PANEL 4 (BOTTOM-RIGHT): Multi-Scale Biological Interpretability (x=93.2..183.2, y=2.0..47.4)
# ==============================================================================
draw_panel(ax, 93.2, 2.0, 90.0, 45.4, "4",
           "Macro-to-Micro Biological Interpretability & Conserved Motif Discovery",
           bg='#FAFBFD', border='#D5DDE6', hdr_bg='#E6F5EF', hdr_tc='#1E6B52')

# Card 4A: Feature Space Visualization (t-SNE & UMAP)
draw_card(ax, 95.0, 4.0, 27.4, 35.4, facecolor='#FFFFFF', edgecolor='#E5E8EC', lw=1.2, zorder=3)
ax.text(108.7, 36.8, "Feature Space (t-SNE/UMAP)", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#2C4A7E', zorder=4)
ax.text(108.7, 34.0, "256-d Penultimate Representation", ha='center', va='center', fontsize=8.3, color='#546E7A', zorder=4)

draw_card(ax, 96.6, 12.2, 24.2, 20.0, facecolor='#FAFBFD', edgecolor='#D5DDE6', lw=1.0, zorder=4)
np.random.seed(42)
neg_x = np.random.normal(102.5, 2.1, 55)
neg_y = np.random.normal(24.5, 3.2, 55)
pos_x = np.random.normal(114.6, 2.1, 55)
pos_y = np.random.normal(19.8, 3.2, 55)
ax.scatter(neg_x, neg_y, s=18, color='#8E44AD', alpha=0.65, label='Non-Methylated', zorder=5)
ax.scatter(pos_x, pos_y, s=18, color='#E74C3C', alpha=0.70, label='Methylated', zorder=5)
ax.plot([108.7, 108.7], [13.2, 31.2], color='#94A3B8', linestyle='--', lw=1.1, zorder=5)
ax.text(102.5, 14.2, "y = 0 Cluster", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#6C3483', zorder=6)
ax.text(114.6, 29.6, "y = 1 Cluster", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#C0392B', zorder=6)
ax.text(108.7, 7.6, "Compact Class Separation\nAcross Diverse Genomes", ha='center', va='center', fontsize=8.6, fontweight='bold', color='#2C4A7E', zorder=5)

# Card 4B: 41-bp Positional Attention Profiling
draw_card(ax, 124.0, 4.0, 27.4, 35.4, facecolor='#FFFFFF', edgecolor='#E5E8EC', lw=1.2, zorder=3)
ax.text(137.7, 36.8, "41-bp Positional Attention", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#8A2B3E', zorder=4)
ax.text(137.7, 34.0, "Autonomous Catalytic Peak at t=21", ha='center', va='center', fontsize=8.3, color='#546E7A', zorder=4)

draw_card(ax, 125.6, 12.2, 24.2, 20.0, facecolor='#FAFBFD', edgecolor='#D5DDE6', lw=1.0, zorder=4)
px_a = np.linspace(127.0, 148.4, 100)
py_a = 15.2 + 13.8 * np.exp(-((px_a - 137.7) / 2.4) ** 2)
ax.fill_between(px_a, 15.2, py_a, color='#F5B7B1', alpha=0.5, zorder=5)
ax.plot(px_a, py_a, color='#C0392B', lw=2.0, zorder=6)
ax.axvline(137.7, ymin=(15.2/104), ymax=(28.8/104), color='#C0392B', linestyle='--', lw=1.1, zorder=6)
ax.text(137.7, 30.3, "α₂₁ ≈ 8× Baseline", ha='center', va='center', fontsize=8.4, fontweight='bold', color='#C0392B', zorder=7)
for off, lbl in [(-8.5, "1"), (-3.2, "18"), (0.0, "21*"), (3.2, "25"), (8.5, "41")]:
    ax.text(137.7 + off, 13.5, lbl, ha='center', va='center', fontsize=7.8, fontweight='bold',
            color=('#C0392B' if lbl == "21*" else '#475569'), zorder=6)
ax.text(137.7, 7.6, "56.8%–57.7% Weight Concentrated\nin Central 8-bp Window (18–25)", ha='center', va='center', fontsize=8.4, fontweight='bold', color='#8A2B3E', zorder=5)

# Card 4C: Single-Nucleotide LIG Attribution & Motifs
draw_card(ax, 153.0, 4.0, 28.4, 35.4, facecolor='#FFFFFF', edgecolor='#E5E8EC', lw=1.2, zorder=3)
ax.text(167.2, 36.8, "Single-Base LIG Attribution", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#1E6B52', zorder=4)
ax.text(167.2, 34.0, "Conserved Catalytic Consensus Cores", ha='center', va='center', fontsize=8.3, color='#546E7A', zorder=4)

# Draw two mini LIG heatmap strips
motifs_demo = [
    ("A*TGCA Core", ["T", "C", "A*", "T", "G", "C", "A", "C"], [0.08, 0.42, 0.78, 0.56, 0.62, 0.92, 0.15, 0.08], 26.2, 2, 5),
    ("GAGA*TTC Core", ["G", "A", "G", "A*", "T", "T", "C", "A"], [0.48, 0.55, 0.65, 0.85, 0.68, 0.62, 0.90, 0.12], 16.4, 0, 7),
]
cmap_blues = plt.cm.Blues
for m_lbl, bases, scores, y_strip, hl_start, hl_len in motifs_demo:
    ax.text(167.2, y_strip + 4.4, m_lbl, ha='center', va='center', fontsize=8.6, fontweight='bold', color='#D35400', zorder=5)
    for b_i, (b_ch, sc) in enumerate(zip(bases, scores)):
        bx = 155.2 + b_i * 3.0
        rgba = cmap_blues(sc * 0.88 + 0.08)
        rect = Rectangle((bx, y_strip), 2.85, 3.0, facecolor=rgba, edgecolor='#CBD5E1', lw=0.6, zorder=5)
        ax.add_patch(rect)
        tc = '#FFFFFF' if sc > 0.52 else '#1E293B'
        ax.text(bx + 1.42, y_strip + 1.5, b_ch, ha='center', va='center', fontsize=8.0, fontweight='bold', color=tc, zorder=6)
    box_hl = Rectangle((155.2 + hl_start * 3.0, y_strip - 0.1), hl_len * 3.0 - 0.15, 3.2,
                       facecolor='none', edgecolor='#E67E22', lw=1.5, zorder=7)
    ax.add_patch(box_hl)

ax.text(167.2, 7.6, "Recovers A*T[G/T]C & [G/A]AGG\nMethyltransferase Target Motifs", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E6B52', zorder=5)

plt.tight_layout(pad=0.2)
for ext, fmt in [('pdf', 'pdf'), ('png', 'png')]:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches='tight', pad_inches=0.06)
    with open(f'/Volumes/Files/Research/DNAPred_Luo/paper_new/Graphical_Abstract.{ext}', 'wb') as f:
        f.write(buf.getvalue())
plt.close(fig)
print("Saved Graphical_Abstract.pdf and Graphical_Abstract.png successfully.")
