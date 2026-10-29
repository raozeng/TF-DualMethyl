#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-grade Figure 4 (fig_4mC_ablation_metrics.pdf) and
Figure 5 (fig_6mA_ablation_metrics.pdf) with a unified bottom legend.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

VARIANTS = ['w/o BiLSTM', 'w/o Transformer', 'w/o Freq', 'w/o Time', 'TF-DualMethyl']

VARIANT_STYLES = {
    'w/o BiLSTM':      {'color': '#6379A8', 'ls': '--', 'marker': 's', 'lw': 1.4, 'ms': 5.2, 'alpha': 0.85, 'zorder': 3},
    'w/o Transformer': {'color': '#72B088', 'ls': '--', 'marker': '^', 'lw': 1.4, 'ms': 5.5, 'alpha': 0.85, 'zorder': 4},
    'w/o Freq':        {'color': '#72C1D6', 'ls': '--', 'marker': 'v', 'lw': 1.4, 'ms': 5.5, 'alpha': 0.85, 'zorder': 5},
    'w/o Time':        {'color': '#E8A2A8', 'ls': '--', 'marker': 'D', 'lw': 1.4, 'ms': 5.0, 'alpha': 0.88, 'zorder': 6},
    'TF-DualMethyl':   {'color': '#D62728', 'ls': '-',  'marker': 'o', 'lw': 2.5, 'ms': 6.8, 'alpha': 1.00, 'zorder': 10},
}

# Order in tables: [ACC, SN, SP, MCC, AUC, F1]
METRICS_INFO = [
    ('ACC', 'Accuracy (ACC)',              'Accuracy (%)'),
    ('SN',  'Sensitivity (SN)',            'Sensitivity (%)'),
    ('SP',  'Specificity (SP)',            'Specificity (%)'),
    ('MCC', 'Matthews Corr. Coeff. (MCC)', 'MCC (%)'),
    ('AUC', 'Area Under ROC (AUC)',        'ROC-AUC (%)'),
    ('F1',  'F1-Score (F1)',               'F1-Score (%)'),
]

# ==============================================================================
# Exact Data from Supplementary Table S3 (4mC Ablation Study: 4 species)
# Columns: [ACC, SN, SP, MCC, AUC, F1]
# ==============================================================================
ABLATION_4MC = {
    'C.equisetifolia': {
        'w/o BiLSTM':      [80.62, 81.47, 79.78, 62.56, 89.73, 80.71],
        'w/o Freq':        [78.46, 78.92, 77.97, 58.29, 87.70, 78.41],
        'w/o Time':        [81.87, 83.08, 80.61, 64.79, 89.51, 82.01],
        'w/o Transformer': [81.04, 83.21, 78.86, 63.54, 89.56, 81.51],
        'TF-DualMethyl':   [81.97, 78.69, 85.25, 64.07, 90.24, 81.36],
    },
    'F.vesca': {
        'w/o BiLSTM':      [84.47, 87.17, 81.76, 69.43, 92.70, 84.88],
        'w/o Freq':        [84.74, 87.08, 82.39, 69.78, 92.64, 85.08],
        'w/o Time':        [85.00, 87.25, 82.75, 70.27, 92.74, 85.31],
        'w/o Transformer': [84.93, 86.97, 82.89, 70.12, 92.80, 85.22],
        'TF-DualMethyl':   [86.15, 87.29, 85.01, 72.32, 93.57, 86.30],
    },
    'S.cerevisiae': {
        'w/o BiLSTM':      [69.05, 71.24, 66.87, 39.15, 77.38, 69.41],
        'w/o Freq':        [68.93, 70.62, 67.24, 39.21, 77.43, 68.90],
        'w/o Time':        [68.85, 68.43, 69.27, 39.06, 77.54, 67.99],
        'w/o Transformer': [68.38, 71.60, 65.16, 38.62, 77.48, 68.32],
        'TF-DualMethyl':   [71.89, 72.90, 70.88, 43.79, 79.00, 72.17],
    },
    'Tolypocladium': {
        'w/o BiLSTM':      [73.41, 73.77, 73.04, 47.39, 81.53, 73.24],
        'w/o Freq':        [72.78, 73.31, 72.25, 46.15, 80.88, 72.76],
        'w/o Time':        [73.23, 72.63, 73.83, 46.78, 81.00, 72.88],
        'w/o Transformer': [73.27, 74.16, 72.38, 46.85, 80.96, 73.41],
        'TF-DualMethyl':   [75.92, 75.30, 76.54, 51.84, 83.90, 75.77],
    },
}

# ==============================================================================
# Exact Data from Supplementary Table S4 (6mA Ablation Study: 11 species)
# Columns: [ACC, SN, SP, MCC, AUC, F1]
# ==============================================================================
ABLATION_6MA = {
    'A.thaliana': {
        'w/o BiLSTM':      [84.76, 83.68, 85.83, 69.65, 92.26, 84.58],
        'w/o Freq':        [84.68, 83.85, 85.52, 69.50, 92.19, 84.56],
        'w/o Time':        [84.61, 83.16, 86.05, 69.37, 92.17, 84.36],
        'w/o Transformer': [84.56, 83.84, 85.27, 69.24, 92.12, 84.43],
        'TF-DualMethyl':   [86.82, 84.42, 89.22, 73.72, 94.01, 86.50],
    },
    'C.elegans': {
        'w/o BiLSTM':      [87.75, 89.38, 86.13, 75.97, 94.97, 87.93],
        'w/o Freq':        [87.45, 89.64, 85.27, 75.25, 94.55, 87.73],
        'w/o Time':        [87.65, 89.52, 85.78, 75.72, 94.82, 87.89],
        'w/o Transformer': [88.15, 89.27, 87.04, 76.55, 94.91, 88.26],
        'TF-DualMethyl':   [92.06, 93.32, 90.80, 84.15, 97.19, 92.16],
    },
    'C.equisetifolia': {
        'w/o BiLSTM':      [70.93, 69.72, 72.13, 42.48, 78.41, 70.37],
        'w/o Freq':        [71.52, 70.19, 72.86, 43.57, 79.07, 70.90],
        'w/o Time':        [71.62, 71.69, 71.56, 44.01, 79.37, 71.51],
        'w/o Transformer': [71.34, 69.43, 73.25, 43.30, 78.92, 70.58],
        'TF-DualMethyl':   [75.02, 74.91, 75.14, 50.05, 82.68, 75.00],
    },
    'D.melanogaster': {
        'w/o BiLSTM':      [91.43, 92.66, 90.20, 83.01, 96.80, 91.54],
        'w/o Freq':        [91.47, 92.37, 90.58, 83.07, 96.82, 91.56],
        'w/o Time':        [91.47, 92.25, 90.69, 83.07, 96.78, 91.55],
        'w/o Transformer': [91.63, 92.52, 90.73, 83.39, 96.86, 91.72],
        'TF-DualMethyl':   [92.98, 92.73, 93.24, 85.97, 97.34, 92.97],
    },
    'F.vesca': {
        'w/o BiLSTM':      [92.01, 93.48, 90.55, 84.46, 97.17, 91.95],
        'w/o Freq':        [92.21, 94.06, 90.36, 84.80, 97.46, 92.14],
        'w/o Time':        [92.57, 93.72, 91.42, 85.42, 97.39, 92.57],
        'w/o Transformer': [93.24, 94.92, 91.56, 86.65, 97.74, 93.37],
        'TF-DualMethyl':   [93.49, 93.68, 93.29, 86.98, 97.72, 93.50],
    },
    'H.sapiens': {
        'w/o BiLSTM':      [88.46, 88.33, 88.59, 77.12, 95.58, 88.44],
        'w/o Freq':        [88.58, 89.17, 87.98, 77.32, 95.64, 88.65],
        'w/o Time':        [88.88, 87.67, 90.10, 77.93, 95.66, 88.73],
        'w/o Transformer': [88.70, 88.32, 89.08, 77.55, 95.71, 88.66],
        'TF-DualMethyl':   [91.14, 88.94, 93.33, 82.35, 97.14, 90.94],
    },
    'R.chinensis': {
        'w/o BiLSTM':      [76.66, 81.38, 71.94, 55.58, 86.86, 76.99],
        'w/o Freq':        [76.84, 79.08, 74.61, 55.50, 86.40, 76.49],
        'w/o Time':        [78.85, 83.64, 74.06, 59.51, 88.39, 79.58],
        'w/o Transformer': [77.51, 79.23, 75.79, 56.68, 87.60, 77.33],
        'TF-DualMethyl':   [82.11, 77.26, 86.96, 64.52, 90.41, 81.20],
    },
    'S.cerevisiae': {
        'w/o BiLSTM':      [79.32, 79.60, 79.05, 59.57, 88.31, 79.27],
        'w/o Freq':        [79.26, 81.24, 77.27, 59.28, 88.09, 79.61],
        'w/o Time':        [80.00, 79.76, 80.23, 60.81, 88.74, 79.85],
        'w/o Transformer': [79.60, 79.70, 79.49, 60.10, 88.55, 79.45],
        'TF-DualMethyl':   [81.91, 79.24, 84.57, 63.91, 90.34, 81.41],
    },
    'T.thermophila': {
        'w/o BiLSTM':      [87.70, 92.29, 83.11, 75.84, 94.27, 88.23],
        'w/o Freq':        [87.53, 92.05, 83.02, 75.49, 94.18, 88.05],
        'w/o Time':        [87.51, 91.83, 83.19, 75.44, 94.17, 88.01],
        'w/o Transformer': [87.55, 91.63, 83.47, 75.46, 94.13, 88.02],
        'TF-DualMethyl':   [88.57, 95.31, 81.83, 77.85, 94.79, 89.29],
    },
    'Tolypocladium': {
        'w/o BiLSTM':      [71.32, 71.23, 71.40, 44.05, 80.03, 70.75],
        'w/o Freq':        [70.36, 72.39, 68.33, 42.31, 79.56, 70.04],
        'w/o Time':        [72.26, 70.15, 74.37, 45.36, 80.68, 71.36],
        'w/o Transformer': [71.64, 72.00, 71.29, 44.29, 80.01, 71.50],
        'TF-DualMethyl':   [78.06, 79.57, 76.55, 56.15, 85.84, 78.39],
    },
    'Xoc BLS256': {
        'w/o BiLSTM':      [88.24, 90.17, 86.31, 76.70, 95.17, 88.47],
        'w/o Freq':        [88.00, 90.18, 85.83, 76.30, 94.87, 88.29],
        'w/o Time':        [88.20, 89.75, 86.66, 76.62, 94.92, 88.39],
        'w/o Transformer': [88.29, 90.31, 86.28, 76.85, 95.13, 88.55],
        'TF-DualMethyl':   [89.36, 89.32, 89.39, 78.70, 95.98, 89.58],
    },
}


def plot_ablation_figure(data_dict, mod_label, out_prefix, rot_deg=22):
    species_list = list(data_dict.keys())
    x_pos = np.arange(len(species_list))
    panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    hspace_val = 0.36 if mod_label == '4mC' else 0.42
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.16, wspace=0.22, hspace=hspace_val)

    for idx, (m_key, m_title, y_label) in enumerate(METRICS_INFO):
        ax = axes[idx // 3, idx % 3]
        p_tag = panel_letters[idx]

        all_vals = []
        for var_name in VARIANTS:
            st = VARIANT_STYLES[var_name]
            y_vals = [data_dict[sp][var_name][idx] for sp in species_list]
            all_vals.extend(y_vals)
            ax.plot(
                x_pos, y_vals,
                label=var_name,
                color=st['color'],
                linestyle=st['ls'],
                linewidth=st['lw'],
                marker=st['marker'],
                markersize=st['ms'],
                alpha=st['alpha'],
                zorder=st['zorder']
            )

        y_min = np.floor(min(all_vals) - 1.2)
        y_max = np.ceil(max(all_vals) + 1.2)
        ax.set_ylim(y_min, y_max)

        ax.yaxis.grid(True, linestyle='--', alpha=0.45, color='#CBD5E1', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(species_list, rotation=rot_deg, ha='right', fontsize=9.2, fontstyle='italic')
        ax.tick_params(axis='y', labelsize=9.2)
        ax.set_ylabel(y_label, fontsize=10.5, fontweight='bold', color='#1E293B')

        ax.set_title(
            m_title,
            fontsize=12.8, fontweight='bold', color='#0F172A', pad=10
        )
        ax.text(
            -0.12, 1.09, p_tag, transform=ax.transAxes,
            fontsize=13.8, fontweight='bold', color='#0F172A', va='top', ha='left'
        )

    # Unified bottom legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc='lower center', bbox_to_anchor=(0.51, 0.01),
        ncol=5, frameon=True, fancybox=True,
        edgecolor='#CBD5E1', facecolor='#F8FAFC',
        fontsize=10.5, handlelength=2.5, columnspacing=1.8
    )
    for text in leg.get_texts():
        if text.get_text() == 'TF-DualMethyl':
            text.set_fontweight('bold')
            text.set_color('#B91C1C')

    fig.savefig(f"/Volumes/Files/Research/DNAPred_Luo/paper_new/{out_prefix}.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(f"/Volumes/Files/Research/DNAPred_Luo/paper_new/{out_prefix}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved {out_prefix}.pdf and .png")


if __name__ == '__main__':
    plot_ablation_figure(ABLATION_4MC, '4mC', 'fig_4mC_ablation_metrics', rot_deg=20)
    plot_ablation_figure(ABLATION_6MA, '6mA', 'fig_6mA_ablation_metrics', rot_deg=32)
