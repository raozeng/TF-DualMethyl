#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-grade Figure 2 (4mC 2x2 Per-Species Multi-Metric Radar Charts)
and Figure 3 (6mA 2x3 Multi-Metric Box + Strip + Macro-Mean Trend Plots) for JMB.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

MODELS = [
    'TextCNN', 'BiLSTM', 'CRNN', 'Transformer',
    'iDNA-ABT', 'DRSN4mCPred', 'PSAC-6mA', 'HD-6mAPred', 'TF-DualMethyl'
]

METRICS = ['ACC', 'SN', 'SP', 'F1', 'MCC', 'AUC']
METRIC_TITLES = [
    'Accuracy (ACC)',
    'Sensitivity (SN)',
    'Specificity (SP)',
    'F1-Score (F1)',
    'Matthews Corr. Coeff. (MCC)',
    'Area Under ROC (AUC)'
]

# Publication color palette: distinct, colorblind-aware progression ending in bold crimson
MODEL_COLORS = {
    'TextCNN':       '#5B9BD5',  # Steel Blue
    'BiLSTM':        '#70AD47',  # Sage Green
    'CRNN':          '#4472C4',  # Royal Blue
    'Transformer':   '#9E7CC1',  # Muted Purple
    'iDNA-ABT':      '#2CA02C',  # Emerald
    'DRSN4mCPred':   '#8C6D31',  # Warm Ochre
    'PSAC-6mA':      '#17BECF',  # Teal Cyan
    'HD-6mAPred':    '#FF7F0E',  # Vivid Amber
    'TF-DualMethyl': '#D62728',  # Bold Crimson
}

MODEL_LINESTYLES = {
    'TextCNN':       (0, (3, 2)),
    'BiLSTM':        (0, (3, 2)),
    'CRNN':          (0, (5, 2)),
    'Transformer':   (0, (5, 2)),
    'iDNA-ABT':      (0, (4, 1, 1, 1)),
    'DRSN4mCPred':   '-',
    'PSAC-6mA':      '-',
    'HD-6mAPred':    '-',
    'TF-DualMethyl': '-',
}

MODEL_MARKERS = {
    'TextCNN':       'v',
    'BiLSTM':        '^',
    'CRNN':          's',
    'Transformer':   'D',
    'iDNA-ABT':      'P',
    'DRSN4mCPred':   'X',
    'PSAC-6mA':      'o',
    'HD-6mAPred':    'h',
    'TF-DualMethyl': 'o',
}

# ==============================================================================
# Exact Data from Supplementary Table S1 (4mC: 4 species)
# Order of columns in raw table: [ACC, SN, SP, MCC, AUC, F1]
# Reordered below to METRICS = ['ACC', 'SN', 'SP', 'F1', 'MCC', 'AUC']
# ==============================================================================
RAW_4MC = {
    'C. equisetifolia': {
        'TextCNN':       [75.68, 73.22, 78.14, 51.43, 86.28, 75.07],
        'BiLSTM':        [77.87, 75.96, 79.78, 55.78, 87.32, 77.44],
        'CRNN':          [78.69, 78.14, 79.23, 57.38, 87.74, 78.57],
        'Transformer':   [74.86, 61.20, 88.52, 51.69, 81.99, 70.89],
        'iDNA-ABT':      [82.51, 79.23, 85.79, 65.17, 85.55, 81.92],
        'DRSN4mCPred':   [79.23, 77.60, 80.87, 58.50, 86.95, 78.89],
        'PSAC-6mA':      [80.05, 78.14, 81.97, 60.15, 87.82, 79.67],
        'HD-6mAPred':    [80.87, 78.69, 83.06, 61.81, 88.65, 80.45],
        'TF-DualMethyl': [81.97, 78.69, 85.25, 64.07, 90.24, 81.36],
    },
    'F. vesca': {
        'TextCNN':       [71.54, 75.70, 67.38, 43.24, 78.91, 72.68],
        'BiLSTM':        [83.51, 86.60, 80.43, 67.16, 91.20, 84.01],
        'CRNN':          [84.52, 84.52, 84.52, 69.03, 92.31, 84.52],
        'Transformer':   [82.94, 84.39, 81.49, 65.91, 90.81, 83.18],
        'iDNA-ABT':      [84.20, 82.91, 85.49, 68.42, 90.70, 83.99],
        'DRSN4mCPred':   [84.52, 84.15, 84.88, 69.03, 91.85, 84.46],
        'PSAC-6mA':      [84.86, 85.12, 84.60, 69.73, 92.24, 84.90],
        'HD-6mAPred':    [85.39, 85.82, 84.96, 70.78, 92.78, 85.45],
        'TF-DualMethyl': [86.15, 87.29, 85.01, 72.32, 93.57, 86.30],
    },
    'S. cerevisiae': {
        'TextCNN':       [60.26, 56.72, 63.80, 20.58, 63.80, 58.81],
        'BiLSTM':        [59.35, 57.23, 61.48, 18.72, 63.49, 58.47],
        'CRNN':          [61.88, 56.72, 67.04, 23.89, 66.82, 59.81],
        'Transformer':   [68.50, 60.36, 76.64, 37.51, 75.38, 65.71],
        'iDNA-ABT':      [70.27, 66.84, 73.71, 40.64, 75.37, 69.21],
        'DRSN4mCPred':   [68.96, 66.84, 71.08, 37.95, 74.62, 68.29],
        'PSAC-6mA':      [69.92, 68.25, 71.59, 39.86, 75.88, 69.41],
        'HD-6mAPred':    [70.73, 69.57, 71.89, 41.47, 77.12, 70.38],
        'TF-DualMethyl': [71.89, 72.90, 70.88, 43.79, 79.00, 72.17],
    },
    'Tolypocladium sp.': {
        'TextCNN':       [58.96, 57.71, 60.21, 17.92, 62.21, 58.44],
        'BiLSTM':        [59.70, 56.06, 63.34, 19.46, 63.59, 58.18],
        'CRNN':          [71.67, 73.33, 70.01, 43.36, 78.76, 72.13],
        'Transformer':   [69.97, 67.35, 72.60, 40.00, 76.87, 69.16],
        'iDNA-ABT':      [73.83, 72.30, 75.36, 47.68, 80.57, 73.42],
        'DRSN4mCPred':   [72.44, 72.18, 72.70, 44.88, 79.48, 72.37],
        'PSAC-6mA':      [73.26, 73.10, 73.42, 46.52, 80.36, 73.22],
        'HD-6mAPred':    [74.25, 74.12, 74.38, 48.51, 81.64, 74.22],
        'TF-DualMethyl': [75.92, 75.30, 76.54, 51.84, 83.90, 75.77],
    }
}

# ==============================================================================
# Exact Data from Supplementary Table S2 (6mA: 11 species)
# Raw order: [ACC, SN, SP, MCC, AUC, F1]
# ==============================================================================
RAW_6MA = {
    'A. thaliana': {
        'TextCNN':       [71.36, 69.95, 72.76, 42.73, 78.62, 70.95],
        'BiLSTM':        [84.50, 81.28, 87.73, 69.15, 92.10, 83.99],
        'CRNN':          [85.70, 84.52, 86.89, 71.42, 93.11, 85.53],
        'Transformer':   [85.23, 82.18, 88.29, 70.60, 92.61, 84.77],
        'iDNA-ABT':      [85.38, 82.45, 88.31, 70.88, 91.84, 84.94],
        'DRSN4mCPred':   [85.42, 83.90, 86.95, 70.88, 92.45, 85.20],
        'PSAC-6mA':      [85.88, 84.45, 87.32, 71.80, 92.92, 85.68],
        'HD-6mAPred':    [86.31, 84.78, 87.84, 72.66, 93.34, 86.10],
        'TF-DualMethyl': [86.82, 84.42, 89.22, 73.72, 94.01, 86.50],
    },
    'C. elegans': {
        'TextCNN':       [72.44, 72.89, 71.98, 44.88, 80.33, 72.56],
        'BiLSTM':        [84.94, 87.49, 82.39, 69.97, 92.04, 85.31],
        'CRNN':          [89.06, 89.70, 88.42, 78.12, 95.31, 89.13],
        'Transformer':   [86.21, 87.64, 84.77, 72.44, 93.68, 86.40],
        'iDNA-ABT':      [89.03, 88.07, 90.00, 78.08, 94.33, 88.93],
        'DRSN4mCPred':   [89.37, 89.55, 89.20, 78.74, 95.18, 89.39],
        'PSAC-6mA':      [90.13, 90.40, 89.85, 80.25, 95.74, 90.15],
        'HD-6mAPred':    [90.75, 91.16, 90.35, 81.51, 96.28, 90.79],
        'TF-DualMethyl': [92.06, 93.32, 90.80, 84.15, 97.19, 92.16],
    },
    'C. equisetifolia': {
        'TextCNN':       [59.91, 57.30, 62.51, 19.84, 64.62, 58.84],
        'BiLSTM':        [59.96, 60.44, 59.48, 19.92, 64.51, 60.15],
        'CRNN':          [65.27, 63.67, 66.86, 30.55, 70.71, 64.70],
        'Transformer':   [70.90, 65.45, 76.36, 42.06, 77.97, 69.22],
        'iDNA-ABT':      [73.28, 68.94, 77.61, 46.73, 79.02, 72.07],
        'DRSN4mCPred':   [71.97, 70.10, 73.85, 43.98, 78.64, 71.44],
        'PSAC-6mA':      [72.93, 71.25, 74.61, 45.89, 79.85, 72.47],
        'HD-6mAPred':    [73.66, 72.44, 74.88, 47.33, 80.92, 73.33],
        'TF-DualMethyl': [75.02, 74.91, 75.14, 50.05, 82.68, 75.00],
    },
    'D. melanogaster': {
        'TextCNN':       [76.46, 75.12, 77.80, 52.94, 84.10, 76.14],
        'BiLSTM':        [89.38, 88.60, 90.17, 78.78, 95.40, 89.30],
        'CRNN':          [91.33, 91.17, 91.49, 82.66, 96.51, 91.32],
        'Transformer':   [90.90, 90.03, 91.78, 81.82, 96.36, 90.82],
        'iDNA-ABT':      [91.22, 90.47, 91.96, 82.44, 95.44, 91.15],
        'DRSN4mCPred':   [91.13, 90.71, 91.56, 82.27, 96.12, 91.10],
        'PSAC-6mA':      [91.64, 91.22, 92.06, 83.29, 96.45, 91.61],
        'HD-6mAPred':    [92.17, 91.85, 92.49, 84.34, 96.76, 92.15],
        'TF-DualMethyl': [92.98, 92.73, 93.24, 85.97, 97.34, 92.97],
    },
    'F. vesca': {
        'TextCNN':       [80.11, 80.72, 79.50, 60.22, 87.97, 80.23],
        'BiLSTM':        [84.11, 82.01, 86.20, 68.27, 91.52, 83.77],
        'CRNN':          [91.13, 90.01, 92.26, 82.29, 96.39, 91.03],
        'Transformer':   [91.91, 93.04, 90.78, 83.84, 97.19, 92.00],
        'iDNA-ABT':      [92.68, 92.13, 93.23, 85.37, 95.44, 92.64],
        'DRSN4mCPred':   [91.72, 91.42, 92.01, 83.43, 96.48, 91.69],
        'PSAC-6mA':      [92.33, 92.13, 92.52, 84.66, 96.95, 92.31],
        'HD-6mAPred':    [92.75, 92.59, 92.91, 85.49, 97.26, 92.73],
        'TF-DualMethyl': [93.49, 93.68, 93.29, 86.98, 97.72, 93.50],
    },
    'H. sapiens': {
        'TextCNN':       [80.07, 78.52, 81.62, 60.17, 88.63, 79.76],
        'BiLSTM':        [88.57, 85.87, 91.26, 77.25, 95.65, 88.25],
        'CRNN':          [90.24, 89.01, 91.46, 80.50, 96.61, 90.12],
        'Transformer':   [89.28, 87.81, 90.74, 78.59, 95.89, 89.12],
        'iDNA-ABT':      [89.80, 89.80, 89.80, 79.60, 95.10, 89.80],
        'DRSN4mCPred':   [89.89, 88.65, 91.12, 79.80, 95.82, 89.76],
        'PSAC-6mA':      [90.41, 89.15, 91.68, 80.85, 96.24, 90.29],
        'HD-6mAPred':    [90.85, 89.61, 92.08, 81.72, 96.59, 90.73],
        'TF-DualMethyl': [91.14, 88.94, 93.33, 82.35, 97.14, 90.94],
    },
    'R. chinensis': {
        'TextCNN':       [72.74, 73.91, 71.57, 45.50, 78.75, 73.06],
        'BiLSTM':        [67.39, 64.21, 70.57, 34.85, 73.46, 66.32],
        'CRNN':          [68.90, 68.23, 69.57, 37.80, 74.71, 68.69],
        'Transformer':   [73.24, 75.25, 71.24, 46.53, 81.50, 73.77],
        'iDNA-ABT':      [82.61, 80.94, 84.28, 65.25, 87.89, 82.31],
        'DRSN4mCPred':   [76.92, 75.59, 78.26, 53.87, 84.15, 76.61],
        'PSAC-6mA':      [78.26, 76.92, 79.60, 56.54, 85.72, 77.97],
        'HD-6mAPred':    [79.60, 77.93, 81.27, 59.23, 87.38, 79.25],
        'TF-DualMethyl': [82.11, 77.26, 86.96, 64.52, 90.41, 81.20],
    },
    'S. cerevisiae': {
        'TextCNN':       [67.88, 62.92, 72.85, 35.94, 74.69, 66.20],
        'BiLSTM':        [67.25, 67.14, 67.35, 34.50, 74.18, 67.21],
        'CRNN':          [77.23, 76.86, 77.60, 54.47, 85.30, 77.15],
        'Transformer':   [78.87, 76.12, 81.62, 57.83, 86.83, 78.27],
        'iDNA-ABT':      [80.11, 72.37, 87.85, 60.96, 87.09, 78.44],
        'DRSN4mCPred':   [79.19, 77.92, 80.45, 58.39, 86.74, 78.92],
        'PSAC-6mA':      [80.08, 78.66, 81.51, 60.19, 87.65, 79.80],
        'HD-6mAPred':    [80.77, 79.08, 82.46, 61.58, 88.49, 80.44],
        'TF-DualMethyl': [81.91, 79.24, 84.57, 63.91, 90.34, 81.41],
    },
    'T. thermophila': {
        'TextCNN':       [67.61, 68.67, 66.54, 35.23, 74.29, 67.95],
        'BiLSTM':        [88.24, 94.49, 82.00, 77.09, 94.44, 88.93],
        'CRNN':          [88.52, 94.94, 82.09, 77.68, 94.69, 89.21],
        'Transformer':   [86.38, 92.59, 80.16, 73.33, 92.81, 87.18],
        'iDNA-ABT':      [87.40, 81.10, 93.70, 75.40, 93.10, 86.55],
        'DRSN4mCPred':   [87.63, 91.85, 83.42, 75.54, 93.88, 88.13],
        'PSAC-6mA':      [87.99, 92.35, 83.62, 76.26, 94.15, 88.49],
        'HD-6mAPred':    [88.34, 93.12, 83.56, 77.03, 94.42, 88.87],
        'TF-DualMethyl': [88.57, 95.31, 81.83, 77.85, 94.79, 89.29],
    },
    'Tolypocladium sp.': {
        'TextCNN':       [64.03, 72.00, 56.07, 28.43, 70.40, 66.68],
        'BiLSTM':        [61.72, 64.36, 59.09, 23.48, 66.65, 62.71],
        'CRNN':          [63.94, 66.96, 60.92, 27.94, 69.78, 65.00],
        'Transformer':   [75.22, 71.05, 79.40, 50.62, 82.07, 74.14],
        'iDNA-ABT':      [77.38, 71.70, 83.07, 55.12, 83.61, 76.02],
        'DRSN4mCPred':   [74.66, 73.12, 76.20, 49.34, 82.19, 74.26],
        'PSAC-6mA':      [75.81, 74.48, 77.15, 51.65, 83.28, 75.49],
        'HD-6mAPred':    [76.76, 75.84, 77.68, 53.53, 84.35, 76.55],
        'TF-DualMethyl': [78.06, 79.57, 76.55, 56.15, 85.84, 78.39],
    },
    'Xoc BLS256': {
        'TextCNN':       [70.52, 66.79, 74.44, 41.31, 78.55, 69.90],
        'BiLSTM':        [85.47, 84.56, 86.43, 70.96, 92.78, 85.64],
        'CRNN':          [87.27, 86.45, 88.13, 74.56, 94.38, 87.44],
        'Transformer':   [84.97, 85.04, 84.90, 69.93, 92.72, 85.29],
        'iDNA-ABT':      [86.94, 84.95, 88.93, 73.94, 92.61, 86.68],
        'DRSN4mCPred':   [87.48, 86.80, 88.15, 74.96, 93.96, 87.39],
        'PSAC-6mA':      [88.04, 87.45, 88.63, 76.08, 94.42, 87.97],
        'HD-6mAPred':    [88.53, 88.11, 88.94, 77.06, 94.95, 88.48],
        'TF-DualMethyl': [89.36, 89.32, 89.39, 78.70, 95.98, 89.58],
    }
}


def reorder_metrics(vec):
    """Convert [ACC, SN, SP, MCC, AUC, F1] -> [ACC, SN, SP, F1, MCC, AUC]."""
    return [vec[0], vec[1], vec[2], vec[5], vec[3], vec[4]]


# ==============================================================================
# PART 1: Figure 2 — 4mC 2x2 Per-Species Multi-Metric Radar Charts
# ==============================================================================
def plot_4mc_radar():
    species_list = [
        ('C. equisetifolia', '(a)', 'Casuarina equisetifolia (4mC)'),
        ('F. vesca',         '(b)', 'Fragaria vesca (4mC)'),
        ('S. cerevisiae',    '(c)', 'Saccharomyces cerevisiae (4mC)'),
        ('Tolypocladium sp.','(d)', 'Tolypocladium sp. SUP5-1 (4mC)'),
    ]

    num_vars = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, axes = plt.subplots(
        2, 2, figsize=(13.2, 10.2),
        subplot_kw=dict(polar=True)
    )
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.09, wspace=0.18, hspace=0.25)

    for idx, (sp_key, panel_tag, sp_title) in enumerate(species_list):
        ax = axes[idx // 2, idx % 2]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        sp_data = {m: reorder_metrics(RAW_4MC[sp_key][m]) for m in MODELS}
        mat = np.array([sp_data[m] for m in MODELS])  # shape (9, 6)

        # Compute per-spoke adaptive bounds so each metric utilizes [0.22, 0.94] radial space
        spoke_min = np.floor(mat.min(axis=0) - 2.0)
        spoke_max = np.ceil(mat.max(axis=0) + 1.0)
        spoke_span = np.maximum(spoke_max - spoke_min, 5.0)

        def to_radius(vals):
            r = 0.22 + 0.72 * (np.array(vals) - spoke_min) / spoke_span
            return r.tolist() + [r[0]]

        # Draw custom hexagonal concentric reference rings
        ring_levels = [0.22, 0.46, 0.70, 0.94]
        ax.set_ylim(0, 1.15)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)

        # Background alternating rings
        for r_idx, r_val in enumerate(ring_levels):
            poly_r = [r_val] * len(angles)
            ax.plot(angles, poly_r, color='#D0D7DE', linewidth=0.85, linestyle='--' if r_idx < 3 else '-', zorder=1)
            if r_idx == 3:
                ax.fill(angles, poly_r, color='#F8FAFC', alpha=0.8, zorder=0)

        # Radial spoke lines and per-spoke min/mid/max tick labels
        for j in range(num_vars):
            ang = angles[j]
            ax.plot([ang, ang], [0.12, 0.96], color='#CBD5E1', linewidth=0.9, zorder=1)
            ang_deg = np.degrees(ang)
            ha = 'center'
            if 15 < ang_deg < 165:
                ha = 'left'
            elif 195 < ang_deg < 345:
                ha = 'right'
            label_r = 1.13 if j in (0, 3) else 1.11
            ax.text(
                ang, label_r,
                f"{METRICS[j]}\n({spoke_min[j]:.0f}–{spoke_max[j]:.0f}%)",
                ha=ha, va='center', fontsize=9.2, fontweight='bold', color='#1E293B'
            )

        # Plot competing models first, then TF-DualMethyl on top
        for m in MODELS:
            r_vals = to_radius(sp_data[m])
            is_ours = (m == 'TF-DualMethyl')
            lw = 2.6 if is_ours else (1.5 if m in ('HD-6mAPred', 'PSAC-6mA', 'iDNA-ABT') else 1.15)
            alpha_line = 1.0 if is_ours else (0.85 if m in ('HD-6mAPred', 'PSAC-6mA', 'iDNA-ABT') else 0.65)
            z = 10 if is_ours else (6 if m in ('HD-6mAPred', 'PSAC-6mA', 'iDNA-ABT') else 3)
            ms = 6.5 if is_ours else 4.2

            ax.plot(
                angles, r_vals,
                color=MODEL_COLORS[m],
                linestyle=MODEL_LINESTYLES[m],
                linewidth=lw,
                marker=MODEL_MARKERS[m],
                markersize=ms,
                markeredgecolor='white' if is_ours else MODEL_COLORS[m],
                markeredgewidth=0.9 if is_ours else 0.5,
                alpha=alpha_line,
                label=m,
                zorder=z
            )
            if is_ours:
                ax.fill(angles, r_vals, color=MODEL_COLORS[m], alpha=0.16, zorder=8)
            elif m == 'HD-6mAPred':
                ax.fill(angles, r_vals, color=MODEL_COLORS[m], alpha=0.05, zorder=4)

        # Annotate TF-DualMethyl exact metric values at each vertex
        r_tf = to_radius(sp_data['TF-DualMethyl'])[:-1]
        for j, (ang, rv) in enumerate(zip(angles[:-1], r_tf)):
            val = sp_data['TF-DualMethyl'][j]
            offset_r = min(rv + 0.045, 0.985)
            ax.text(
                ang, offset_r, f"{val:.1f}",
                ha='center', va='center',
                fontsize=8.0, fontweight='bold', color='#B91C1C',
                bbox=dict(boxstyle='round,pad=0.13', facecolor='white', edgecolor='#FCA5A5', alpha=0.92, lw=0.6),
                zorder=15
            )

        # Panel title
        ax.text(
            -0.10, 1.12, panel_tag, transform=ax.transAxes,
            fontsize=14.5, fontweight='bold', color='#0F172A', va='top', ha='left'
        )
        ax.set_title(
            sp_title, y=1.09, fontsize=13.8, fontweight='bold', fontstyle='italic', color='#0F172A'
        )

    # Bottom unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc='lower center', bbox_to_anchor=(0.5, 0.012),
        ncol=5, frameon=True, fancybox=True,
        edgecolor='#CBD5E1', facecolor='#F8FAFC',
        fontsize=10.0, handlelength=2.4, columnspacing=1.5
    )
    for text in leg.get_texts():
        if text.get_text() == 'TF-DualMethyl':
            text.set_fontweight('bold')
            text.set_color('#B91C1C')

    fig.savefig('/Volumes/Files/Research/DNAPred_Luo/paper_new/fig_4mC_comparison.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('/Volumes/Files/Research/DNAPred_Luo/paper_new/fig_4mC_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved fig_4mC_comparison.pdf and .png")


# ==============================================================================
# PART 2: Figure 3 — 6mA 2x3 Box + Jittered Strip + Macro-Mean Trend Plots
# ==============================================================================
def plot_6ma_boxstrip():
    species_keys = list(RAW_6MA.keys())  # 11 species
    panel_tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.2))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.17, wspace=0.20, hspace=0.37)

    rng = np.random.default_rng(42)
    x_pos = np.arange(len(MODELS))

    for m_idx, (metric_key, metric_title, p_tag) in enumerate(zip(METRICS, METRIC_TITLES, panel_tags)):
        ax = axes[m_idx // 3, m_idx % 3]

        # Collect 11-species distributions for each of the 9 models
        dist_list = []
        means = []
        medians = []
        for model in MODELS:
            vals = [reorder_metrics(RAW_6MA[sp][model])[m_idx] for sp in species_keys]
            dist_list.append(vals)
            means.append(np.mean(vals))
            medians.append(np.median(vals))

        # Highlight column background for TF-DualMethyl (x = 8)
        ax.axvspan(7.52, 8.48, color='#FEE2E2', alpha=0.55, zorder=0)

        # Draw boxplot
        bp = ax.boxplot(
            dist_list,
            positions=x_pos,
            widths=0.54,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            zorder=2
        )

        for i, (patch, model) in enumerate(zip(bp['boxes'], MODELS)):
            col = MODEL_COLORS[model]
            is_ours = (model == 'TF-DualMethyl')
            patch.set_facecolor(col)
            patch.set_alpha(0.38 if not is_ours else 0.62)
            patch.set_edgecolor(col if not is_ours else '#991B1B')
            patch.set_linewidth(1.2 if not is_ours else 1.8)

        for i, med in enumerate(bp['medians']):
            is_ours = (i == len(MODELS) - 1)
            med.set_color('#7F1D1D' if is_ours else '#1E293B')
            med.set_linewidth(2.0 if is_ours else 1.4)

        for w in bp['whiskers'] + bp['caps']:
            w.set_color('#64748B')
            w.set_linewidth(1.0)

        # Overlay jittered 11-species points
        for i, (vals, model) in enumerate(zip(dist_list, MODELS)):
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            is_ours = (model == 'TF-DualMethyl')
            ax.scatter(
                np.full(len(vals), x_pos[i]) + jitter,
                vals,
                s=28 if is_ours else 20,
                color=MODEL_COLORS[model],
                alpha=0.90 if is_ours else 0.72,
                edgecolors='white',
                linewidths=0.5,
                zorder=4
            )

        # Plot Macro-Average Mean trajectory line + diamond markers
        ax.plot(
            x_pos, means,
            color='#B91C1C', linestyle='--', linewidth=1.6, alpha=0.85,
            label='Macro-Average Mean Trajectory', zorder=6
        )
        ax.scatter(
            x_pos[:-1], means[:-1],
            marker='D', s=38, color='#FFFBEB', edgecolors='#B91C1C', linewidths=1.3, zorder=7
        )
        ax.scatter(
            [x_pos[-1]], [means[-1]],
            marker='D', s=62, color='#DC2626', edgecolors='white', linewidths=1.2, zorder=8
        )

        # Annotate TF-DualMethyl macro-average cleanly at the top of the highlighted column
        all_vals = np.concatenate(dist_list)
        y_min = max(0.0, np.floor(all_vals.min() - 2.5))
        y_max = np.ceil(all_vals.max() + 5.8)
        ax.set_ylim(y_min, y_max)

        ax.text(
            x_pos[-1], y_max - 0.055 * (y_max - y_min),
            f"{means[-1]:.2f}%",
            ha='center', va='center',
            fontsize=8.4, fontweight='bold', color='#991B1B',
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white', edgecolor='#DC2626', lw=1.0),
            zorder=10
        )

        # Grid & styling
        ax.yaxis.grid(True, linestyle='--', alpha=0.45, color='#CBD5E1', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(MODELS, rotation=32, ha='right', fontsize=9.0)
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text() == 'TF-DualMethyl':
                tick_label.set_fontweight('bold')
                tick_label.set_color('#B91C1C')

        ax.set_ylabel(metric_key + ' (%)', fontsize=10.5, fontweight='bold', color='#1E293B')
        ax.set_title(metric_title, fontsize=13.2, fontweight='bold', color='#0F172A', pad=10)
        ax.text(
            -0.12, 1.09, p_tag, transform=ax.transAxes,
            fontsize=14.0, fontweight='bold', color='#0F172A', va='top', ha='left'
        )

    # Bottom unified legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=MODEL_COLORS[m], edgecolor=MODEL_COLORS[m], alpha=0.55, label=m)
        for m in MODELS
    ] + [
        Line2D([0], [0], color='#B91C1C', linestyle='--', marker='D', markersize=6,
               markerfacecolor='#DC2626', label='Mean across 11 Species')
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc='lower center', bbox_to_anchor=(0.51, 0.005),
        ncol=5, frameon=True, fancybox=True,
        edgecolor='#CBD5E1', facecolor='#F8FAFC',
        fontsize=10.2, columnspacing=1.5
    )
    for text in leg.get_texts():
        if 'TF-DualMethyl' in text.get_text():
            text.set_fontweight('bold')
            text.set_color('#B91C1C')

    fig.savefig('/Volumes/Files/Research/DNAPred_Luo/paper_new/fig_6mA_comparison.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('/Volumes/Files/Research/DNAPred_Luo/paper_new/fig_6mA_comparison.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Saved fig_6mA_comparison.pdf and .png")


if __name__ == '__main__':
    plot_4mc_radar()
    plot_6ma_boxstrip()
