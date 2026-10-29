#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a soft-pastel, large-typography 5-stage, 2-row vector PDF architecture
diagram for TF-DualMethyl, with clean conceptual labels, zero formula clutter,
and generous padding inside every card.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

# Configure publication-grade sans-serif typography
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
mpl.rcParams['mathtext.fontset'] = 'dejavusans'

fig, ax = plt.subplots(figsize=(16.8, 10.1), dpi=300)
ax.set_xlim(0, 210)
ax.set_ylim(0, 126)
ax.axis('off')

# ==============================================================================
# Soft Pastel / Morandi Color Palettes (Low Saturation, Gentle & Harmonious)
# ==============================================================================
PASTEL_WARM = [
    '#F5CBA7', '#FAD7A0', '#F8C471', '#F0B27A', '#EDBB99',
    '#F5B7B1', '#F1948A', '#F9E79F', '#FDEBD0', '#F6DDCC'
]
PASTEL_COOL = [
    '#A9CCE3', '#AED6F1', '#85C1E9', '#A3E4D7', '#A2D9CE',
    '#BFC9CA', '#C39BD3', '#BB8FCE', '#D4E6F1', '#D1F2EB'
]
PASTEL_MHA = [
    '#C5CAE9', '#DCE1F6', '#9FA8DA', '#E8EAF6', '#90CAF9',
    '#BBDEFB', '#B39DDB', '#E3F2FD', '#7986CB', '#D1C4E9'
]
PASTEL_MIX = [
    '#F5CBA7', '#FAD7A0', '#F0B27A', '#F5B7B1', '#F9E79F',
    '#A9CCE3', '#AED6F1', '#A3E4D7', '#C5CAE9', '#D1F2EB'
]

# ==============================================================================
# Core Drawing Primitives
# ==============================================================================

def draw_stage_panel(ax, x, y, w, h, stage_num, stage_title,
                     bg_color='#FAFBFD', border_color='#D5DDE6',
                     header_bg='#EDF2F8', header_text_color='#2C4A7E'):
    """Draws a major stage container with a soft drop shadow and pastel top header."""
    shadow = FancyBboxPatch(
        (x + 0.3, y - 0.3), w, h,
        boxstyle="round,pad=0,rounding_size=1.8",
        facecolor='#E8EEF4', edgecolor='none', alpha=0.5, zorder=1
    )
    ax.add_patch(shadow)

    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=1.8",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.3, zorder=2
    )
    ax.add_patch(body)

    hh = 6.0
    header = FancyBboxPatch(
        (x, y + h - hh), w, hh,
        boxstyle="round,pad=0,rounding_size=1.8",
        facecolor=header_bg, edgecolor=border_color, linewidth=1.3, zorder=3
    )
    ax.add_patch(header)
    rect_patch = Rectangle(
        (x + 0.15, y + h - hh), w - 0.3, 2.0,
        facecolor=header_bg, edgecolor='none', zorder=3
    )
    ax.add_patch(rect_patch)
    ax.plot([x, x + w], [y + h - hh, y + h - hh],
            color=border_color, linewidth=1.15, zorder=4)

    full_title = f"Stage {stage_num}  |  {stage_title}"
    ax.text(x + w / 2, y + h - hh / 2, full_title,
            ha='center', va='center', fontsize=12.6, fontweight='bold',
            color=header_text_color, zorder=5)


def draw_card(ax, x, y, w, h, facecolor='#FFFFFF', edgecolor='#D5DDE6',
              lw=1.15, linestyle='-', rounding=1.2, zorder=3, alpha=1.0):
    """Draws a rounded sub-card."""
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, linestyle=linestyle, alpha=alpha, zorder=zorder
    )
    ax.add_patch(card)
    return card


def draw_arrow(ax, x1, y1, x2, y2, color='#64748B', lw=1.4,
               head_w=0.95, head_l=1.2, linestyle='-', zorder=8):
    """Draws a soft-colored crisp arrow from (x1, y1) to (x2, y2)."""
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=f'-|>,head_width={head_w*0.42},head_length={head_l*0.42}',
            color=color, lw=lw, linestyle=linestyle, shrinkA=0, shrinkB=0
        ),
        zorder=zorder
    )


def draw_polyline_arrow(ax, points, color='#64748B', lw=1.45,
                        head_w=1.0, head_l=1.25, linestyle='-', zorder=8):
    """Draws a multi-segment orthogonal path ending with an arrowhead."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, linewidth=lw, linestyle=linestyle,
                solid_capstyle='round', solid_joinstyle='round', zorder=zorder)
    draw_arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1],
               color=color, lw=lw, head_w=head_w, head_l=head_l,
               linestyle=linestyle, zorder=zorder)


def draw_2d_token_matrix(ax, x, y, w, h, row_labels, palette, zorder=5):
    """Draws a soft-pastel 2D token embedding matrix grid with large row labels."""
    n_rows = len(row_labels)
    n_cols = 5
    label_w = 5.8
    grid_x = x + label_w
    grid_w = w - label_w - 0.5
    cell_w = grid_w / n_cols
    cell_h = h / n_rows
    pad_x, pad_y = 0.22, 0.22

    for r, r_label in enumerate(row_labels):
        ry = y + h - (r + 1) * cell_h
        is_pad = ('PAD' in r_label)
        is_target = ('*' in r_label)
        lbl_color = '#C0392B' if is_target else ('#7F8C8D' if is_pad else '#34495E')
        ax.text(x + label_w - 0.4, ry + cell_h / 2, r_label,
                ha='right', va='center', fontsize=9.2,
                fontweight='bold', color=lbl_color, zorder=zorder + 1)

        if is_pad:
            for c in range(n_cols):
                cx = grid_x + c * cell_w
                rect = FancyBboxPatch(
                    (cx + pad_x, ry + pad_y), cell_w - 2 * pad_x, cell_h - 2 * pad_y,
                    boxstyle="round,pad=0,rounding_size=0.2",
                    facecolor='#F4F6F8', edgecolor='#E5E8EC', linewidth=0.6, zorder=zorder
                )
                ax.add_patch(rect)
            ax.text(grid_x + grid_w / 2, ry + cell_h / 2, "0",
                    ha='center', va='center', fontsize=9.0, fontweight='bold',
                    color='#7F8C8D', zorder=zorder + 2)
        else:
            for c in range(n_cols):
                cx = grid_x + c * cell_w
                color_idx = (r * 3 + c * 2) % len(palette)
                fc = palette[color_idx]
                ec = '#E07A72' if is_target else 'none'
                lw = 0.85 if is_target else 0.0
                rect = FancyBboxPatch(
                    (cx + pad_x, ry + pad_y), cell_w - 2 * pad_x, cell_h - 2 * pad_y,
                    boxstyle="round,pad=0,rounding_size=0.22",
                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder
                )
                ax.add_patch(rect)


def draw_matrix_grid(ax, x, y, w, h, n_rows, n_cols, palette, zorder=5):
    """Draws a clean 2D pastel feature matrix or attention heatmap grid."""
    cell_w = w / n_cols
    cell_h = h / n_rows
    px, py = 0.22, 0.22
    for r in range(n_rows):
        for c in range(n_cols):
            cx = x + c * cell_w
            cy = y + h - (r + 1) * cell_h
            fc = palette[(r * n_cols + c) % len(palette)]
            rect = FancyBboxPatch(
                (cx + px, cy + py), cell_w - 2 * px, cell_h - 2 * py,
                boxstyle="round,pad=0,rounding_size=0.22",
                facecolor=fc, edgecolor='none', zorder=zorder
            )
            ax.add_patch(rect)


def draw_horiz_strip(ax, x, y, w, h, colors, border_color='#B0BEC5', zorder=5):
    """Draws a horizontal vector strip container with soft pastel cells inside."""
    draw_card(ax, x, y, w, h, facecolor='#FFFFFF', edgecolor=border_color,
              lw=1.05, rounding=0.7, zorder=zorder)
    n = len(colors)
    pad = 0.45
    inner_w = w - 2 * pad
    inner_h = h - 2 * pad
    cw = inner_w / n
    for i, c in enumerate(colors):
        rect = FancyBboxPatch(
            (x + pad + i * cw + 0.15, y + pad + 0.1), cw - 0.3, inner_h - 0.2,
            boxstyle="round,pad=0,rounding_size=0.2",
            facecolor=c, edgecolor='none', zorder=zorder + 1
        )
        ax.add_patch(rect)


def draw_vert_strip(ax, x, y, w, h, colors, border_color='#B0BEC5', zorder=5):
    """Draws a vertical vector stack container with horizontal pastel bars inside."""
    draw_card(ax, x, y, w, h, facecolor='#FFFFFF', edgecolor=border_color,
              lw=1.05, rounding=0.8, zorder=zorder)
    n = len(colors)
    pad = 0.55
    inner_w = w - 2 * pad
    inner_h = h - 2 * pad
    ch = inner_h / n
    for i, c in enumerate(colors):
        ry = y + h - pad - (i + 1) * ch
        rect = FancyBboxPatch(
            (x + pad + 0.1, ry + 0.18), inner_w - 0.2, ch - 0.36,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=c, edgecolor='none', zorder=zorder + 1
        )
        ax.add_patch(rect)


def draw_mlp_network(ax, x, y, w, h, layer_sizes, node_colors, zorder=5):
    """Draws a multi-layer neural network node-link graph with soft pastel nodes."""
    n_layers = len(layer_sizes)
    lx = np.linspace(x + 1.2, x + w - 1.2, n_layers)
    coords = []
    for li, n_nodes in enumerate(layer_sizes):
        ly = np.linspace(y + 1.2, y + h - 1.2, n_nodes)
        coords.append([(lx[li], yy) for yy in ly])

    for li in range(n_layers - 1):
        for p1 in coords[li]:
            for p2 in coords[li + 1]:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color='#B0BEC5', linewidth=0.8, alpha=0.8, zorder=zorder)

    radius = 0.68 if max(layer_sizes) <= 5 else 0.56
    for li, layer_pts in enumerate(coords):
        col = node_colors[li % len(node_colors)]
        for pt in layer_pts:
            circ = Circle(pt, radius, facecolor=col, edgecolor='#546E7A',
                          linewidth=0.85, zorder=zorder + 1)
            ax.add_patch(circ)


# ==============================================================================
# ROW 1, STAGE 1: Sequence & Positional Encoding (x = 1.5 to 64.5, y = 60.5 to 124.2)
# ==============================================================================
s1_x, s1_y, s1_w, s1_h = 1.5, 60.5, 63.0, 63.7
draw_stage_panel(ax, s1_x, s1_y, s1_w, s1_h,
                 1, "Sequence & Positional Encoding",
                 bg_color='#FAFBFD', border_color='#D5DDE6',
                 header_bg='#EBF2FA', header_text_color='#2C4A7E')

# --- Top Sub-container: 41-bp DNA Sequence Window ---
draw_card(ax, 3.0, 90.0, 60.0, 27.0, facecolor='#FFFDF9', edgecolor='#EED2B5', lw=1.2)
ax.text(4.5, 114.6, "41-bp DNA Sequence",
        ha='left', va='center', fontsize=11.5, fontweight='bold', color='#AD5D28')

# Mini-card 1A: DNA Duplex & Target Base Illustration
draw_card(ax, 4.3, 102.2, 28.5, 10.4, facecolor='#FFFFFF', edgecolor='#F3DFC8', lw=1.05)
hx = np.linspace(5.2, 12.6, 60)
hy1 = 107.4 + 2.5 * np.sin((hx - 5.2) * 0.98)
hy2 = 107.4 - 2.5 * np.sin((hx - 5.2) * 0.98)
ax.plot(hx, hy1, color='#E59866', lw=1.7, zorder=5)
ax.plot(hx, hy2, color='#7FB3D5', lw=1.7, zorder=5)
for bx_pos in np.linspace(5.7, 12.1, 7):
    by1 = 107.4 + 2.5 * np.sin((bx_pos - 5.2) * 0.98)
    by2 = 107.4 - 2.5 * np.sin((bx_pos - 5.2) * 0.98)
    is_mid = abs(bx_pos - 8.9) < 0.6
    ax.plot([bx_pos, bx_pos], [by1, by2],
            color=('#D96B62' if is_mid else '#BDC3C7'),
            lw=(1.9 if is_mid else 0.95), zorder=4)
ax.scatter([8.9], [107.4], s=55, color='#F1948A', alpha=0.45, zorder=6)
ax.scatter([8.9], [107.4], s=18, color='#C0392B', zorder=7)

ax.text(23.0, 109.1, "41-bp Window",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#8E4B1E')
ax.text(23.0, 105.4, "Target Site C / A*",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#C0392B')

# Mini-card 1B: Token Embedding Layer
draw_card(ax, 4.3, 91.2, 28.5, 9.4, facecolor='#FDF6EE', edgecolor='#E8B88A', lw=1.15)
ax.text(18.55, 98.3, "Token Embedding",
        ha='center', va='center', fontsize=11.0, fontweight='bold', color='#8E4B1E')
ax.text(18.55, 95.5, "DNA Vocabulary (d = 128)",
        ha='center', va='center', fontsize=9.6, color='#7E5130')
for idx, lbl in enumerate(["Embed", "Scale", "Dropout"]):
    px = 5.5 + idx * 8.9
    draw_card(ax, px, 91.9, 8.1, 2.4, facecolor='#FFFFFF', edgecolor='#E5A970', lw=0.95, rounding=0.6, zorder=5)
    ax.text(px + 4.05, 93.1, lbl, ha='center', va='center', fontsize=8.8, fontweight='bold', color='#8E4B1E', zorder=6)

draw_arrow(ax, 18.55, 102.2, 18.55, 100.6, color='#D38B56', lw=1.3)
draw_arrow(ax, 32.8, 95.9, 35.8, 95.9, color='#D38B56', lw=1.35)

# Right side of Top Sub-container: 2D Nucleotide Token Matrix
draw_card(ax, 36.0, 91.2, 25.6, 23.6, facecolor='#FFFFFF', edgecolor='#F3DFC8', lw=1.1)
ax.text(48.8, 112.8, "Token Matrix",
        ha='center', va='center', fontsize=11.0, fontweight='bold', color='#34495E')
draw_2d_token_matrix(
    ax, 36.8, 94.3, 24.0, 16.6,
    row_labels=['A1', 'C2', 'G20', 'C/A21*', 'T41', '[PAD]'],
    palette=PASTEL_WARM
)
ax.text(48.8, 92.6, "41 × 128",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#4A5568')


# --- Bottom Sub-container: Sinusoidal Positional Encoding ---
draw_card(ax, 3.0, 62.0, 60.0, 26.6, facecolor='#F7FBFE', edgecolor='#B5D6EC', lw=1.2)
ax.text(4.5, 86.2, "Positional Encoding",
        ha='left', va='center', fontsize=11.5, fontweight='bold', color='#286690')

# Mini-card 1C: Sinusoidal Multi-Frequency Wave Plot
draw_card(ax, 4.3, 74.0, 28.5, 10.2, facecolor='#FFFFFF', edgecolor='#CBE3F5', lw=1.05)
wx = np.linspace(5.2, 12.6, 80)
for freq, col in [(1.0, '#6CA6CD'), (2.2, '#8E99E6'), (3.8, '#66BFA6')]:
    wy = 79.1 + 2.4 * np.sin((wx - 5.2) * freq)
    ax.plot(wx, wy, color=col, lw=1.4, alpha=0.85, zorder=5)
ax.axvline(8.9, ymin=(75.0/126), ymax=(83.2/126), color='#D96B62', linestyle='--', lw=1.05, zorder=6)
ax.text(23.0, 80.7, "Sinusoidal PE",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#24557A')
ax.text(23.0, 77.1, "Position Phase",
        ha='center', va='center', fontsize=10.0, color='#286690')

# Mini-card 1D: Positional Encoding Fusion
draw_card(ax, 4.3, 63.2, 28.5, 9.2, facecolor='#EDF6FC', edgecolor='#8EC2E6', lw=1.15)
ax.text(18.55, 70.1, "Position-Aware Input",
        ha='center', va='center', fontsize=11.0, fontweight='bold', color='#24557A')
ax.text(18.55, 67.4, "Token + Positional Fusion",
        ha='center', va='center', fontsize=9.6, color='#286690')
for idx, lbl in enumerate(["Sin", "Cos", "Norm"]):
    px = 5.5 + idx * 8.9
    draw_card(ax, px, 63.9, 8.1, 2.4, facecolor='#FFFFFF', edgecolor='#7BB4DC', lw=0.95, rounding=0.6, zorder=5)
    ax.text(px + 4.05, 65.1, lbl, ha='center', va='center', fontsize=8.8, fontweight='bold', color='#286690', zorder=6)

draw_arrow(ax, 18.55, 74.0, 18.55, 72.4, color='#5B9BD5', lw=1.3)
draw_arrow(ax, 32.8, 67.8, 35.8, 67.8, color='#5B9BD5', lw=1.35)

# Right side of Bottom Sub-container: 2D Position-Encoded Matrix
draw_card(ax, 36.0, 63.2, 25.6, 23.4, facecolor='#FFFFFF', edgecolor='#CBE3F5', lw=1.1)
ax.text(48.8, 84.6, "Encoded Sequence",
        ha='center', va='center', fontsize=11.0, fontweight='bold', color='#34495E')
draw_2d_token_matrix(
    ax, 36.8, 66.3, 24.0, 16.5,
    row_labels=['t=1', 't=2', 't=20', 't=21*', 't=41', '[PAD]'],
    palette=PASTEL_COOL
)
ax.text(48.8, 64.6, "41 × 128",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#4A5568')


# ==============================================================================
# ROW 1, STAGE 2: Time-Frequency Front-End (x = 66.8 to 118.8, y = 60.5 to 124.2)
# ==============================================================================
s2_x, s2_y, s2_w, s2_h = 66.8, 60.5, 52.0, 63.7
draw_stage_panel(ax, s2_x, s2_y, s2_w, s2_h,
                 2, "Time-Frequency Front-End",
                 bg_color='#FAFBFD', border_color='#D5DDE6',
                 header_bg='#FAF1E2', header_text_color='#7E5119')

# --- Top Branch: Temporal-Spatial Branch ---
draw_card(ax, 68.3, 94.2, 37.2, 22.8, facecolor='#FFFDF7', edgecolor='#E6BE8A', lw=1.2)
ax.text(86.9, 114.4, "Temporal-Spatial Branch",
        ha='center', va='center', fontsize=11.5, fontweight='bold', color='#9C640C')
ax.text(86.9, 111.2, "Dilated TCN + Multi-Scale CNN",
        ha='center', va='center', fontsize=10.0, color='#7E5109')

draw_card(ax, 70.0, 103.4, 33.8, 6.4, facecolor='#FFFFFF', edgecolor='#F2D7B6', lw=1.0)
for idx, (lbl, col, w_bar) in enumerate([
    ("TCN", '#E59866', 9.6),
    ("MS-CNN", '#EB984E', 9.6),
    ("GeLU", '#D4AC6E', 9.2)
]):
    bx = 71.2 + idx * 10.9
    draw_card(ax, bx, 104.5, w_bar, 4.2, facecolor=col, edgecolor='none', rounding=0.6, alpha=0.18, zorder=4)
    draw_card(ax, bx, 104.5, w_bar, 4.2, facecolor='none', edgecolor=col, lw=1.0, rounding=0.6, zorder=5)
    ax.text(bx + w_bar / 2, 106.6, lbl, ha='center', va='center', fontsize=9.6, fontweight='bold', color='#8E4B1E', zorder=6)

draw_horiz_strip(ax, 70.2, 98.0, 33.4, 3.8,
                 ['#F5CBA7', '#FAD7A0', '#F0B27A', '#EDBB99', '#F5B7B1', '#F8C471'],
                 border_color='#DEB887')
ax.text(86.9, 95.8, "Multi-Scale Motif Features",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#9C640C')

# --- Middle Dashed Anchor Pill ---
draw_card(ax, 68.3, 88.0, 37.2, 4.6, facecolor='#FDF2F4', edgecolor='#E69A9A',
          lw=1.2, linestyle='--', rounding=0.9)
ax.text(86.9, 90.3, "Adaptive Gated Fusion",
        ha='center', va='center', fontsize=10.8, fontweight='bold', color='#A93226')

# --- Bottom Branch: Frequency-Wavelet Branch ---
draw_card(ax, 68.3, 62.0, 37.2, 24.2, facecolor='#F6FCFA', edgecolor='#95D1C4', lw=1.2)
ax.text(86.9, 83.6, "Frequency-Wavelet Branch",
        ha='center', va='center', fontsize=11.5, fontweight='bold', color='#1E7065')
ax.text(86.9, 80.4, "Learnable VMD + Orthogonal Wavelet",
        ha='center', va='center', fontsize=10.0, color='#1B5E55')

draw_card(ax, 70.0, 72.2, 33.8, 6.6, facecolor='#FFFFFF', edgecolor='#B8E2D8', lw=1.0)
for idx, (lbl, col, w_bar) in enumerate([
    ("VMD", '#56B4A9', 9.6),
    ("Wavelet", '#6CA6CD', 9.6),
    ("Shrink", '#8E99E6', 9.2)
]):
    bx = 71.2 + idx * 10.9
    draw_card(ax, bx, 73.3, w_bar, 4.3, facecolor=col, edgecolor='none', rounding=0.6, alpha=0.16, zorder=4)
    draw_card(ax, bx, 73.3, w_bar, 4.3, facecolor='none', edgecolor=col, lw=1.0, rounding=0.6, zorder=5)
    ax.text(bx + w_bar / 2, 75.45, lbl, ha='center', va='center', fontsize=9.6, fontweight='bold', color='#1B5E55', zorder=6)

draw_horiz_strip(ax, 70.2, 66.6, 33.4, 3.8,
                 ['#A9CCE3', '#AED6F1', '#A3E4D7', '#A2D9CE', '#C5CAE9', '#D1F2EB'],
                 border_color='#85C1B5')
ax.text(86.9, 64.2, "Wavelet-Spectral Features",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#1E7065')

# --- Right Side of Stage 2: Vertical Feature Stacks ---
ax.text(112.4, 115.2, "Temporal",
        ha='center', va='center', fontsize=10.4, fontweight='bold', color='#9C640C')
draw_vert_strip(ax, 108.6, 94.6, 7.6, 18.8,
                ['#F5CBA7', '#FAD7A0', '#F0B27A', '#EDBB99', '#F5B7B1', '#F8C471', '#F9E79F'],
                border_color='#DEB887')
ax.text(112.4, 92.6, "41 × 128",
        ha='center', va='center', fontsize=9.8, fontweight='bold', color='#4A5568')

ax.text(112.4, 86.0, "Spectral",
        ha='center', va='center', fontsize=10.4, fontweight='bold', color='#1E7065')
draw_vert_strip(ax, 108.6, 65.4, 7.6, 18.8,
                ['#A9CCE3', '#AED6F1', '#C5CAE9', '#A3E4D7', '#A2D9CE', '#D4E6F1', '#D1F2EB'],
                border_color='#85C1B5')
ax.text(112.4, 63.4, "41 × 128",
        ha='center', va='center', fontsize=9.8, fontweight='bold', color='#4A5568')

draw_arrow(ax, 61.6, 103.5, 68.3, 103.5, color='#C88A4B', lw=1.45)
draw_arrow(ax, 61.6, 74.5, 68.3, 74.5, color='#5B9BD5', lw=1.45)
draw_arrow(ax, 105.5, 104.2, 108.6, 104.2, color='#C88A4B', lw=1.4)
draw_arrow(ax, 105.5, 74.8, 108.6, 74.8, color='#56B4A9', lw=1.4)


# ==============================================================================
# ROW 1, STAGE 3: Dual-Path Context Backbone & SE Fusion (x = 121.0 to 208.5, y = 60.5 to 124.2)
# ==============================================================================
s3_x, s3_y, s3_w, s3_h = 121.0, 60.5, 87.5, 63.7
draw_stage_panel(ax, s3_x, s3_y, s3_w, s3_h,
                 3, "Dual-Path Context Backbone & SE Fusion",
                 bg_color='#FAFBFD', border_color='#D5DDE6',
                 header_bg='#ECEFFA', header_text_color='#393B80')

# --- Top Sub-panel: Sequential BiLSTM Pathway ---
draw_card(ax, 127.2, 93.0, 52.8, 24.0, facecolor='#FFFDF7', edgecolor='#E6BE8A', lw=1.2)
draw_card(ax, 129.0, 112.0, 49.2, 3.8, facecolor='#FAF1E2', edgecolor='none', rounding=0.7, zorder=4)
ax.text(153.6, 113.9, "Sequential BiLSTM Pathway",
        ha='center', va='center', fontsize=11.4, fontweight='bold', color='#8E4B1E', zorder=5)

ax.text(134.8, 106.0, "Forward",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#9C640C')
ax.text(134.8, 99.2, "Backward",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#9C640C')

draw_card(ax, 142.2, 94.6, 16.4, 15.8, facecolor='#FFFFFF', edgecolor='#E6BE8A', lw=1.1, zorder=4)
ax.text(150.4, 108.3, "2-Layer BiLSTM",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#8E4B1E', zorder=5)
draw_matrix_grid(ax, 144.0, 96.0, 12.8, 10.2, 4, 4, PASTEL_WARM, zorder=5)

ax.text(160.8, 104.0, "+", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#6C7A89')
draw_arrow(ax, 158.6, 102.0, 163.0, 102.0, color='#64748B', lw=1.35)

ax.text(170.5, 110.0, "BiLSTM Out",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#34495E')
draw_matrix_grid(ax, 164.0, 96.6, 13.0, 11.4, 4, 4,
                 ['#F5CBA7', '#F5B7B1', '#A3E4D7', '#FAD7A0', '#F8C471', '#F0B27A', '#A2D9CE', '#F9E79F'], zorder=5)
ax.text(170.5, 94.8, "41 × 256",
        ha='center', va='center', fontsize=10.0, fontweight='bold', color='#4A5568')


# --- Bottom Sub-panel: Global Self-Attention Pathway ---
draw_card(ax, 127.2, 62.0, 52.8, 28.0, facecolor='#F7FBFE', edgecolor='#A8C9E8', lw=1.2)
draw_card(ax, 129.0, 85.0, 49.2, 3.8, facecolor='#EBF4FC', edgecolor='none', rounding=0.7, zorder=4)
ax.text(153.6, 86.9, "Global Self-Attention Pathway",
        ha='center', va='center', fontsize=11.4, fontweight='bold', color='#24557A', zorder=5)

ax.text(134.8, 78.2, "Query",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#286690')
ax.text(134.8, 70.8, "Key / Val",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#9C640C')

draw_card(ax, 142.2, 63.8, 16.4, 19.0, facecolor='#FFFFFF', edgecolor='#A5B4DF', lw=1.1, zorder=4)
ax.text(150.4, 80.5, "8-Head MHSA",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#393B80', zorder=5)
draw_matrix_grid(ax, 144.0, 65.5, 12.8, 12.5, 4, 4, PASTEL_MHA, zorder=5)

ax.text(160.8, 74.8, "+", ha='center', va='center', fontsize=13.0, fontweight='bold', color='#6C7A89')
draw_arrow(ax, 158.6, 72.8, 163.0, 72.8, color='#64748B', lw=1.35)

ax.text(170.5, 81.6, "Attention Out",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#34495E')
draw_matrix_grid(ax, 164.0, 66.8, 13.0, 12.6, 4, 4, PASTEL_COOL, zorder=5)
ax.text(170.5, 64.6, "41 × 128",
        ha='center', va='center', fontsize=10.0, fontweight='bold', color='#4A5568')

# Cross-routing arrows between Stage 2 and Stage 3
draw_arrow(ax, 116.2, 105.5, 127.2, 105.5, color='#C88A4B', lw=1.45)
draw_arrow(ax, 116.2, 75.5, 127.2, 75.5, color='#5B9BD5', lw=1.45)
draw_arrow(ax, 116.2, 101.5, 127.2, 71.2, color='#C88A4B', lw=1.3, linestyle='--')
draw_arrow(ax, 116.2, 79.5, 127.2, 100.2, color='#5B9BD5', lw=1.3, linestyle='--')


# --- Right Column of Stage 3: 3 Stacked Pooling/Fusion Cards ---
draw_card(ax, 182.2, 95.8, 24.6, 18.8, facecolor='#F4FBF8', edgecolor='#8FD3C1', lw=1.2)
ax.text(194.5, 111.4, "Sequential Context",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#1E6B52')
draw_horiz_strip(ax, 183.8, 103.4, 21.4, 4.2,
                 ['#F5CBA7', '#FAD7A0', '#F0B27A', '#A3E4D7', '#EDBB99', '#F5B7B1'],
                 border_color='#8FD3C1')
ax.text(194.5, 99.0, "41 × 256",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#1E6B52')

draw_card(ax, 182.2, 81.6, 24.6, 11.4, facecolor='#FFFDF6', edgecolor='#E6BE8A', lw=1.25)
ax.text(194.5, 90.3, "SE Channel Fusion",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#8E4B1E')
draw_horiz_strip(ax, 183.4, 86.0, 22.2, 2.8,
                 ['#F5CBA7', '#F0B27A', '#F5B7B1', '#A3E4D7', '#A9CCE3', '#C5CAE9', '#AED6F1', '#A2D9CE'],
                 border_color='#DEB887')
ax.text(194.5, 83.6, "Channel Attention",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#7E5109')

draw_card(ax, 182.2, 62.0, 24.6, 16.8, facecolor='#F4FBF8', edgecolor='#8FD3C1', lw=1.2)
ax.text(194.5, 75.8, "Global Context",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#1E6B52')
draw_horiz_strip(ax, 183.8, 68.8, 21.4, 4.2,
                 ['#A9CCE3', '#C5CAE9', '#A3E4D7', '#AED6F1', '#A2D9CE', '#D4E6F1'],
                 border_color='#8FD3C1')
ax.text(194.5, 64.8, "41 × 128",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#1E6B52')

draw_arrow(ax, 180.0, 104.0, 182.2, 104.0, color='#66BFA6', lw=1.45)
draw_arrow(ax, 180.0, 72.8, 182.2, 72.8, color='#66BFA6', lw=1.45)
draw_arrow(ax, 194.5, 95.8, 194.5, 93.0, color='#C88A4B', lw=1.45)
draw_arrow(ax, 194.5, 78.8, 194.5, 81.6, color='#C88A4B', lw=1.45)


# ==============================================================================
# SOFT GOLDEN WRAP-AROUND BUS FROM STAGE 3 (ROW 1 RIGHT) TO STAGE 4 (ROW 2 LEFT)
# ==============================================================================
draw_polyline_arrow(
    ax,
    [
        (206.8, 87.3),
        (209.2, 87.3),
        (209.2, 57.6),
        (0.6, 57.6),
        (0.6, 25.0),
        (3.2, 25.0)
    ],
    color='#D49B4B', lw=1.7, head_w=1.05, head_l=1.3, zorder=10
)


# ==============================================================================
# ROW 2, STAGE 4: Target-Anchored Readout & Ensemble Training (x = 1.5 to 113.5, y = 1.8 to 55.2)
# ==============================================================================
s4_x, s4_y, s4_w, s4_h = 1.5, 1.8, 112.0, 53.4
draw_stage_panel(ax, s4_x, s4_y, s4_w, s4_h,
                 4, "Target-Anchored Pooling & Ensemble Readout",
                 bg_color='#FAFBFD', border_color='#D5DDE6',
                 header_bg='#FBECEE', header_text_color='#8A2B3E')

# --- Far Left of Stage 4: Vertical Multi-Color Stack ---
draw_card(ax, 3.2, 3.8, 10.8, 44.2, facecolor='#FFFDF7', edgecolor='#DEB887', lw=1.2)
ax.text(8.6, 45.2, "Context",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#34495E')
draw_vert_strip(ax, 4.5, 8.2, 8.2, 34.6, PASTEL_MIX, border_color='#D5DDE6')
ax.text(8.6, 5.8, "41 × 384",
        ha='center', va='center', fontsize=10.0, fontweight='bold', color='#4A5568')

# --- Top Branch of Stage 4: 1. Target-Anchored MLP Classifier ---
draw_card(ax, 16.0, 27.2, 58.0, 20.8, facecolor='#F6FAFE', edgecolor='#9BC2E6', lw=1.2)
ax.text(17.8, 45.0, "1. Deep MLP Classification Branch",
        ha='left', va='center', fontsize=11.4, fontweight='bold', color='#24557A')

draw_mlp_network(ax, 18.2, 29.2, 24.5, 13.8,
                 layer_sizes=[4, 5, 4, 1],
                 node_colors=['#AED6F1', '#A3E4D7', '#D2B4DE', '#A9CCE3'],
                 zorder=5)

draw_arrow(ax, 43.2, 36.1, 46.8, 36.1, color='#5B9BD5', lw=1.45)

draw_card(ax, 47.0, 30.2, 24.8, 11.8, facecolor='#EBF4FC', edgecolor='#85B8E0', lw=1.15, zorder=4)
ax.text(59.4, 38.2, "Methylation Score",
        ha='center', va='center', fontsize=11.2, fontweight='bold', color='#24557A', zorder=5)
ax.text(59.4, 33.4, "Classifier Head",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#2E6B9E', zorder=5)


# --- Bottom Branch of Stage 4: 2. Positional Attention & Latent Manifold Readout ---
draw_card(ax, 16.0, 3.8, 58.0, 21.4, facecolor='#FDF6F7', edgecolor='#E6A8B2', lw=1.2)
ax.text(17.8, 22.6, "2. Positional Attention & Latent Readout",
        ha='left', va='center', fontsize=11.4, fontweight='bold', color='#8A2B3E')

draw_mlp_network(ax, 18.0, 13.8, 12.6, 6.8,
                 layer_sizes=[3, 4, 1],
                 node_colors=['#F5CBA7', '#A3E4D7', '#F5B7B1'], zorder=5)
draw_mlp_network(ax, 18.0, 5.2, 12.6, 6.8,
                 layer_sizes=[3, 4, 1],
                 node_colors=['#AED6F1', '#A3E4D7', '#A9CCE3'], zorder=5)

draw_arrow(ax, 31.0, 17.2, 34.0, 17.2, color='#D97B88', lw=1.35)
draw_arrow(ax, 31.0, 8.6, 34.0, 8.6, color='#5B9BD5', lw=1.35)

draw_card(ax, 34.2, 13.8, 16.8, 6.8, facecolor='#FBECEE', edgecolor='#E6A8B2', lw=1.15, zorder=4)
ax.text(42.6, 17.2, "Attention",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#8A2B3E', zorder=5)

draw_card(ax, 34.2, 5.2, 16.8, 6.8, facecolor='#EDF6FC', edgecolor='#8EC2E6', lw=1.15, zorder=4)
ax.text(42.6, 8.6, "Latent Space",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#24557A', zorder=5)

draw_card(ax, 53.6, 6.8, 18.6, 12.2, facecolor='#FFF9FA', edgecolor='#E098A4', lw=1.15, zorder=4)
ax.text(62.9, 15.2, "Target Readout",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#8A2B3E', zorder=5)
ax.text(62.9, 10.2, "Context Vector",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#A93226', zorder=5)

draw_polyline_arrow(ax, [(51.0, 17.2), (52.3, 17.2), (52.3, 14.5), (53.6, 14.5)], color='#D97B88', lw=1.3)
draw_polyline_arrow(ax, [(51.0, 8.6), (52.3, 8.6), (52.3, 11.2), (53.6, 11.2)], color='#5B9BD5', lw=1.3)


# --- Right Column of Stage 4: 5-Fold Ensemble & Supervision ---
draw_card(ax, 76.2, 3.8, 35.4, 44.2, facecolor='#FFFDF9', edgecolor='#EED2B5', lw=1.2)
ax.text(93.9, 45.0, "5-Fold Ensemble & Training",
        ha='center', va='center', fontsize=11.4, fontweight='bold', color='#8E4B1E')

draw_card(ax, 78.4, 33.4, 31.0, 8.4, facecolor='#FFFFFF', edgecolor='#9BC2E6', lw=1.15, zorder=4)
ax.text(93.9, 37.6, "Cross-Entropy Loss",
        ha='center', va='center', fontsize=11.2, fontweight='bold', color='#24557A', zorder=5)

draw_card(ax, 78.4, 18.2, 31.0, 11.2, facecolor='#FAF1E2', edgecolor='#E6BE8A', lw=1.25, zorder=4)
ax.text(93.9, 26.0, "5-Fold Soft-Voting",
        ha='center', va='center', fontsize=11.2, fontweight='bold', color='#8E4B1E', zorder=5)
ax.text(93.9, 21.2, "Ensemble Probability",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#9C640C', zorder=5)

draw_card(ax, 78.4, 6.2, 31.0, 8.2, facecolor='#FFFFFF', edgecolor='#E6A8B2', lw=1.15, zorder=4)
ax.text(93.9, 10.3, "AdamW + Cosine LR",
        ha='center', va='center', fontsize=11.0, fontweight='bold', color='#8A2B3E', zorder=5)

draw_arrow(ax, 14.0, 36.8, 16.0, 36.8, color='#5B9BD5', lw=1.45)
draw_arrow(ax, 14.0, 14.5, 16.0, 14.5, color='#D97B88', lw=1.45)
draw_arrow(ax, 74.0, 36.8, 78.4, 37.6, color='#5B9BD5', lw=1.45)
draw_arrow(ax, 72.2, 12.8, 78.4, 10.3, color='#D97B88', lw=1.45)
draw_arrow(ax, 93.9, 33.4, 93.9, 29.4, color='#C88A4B', lw=1.45)
draw_arrow(ax, 93.9, 14.4, 93.9, 18.2, color='#C88A4B', lw=1.45)


# ==============================================================================
# ROW 2, STAGE 5: Epigenomic Attribution & Generalization (x = 116.2 to 208.5, y = 1.8 to 55.2)
# ==============================================================================
s5_x, s5_y, s5_w, s5_h = 116.2, 1.8, 92.3, 53.4
draw_stage_panel(ax, s5_x, s5_y, s5_w, s5_h,
                 5, "Epigenomic Attribution & Generalization",
                 bg_color='#FAFBFD', border_color='#D5DDE6',
                 header_bg='#E6F5EF', header_text_color='#1E6B52')

draw_arrow(ax, 111.6, 23.8, 116.2, 23.8, color='#66BFA6', lw=1.55, linestyle='--')

# --- Card 5.1: Target Locus & Motif Recovery ---
draw_card(ax, 118.0, 3.8, 28.6, 44.2, facecolor='#FFFFFF', edgecolor='#D5DDE6', lw=1.15)
ax.text(132.3, 45.0, "Motif Recovery",
        ha='center', va='center', fontsize=11.4, fontweight='bold', color='#34495E')

draw_card(ax, 119.6, 13.0, 25.4, 29.2, facecolor='#FAFBFD', edgecolor='#E5E8EC', lw=1.0, zorder=4)
px_vals = np.linspace(121.0, 143.6, 100)
py_base = 17.5
py_curve = py_base + 18.0 * np.exp(-((px_vals - 132.3) / 3.2) ** 2) + \
           2.2 * np.exp(-((px_vals - 127.0) / 2.0) ** 2) + \
           2.0 * np.exp(-((px_vals - 137.5) / 2.0) ** 2)
ax.fill_between(px_vals, py_base, py_curve, color='#F5B7B1', alpha=0.45, zorder=5)
ax.plot(px_vals, py_curve, color='#D96B62', lw=1.8, zorder=6)

motif_box = Rectangle((129.1, 16.5), 6.4, 20.8, facecolor='none',
                      edgecolor='#9B82C8', linewidth=1.2, linestyle='--', zorder=7)
ax.add_patch(motif_box)
ax.scatter([132.3], [35.5], s=44, color='#C0392B', zorder=8)

for cx_b, cy_b, c_txt in [
    (124.2, 34.4, "A*TGC"),
    (140.4, 34.4, "GAGG"),
    (132.3, 39.3, "Center Peak")
]:
    draw_card(ax, cx_b - 3.8, cy_b - 1.2, 7.6, 2.4, facecolor='#FDF2F4', edgecolor='#E6A8B2', lw=0.8, rounding=0.4, zorder=8)
    ax.text(cx_b, cy_b, c_txt, ha='center', va='center', fontsize=8.0, fontweight='bold', color='#8A2B3E', zorder=9)

for bx_off, b_lbl in [(-9.0, "-20"), (-4.5, "-5"), (0.0, "0*"), (4.5, "+5"), (9.0, "+20")]:
    ax.text(132.3 + bx_off, 14.8, b_lbl, ha='center', va='center',
            fontsize=8.6, fontweight='bold',
            color=('#C0392B' if b_lbl == "0*" else '#546E7A'), zorder=7)

ax.text(132.3, 8.8, "Core Cis-Element Loci",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#8A2B3E')
ax.text(132.3, 5.8, "(±5 bp Target Window)",
        ha='center', va='center', fontsize=10.0, fontweight='bold', color='#8A2B3E')


# --- Card 5.2: Time-Frequency & Manifold Attribution ---
draw_card(ax, 148.2, 3.8, 29.8, 44.2, facecolor='#FFFFFF', edgecolor='#D5DDE6', lw=1.15)
ax.text(163.1, 45.0, "Domain Attribution",
        ha='center', va='center', fontsize=11.4, fontweight='bold', color='#34495E')

draw_card(ax, 149.6, 28.2, 8.2, 13.8, facecolor='#FDF2F4', edgecolor='#E6A8B2', lw=1.05, zorder=4)
ax.text(153.7, 36.8, "Temporal",
        ha='center', va='center', fontsize=9.0, fontweight='bold', color='#8A2B3E', zorder=5)
ax.text(153.7, 32.2, "Local\nMotif",
        ha='center', va='center', fontsize=9.2, fontweight='bold', color='#8A2B3E', zorder=5)

draw_card(ax, 158.4, 28.2, 18.2, 13.8, facecolor='#FFFFFF', edgecolor='#E5E8EC', lw=1.0, zorder=4)
pts_top = [(160.2, 33.0), (162.8, 36.2), (165.6, 34.0), (168.4, 37.2), (171.2, 34.5), (174.2, 36.8)]
ax.plot([p[0] for p in pts_top], [p[1] for p in pts_top], color='#64748B', lw=1.45, zorder=5)
for gx, gy, gs in [(162.8, 36.2, 140), (165.6, 34.0, 210), (168.4, 37.2, 250), (171.2, 34.5, 145)]:
    ax.scatter([gx], [gy], s=gs, color='#F1948A', alpha=0.35, edgecolors='none', zorder=6)
    ax.scatter([gx], [gy], s=gs*0.35, color='#D96B62', alpha=0.75, edgecolors='none', zorder=7)
ax.text(167.5, 30.0, "Core Motif Loci",
        ha='center', va='center', fontsize=9.0, fontweight='bold', color='#8A2B3E', zorder=7)

draw_card(ax, 149.6, 13.0, 8.2, 13.8, facecolor='#F2F8FD', edgecolor='#9BC2E6', lw=1.05, zorder=4)
ax.text(153.7, 21.6, "Spectral",
        ha='center', va='center', fontsize=9.0, fontweight='bold', color='#24557A', zorder=5)
ax.text(153.7, 17.0, "Wavelet\nFlank",
        ha='center', va='center', fontsize=9.2, fontweight='bold', color='#24557A', zorder=5)

draw_card(ax, 158.4, 13.0, 18.2, 13.8, facecolor='#FFFFFF', edgecolor='#E5E8EC', lw=1.0, zorder=4)
pts_bot = [(160.2, 18.2), (162.8, 21.4), (165.6, 19.0), (168.4, 22.0), (171.2, 19.4), (174.2, 21.6)]
ax.plot([p[0] for p in pts_bot], [p[1] for p in pts_bot], color='#64748B', lw=1.45, zorder=5)
for gx, gy, gs in [(160.2, 18.2, 175), (162.8, 21.4, 215), (171.2, 19.4, 205), (174.2, 21.6, 235)]:
    ax.scatter([gx], [gy], s=gs, color='#85C1E9', alpha=0.35, edgecolors='none', zorder=6)
    ax.scatter([gx], [gy], s=gs*0.35, color='#5B9BD5', alpha=0.75, edgecolors='none', zorder=7)
ax.text(167.5, 14.8, "Flanking Envelope",
        ha='center', va='center', fontsize=9.0, fontweight='bold', color='#24557A', zorder=7)

ax.text(163.1, 8.8, "Time-Frequency",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#8E4B1E')
ax.text(163.1, 5.8, "Domain Separation",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#8E4B1E')


# --- Card 5.3: Cross-Species Generalization & Transfer ---
draw_card(ax, 179.6, 3.8, 27.2, 44.2, facecolor='#FFFFFF', edgecolor='#D5DDE6', lw=1.15)
ax.text(193.2, 45.0, "Species Transfer",
        ha='center', va='center', fontsize=11.4, fontweight='bold', color='#34495E')

badges = [
    ("4mC (4 Species)",    '#EFF9F5', '#8FD3C1', '#1E6B52', 35.8),
    ("6mA (11 Species)",   '#F2F8FD', '#9BC2E6', '#24557A', 28.4),
    ("Cross-Species AUC",  '#FFFDF5', '#E6BE8A', '#8E4B1E', 21.0),
    ("Noise Robustness",   '#FDF4F6', '#E6A8B2', '#8A2B3E', 13.6),
]
for b_text, b_bg, b_ec, b_tc, b_y in badges:
    draw_card(ax, 181.4, b_y, 23.6, 5.6, facecolor=b_bg, edgecolor=b_ec, lw=1.2, rounding=0.9, zorder=4)
    ax.text(193.2, b_y + 2.8, b_text,
            ha='center', va='center', fontsize=10.6, fontweight='bold', color=b_tc, zorder=5)

ax.text(193.2, 8.8, "15-Species Benchmark",
        ha='center', va='center', fontsize=10.6, fontweight='bold', color='#1E6B52')
ax.text(193.2, 5.8, "Generalization",
        ha='center', va='center', fontsize=10.2, fontweight='bold', color='#1E6B52')


import io
plt.tight_layout(pad=0.2)
for ext, fmt in [('pdf', 'pdf'), ('png', 'png')]:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches='tight', pad_inches=0.06)
    with open(f'/Volumes/Files/Research/DNAPred_Luo/paper_new/fig_architecture.{ext}', 'wb') as f:
        f.write(buf.getvalue())
print("Saved final soft-pastel large-font /Volumes/Files/Research/DNAPred_Luo/paper_new/fig_architecture.pdf and .png successfully.")
