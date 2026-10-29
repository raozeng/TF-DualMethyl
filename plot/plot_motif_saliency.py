import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Exact 41-bp normalized Layer Integrated Gradients (LIG) saliency scores
# extracted from the trained TF-DualMethyl model on Fragaria vesca (6mA) independent test set
SAMPLES = [
    {
        'sample_id': 1079,
        'confidence': 0.9489,
        'seq': 'TTATTCTTCATCACGTTATCATGCACTTTGTTCTAACCCTA',
        'attr': [
            0.016, 0.020, 0.012, 0.012, 0.016, 0.024, 0.036, 0.020, 0.020, 0.020,
            0.024, 0.036, 0.036, 0.024, 0.067, 0.059, 0.075, 0.294, 0.059, 0.075,
            0.616, 0.635, 0.439, 0.423, 1.000, 0.114, 0.063, 0.189, 0.353, 0.161,
            0.075, 0.067, 0.036, 0.012, 0.020, 0.016, 0.008, 0.016, 0.024, 0.012,
            0.000
        ],
        # (start, end, label, label_x)
        'motifs': [(20, 25, 'A*TGCA core', 23.3)],
    },
    {
        'sample_id': 621,
        'confidence': 0.9476,
        'seq': 'TGTGAAGTATCCTTCAGGCTATGCAGGAAATATAGCTCGAT',
        'attr': [
            0.012, 0.012, 0.008, 0.012, 0.020, 0.016, 0.020, 0.016, 0.012, 0.016,
            0.016, 0.012, 0.016, 0.012, 0.083, 0.263, 0.047, 0.024, 0.047, 0.538,
            0.949, 0.686, 0.141, 0.205, 0.302, 0.334, 1.000, 0.592, 0.255, 0.526,
            0.012, 0.165, 0.036, 0.032, 0.024, 0.036, 0.028, 0.020, 0.008, 0.000,
            0.012
        ],
        'motifs': [(19, 24, 'TA*TGC core', 22.8), (24, 30, 'AGGAAA motif', 27.6)],
    },
    {
        'sample_id': 913,
        'confidence': 0.9476,
        'seq': 'GGCAGTAACTTTCCTGGATGATTCAGCACATCACTGGTGGT',
        'attr': [
            0.047, 0.047, 0.067, 0.020, 0.059, 0.032, 0.012, 0.012, 0.000, 0.012,
            0.016, 0.016, 0.020, 0.032, 0.012, 0.024, 0.004, 0.251, 0.047, 0.133,
            0.894, 0.639, 0.968, 0.118, 0.408, 0.291, 1.000, 0.581, 0.612, 0.486,
            0.075, 0.133, 0.149, 0.098, 0.181, 0.063, 0.075, 0.036, 0.040, 0.028,
            0.008
        ],
        'motifs': [(20, 24, 'A*TTC core', 22.8), (24, 30, 'AGCACA motif', 27.6)],
    },
    {
        'sample_id': 198,
        'confidence': 0.9471,
        'seq': 'CTCATGGATGTTTGGGAGAGATTCACCTGCTGCATTCATTT',
        'attr': [
            0.020, 0.008, 0.024, 0.020, 0.028, 0.032, 0.016, 0.016, 0.024, 0.032,
            0.028, 0.024, 0.059, 0.063, 0.157, 0.251, 0.585, 0.240, 0.756, 0.043,
            0.439, 0.631, 0.756, 0.698, 0.729, 0.024, 1.000, 0.400, 0.094, 0.279,
            0.004, 0.028, 0.047, 0.008, 0.020, 0.032, 0.000, 0.012, 0.004, 0.004,
            0.004
        ],
        'motifs': [(16, 28, 'GAGA*TTCACCT motif', 24.5)],
    },
    {
        'sample_id': 606,
        'confidence': 0.9465,
        'seq': 'GAGGAGGCGGAGACCGGGAGAGGGCGGCGACCGGGAGAGGG',
        'attr': [
            0.047, 0.055, 0.036, 0.059, 0.016, 0.024, 0.036, 0.024, 0.083, 0.036,
            0.020, 0.032, 0.008, 0.020, 0.032, 0.036, 0.283, 0.337, 1.000, 0.055,
            0.514, 0.577, 0.197, 0.012, 0.392, 0.137, 0.161, 0.224, 0.125, 0.008,
            0.000, 0.020, 0.024, 0.020, 0.032, 0.012, 0.020, 0.012, 0.016, 0.012,
            0.016
        ],
        'motifs': [(16, 23, 'GGAGA*GG (GAGG) motif', 24.4)],
    },
    {
        'sample_id': 1477,
        'confidence': 0.9464,
        'seq': 'AATTACGATTGATGTTGTTTACTTACTATATACAATCGGAA',
        'attr': [
            0.000, 0.004, 0.028, 0.016, 0.020, 0.036, 0.012, 0.016, 0.024, 0.020,
            0.032, 0.047, 0.047, 0.047, 0.122, 0.040, 0.008, 0.075, 0.106, 0.181,
            0.416, 0.291, 0.498, 0.205, 1.000, 0.137, 0.071, 0.337, 0.298, 0.349,
            0.028, 0.020, 0.051, 0.024, 0.016, 0.024, 0.028, 0.020, 0.012, 0.012,
            0.012
        ],
        'motifs': [(20, 26, 'A*CTTAC core', 23.6)],
    },
]


def plot_motif_saliency():
    out_dir = '/Volumes/Files/Research/DNAPred_Luo/paper_new'
    fig, axes = plt.subplots(len(SAMPLES), 1, figsize=(15.0, 12.2))
    fig.suptitle(
        'TF-DualMethyl Nucleotide-Level Motif Saliency Heatmap (Layer Integrated Gradients, 6mA F. vesca)',
        fontsize=16.5, fontweight='bold', y=0.985
    )

    cmap = cm.Blues
    for i, res in enumerate(SAMPLES):
        ax = axes[i]
        attr = np.array(res['attr'], dtype=float)
        seq = res['seq']

        # Draw 41 cells
        for j in range(41):
            val = attr[j]
            face_col = cmap(val)
            rect = Rectangle((j, 0), 1, 1, facecolor=face_col, edgecolor='#D0D5DD', linewidth=0.95, zorder=1)
            ax.add_patch(rect)
            txt_col = 'white' if val > 0.56 else '#111827'
            ax.text(j + 0.5, 0.5, seq[j], ha='center', va='center',
                    color=txt_col, fontsize=12.5, fontweight='bold', zorder=3)

        # Highlight high-saliency core 6mA motifs with orange bounding boxes
        for idx_m, (start, end, m_label, lbl_x) in enumerate(res['motifs']):
            m_rect = Rectangle(
                (start + 0.04, 0.03), (end - start) - 0.08, 0.94,
                fill=False, edgecolor='#EA580C', linewidth=2.6, zorder=5
            )
            ax.add_patch(m_rect)
            ax.text(
                lbl_x, 1.07, m_label,
                ha='center', va='bottom', color='#C2410C',
                fontsize=10.8, fontweight='bold', zorder=6
            )

        # Center methylated adenine A* at position 21 (index 20 -> x = 20.5)
        ax.axvline(x=20.5, color='#DC2626', linestyle='--', linewidth=1.8, zorder=6)
        ax.text(20.5, 1.07, 'A*', ha='center', va='bottom', color='#DC2626',
                fontsize=12.0, fontweight='bold', zorder=7)

        ax.set_xlim(0, 41)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(0.5, 41.5, 5))
        ax.set_xticklabels(np.arange(1, 42, 5), fontsize=11)
        ax.tick_params(axis='x', length=4.0, width=1.0, colors='#1F2937', direction='out', pad=2.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        motif_names = ', '.join([m[2] for m in res['motifs']])
        ax.set_title(
            f"Sample #{res['sample_id']}  |  Confidence: {res['confidence']:.4f}  |  High-Saliency Core Motif: {motif_names}",
            fontsize=13.0, fontweight='bold', pad=25, color='#111827'
        )

    plt.subplots_adjust(left=0.03, right=0.97, top=0.915, bottom=0.115, hspace=1.08)

    # Shared horizontal colorbar at the bottom left
    cbar_ax = fig.add_axes([0.06, 0.048, 0.31, 0.017])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Normalized LIG Attribution Saliency Score', fontsize=12.0, fontweight='bold', labelpad=4)
    cbar.ax.tick_params(labelsize=10.5)

    # Shared legend at the bottom right (well separated from colorbar)
    legend_elements = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor='#EA580C', lw=2.6, label='High-saliency core 6mA motif region'),
        Line2D([0], [0], color='#DC2626', linestyle='--', lw=1.8, label='Central methylated adenine (A*, pos 21)')
    ]
    fig.legend(
        handles=legend_elements, loc='lower center', ncol=2,
        bbox_to_anchor=(0.68, 0.026), fontsize=12.0, frameon=True, edgecolor='#D0D5DD'
    )

    pdf_path = os.path.join(out_dir, 'fig_motif_saliency_6mA_F.vesca.pdf')
    png_path = os.path.join(out_dir, 'fig_motif_saliency_6mA_F.vesca.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', pdf_path, png_path)


if __name__ == '__main__':
    plot_motif_saliency()
