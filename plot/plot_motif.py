import argparse
import os
import sys
import re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_model_by_name, Config
from dataset import DNADataset
from run_all import resolve_dirs, prepare_all_tsv

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_args():
    script_dir, raw_data_root, base_result_dir = resolve_dirs()
    project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == "model_plus" else script_dir
    tsv_root = prepare_all_tsv(raw_data_root, categories=("4mC", "6mA"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Ultimate_Fusion')
    parser.add_argument('--data_path', type=str, default=os.path.join(tsv_root, '6mA', '6mA_F.vesca', 'test.tsv'))
    parser.add_argument('--model_path', type=str, default=os.path.join(base_result_dir, 'Ultimate_Fusion', '6mA', '6mA_F.vesca', 'model_fold_1.pth'))
    parser.add_argument('--category', type=str, default='6mA')
    parser.add_argument('--species', type=str, default='F.vesca')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'page', 'page_motif'))
    parser.add_argument('--paper_fig_dir', type=str, default=os.path.join(project_root, 'paper_new', 'figures'))
    parser.add_argument('--n_steps', type=int, default=50, help='IG steps')
    parser.add_argument('--top_n', type=int, default=6, help='Number of top confidence sequences to visualize')
    return parser.parse_args()


def find_drach_motifs(seq_str):
    pattern = re.compile(r'(?=([AGT][AG]AC[ACT]))')
    matches = []
    for match in pattern.finditer(seq_str):
        start = match.start()
        end = start + 5
        matches.append((start, end))
    return matches


def main():
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        print("captum is not installed; skipping motif saliency IG plot.")
        return

    args = get_args()
    set_seed()

    test_file = args.data_path
    model_path = args.model_path
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.paper_fig_dir, exist_ok=True)

    if not os.path.exists(test_file) or not os.path.exists(model_path):
        print(f"Data or model not found.\nChecked data: {test_file}\nChecked model: {model_path}")
        return

    dataset = DNADataset(test_file)
    vocab = dataset.vocab
    inv_vocab = {v: k for k, v in vocab.items()}

    config = Config()
    model = get_model_by_name(args.model_name, config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if hasattr(model, 'lstm') and model.lstm is not None:
        model.lstm.flatten_parameters()
        torch.backends.cudnn.enabled = False

    lig = LayerIntegratedGradients(model, model.embedding)
    pos_indices = [i for i in range(len(dataset)) if dataset.data_frame.iloc[i]['label'] == 1]

    confidences = []
    with torch.no_grad():
        for idx in pos_indices:
            seq, label = dataset[idx]
            seq_t = seq.unsqueeze(0).to(device)
            if (seq_t == 0).all():
                confidences.append(-1)
                continue
            logits = model(seq_t)
            probs = torch.softmax(logits, dim=1)
            confidences.append(probs[0, 1].item())

    valid_samples = [(idx, conf) for idx, conf in zip(pos_indices, confidences) if 0.8 <= conf <= 0.95]
    sorted_idx_conf = sorted(valid_samples, key=lambda x: x[1], reverse=True)
    top_samples = sorted_idx_conf[:args.top_n]

    results = []
    for idx, conf in top_samples:
        seq, label = dataset[idx]
        seq_t = seq.unsqueeze(0).to(device)
        attr, delta = lig.attribute(inputs=seq_t, target=1, return_convergence_delta=True, n_steps=args.n_steps)
        attr = np.abs(attr.sum(dim=-1).squeeze(0).cpu().detach().numpy())
        if attr.max() != attr.min():
            attr = (attr - attr.min()) / (attr.max() - attr.min())
        seq_chars = [inv_vocab.get(c, 'N') for c in seq.cpu().numpy()]
        seq_str = "".join(seq_chars)
        results.append({
            'sample_id': idx,
            'confidence': conf,
            'seq_chars': seq_chars,
            'seq_str': seq_str,
            'attr': attr,
            'motifs': find_drach_motifs(seq_str)
        })

    fig, axes = plt.subplots(len(results), 1, figsize=(16, 2.5 * len(results)))
    fig.suptitle(f'TF-DualMethyl Motif Saliency Heatmap — {args.category} ({args.species})', fontsize=18, fontweight='bold', y=0.96)
    if len(results) == 1:
        axes = [axes]

    for i, res in enumerate(results):
        ax = axes[i]
        attr = res['attr'].reshape(1, -1)
        sns.heatmap(attr, cmap='Blues', cbar=False, ax=ax, xticklabels=False, yticklabels=False, linewidths=1.0, linecolor='lightgray')
        for j, char in enumerate(res['seq_chars']):
            color = 'white' if attr[0, j] > 0.6 else 'black'
            ax.text(j + 0.5, 0.5, char, ha='center', va='center', color=color, fontsize=11, fontweight='bold')
        for start, end in res['motifs']:
            rect = Rectangle((start, 0), end - start, 1, fill=False, edgecolor='darkorange', lw=2.5)
            ax.add_patch(rect)
            ax.text((start + end) / 2, 1.05, 'DRACH', ha='center', va='top', color='darkorange', fontsize=9, fontweight='bold')
        ax.axvline(x=20.5, color='red', linestyle='--', lw=1.5)
        ax.text(20.5, -0.05, 'A*', ha='center', va='bottom', color='red', fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(0.5, 41.5, 5))
        ax.set_xticklabels(np.arange(1, 42, 5), fontsize=9)
        ax.tick_params(axis='x', length=4, width=1, colors='black', direction='out')
        ax.set_title(
            f"Sample #{res['sample_id']} | Confidence: {res['confidence']:.4f} | DRACH motifs found: {len(res['motifs'])} | Seq: {res['seq_str'][:10]}...",
            fontsize=11, fontweight='bold', pad=20
        )
        cbaxes = inset_axes(ax, width="20%", height="15%", loc='lower center', bbox_to_anchor=(0, -0.55, 1, 1), bbox_transform=ax.transAxes)
        sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbaxes, orientation='horizontal')
        cbar.set_label('Normalized Saliency Score', fontsize=9, labelpad=2)
        cbar.ax.tick_params(labelsize=8)

    legend_elements = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor='darkorange', lw=2.5, label='DRACH motif region'),
        Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='Center site (A*)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=12, frameon=True)
    plt.subplots_adjust(hspace=1.2, bottom=0.08, top=0.9)

    for out_p in [
        os.path.join(args.output_dir, f"motif_saliency_heatmap_{args.category}_{args.species}.pdf"),
        os.path.join(args.output_dir, f"motif_saliency_heatmap_{args.category}_{args.species}.png"),
        os.path.join(args.paper_fig_dir, f"fig_motif_saliency_{args.category}_{args.species}.pdf"),
    ]:
        fig.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Motif saliency plot saved.")


if __name__ == "__main__":
    main()

