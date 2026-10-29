import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import DNADataset
from model import get_model_by_name, Config
from run_all import resolve_dirs, prepare_all_tsv


def run_probability_for_all():
    script_dir, raw_data_root, save_dir = resolve_dirs()
    base_data_dir = prepare_all_tsv(raw_data_root, categories=("4mC", "6mA"))

    models_to_run = [
        "Baseline_TextCNN", "Baseline_BiLSTM", "Baseline_CRNN", "Baseline_Transformer",
        "Ultimate_Fusion",
        "Ablation_No_BiLSTM", "Ablation_No_Transformer", "Ablation_No_Freq", "Ablation_No_Time",
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = Config()

    datasets = []
    if os.path.exists(base_data_dir):
        for category in sorted(os.listdir(base_data_dir)):
            category_path = os.path.join(base_data_dir, category)
            if os.path.isdir(category_path) and not category.startswith("."):
                for species in sorted(os.listdir(category_path)):
                    species_path = os.path.join(category_path, species)
                    if os.path.isdir(species_path) and not species.startswith("."):
                        if os.path.exists(os.path.join(species_path, "test.tsv")):
                            datasets.append({
                                "data_dir": species_path,
                                "category": category,
                                "species": species
                            })

    for dataset in datasets:
        test_file = os.path.join(dataset["data_dir"], "test.tsv")
        try:
            test_dataset = DNADataset(tsv_file=test_file, max_len=41)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        except Exception as e:
            print(f"Error loading {test_file}: {e}")
            continue

        for model_name in models_to_run:
            model_dir = os.path.join(save_dir, model_name, dataset["category"], dataset["species"])
            if not os.path.exists(model_dir):
                continue

            out_file = os.path.join(model_dir, "probability.txt")
            if os.path.exists(out_file) and os.path.getsize(out_file) > 100:
                # Already generated
                continue

            models = []
            for i in range(1, 6):
                path = os.path.join(model_dir, f'model_fold_{i}.pth')
                if os.path.exists(path):
                    model = get_model_by_name(model_name, config).to(device)
                    model.load_state_dict(torch.load(path, map_location=device))
                    model.eval()
                    models.append(model)

            if not models:
                for i in range(1, 6):
                    path = os.path.join(model_dir, f'fold_{i}.pth')
                    if os.path.exists(path):
                        model = get_model_by_name(model_name, config).to(device)
                        model.load_state_dict(torch.load(path, map_location=device))
                        model.eval()
                        models.append(model)

            if not models:
                print(f"Skipping {model_name} on {dataset['species']}, no fold weights found.")
                continue

            test_targets = []
            all_probs = []

            print(f"Extracting probabilities for {model_name} on {dataset['species']}...")
            with torch.no_grad():
                for seqs, labels in test_loader:
                    seqs, labels = seqs.to(device), labels.to(device)
                    test_targets.extend(labels.cpu().numpy())

                    batch_probs = []
                    for model in models:
                        logits = model(seqs)
                        probs = torch.softmax(logits, dim=1)[:, 1]
                        batch_probs.append(probs.cpu().numpy())

                    avg_batch_prob = np.mean(batch_probs, axis=0)
                    all_probs.extend(avg_batch_prob)

            with open(out_file, 'w') as f:
                f.write("True_Label\tProbability\tPrediction\n")
                for t, p in zip(test_targets, all_probs):
                    pred = 1 if p > 0.5 else 0
                    f.write(f"{t}\t{p:.6f}\t{pred}\n")
            print(f"Saved {out_file}")


if __name__ == '__main__':
    run_probability_for_all()

