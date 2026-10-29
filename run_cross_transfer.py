import os
import gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import DNADataset
from model import get_model_by_name, Config
from run_all import resolve_dirs, prepare_all_tsv


def run_cross_species_eval():
    script_dir, raw_data_root, base_result_dir = resolve_dirs()
    base_data_dir = prepare_all_tsv(raw_data_root, categories=("4mC", "6mA"))
    model_name = "Ultimate_Fusion"
    save_root = os.path.join(base_result_dir, "Cross_Transfer")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = Config()

    print(f"Starting Cross-Species Transferability Evaluation...")
    print(f"   Model: TF-DualMethyl ({model_name})")
    print(f"   Device: {device}")
    print(f"   Save directory: {save_root}")

    categories = ["4mC", "6mA"]

    for category in categories:
        cat_data_path = os.path.join(base_data_dir, category)
        cat_model_path = os.path.join(base_result_dir, model_name, category)

        if not os.path.exists(cat_data_path) or not os.path.exists(cat_model_path):
            print(f"Skipping category {category} (missing data or models)")
            continue

        trained_species = sorted([
            s for s in os.listdir(cat_model_path)
            if os.path.isdir(os.path.join(cat_model_path, s)) and not s.startswith(".")
        ])
        target_species = sorted([
            s for s in os.listdir(cat_data_path)
            if os.path.isdir(os.path.join(cat_data_path, s))
            and not s.startswith(".")
            and os.path.exists(os.path.join(cat_data_path, s, "test.tsv"))
        ])

        print(f"\nProcessing Category: {category}")
        print(f"   Trained Species Found: {len(trained_species)}")
        print(f"   Target Species Found: {len(target_species)}")

        for s_train in trained_species:
            model_dir = os.path.join(cat_model_path, s_train)
            models = []
            for i in range(1, 6):
                path = os.path.join(model_dir, f'model_fold_{i}.pth')
                if not os.path.exists(path):
                    path = os.path.join(model_dir, f'fold_{i}.pth')

                if os.path.exists(path):
                    m = get_model_by_name(model_name, config).to(device)
                    m.load_state_dict(torch.load(path, map_location=device))
                    m.eval()
                    models.append(m)

            if not models:
                print(f"   No weights found for {s_train}, skipping.")
                continue

            print(f"   Trained on: {s_train} (ensemble of {len(models)} folds)")

            for s_test in target_species:
                clean_train = s_train.replace(f"{category}_", "")
                clean_test = s_test.replace(f"{category}_", "")

                output_dir = os.path.join(save_root, category, f"Train_{clean_train}_Test_{clean_test}")
                prob_file = os.path.join(output_dir, "probability.txt")

                if os.path.exists(prob_file) and os.path.getsize(prob_file) > 100:
                    continue

                os.makedirs(output_dir, exist_ok=True)

                test_file = os.path.join(cat_data_path, s_test, "test.tsv")
                try:
                    test_dataset = DNADataset(tsv_file=test_file, max_len=41)
                    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
                except Exception as e:
                    print(f"      Error loading {s_test}: {e}")
                    continue

                with open(prob_file, 'w') as f_out:
                    f_out.write("True_Label\tProbability\tPrediction\n")

                    with torch.no_grad():
                        for seqs, labels in test_loader:
                            seqs = seqs.to(device)

                            batch_probs = []
                            for m in models:
                                logits = m(seqs)
                                p = torch.softmax(logits, dim=1)[:, 1]
                                batch_probs.append(p.cpu().numpy())
                                del logits, p

                            avg_p = np.mean(batch_probs, axis=0)

                            for t, p in zip(labels.numpy(), avg_p):
                                pred = 1 if p > 0.5 else 0
                                f_out.write(f"{t}\t{p:.6f}\t{pred}\n")

                            del seqs, batch_probs, avg_p
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()

                del test_dataset, test_loader
                gc.collect()
                print(f"      Tested on {clean_test} -> Saved to {prob_file}")

            del models
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print("\nCross-Species Evaluation Completed!")


if __name__ == "__main__":
    run_cross_species_eval()

