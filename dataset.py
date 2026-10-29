import os
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


def ensure_tsv_from_pos_neg(path):
    """
    If train.tsv or test.tsv does not exist, automatically construct it from
    train_pos.txt + train_neg.txt or test_pos.txt + test_neg.txt in the same folder.
    Accepts either a directory path or a specific .tsv file path.
    """
    if os.path.isdir(path) or not path.endswith(".tsv"):
        ensure_tsv_from_pos_neg(os.path.join(path, "train.tsv"))
        ensure_tsv_from_pos_neg(os.path.join(path, "test.tsv"))
        return path

    tsv_file = path
    if os.path.exists(tsv_file):
        return tsv_file

    folder = os.path.dirname(tsv_file)
    basename = os.path.basename(tsv_file)
    if basename == "train.tsv":
        pos_file = os.path.join(folder, "train_pos.txt")
        neg_file = os.path.join(folder, "train_neg.txt")
    elif basename == "test.tsv":
        pos_file = os.path.join(folder, "test_pos.txt")
        neg_file = os.path.join(folder, "test_neg.txt")
    else:
        return tsv_file

    if os.path.exists(pos_file) and os.path.exists(neg_file):
        rows = []
        idx = 0
        with open(pos_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                seq = line.strip()
                if seq and not seq.startswith(">"):
                    rows.append((idx, 1, seq))
                    idx += 1
        with open(neg_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                seq = line.strip()
                if seq and not seq.startswith(">"):
                    rows.append((idx, 0, seq))
                    idx += 1
        os.makedirs(folder, exist_ok=True)
        df = pd.DataFrame(rows, columns=["index", "label", "text"])
        df.to_csv(tsv_file, sep="\t", index=False)
        print(f"[Auto-Generated TSV] Created {tsv_file} ({len(df)} sequences)")
    return tsv_file


class DNADataset(Dataset):
    def __init__(self, tsv_file, max_len=41):
        """
        Args:
            tsv_file (string): Path to the tsv file with indices, label, and text columns.
            max_len (int): Maximum length of DNA sequence.
        """
        tsv_file = ensure_tsv_from_pos_neg(tsv_file)
        self.data_frame = pd.read_csv(tsv_file, sep='\t')
        self.max_len = max_len

        # Setup vocabulary (unigram mapping)
        self.vocab = {'PAD': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}

    def __len__(self):
        return len(self.data_frame)

    def _encode_sequence(self, seq):
        # Convert Characters to Integers based on the vocab
        indices = [self.vocab.get(char.upper(), self.vocab['N']) for char in seq]

        # Padding or Truncating
        if len(indices) >= self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.vocab['PAD']] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = str(self.data_frame.iloc[idx]['text'])
        label = int(self.data_frame.iloc[idx]['label'])

        encoded_seq = self._encode_sequence(seq)

        return encoded_seq, torch.tensor(label, dtype=torch.long)
