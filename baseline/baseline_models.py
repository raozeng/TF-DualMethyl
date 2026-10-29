import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Baseline: TextCNN (Local Motif Extractor)
# Represents architectures like iDNA-Methyl, DeepCpG
# ==========================================
class Baseline_TextCNN(nn.Module):
    def __init__(self, config):
        super(Baseline_TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim_embedding, padding_idx=0)
        
        # Multiple kernel sizes for k-mers
        self.conv1 = nn.Conv1d(config.dim_embedding, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.dim_embedding, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(config.dim_embedding, 128, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(128 * 3)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_class)
        )

    def forward(self, seqs):
        x = self.embedding(seqs)  # [B, L, C]
        x = x.permute(0, 2, 1)    # [B, C, L]
        
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))
        
        # Global Max Pooling Over Time
        c1 = torch.max(c1, dim=2)[0]
        c2 = torch.max(c2, dim=2)[0]
        c3 = torch.max(c3, dim=2)[0]
        
        out = torch.cat([c1, c2, c3], dim=1) # [B, 128*3]
        out = self.bn(out)
        return self.classifier(out)


# ==========================================
# 2. Baseline: BiLSTM (Long Range Sequential)
# Represents architectures like MethyRNN
# ==========================================
class Baseline_BiLSTM(nn.Module):
    def __init__(self, config):
        super(Baseline_BiLSTM, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim_embedding, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=config.dim_embedding,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_class)
        )

    def forward(self, seqs):
        x = self.embedding(seqs)
        out, _ = self.lstm(x) # [B, L, hidden_dim * 2]
        
        # Global Average Pooling over time (or taking the last step)
        # Using simple mean pooling here ignoring padding for baseline purity
        out = torch.mean(out, dim=1) 
        
        return self.classifier(out)


# ==========================================
# 3. Baseline: CRNN (CNN + BiLSTM Hybrid)
# Represents architectures like DanQ, DeepBiLSTM-CNN
# ==========================================
class Baseline_CRNN(nn.Module):
    def __init__(self, config):
        super(Baseline_CRNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim_embedding, padding_idx=0)
        self.conv = nn.Conv1d(config.dim_embedding, 128, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_class)
        )

    def forward(self, seqs):
        x = self.embedding(seqs).permute(0, 2, 1) # [B, C, L]
        x = F.relu(self.bn(self.conv(x)))
        x = x.permute(0, 2, 1) # [B, L, C]
        
        out, _ = self.lstm(x)
        out = torch.mean(out, dim=1) # Global pooling
        
        return self.classifier(out)


# ==========================================
# 4. Baseline: Transformer (Attention Only)
# Represents architectures like small DNABERT
# ==========================================
class Baseline_Transformer(nn.Module):
    def __init__(self, config):
        super(Baseline_Transformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim_embedding, padding_idx=0)
        
        # Simple positional Encoding for baseline
        self.pos_emb = nn.Parameter(torch.zeros(1, 1000, config.dim_embedding))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_embedding,
            nhead=config.num_heads,
            dim_feedforward=config.dim_embedding * 4,
            dropout=0.3,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(config.dim_embedding, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_class)
        )

    def forward(self, seqs):
        seq_len = seqs.size(1)
        x = self.embedding(seqs) + self.pos_emb[:, :seq_len, :]
        
        padding_mask = (seqs == 0) # [B, L]
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Pool non-padded token outputs
        mask = (~padding_mask).unsqueeze(-1).float()
        out = torch.sum(out * mask, dim=1) / torch.sum(mask, dim=1).clamp(min=1e-9)
        
        return self.classifier(out)

