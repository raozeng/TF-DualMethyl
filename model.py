import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Config:
    def __init__(
        self,
        vocab_size=6,
        dim_embedding=128,
        hidden_dim=128,
        num_class=2,
        lstm_num_layers=2,
        num_heads=8,
        transformer_layers=2,
    ):
        self.vocab_size = vocab_size         # 'PAD', 'A', 'C', 'G', 'T', 'N' (0-5)
        self.dim_embedding = dim_embedding   # d = 128
        self.hidden_dim = hidden_dim         # d_h = 128 (BiLSTM output = 256)
        self.num_class = num_class
        self.lstm_num_layers = lstm_num_layers
        self.num_heads = num_heads           # M = 8 attention heads
        self.transformer_layers = transformer_layers


# ==========================================
# 1. Attention Pooling Layer
# ==========================================
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.last_attn_weights = None

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        attn_weights = self.attention(x)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn_weights, dim=1)
        self.last_attn_weights = attn_weights.detach()
        pooled_output = torch.sum(x * attn_weights, dim=1)
        return pooled_output


# ==========================================
# 2. Positional Encoding
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


# ==========================================
# 3. Genuine Differentiable Time-Frequency Front-End Modules
# ==========================================
class LearnableWaveletTransform(nn.Module):
    """
    Multi-Scale Discrete Wavelet Transform (scales s in {2, 4, 8}) with Mathematical
    Orthogonal/Wavelet-Packet Basis Initialization & Learnable Soft-Thresholding Shrinkage:
      - Scale s=2: Initialized with Haar low-pass [1/sqrt(2), 1/sqrt(2)] and
                   high-pass [1/sqrt(2), -1/sqrt(2)] difference filters.
      - Scale s=4: Initialized with Daubechies-2 (Db2) orthogonal scaling and wavelet filters.
      - Scale s=8: Initialized with sinusoidal-modulated Gaussian wavelet packets
                   cos(2*pi*t/8)*exp(-t^2/4) and sin(2*pi*t/8)*exp(-t^2/4).
      - Applies learnable soft-thresholding shrinkage T_tau(cD) = sign(cD) * ReLU(|cD| - tau)
        on high-frequency detail sub-bands to suppress non-stationary sequence noise.
    """
    def __init__(self, features):
        super(LearnableWaveletTransform, self).__init__()
        self.features = features

        # Scale s=2 (Haar low-pass & high-pass QMF)
        self.haar_low = nn.Conv1d(features, features, kernel_size=2, padding=0, groups=features, bias=False)
        self.haar_high = nn.Conv1d(features, features, kernel_size=2, padding=0, groups=features, bias=False)

        # Scale s=4 (Daubechies-2 scaling & wavelet QMF)
        self.db2_low = nn.Conv1d(features, features, kernel_size=4, padding=0, groups=features, bias=False)
        self.db2_high = nn.Conv1d(features, features, kernel_size=4, padding=0, groups=features, bias=False)

        # Scale s=8 (Gabor/Morlet Gaussian wavelet packet even & odd phase)
        self.pkt8_low = nn.Conv1d(features, features, kernel_size=8, padding=0, groups=features, bias=False)
        self.pkt8_high = nn.Conv1d(features, features, kernel_size=8, padding=0, groups=features, bias=False)

        self._init_wavelet_bases()

        # Learnable soft-thresholding shrinkage parameters for high-frequency detail bands
        self.raw_tau_s2 = nn.Parameter(torch.full((1, features, 1), -2.0))
        self.raw_tau_s4 = nn.Parameter(torch.full((1, features, 1), -2.0))
        self.raw_tau_s8 = nn.Parameter(torch.full((1, features, 1), -2.0))

        # 1x1 projection across the 6 multi-resolution wavelet sub-bands (3 approx + 3 detail)
        self.subband_proj = nn.Conv1d(features * 6, features, kernel_size=1)
        self.bn = nn.BatchNorm1d(features)
        self.act = nn.GELU()

    def _init_wavelet_bases(self):
        with torch.no_grad():
            # 1. Haar (s=2)
            s2 = 1.0 / math.sqrt(2.0)
            h_low = torch.tensor([s2, s2], dtype=torch.float32).view(1, 1, 2).repeat(self.features, 1, 1)
            h_high = torch.tensor([s2, -s2], dtype=torch.float32).view(1, 1, 2).repeat(self.features, 1, 1)
            self.haar_low.weight.copy_(h_low)
            self.haar_high.weight.copy_(h_high)

            # 2. Daubechies-2 (s=4)
            denom = 4.0 * math.sqrt(2.0)
            rt3 = math.sqrt(3.0)
            d_low = torch.tensor(
                [(1.0 + rt3) / denom, (3.0 + rt3) / denom, (3.0 - rt3) / denom, (1.0 - rt3) / denom],
                dtype=torch.float32
            ).view(1, 1, 4).repeat(self.features, 1, 1)
            d_high = torch.tensor(
                [(1.0 - rt3) / denom, -(3.0 - rt3) / denom, (3.0 + rt3) / denom, -(1.0 + rt3) / denom],
                dtype=torch.float32
            ).view(1, 1, 4).repeat(self.features, 1, 1)
            self.db2_low.weight.copy_(d_low)
            self.db2_high.weight.copy_(d_high)

            # 3. Sinusoidal-modulated Gaussian Wavelet Packet (s=8)
            t = torch.linspace(-3.5, 3.5, steps=8, dtype=torch.float32)
            env = torch.exp(-(t ** 2) / 4.0)
            p_low = (torch.cos(2.0 * math.pi * t / 8.0) * env)
            p_high = (torch.sin(2.0 * math.pi * t / 8.0) * env)
            p_low = (p_low / (p_low.norm() + 1e-8)).view(1, 1, 8).repeat(self.features, 1, 1)
            p_high = (p_high / (p_high.norm() + 1e-8)).view(1, 1, 8).repeat(self.features, 1, 1)
            self.pkt8_low.weight.copy_(p_low)
            self.pkt8_high.weight.copy_(p_high)

    @staticmethod
    def _soft_threshold(detail_coeff, raw_tau):
        tau = F.softplus(raw_tau) * 0.1
        return torch.sign(detail_coeff) * F.relu(torch.abs(detail_coeff) - tau)

    def forward(self, x):
        # x: [B, C, L], preserve exact sequence length L via symmetric/replicate padding
        x_s2 = F.pad(x, (0, 1), mode='replicate')
        cA2 = self.haar_low(x_s2)
        cD2 = self._soft_threshold(self.haar_high(x_s2), self.raw_tau_s2)

        x_s4 = F.pad(x, (1, 2), mode='replicate')
        cA4 = self.db2_low(x_s4)
        cD4 = self._soft_threshold(self.db2_high(x_s4), self.raw_tau_s4)

        x_s8 = F.pad(x, (3, 4), mode='replicate')
        cA8 = self.pkt8_low(x_s8)
        cD8 = self._soft_threshold(self.pkt8_high(x_s8), self.raw_tau_s8)

        wavelet_cat = torch.cat([cA2, cD2, cA4, cD4, cA8, cD8], dim=1)  # [B, 6*C, L]
        return self.act(self.bn(self.subband_proj(wavelet_cat)))


class LearnableVMD(nn.Module):
    """
    Differentiable Variational Mode Decomposition (Learnable VMD) in the Frequency Domain:
      1. Applies 1D Real Fast Fourier Transform: X_hat(f) = rFFT(x) in C^{B x d x L_f}.
      2. Decomposes the spectrum into K=4 intrinsic mode sub-bands using the analytical
         VMD quadratic Wiener filter centered at learnable frequencies omega_k in (0, 0.5)
         with bandwidth penalty alpha_k > 0, modulated by learnable spectral weights W_mode:
             H_k(f) = W_mode[k, f] / (1 + 2 * alpha_k * (f - omega_k)^2)
      3. Reconstructs each mode M_k(t) = irFFT(X_hat(f) * H_k(f), n=L) in the sequence domain.
      4. Fuses all K modes via 1x1 projection + BatchNorm + GELU + residual connection.
    """
    def __init__(self, channels, num_modes=4, max_freq_bins=32):
        super(LearnableVMD, self).__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.max_freq_bins = max_freq_bins

        # Learnable mode center frequencies omega_k in (0, 0.5)
        init_omegas = torch.linspace(0.15, 0.85, steps=num_modes)
        init_logits = torch.log(init_omegas / (1.0 - init_omegas))
        self.raw_omega = nn.Parameter(init_logits.view(1, num_modes, 1, 1).repeat(1, 1, channels, 1))

        # Learnable VMD quadratic bandwidth penalty alpha_k > 0
        self.raw_alpha = nn.Parameter(torch.full((1, num_modes, channels, 1), 2.5))

        # Learnable spectral filter bank W_mode in R^{1 x K x C x F_max}
        self.spectral_weights = nn.Parameter(torch.ones(1, num_modes, channels, max_freq_bins))

        # Depthwise spatial phase refinement per reconstructed IMF mode
        self.modes = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d, groups=channels)
            for d in range(1, num_modes + 1)
        ])
        self.reconstruct = nn.Conv1d(channels * num_modes, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        orig_dtype = x.dtype

        x_fp32 = x.float()
        x_fft = torch.fft.rfft(x_fp32, dim=-1)  # [B, C, L_f], where L_f = L // 2 + 1 (21 for L=41)
        L_f = x_fft.size(-1)

        freq_grid = torch.linspace(0.0, 0.5, steps=L_f, device=x.device, dtype=torch.float32).view(1, 1, 1, L_f)
        omega_k = 0.5 * torch.sigmoid(self.raw_omega)  # [1, K, C, 1]
        alpha_k = F.softplus(self.raw_alpha) + 4.0     # [1, K, C, 1]

        if L_f <= self.max_freq_bins:
            w_mode = self.spectral_weights[..., :L_f]
        else:
            w_mode = F.interpolate(
                self.spectral_weights.view(self.num_modes, C, self.max_freq_bins),
                size=L_f,
                mode='linear',
                align_corners=False
            ).unsqueeze(0)

        # Analytical VMD Wiener transfer function modulated by learnable spectral weights
        transfer_fn = w_mode / (1.0 + 2.0 * alpha_k * (freq_grid - omega_k).pow(2))  # [1, K, C, L_f]

        modes_fft = x_fft.unsqueeze(1) * transfer_fn                                 # [B, K, C, L_f]
        modes_time = torch.fft.irfft(modes_fft, n=L, dim=-1).to(orig_dtype)          # [B, K, C, L]

        mode_outs = []
        for k, mode_conv in enumerate(self.modes):
            m_k = modes_time[:, k, :, :]
            mode_outs.append(m_k + mode_conv(m_k))

        out = torch.cat(mode_outs, dim=1)  # [B, K*C, L]
        return self.act(self.bn(self.reconstruct(out))) + x


class MultiScaleCNN(nn.Module):
    def __init__(self, channels):
        super(MultiScaleCNN, self).__init__()
        self.branch1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.branch3 = nn.Conv1d(channels, channels, kernel_size=7, padding=3)
        self.fusion = nn.Conv1d(channels * 3, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.act(self.bn(self.fusion(out)))


class CausalConv1dBlock(nn.Module):
    """
    Single causal dilated convolution stage with Chomp1d truncation of trailing
    padding positions p = (k - 1) * d to preserve exact sequence length L.
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super(CausalConv1dBlock, self).__init__()
        self.chomp_size = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.chomp_size,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        if self.chomp_size > 0:
            out = out[:, :, :-self.chomp_size]
        return self.act(self.bn(out) + x)


class TCNBlock(nn.Module):
    """
    3-Stage Causal Dilated Temporal Convolutional Network (d_l in {1, 2, 4})
    with Chomp1d causal padding truncation and residual connections.
    """
    def __init__(self, channels, dilations=(1, 2, 4)):
        super(TCNBlock, self).__init__()
        self.stages = nn.ModuleList([
            CausalConv1dBlock(channels, kernel_size=3, dilation=d)
            for d in dilations
        ])

    def forward(self, x):
        out = x
        for stage in self.stages:
            out = stage(out)
        return out


class AdaptiveGatedFusion(nn.Module):
    """
    Learnable multi-branch gating unit that dynamically weights the base sequence
    representation and extracted time-frequency branches at each nucleotide position.
    """
    def __init__(self, channels, num_branches=3):
        super(AdaptiveGatedFusion, self).__init__()
        self.num_branches = num_branches
        self.gate_net = nn.Sequential(
            nn.Conv1d(channels * num_branches, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels * num_branches, kernel_size=1),
        )

    def forward(self, branches):
        B, C, L = branches[0].shape
        concat = torch.cat(branches, dim=1)
        gates = self.gate_net(concat).view(B, self.num_branches, C, L)
        weights = F.softmax(gates, dim=1) * self.num_branches
        stacked = torch.stack(branches, dim=1)
        return torch.sum(stacked * weights, dim=1)


class SEChannelRecalibration(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention block for recalibrating
    dual-path (BiLSTM + Transformer) feature channels.
    """
    def __init__(self, d_model, reduction=4):
        super(SEChannelRecalibration, self).__init__()
        mid_dim = max(d_model // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(d_model, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, d_model),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = x.mean(dim=1)
        excitation = self.fc(squeeze).unsqueeze(1)
        return x * excitation


# ==========================================
# TF-DualMethyl (Ultimate_Fusion) Network
# ==========================================
class Base_Ultimate_Fusion(nn.Module):
    def __init__(self, config=None):
        super(Base_Ultimate_Fusion, self).__init__()
        if config is None:
            config = Config()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.dim_embedding, padding_idx=0)
        self.pos_encoder = PositionalEncoding(config.dim_embedding)
        self.dropout_emb = nn.Dropout(0.2)

        self.wavelet = LearnableWaveletTransform(features=config.dim_embedding)
        self.vmd = LearnableVMD(channels=config.dim_embedding, num_modes=4)
        self.mscnn = MultiScaleCNN(channels=config.dim_embedding)
        self.tcn = TCNBlock(channels=config.dim_embedding, dilations=(1, 2, 4))

        self.fusion_norm = nn.LayerNorm(config.dim_embedding)

        self.lstm = nn.LSTM(
            input_size=config.dim_embedding,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        lstm_out_dim = config.hidden_dim * 2  # 256

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_embedding,
            nhead=config.num_heads,
            dim_feedforward=config.dim_embedding * 4,
            dropout=0.3,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.transformer.use_nested_tensor = False
        trans_out_dim = config.dim_embedding  # 128

        total_dim = lstm_out_dim + trans_out_dim  # 384
        self.se_recalibration = SEChannelRecalibration(d_model=total_dim, reduction=4)
        self.attn_pooling = AttentionPooling(d_model=total_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_class)
        )

    def forward(self, seqs):
        x_emb = self.embedding(seqs)  # [B, L, 128]
        x_emb_pe = self.pos_encoder(x_emb)
        x_emb_pe = self.dropout_emb(x_emb_pe)

        x_cnn_in = x_emb_pe.permute(0, 2, 1)  # [B, 128, L]

        wav_out = self.wavelet(x_cnn_in)
        vmd_out = self.vmd(x_cnn_in)
        mscnn_out = self.mscnn(x_cnn_in)
        tcn_out = self.tcn(x_cnn_in)

        x_fused_cnn = x_cnn_in + wav_out + vmd_out + mscnn_out + tcn_out
        x_fused = x_fused_cnn.permute(0, 2, 1)  # [B, L, 128]
        x_fused = self.fusion_norm(x_fused)

        padding_mask = (seqs == 0)  # [B, L]
        attn_mask = (seqs != 0).unsqueeze(-1)  # [B, L, 1]

        lstm_out, _ = self.lstm(x_fused)  # [B, L, 256]

        trans_out = self.transformer(
            x_fused,
            src_key_padding_mask=padding_mask
        )  # [B, L, 128]

        dual_path_out = torch.cat([lstm_out, trans_out], dim=-1)  # [B, L, 384]
        dual_path_out = self.se_recalibration(dual_path_out)

        pooled_hidden = self.attn_pooling(dual_path_out, mask=attn_mask)  # [B, 384]
        logits = self.classifier(pooled_hidden)
        return logits


class Ultimate_Fusion(Base_Ultimate_Fusion):
    """
    Full TF-DualMethyl Model with Genuine Time-Frequency Decomposition & Dual-Path Gated Fusion:
      - Temporal-Spatial Serial Branch: F_ts = MSCNN(TCN(X_conv))
      - Frequency-Multi-resolution Serial Branch: F_fw = MultiWavelet(SpectralVMD(X_conv))
      - Adaptive Gated Residual Combination: H = LayerNorm(GatedFusion(X_conv, F_ts, F_fw)^T)
      - Dual-Path Context Encoder: SE_Recalibration(BiLSTM || Transformer) -> AttentionPooling -> Classifier
    """
    def __init__(self, config=None):
        super(Ultimate_Fusion, self).__init__(config)
        self.front_gate = AdaptiveGatedFusion(channels=self.config.dim_embedding, num_branches=3)

    def forward(self, seqs):
        x_emb = self.embedding(seqs)
        x_emb_pe = self.pos_encoder(x_emb)
        x_emb_pe = self.dropout_emb(x_emb_pe)
        x_cnn_in = x_emb_pe.permute(0, 2, 1)

        # 1. TCN + MSCNN Serial Fusion (Temporal-Spatial Branch)
        tcn_out = self.tcn(x_cnn_in)
        mscnn_out = self.mscnn(tcn_out)

        # 2. Spectral VMD + Multi-Scale Wavelet Serial Fusion (Frequency-Wavelet Branch)
        vmd_out = self.vmd(x_cnn_in)
        wav_out = self.wavelet(vmd_out)

        # 3. Adaptive Gated Multi-Branch Combination
        x_fused_cnn = self.front_gate([x_cnn_in, mscnn_out, wav_out])
        x_fused = x_fused_cnn.permute(0, 2, 1)
        x_fused = self.fusion_norm(x_fused)

        padding_mask = (seqs == 0)
        attn_mask = (seqs != 0).unsqueeze(-1)

        lstm_out, _ = self.lstm(x_fused)
        trans_out = self.transformer(x_fused, src_key_padding_mask=padding_mask)

        dual_path_out = torch.cat([lstm_out, trans_out], dim=-1)
        dual_path_out = self.se_recalibration(dual_path_out)
        pooled_hidden = self.attn_pooling(dual_path_out, mask=attn_mask)
        logits = self.classifier(pooled_hidden)
        return logits


class Ablation_No_BiLSTM(Ultimate_Fusion):
    """
    Ablation variant w/o BiLSTM:
    Keeps the exact Serial-Parallel Time-Frequency front-end of TF-DualMethyl,
    removes only the BiLSTM path in the dual-path encoder.
    """
    def __init__(self, config=None):
        super(Ablation_No_BiLSTM, self).__init__(config)
        self.lstm = None
        total_dim = self.config.dim_embedding
        self.se_recalibration = SEChannelRecalibration(d_model=total_dim, reduction=4)
        self.attn_pooling = AttentionPooling(d_model=total_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.num_class)
        )

    def forward(self, seqs):
        x_emb = self.embedding(seqs)
        x_emb_pe = self.pos_encoder(x_emb)
        x_emb_pe = self.dropout_emb(x_emb_pe)
        x_cnn_in = x_emb_pe.permute(0, 2, 1)

        tcn_out = self.tcn(x_cnn_in)
        mscnn_out = self.mscnn(tcn_out)

        vmd_out = self.vmd(x_cnn_in)
        wav_out = self.wavelet(vmd_out)

        x_fused_cnn = self.front_gate([x_cnn_in, mscnn_out, wav_out])
        x_fused = x_fused_cnn.permute(0, 2, 1)
        x_fused = self.fusion_norm(x_fused)

        padding_mask = (seqs == 0)
        attn_mask = (seqs != 0).unsqueeze(-1)

        trans_out = self.transformer(x_fused, src_key_padding_mask=padding_mask)
        trans_out = self.se_recalibration(trans_out)
        pooled_hidden = self.attn_pooling(trans_out, mask=attn_mask)
        logits = self.classifier(pooled_hidden)
        return logits


class Ablation_No_Transformer(Ultimate_Fusion):
    """
    Ablation variant w/o Transformer:
    Keeps the exact Serial-Parallel Time-Frequency front-end of TF-DualMethyl,
    removes only the Transformer path in the dual-path encoder.
    """
    def __init__(self, config=None):
        super(Ablation_No_Transformer, self).__init__(config)
        self.transformer = None
        total_dim = self.config.hidden_dim * 2
        self.se_recalibration = SEChannelRecalibration(d_model=total_dim, reduction=4)
        self.attn_pooling = AttentionPooling(d_model=total_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.config.num_class)
        )

    def forward(self, seqs):
        x_emb = self.embedding(seqs)
        x_emb_pe = self.pos_encoder(x_emb)
        x_emb_pe = self.dropout_emb(x_emb_pe)
        x_cnn_in = x_emb_pe.permute(0, 2, 1)

        tcn_out = self.tcn(x_cnn_in)
        mscnn_out = self.mscnn(tcn_out)

        vmd_out = self.vmd(x_cnn_in)
        wav_out = self.wavelet(vmd_out)

        x_fused_cnn = self.front_gate([x_cnn_in, mscnn_out, wav_out])
        x_fused = x_fused_cnn.permute(0, 2, 1)
        x_fused = self.fusion_norm(x_fused)

        attn_mask = (seqs != 0).unsqueeze(-1)
        lstm_out, _ = self.lstm(x_fused)
        lstm_out = self.se_recalibration(lstm_out)

        pooled_hidden = self.attn_pooling(lstm_out, mask=attn_mask)
        logits = self.classifier(pooled_hidden)
        return logits


class Ablation_No_Freq(Ultimate_Fusion):
    """
    Ablation variant w/o Freq (removes Spectral VMD + Multi-Scale Wavelet branch):
    Retains the serial Temporal-Spatial branch F_ts = MSCNN(TCN(X_conv)) and dual-path encoder.
    """
    def __init__(self, config=None):
        super(Ablation_No_Freq, self).__init__(config)
        self.wavelet = None
        self.vmd = None
        self.front_gate = AdaptiveGatedFusion(channels=self.config.dim_embedding, num_branches=2)

    def forward(self, seqs):
        x_emb = self.embedding(seqs)
        x_emb_pe = self.pos_encoder(x_emb)
        x_emb_pe = self.dropout_emb(x_emb_pe)
        x_cnn_in = x_emb_pe.permute(0, 2, 1)

        tcn_out = self.tcn(x_cnn_in)
        mscnn_out = self.mscnn(tcn_out)

        x_fused_cnn = self.front_gate([x_cnn_in, mscnn_out])
        x_fused = x_fused_cnn.permute(0, 2, 1)
        x_fused = self.fusion_norm(x_fused)

        padding_mask = (seqs == 0)
        attn_mask = (seqs != 0).unsqueeze(-1)

        lstm_out, _ = self.lstm(x_fused)
        trans_out = self.transformer(x_fused, src_key_padding_mask=padding_mask)

        dual_path_out = torch.cat([lstm_out, trans_out], dim=-1)
        dual_path_out = self.se_recalibration(dual_path_out)
        pooled_hidden = self.attn_pooling(dual_path_out, mask=attn_mask)
        logits = self.classifier(pooled_hidden)
        return logits


class Ablation_No_Time(Ultimate_Fusion):
    """
    Ablation variant w/o Time (removes TCN + MSCNN branch):
    Retains the serial Frequency-Wavelet branch F_fw = Wavelet(VMD(X_conv)) and dual-path encoder.
    """
    def __init__(self, config=None):
        super(Ablation_No_Time, self).__init__(config)
        self.mscnn = None
        self.tcn = None
        self.front_gate = AdaptiveGatedFusion(channels=self.config.dim_embedding, num_branches=2)

    def forward(self, seqs):
        x_emb = self.embedding(seqs)
        x_emb_pe = self.pos_encoder(x_emb)
        x_emb_pe = self.dropout_emb(x_emb_pe)
        x_cnn_in = x_emb_pe.permute(0, 2, 1)

        vmd_out = self.vmd(x_cnn_in)
        wav_out = self.wavelet(vmd_out)

        x_fused_cnn = self.front_gate([x_cnn_in, wav_out])
        x_fused = x_fused_cnn.permute(0, 2, 1)
        x_fused = self.fusion_norm(x_fused)

        padding_mask = (seqs == 0)
        attn_mask = (seqs != 0).unsqueeze(-1)

        lstm_out, _ = self.lstm(x_fused)
        trans_out = self.transformer(x_fused, src_key_padding_mask=padding_mask)

        dual_path_out = torch.cat([lstm_out, trans_out], dim=-1)
        dual_path_out = self.se_recalibration(dual_path_out)
        pooled_hidden = self.attn_pooling(dual_path_out, mask=attn_mask)
        logits = self.classifier(pooled_hidden)
        return logits


class iDNA_ABT_Wrapper(nn.Module):
    def __init__(self, config):
        super(iDNA_ABT_Wrapper, self).__init__()
        from iDNA_ABT import BERT

        if not hasattr(config, 'max_len'): config.max_len = 41
        if not hasattr(config, 'num_layer'): config.num_layer = 1
        if not hasattr(config, 'num_head'): config.num_head = config.num_heads
        if not hasattr(config, 'dim_feedforward'): config.dim_feedforward = config.dim_embedding * 4
        if not hasattr(config, 'dim_k'): config.dim_k = 16
        if not hasattr(config, 'dim_v'): config.dim_v = 16
        if not hasattr(config, 'cuda'): config.cuda = torch.cuda.is_available()

        self.model = BERT(config)

    def forward(self, seqs):
        logits, rep = self.model(seqs)
        return logits


def get_model_by_name(model_name, config):
    if model_name == "Base_Ultimate_Fusion":
        return Base_Ultimate_Fusion(config)
    elif model_name in ["Ultimate_Fusion", "TF_DualMethyl", "TF-DualMethyl"]:
        return Ultimate_Fusion(config)
    elif model_name == "Ablation_No_BiLSTM":
        return Ablation_No_BiLSTM(config)
    elif model_name == "Ablation_No_Transformer":
        return Ablation_No_Transformer(config)
    elif model_name == "Ablation_No_Freq":
        return Ablation_No_Freq(config)
    elif model_name == "Ablation_No_Time":
        return Ablation_No_Time(config)
    elif model_name in ["Baseline_TextCNN", "CNN"]:
        from baseline.baseline_models import Baseline_TextCNN
        return Baseline_TextCNN(config)
    elif model_name in ["Baseline_BiLSTM", "LSTM"]:
        from baseline.baseline_models import Baseline_BiLSTM
        return Baseline_BiLSTM(config)
    elif model_name == "Baseline_CRNN":
        from baseline.baseline_models import Baseline_CRNN
        return Baseline_CRNN(config)
    elif model_name in ["Baseline_Transformer", "Transformer"]:
        from baseline.baseline_models import Baseline_Transformer
        return Baseline_Transformer(config)
    elif model_name == "iDNA_ABT":
        return iDNA_ABT_Wrapper(config)
    elif model_name == "TCN":
        from baseline.baseline_models import Baseline_TextCNN
        return Baseline_TextCNN(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
