"""
Basic Neural Network Layers for H²-CAD.

This module contains the fundamental building blocks for the H²-CAD architecture:
- Transformer components (Encoder, Decoder, Cross-Attention)
- CrossModalCompensator: Generates proxy language features from audio-visual
- CompletenessEstimator: Predicts sample completeness (certainty) score
- LanguageGuidedFusion: Performs hyper-modality learning and cross-modal fusion

These components work together to handle incomplete multimodal data by:
1. Compensating for missing language modality using audio-visual information
2. Estimating how complete/certain each sample is
3. Fusing information across modalities with language guidance
"""

import torch
from torch import nn, einsum
from torch.nn import functional as F
from einops import rearrange, repeat


# ==================== Normalization Layers ====================

class PreNorm(nn.Module):
    """Pre-normalization wrapper for any layer."""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    """Pre-normalization for Q, K, V inputs separately."""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return self.fn(q, k, v)


class PreNorm_hyper(nn.Module):
    """Pre-normalization for hyper-modality learning (4 inputs)."""

    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_dominate, h_a, h_v, h_hyper):
        h_dominate = self.norm1(h_dominate)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)
        return self.fn(h_dominate, h_a, h_v, h_hyper)


# ==================== Basic Transformer Components ====================

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HyperModalityLearningLayer(nn.Module):
    """
    Hyper-modality learning layer for language-guided cross-modal fusion.

    Uses the dominant modality (language) as Query to attend to
    audio and visual modalities, aggregating information into
    a hyper-modality representation.

    Δh = CrossAttn(h_dominant → h_audio) + CrossAttn(h_dominant → h_visual)
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_audio = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_visual = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_audio = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_visual = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_dominate, h_a, h_v, h_hyper):
        h = self.heads

        q = self.to_q(h_dominate)
        k_a = self.to_k_audio(h_a)
        k_v = self.to_k_visual(h_v)
        v_a = self.to_v_audio(h_a)
        v_v = self.to_v_visual(h_v)

        q, k_a, k_v, v_a, v_v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
            (q, k_a, k_v, v_a, v_v)
        )

        # Cross-attention: Language → Audio
        dots_q_ka = einsum('b h i d, b h j d -> b h i j', q, k_a) * self.scale
        attn_q_ka = self.attend(dots_q_ka)
        out_q_ka = einsum('b h i j, b h j d -> b h i d', attn_q_ka, v_a)
        out_q_ka = rearrange(out_q_ka, 'b h n d -> b n (h d)')

        # Cross-attention: Language → Visual
        dots_q_kv = einsum('b h i d, b h j d -> b h i j', q, k_v) * self.scale
        attn_q_kv = self.attend(dots_q_kv)
        out_q_kv = einsum('b h i j, b h j d -> b h i d', attn_q_kv, v_v)
        out_q_kv = rearrange(out_q_kv, 'b h n d -> b n (h d)')

        # Aggregate and update hyper-modality representation
        h_hyper_shift = self.to_out(out_q_ka + out_q_kv)
        h_hyper = h_hyper + h_hyper_shift

        return h_hyper


class HyperModalityEncoder(nn.Module):
    """
    Multi-layer hyper-modality encoder.

    Iteratively refines the hyper-modality representation by
    attending to audio and visual features guided by language.
    """

    def __init__(self, dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_hyper(dim, HyperModalityLearningLayer(
                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                ))
            ]))

    def forward(self, h_dominate_list, h_a, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_dominate_list[i], h_a, h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder with self-attention."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden:
            hidden_list = [x]
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class TransformerDecoder(nn.Module):
    """Transformer decoder with self-attention and cross-attention."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, tgt, memory):
        for attn1, attn2, ff in self.layers:
            tgt = attn1(tgt, tgt, tgt) + tgt
            tgt = attn2(tgt, memory, memory) + tgt
            tgt = ff(tgt) + tgt
        return tgt


class CrossAttentionEncoder(nn.Module):
    """Cross-attention encoder for attending to external context."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    """
    Transformer with optional learnable tokens.

    Can prepend learnable tokens (e.g., CLS token) to the input sequence
    and optionally save intermediate hidden states.
    """

    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b=b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n + self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    """
    Cross-modal Transformer for fusing two sequences.

    Adds learnable CLS tokens and performs cross-attention
    from source to target sequence.
    """

    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads,
                 mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.CrossTransformerEncoder = CrossAttentionEncoder(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b=b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, :n_s + 1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, :n_t + 1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t


# ==================== H²-CAD Core Components ====================

class CrossModalCompensator(nn.Module):
    """
    Cross-Modal Compensation (CMC) Module.

    When language modality is severely missing, this module generates
    proxy language features from audio and visual modalities.

    Architecture:
        1. Learnable proxy tokens attend to concatenated [audio; visual]
        2. Gate network determines mixing ratio
        3. Output: Ĥ^l = g · H_proxy + (1-g) · H^l

    Args:
        dim: Feature dimension
        token_len: Number of tokens
        depth: Transformer depth
        heads: Number of attention heads
        dim_head: Dimension per head
        mlp_dim: FFN hidden dimension
    """

    def __init__(self, dim=128, token_len=8, depth=2, heads=8, dim_head=16, mlp_dim=None):
        super().__init__()
        mlp_dim = mlp_dim or dim

        # Learnable proxy tokens
        self.proxy_tokens = nn.Parameter(torch.randn(1, token_len, dim))

        # Cross-attention to audio-visual features
        self.cross_attention = CrossAttentionEncoder(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim
        )

        # Gate network: g = σ(W_g[H̄_proxy; H̄^a; H̄^v] + b_g)
        self.gate_network = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

    def forward(self, H_audio, H_visual, H_language_incomplete):
        """
        Generate compensated language features.

        Args:
            H_audio: Audio features [B, T, D]
            H_visual: Visual features [B, T, D]
            H_language_incomplete: Incomplete language features [B, T, D]

        Returns:
            Compensated language features Ĥ^l [B, T, D]
        """
        b = H_audio.size(0)

        # Expand proxy tokens for batch
        proxy = self.proxy_tokens.expand(b, -1, -1)

        # Cross-attend to audio-visual features
        H_proxy = self.cross_attention(
            torch.cat([H_audio, H_visual], dim=1),
            proxy
        )

        # Compute gating weights from global representations
        gate_input = torch.cat([
            H_proxy.mean(dim=1),
            H_audio.mean(dim=1),
            H_visual.mean(dim=1)
        ], dim=-1)
        g = self.gate_network(gate_input).unsqueeze(1)

        # Gated fusion: Ĥ^l = g · H_proxy + (1-g) · H^l
        H_language_compensated = g * H_proxy + (1 - g) * H_language_incomplete

        return H_language_compensated


class LanguageGuidedFusion(nn.Module):
    """
    Language-Guided Fusion Module with Completeness Estimation.

    This module:
    1. Estimates sample completeness (certainty) score ŵ
    2. Performs hyper-modality learning with language guidance
    3. Fuses information via cross-modal attention

    Architecture:
        - CompletenessEstimator: Transformer + MLP → ŵ ∈ [0,1]
        - HyperModalityEncoder: Iterative cross-attention
        - CrossTransformer: Final fusion

    Args:
        dim: Feature dimension
        token_len: Number of tokens
        depth: Transformer depth
        heads: Number of attention heads
        dim_head: Dimension per head
        hyper_depth: Depth of hyper-modality encoder
    """

    def __init__(self, dim=128, token_len=8, depth=2, heads=8, dim_head=16, hyper_depth=3):
        super().__init__()
        self.hyper_depth = hyper_depth

        # ===== Completeness Estimator =====
        self.completeness_encoder = Transformer(
            num_frames=token_len,
            token_len=1,  # Single CLS token
            save_hidden=False,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=dim
        )
        self.completeness_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()  # Output ŵ ∈ [0, 1]
        )

        # ===== Hyper-Modality Learning =====
        self.hyper_tokens = nn.Parameter(torch.randn(1, token_len, dim))
        self.hyper_encoder = HyperModalityEncoder(
            dim=dim, depth=hyper_depth, heads=heads, dim_head=dim_head
        )

        # ===== Cross-Modal Fusion =====
        self.fusion_transformer = CrossTransformer(
            source_num_frames=token_len,
            tgt_num_frames=token_len,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=dim
        )

    def forward(self, H_dominant, H_audio, H_visual):
        """
        Perform completeness estimation and language-guided fusion.

        Args:
            H_dominant: Dominant (compensated language) features [B, T, D]
            H_audio: Audio features [B, T, D]
            H_visual: Visual features [B, T, D]

        Returns:
            H_fused: Fused features [B, T-1, D] (excluding CLS token)
            w_hat: Completeness (certainty) scores [B, 1]
        """
        b = H_dominant.size(0)

        # ===== Step 1: Estimate Completeness =====
        # Extract CLS token after encoding
        comp_features = self.completeness_encoder(H_dominant)[:, :1].squeeze(1)
        w_hat = self.completeness_head(comp_features)

        # ===== Step 2: Hyper-Modality Learning =====
        # Initialize hyper-modality tokens
        h_hyper = self.hyper_tokens.expand(b, -1, -1)

        # Create list of dominant features for each layer
        H_dominant_list = [H_dominant for _ in range(self.hyper_depth)]

        # Iteratively refine hyper-modality representation
        h_hyper = self.hyper_encoder(H_dominant_list, H_audio, H_visual, h_hyper)

        # ===== Step 3: Cross-Modal Fusion =====
        H_fused = self.fusion_transformer(h_hyper, H_dominant)

        # Return features excluding the CLS token
        return H_fused[:, 1:], w_hat


def pool_sequence_to_tokens(x, token_len):
    """
    Pool sequence to fixed token length using adaptive average pooling.

    Args:
        x: Input tensor [B, T, D]
        token_len: Target sequence length

    Returns:
        Pooled tensor [B, token_len, D]
    """
    if x is None:
        return None
    return F.adaptive_avg_pool1d(x.transpose(1, 2), token_len).transpose(1, 2)


# ==================== Backward Compatibility Aliases ====================

# Old names → New names
GatedProxyGenerator = CrossModalCompensator
CompletenessAwareFusion = LanguageGuidedFusion
HhyperLearningLayer = HyperModalityLearningLayer
HhyperLearningEncoder = HyperModalityEncoder
CrossTransformerEncoder = CrossAttentionEncoder
