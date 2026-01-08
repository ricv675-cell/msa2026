"""
H²-CAD: Hierarchical Hyperbolic Certainty-Aware Distillation
层次化双曲确定性感知蒸馏

This module implements the H²-CAD architecture for multimodal sentiment analysis
with incomplete data. The framework uses:
- Poincaré ball embeddings for hierarchical uncertainty modeling
- EMA (Exponential Moving Average) Teacher for knowledge distillation
- Cross-Modal Compensation (CMC) for robust missing modality handling
- Completeness-Aware Fusion for adaptive multimodal integration

Key Components:
    - CrossModalCompensator: Generates proxy features from audio-visual modalities
    - CompletenessEstimator: Predicts sample completeness score
    - LanguageGuidedFusion: Performs hyper-modality learning and cross-modal fusion
    - PoincareEmbedding: Projects features to hyperbolic space

Reference:
    Hierarchical Hyperbolic Certainty-Aware Distillation for
    Multimodal Sentiment Analysis with Incomplete Data
"""

import torch
from torch import nn
from .basic_layers import (
    Transformer,
    CrossModalCompensator,
    LanguageGuidedFusion,
    pool_sequence_to_tokens
)
from .bert import BertTextEncoder
from .hyperbolic import PoincareEmbedding


class H2CAD(nn.Module):
    """
    H²-CAD: Hierarchical Hyperbolic Certainty-Aware Distillation Model.

    This model addresses multimodal sentiment analysis under incomplete data
    by leveraging hyperbolic geometry to encode hierarchical uncertainty.

    Architecture Overview:
        1. Multimodal Encoders: Extract features from text, audio, visual modalities
        2. Cross-Modal Compensator (CMC): Generate proxy features when language is missing
        3. Completeness-Aware Fusion: Estimate completeness and perform guided fusion
        4. Poincaré Embedding: Project to hyperbolic space for uncertainty-aware learning
        5. Sentiment Regressor: Predict sentiment from hyperbolic embeddings

    Args:
        args: Configuration dictionary containing model hyperparameters

    Input:
        complete_input: Tuple of (vision, audio, text) - complete modality tensors
        incomplete_input: Tuple of (vision_m, audio_m, text_m) - incomplete modality tensors
        return_teacher: If True, compute teacher embeddings from complete input

    Output:
        Dictionary containing:
            - sentiment_preds: Predicted sentiment scores [B, 1]
            - w: Completeness scores (certainty) [B, 1]
            - z_hyp: Poincaré ball embeddings [B, hyp_dim]
            - recons: Reconstructed features for each modality
            - recon_targets: Target features for reconstruction loss
    """

    def __init__(self, args):
        super(H2CAD, self).__init__()

        fe = args['model']['feature_extractor']
        hyper_cfg = args['model']['hyper_params']

        dim = fe['hidden_dims'][0]
        token_len = fe['token_length'][0]
        heads = fe['heads']
        depth = fe['depth']
        dim_head = hyper_cfg.get('dim_head', 16)

        self.token_len = token_len

        # ============ Multimodal Encoders ============
        # Language Encoder (BERT-based)
        self.text_encoder = BertTextEncoder(
            use_finetune=True,
            transformers='bert',
            pretrained=fe['bert_pretrained']
        )

        # Language Feature Projector
        self.proj_language = nn.Sequential(
            nn.Linear(fe['input_dims'][0], fe['hidden_dims'][0]),
            Transformer(
                num_frames=fe['input_length'][0],
                save_hidden=False,
                token_len=token_len,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=dim
            )
        )

        # Audio Feature Projector
        self.proj_audio = nn.Sequential(
            nn.Linear(fe['input_dims'][2], fe['hidden_dims'][2]),
            Transformer(
                num_frames=fe['input_length'][2],
                save_hidden=False,
                token_len=token_len,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=dim
            )
        )

        # Visual Feature Projector
        self.proj_visual = nn.Sequential(
            nn.Linear(fe['input_dims'][1], fe['hidden_dims'][1]),
            Transformer(
                num_frames=fe['input_length'][1],
                save_hidden=False,
                token_len=token_len,
                dim=dim,
                depth=depth,
                heads=heads,
                mlp_dim=dim
            )
        )

        # ============ Cross-Modal Compensation (CMC) ============
        self.cross_modal_compensator = CrossModalCompensator(
            dim=dim,
            token_len=token_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=dim
        )

        # ============ Completeness-Aware Fusion ============
        self.language_guided_fusion = LanguageGuidedFusion(
            dim=dim,
            token_len=token_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            hyper_depth=hyper_cfg['hyper_depth']
        )

        # ============ Poincaré Embedding ============
        self.poincare_embedding = PoincareEmbedding(
            input_dim=dim,
            output_dim=hyper_cfg['hyp_dim'],
            curvature=args['base']['curvature'],
            safety_margin=args['base'].get('safety_margin', 1e-5)
        )

        # ============ Sentiment Regressor ============
        self.sentiment_regressor = nn.Linear(hyper_cfg['hyp_dim'], 1)

        # ============ Reconstruction Decoders ============
        self.decoders = nn.ModuleDict({
            'text': nn.Linear(dim, fe['input_dims'][0]),
            'audio': nn.Linear(dim, fe['input_dims'][2]),
            'vision': nn.Linear(dim, fe['input_dims'][1])
        })

    def forward(self, complete_input, incomplete_input, return_teacher=False):
        """
        Forward pass of H²-CAD model.

        The forward pass consists of:
        1. Feature extraction from incomplete modalities
        2. Cross-modal compensation to generate proxy language features
        3. Completeness estimation and language-guided fusion
        4. Projection to Poincaré ball
        5. Sentiment prediction from hyperbolic embeddings

        Args:
            complete_input: Tuple (vision, audio, text) of complete modality tensors
            incomplete_input: Tuple (vision_m, audio_m, text_m) of incomplete tensors
            return_teacher: Whether to compute teacher embeddings

        Returns:
            Dictionary with predictions, embeddings, and intermediate outputs
        """
        vision, audio, text = complete_input
        vision_m, audio_m, text_m = incomplete_input

        # ===== Step 1: Multimodal Feature Extraction =====
        text_hidden_m = self.text_encoder(text_m)
        H_l = self.proj_language(text_hidden_m)[:, :self.token_len]  # Language features
        H_a = self.proj_audio(audio_m)[:, :self.token_len]           # Audio features
        H_v = self.proj_visual(vision_m)[:, :self.token_len]         # Visual features

        # ===== Step 2: Cross-Modal Compensation (CMC) =====
        # Generate compensated language features: Ĥ^l = g·H_proxy + (1-g)·H^l
        H_l_hat = self.cross_modal_compensator(H_a, H_v, H_l)

        # ===== Step 3: Completeness-Aware Fusion =====
        # Estimate completeness ŵ and perform language-guided fusion
        H_fused, w_hat = self.language_guided_fusion(H_l_hat, H_a, H_v)

        # ===== Step 4: Poincaré Ball Projection =====
        # Global representation: G = mean(H_fused)
        G = H_fused.mean(dim=1)

        # Project to hyperbolic space: z = Π_c(exp_0(W·G + b))
        z_hyp = self.poincare_embedding(G)

        # Map back to Euclidean for regression
        z_euc = self.poincare_embedding.to_euclidean(z_hyp)

        # ===== Step 5: Sentiment Prediction =====
        sentiment_pred = self.sentiment_regressor(z_euc)

        # ===== Reconstruction Targets =====
        if text is not None:
            text_hidden_complete = self.text_encoder(text)
        else:
            text_hidden_complete = text_hidden_m

        text_target = pool_sequence_to_tokens(text_hidden_complete.detach(), self.token_len)
        audio_target = pool_sequence_to_tokens(
            (audio if audio is not None else audio_m).detach(), self.token_len
        )
        vision_target = pool_sequence_to_tokens(
            (vision if vision is not None else vision_m).detach(), self.token_len
        )

        # Compute reconstructions
        rec_text = self.decoders['text'](H_l)
        rec_audio = self.decoders['audio'](H_a)
        rec_vision = self.decoders['vision'](H_v)

        result = {
            'sentiment_preds': sentiment_pred,
            'w': w_hat,                          # Completeness (certainty) score
            'z_hyp': z_hyp,                      # Poincaré ball embedding
            'recons': {
                'text': rec_text,
                'audio': rec_audio,
                'vision': rec_vision
            },
            'recon_targets': {
                'text': text_target,
                'audio': audio_target,
                'vision': vision_target
            }
        }

        # ===== Optional: Teacher Embeddings =====
        if return_teacher and all(x is not None for x in [text, audio, vision]):
            H_l_t = self.proj_language(self.text_encoder(text))[:, :self.token_len]
            H_a_t = self.proj_audio(audio)[:, :self.token_len]
            H_v_t = self.proj_visual(vision)[:, :self.token_len]

            H_l_hat_t = self.cross_modal_compensator(H_a_t, H_v_t, H_l_t)
            H_fused_t, _ = self.language_guided_fusion(H_l_hat_t, H_a_t, H_v_t)

            G_t = H_fused_t.mean(dim=1)
            z_hyp_t = self.poincare_embedding(G_t)
            result['z_hyp_teacher'] = z_hyp_t.detach()

        return result


def build_model(args):
    """Build and return H²-CAD model."""
    return H2CAD(args)
