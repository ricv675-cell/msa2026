"""
Loss Functions for H²-CAD.

This module implements the composite loss function for H²-CAD:
    L = L_task + λ₁·L_comp + β(t)·L_hyp + λ₂·L_recon

Components:
    - L_task: Sentiment prediction loss (MSE)
    - L_comp: Completeness estimation loss (MSE)
    - L_hyp: Hyperbolic distillation loss (certainty-weighted)
    - L_recon: Feature reconstruction loss (MSE)

The hyperbolic distillation loss uses:
    L_hyp = max(0, d_H(z_S, z_T) - τ) · (1 - ŵ)

Where:
    - d_H: Hyperbolic distance in Poincaré ball
    - z_S, z_T: Student and teacher embeddings
    - τ: Relaxation margin
    - ŵ: Completeness (certainty) score
"""

import torch
from torch import nn
from models.hyperbolic import HyperbolicDistillationLoss


class H2CADLoss(nn.Module):
    """
    Composite Loss Function for H²-CAD.

    Total Loss = L_task + α·L_comp + β(t)·L_hyp + γ·L_recon

    The hyperbolic distillation loss is weighted by (1 - completeness),
    focusing correction on uncertain/incomplete samples.

    Args:
        args: Configuration dictionary containing:
            - base.alpha: Weight for completeness loss
            - base.gamma: Weight for reconstruction loss
            - base.curvature: Poincaré ball curvature
            - base.hyp_margin: Margin for hyperbolic loss (optional)
    """

    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']      # Completeness loss weight (λ₁)
        self.gamma = args['base']['gamma']      # Reconstruction loss weight (λ₂)

        # Dataset-specific optimization for high-dimensional features
        dataset_name = args.get('dataset', {}).get('datasetName', '').lower()
        self.use_balanced_recon = (dataset_name == 'sims')

        # Hyperbolic distillation loss
        self.hyperbolic_loss = HyperbolicDistillationLoss(
            curvature=args['base']['curvature'],
            margin=args['base'].get('hyp_margin', 0.1)
        )

        self.mse = nn.MSELoss()

    def forward(self, student_out, teacher_z, labels, completeness_labels, beta_weight):
        """
        Compute H²-CAD loss.

        Args:
            student_out: Dictionary from model forward pass containing:
                - sentiment_preds: Predicted sentiment [B, 1]
                - w: Completeness (certainty) scores [B, 1]
                - z_hyp: Poincaré ball embeddings [B, D]
                - recons: Dict of reconstructed features
                - recon_targets: Dict of target features
            teacher_z: Teacher's Poincaré embeddings [B, D] (detached)
            labels: Ground truth sentiment labels [B, 1]
            completeness_labels: Ground truth completeness [B, 1]
            beta_weight: Current weight for hyperbolic loss β(t)

        Returns:
            Dictionary containing total loss and individual components
        """
        # ===== Task Loss: Sentiment Prediction =====
        L_task = self.mse(student_out['sentiment_preds'], labels)

        # ===== Completeness Loss =====
        L_comp = self.mse(student_out['w'], completeness_labels)

        # ===== Reconstruction Loss =====
        rec_losses = []
        if student_out.get('recons') and student_out.get('recon_targets'):
            for key in student_out['recons']:
                # Skip high-dimensional modalities for numerical stability
                if self.use_balanced_recon and key == 'vision':
                    continue
                rec_target = student_out['recon_targets'][key]
                rec_pred = student_out['recons'][key]
                if rec_target is not None and rec_pred is not None:
                    rec_losses.append(self.mse(rec_pred, rec_target))

        L_recon = torch.stack(rec_losses).mean() if rec_losses else \
                  torch.tensor(0.0, device=labels.device)

        # ===== Hyperbolic Distillation Loss =====
        L_hyp = torch.tensor(0.0, device=labels.device)
        if (teacher_z is not None) and (beta_weight > 0):
            L_hyp = self.hyperbolic_loss(
                student_out['z_hyp'],
                teacher_z,
                student_out['w']
            )

        # ===== Total Loss =====
        total_loss = L_task + self.alpha * L_comp + beta_weight * L_hyp + self.gamma * L_recon

        return {
            'loss': total_loss,
            'L_task': L_task.item() if isinstance(L_task, torch.Tensor) else L_task,
            'L_comp': L_comp.item() if isinstance(L_comp, torch.Tensor) else L_comp,
            'L_hyp': L_hyp.item() if isinstance(L_hyp, torch.Tensor) else L_hyp,
            'L_recon': L_recon.item() if isinstance(L_recon, torch.Tensor) else L_recon
        }


# Backward compatibility alias
H2EMTLoss = H2CADLoss
MultimodalLoss = H2CADLoss
