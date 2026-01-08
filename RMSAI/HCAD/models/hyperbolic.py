"""
Hyperbolic Geometry Operations for H²-CAD.

This module implements operations in the Poincaré ball model of hyperbolic space,
which is used for hierarchical uncertainty encoding in H²-CAD.

Key Components:
    - HyperbolicOperations: Core geometric operations (exp_map, log_map, distance)
    - PoincareEmbedding: Neural network layer for projecting to Poincaré ball
    - HyperbolicDistillationLoss: Certainty-weighted hyperbolic distance loss
    - ProgressiveDistillationScheduler: Warmup scheduler for distillation weight

Mathematical Background:
    The Poincaré ball model D^n_c = {z ∈ R^n : c||z||² < 1} provides:
    - Exponential volume growth for hierarchical representation
    - Conformal factor λ_z = 2/(1 - c||z||²) that amplifies gradients near boundary
    - Natural encoding of uncertainty: high uncertainty → near boundary
"""

import torch
import torch.nn as nn


class HyperbolicOperations:
    """
    Core hyperbolic space operations in the Poincaré ball model.

    All operations are implemented with numerical stability safeguards
    to prevent NaN values near the boundary of the ball.

    The Poincaré ball D^n_c has curvature -c (c > 0).
    Points with ||z|| close to 1/√c are near the boundary.
    """

    @staticmethod
    def safe_norm(x, eps=1e-8, dim=-1):
        """
        Compute norm with numerical stability.

        Args:
            x: Input tensor
            eps: Small constant for numerical stability
            dim: Dimension along which to compute norm

        Returns:
            Clamped norm tensor
        """
        return torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)

    @staticmethod
    def project_to_ball(x, c=1.0, eps=1e-5):
        """
        Project points to the interior of the Poincaré ball.

        Ensures ||z|| < 1/√c - ε for numerical stability.

        Args:
            x: Points to project
            c: Curvature parameter
            eps: Safety margin from boundary

        Returns:
            Projected points safely inside the ball
        """
        max_norm = (1.0 / (c ** 0.5)) - eps
        norm = HyperbolicOperations.safe_norm(x)
        cond = norm > max_norm
        x_proj = x / norm * max_norm
        return torch.where(cond, x_proj, x)

    @staticmethod
    def exp_map_zero(v, c=1.0, eps=1e-8):
        """
        Exponential map from tangent space at origin to Poincaré ball.

        exp_0(v) = tanh(√c ||v||) · v / (√c ||v||)

        This maps Euclidean vectors to points in hyperbolic space.

        Args:
            v: Tangent vector at origin
            c: Curvature parameter
            eps: Numerical stability constant

        Returns:
            Point in Poincaré ball
        """
        sqrt_c = c ** 0.5
        norm = HyperbolicOperations.safe_norm(v, eps=eps)
        coef = torch.tanh(sqrt_c * norm) / (sqrt_c * norm + eps)
        return HyperbolicOperations.project_to_ball(coef * v, c)

    @staticmethod
    def log_map_zero(z, c=1.0, eps=1e-8):
        """
        Logarithmic map from Poincaré ball to tangent space at origin.

        log_0(z) = artanh(√c ||z||) · z / (√c ||z||)

        This maps points in hyperbolic space back to Euclidean vectors.

        Args:
            z: Point in Poincaré ball
            c: Curvature parameter
            eps: Numerical stability constant

        Returns:
            Tangent vector at origin
        """
        sqrt_c = c ** 0.5
        norm = HyperbolicOperations.safe_norm(z, eps=eps)
        norm_clamped = torch.clamp(sqrt_c * norm, max=1.0 - eps)
        coef = torch.atanh(norm_clamped) / (sqrt_c * norm + eps)
        return coef * z

    @staticmethod
    def mobius_addition(x, y, c=1.0, eps=1e-8):
        """
        Möbius addition in the Poincaré ball.

        x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) /
                  (1 + 2c⟨x,y⟩ + c²||x||²||y||²)

        This is the hyperbolic analog of vector addition.

        Args:
            x, y: Points in Poincaré ball
            c: Curvature parameter
            eps: Numerical stability constant

        Returns:
            Result of Möbius addition
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c * c * x2 * y2
        return HyperbolicOperations.project_to_ball(num / (denom + eps), c)

    @staticmethod
    def hyperbolic_distance(x, y, c=1.0, eps=1e-8):
        """
        Compute hyperbolic distance between two points in Poincaré ball.

        d_H(x, y) = (2/√c) · artanh(√c ||-x ⊕_c y||)

        Key property: distance grows exponentially near boundary,
        providing position-aware gradient amplification.

        Args:
            x, y: Points in Poincaré ball
            c: Curvature parameter
            eps: Numerical stability constant

        Returns:
            Hyperbolic distance (scalar per sample)
        """
        sqrt_c = c ** 0.5
        add_result = HyperbolicOperations.mobius_addition(-x, y, c, eps)
        norm = HyperbolicOperations.safe_norm(add_result, eps=eps)
        norm_clamped = torch.clamp(sqrt_c * norm, max=1.0 - eps)
        return (2.0 / sqrt_c) * torch.atanh(norm_clamped)


class PoincareEmbedding(nn.Module):
    """
    Neural network layer for projecting features to Poincaré ball.

    Maps Euclidean features G to hyperbolic embeddings z:
        z = Π_c(exp_0(W·G + b))

    where Π_c ensures the result stays inside the ball.

    Args:
        input_dim: Dimension of input Euclidean features
        output_dim: Dimension of hyperbolic embeddings (d')
        curvature: Curvature parameter c of the Poincaré ball
        safety_margin: Safety margin ε for numerical stability
    """

    def __init__(self, input_dim=128, output_dim=128, curvature=1.0, safety_margin=1e-5):
        super().__init__()
        self.curvature = curvature
        self.safety_margin = safety_margin
        self.projection = nn.Linear(input_dim, output_dim)
        self.ops = HyperbolicOperations()

    def forward(self, x):
        """
        Project Euclidean features to Poincaré ball.

        Args:
            x: Euclidean features [B, input_dim]

        Returns:
            Poincaré ball embeddings [B, output_dim]
        """
        z_euc = self.projection(x)
        z_hyp = self.ops.exp_map_zero(z_euc, self.curvature)
        z_hyp = self.ops.project_to_ball(z_hyp, self.curvature, eps=self.safety_margin)
        return z_hyp

    def to_euclidean(self, z_hyp):
        """
        Map hyperbolic embeddings back to Euclidean space.

        Args:
            z_hyp: Poincaré ball embeddings [B, output_dim]

        Returns:
            Euclidean features [B, output_dim]
        """
        return self.ops.log_map_zero(z_hyp, self.curvature)


class HyperbolicDistillationLoss(nn.Module):
    """
    Certainty-Aware Hyperbolic Distillation Loss.

    L_hyp = max(0, d_H(z_S, z_T) - τ) · (1 - ŵ)

    Key properties:
    - Uses hyperbolic distance for position-aware gradient amplification
    - Weights loss by (1 - completeness) to focus on uncertain samples
    - Margin τ prevents over-optimization of already close embeddings

    Args:
        curvature: Curvature parameter of Poincaré ball
        margin: Relaxation margin τ
    """

    def __init__(self, curvature=1.0, margin=0.1):
        super().__init__()
        self.curvature = curvature
        self.margin = margin
        self.ops = HyperbolicOperations()

    def forward(self, z_student, z_teacher_detached, w_completeness):
        """
        Compute hyperbolic distillation loss.

        Args:
            z_student: Student's Poincaré embeddings [B, D]
            z_teacher_detached: Teacher's embeddings (detached) [B, D]
            w_completeness: Completeness (certainty) scores [B, 1]

        Returns:
            Scalar loss value
        """
        # Compute hyperbolic distance
        dist = self.ops.hyperbolic_distance(
            z_student, z_teacher_detached, self.curvature
        ).squeeze(-1)

        # Weight by incompleteness (uncertainty): (1 - ŵ)
        # Higher uncertainty → higher weight → stronger correction
        uncertainty_weight = 1.0 - w_completeness.squeeze(-1)

        # Apply margin and weighting
        loss = torch.relu(dist - self.margin) * uncertainty_weight

        return loss.mean()


class ProgressiveDistillationScheduler:
    """
    Progressive warmup scheduler for hyperbolic distillation weight.

    β(t) = 0                                    if t < T_warmup
         = β_max · (t - T_warmup) / T_warmup   if t ≥ T_warmup

    This prevents the model from being constrained by hyperbolic geometry
    before it learns basic representations.

    Args:
        warmup_epochs: Number of epochs before distillation begins
        max_weight: Maximum distillation weight β_max
    """

    def __init__(self, warmup_epochs=20, max_weight=1.0):
        self.warmup_epochs = warmup_epochs
        self.max_weight = max_weight

    def get_weight(self, epoch):
        """
        Get distillation weight for current epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Distillation weight β(t)
        """
        if epoch <= self.warmup_epochs:
            return 0.0
        progress = (epoch - self.warmup_epochs) / float(self.warmup_epochs)
        return min(progress * self.max_weight, self.max_weight)


# Aliases for backward compatibility
SafeHyperbolicOps = HyperbolicOperations
HyperbolicProjector = PoincareEmbedding
HyperbolicAttractionLoss = HyperbolicDistillationLoss
HyperbolicScheduler = ProgressiveDistillationScheduler
