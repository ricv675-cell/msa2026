"""
EMA (Exponential Moving Average) Teacher for H2-EMT model.

The EMA teacher provides stable learning targets by maintaining
an exponentially weighted moving average of the student model parameters.
"""

import copy
import torch


class EMATeacher:
    """
    Exponential Moving Average Teacher for knowledge distillation.

    The teacher model is a copy of the student model whose parameters
    are updated using exponential moving average of the student's parameters.
    This provides more stable and less noisy targets for the student to learn from.

    Usage:
        teacher = EMATeacher(student_model, decay=0.999)

        for batch in dataloader:
            # Forward pass with student
            student_out = student_model(batch)

            # Get teacher's output (no grad)
            with torch.no_grad():
                teacher_out = teacher.model(batch)

            # Compute loss using both outputs
            loss = loss_fn(student_out, teacher_out)

            # Update student
            optimizer.step()

            # Update teacher (after optimizer step)
            teacher.update(student_model)
    """

    def __init__(self, model, decay=0.999):
        """
        Initialize EMA teacher.

        Args:
            model: Student model to copy
            decay: EMA decay rate (higher = slower updates, more stable)
        """
        self.model = copy.deepcopy(model)
        self.decay = decay

        # Disable gradients for teacher model
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, student_model):
        """
        Update teacher parameters using EMA of student parameters.

        new_teacher_param = decay * teacher_param + (1 - decay) * student_param

        Args:
            student_model: Current student model
        """
        for ema_param, param in zip(self.model.parameters(), student_model.parameters()):
            ema_param.data = self.decay * ema_param.data + (1 - self.decay) * param.data

    def state_dict(self):
        """Return teacher model state dict for checkpointing."""
        return {
            'model_state_dict': self.model.state_dict(),
            'decay': self.decay
        }

    def load_state_dict(self, state_dict):
        """Load teacher model state dict from checkpoint."""
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.decay = state_dict['decay']
