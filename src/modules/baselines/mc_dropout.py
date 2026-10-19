import torch
import torch.nn as nn


class MCDropoutScheduler(nn.Module):
    """Trivial scheduler for MC Dropout time-stepper.

    Training: MSE(model(z0), z1).
    Inference: single forward pass (dropout stays active for MC sampling).
    """

    def __init__(self, **kwargs):
        super().__init__()

    def compute_loss(self, z0, z1, model, cond=None, **kwargs):
        """MSE loss: model(z0) vs z1."""
        z_pred = model(z0, cond=cond)
        loss = torch.nn.functional.mse_loss(z_pred, z1)
        return loss

    def sample(self, z_input, model, cond=None, **kwargs):
        """Single forward pass. Dropout stays active for MC uncertainty."""
        return model(z_input, cond=cond)
