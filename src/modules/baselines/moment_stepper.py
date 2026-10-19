import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MomentStepperScheduler(nn.Module):
    """
    Baseline stepper that matches the moments of the predicted Gaussian
    to the moments of the next VAE posterior (post1), instead of using NLL
    against a sample from post1.

    Loss: KL( N(μ1, Σ1) || N(μ_pred, σ_pred²) )
        = 0.5 * [ log(σ_pred²/Σ1) + Σ1/σ_pred² + (μ1 - μ_pred)²/σ_pred² - 1 ]

    This requires no sampling from post1 — it uses post1.mean and post1.var
    directly, just like TORNADO.

    Model interface: identical to GaussianStepperScheduler
        input:  cat(mu0, sigma0)  shape (B, H, W, 2*C)
        output: (mu_pred, sigma_pred_raw)  each (B, H, W, C)
    """

    def __init__(self, loss_type: str = "kl", **kwargs):
        super().__init__()
        self.loss_type = loss_type  # "kl" or "mse_moments"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_input(post0):
        mu0 = rearrange(post0.mean, "b c h w -> b h w c")
        sigma0 = rearrange(torch.sqrt(post0.var), "b c h w -> b h w c")
        return torch.cat([mu0, sigma0], dim=-1), mu0, sigma0

    @staticmethod
    def _sigma_from_raw(raw):
        return F.softplus(raw) + 1e-5

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, post0, post1, model, cond=None, **kwargs):
        """
        Computes moment-matching loss between the predicted Gaussian
        and the target posterior post1.

        Parameters
        ----------
        post0 : VAE posterior for the current frame  (has .mean, .var)
        post1 : VAE posterior for the next frame     (has .mean, .var)
        model : DiT model with tornado=True (returns two outputs)
        cond  : optional conditioning tensor
        """
        # ----- Target moments (from post1, no sampling needed) -----
        mu1    = rearrange(post1.mean, "b c h w -> b h w c")
        Sigma1 = rearrange(post1.var,  "b c h w -> b h w c")  # variance

        # ----- Model input -----
        model_in, _, _ = self._make_input(post0)

        output = model(
            model_in,
            sigma_t=torch.ones(model_in.shape[0], device=model_in.device),
            cond=cond,
        )

        if isinstance(output, tuple):
            mu_pred, sigma_pred_raw = output
        else:
            raise ValueError("Model must return tuple (mu, sigma_raw)")

        sigma_pred = self._sigma_from_raw(sigma_pred_raw)   # σ  (std)
        var_pred   = sigma_pred.pow(2)                       # σ² (variance)

        # ----- Loss -----
        if self.loss_type == "kl":
            # KL( N(μ1,Σ1) || N(μ_pred, σ_pred²) )
            # = 0.5 * [ log(σ²/Σ1) + Σ1/σ² + (μ1-μ_pred)²/σ² - 1 ]
            kl = 0.5 * (
                torch.log(var_pred / (Sigma1 + 1e-10))
                + (Sigma1 + 1e-10) / var_pred
                + (mu1 - mu_pred).pow(2) / var_pred
                - 1.0
            )
            loss = kl.mean()
        elif self.loss_type == "mse_moments":
            loss_mu  = F.mse_loss(mu_pred, mu1)
            loss_var = F.mse_loss(var_pred, Sigma1)
            loss = loss_mu + loss_var
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}")

        # ----- Diagnostics (no-grad) -----
        with torch.no_grad():
            mu_err   = (mu_pred - mu1).pow(2).mean().sqrt().item()
            var_err  = (var_pred - Sigma1).pow(2).mean().sqrt().item()
            mu1_rms  = mu1.pow(2).mean().sqrt().item()
            Σ1_mean  = Sigma1.mean().item()
            σ_pred_mean = sigma_pred.mean().item()

        log_dict = {
            "loss": loss.item(),
            "mu_err_rms": mu_err,
            "var_err_rms": var_err,
            "mu1_rms": mu1_rms,
            "Sigma1_mean": Σ1_mean,
            "sigma_pred_mean": σ_pred_mean,
        }

        return loss, log_dict

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def sample(self, input_stats, model, cond=None, **kwargs):
        """
        One-step inference.

        Parameters
        ----------
        input_stats : (mu, sigma) tuple  — (B, H, W, C) each
        model       : DiT model
        cond        : optional conditioning tensor

        Returns
        -------
        (mu_pred, sigma_pred)  — predicted mean and std  (B, H, W, C)
        """
        mu, sigma = input_stats

        model_in = torch.cat([mu, sigma], dim=-1)

        output = model(
            model_in,
            sigma_t=torch.ones(model_in.shape[0], device=model_in.device),
            cond=cond,
        )

        mu_pred, sigma_pred_raw = output
        sigma_pred = self._sigma_from_raw(sigma_pred_raw)

        return mu_pred, sigma_pred
