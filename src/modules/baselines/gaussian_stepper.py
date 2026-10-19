import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GaussianStepperScheduler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # We can add parameters here if needed in the future

    def compute_loss(self, post0, post1, model, cond=None, **kwargs):
        """
        Computes Gaussian NLL loss.
        input to model: cat(mu0, sigma0)
        target: z1 (sample from post1)
        model output: mu_pred, sigma_pred
        """
        
        mu0 = rearrange(post0.mean, "b c h w -> b h w c")
        sigma0 = torch.sqrt(post0.var)
        sigma0 = rearrange(sigma0, "b c h w -> b h w c")

        # Target sample z1
        z1 = post1.sample()
        z1 = rearrange(z1, "b c h w -> b h w c")

        # Prepare input
        model_in = torch.cat([mu0, sigma0], dim=-1) # (B, H, W, 2*C)

        
        output = model(model_in, sigma_t=torch.ones(model_in.shape[0], device=model_in.device), cond=cond)

        if isinstance(output, tuple):
            mu_pred, sigma_pred_raw = output
        else:
            
            raise ValueError("Model must return tuple (mu, sigma)")



        sigma_pred = F.softplus(sigma_pred_raw) + 1e-5

        # Gaussian NLL
        # nll = 0.5 * (log(sigma^2) + (x - mu)^2 / sigma^2)
        #     = log(sigma) + 0.5 * (x - mu)^2 / sigma^2
        nll = torch.log(sigma_pred) + 0.5 * (z1 - mu_pred).pow(2) / sigma_pred.pow(2)
        
        loss = nll.mean()
        
        log_dict = {
            "nll": loss.item(),
            "sigma_pred_mean": sigma_pred.mean().item(),
            "mu_pred_mean": mu_pred.mean().item()
        }

        return loss, log_dict

    def sample(self, input_stats, model, cond=None, **kwargs):
        """
        Step function for inference.
        input_stats: (mu, sigma) tuple
        Returns: (next_mu, next_sigma)
        """
        mu, sigma = input_stats
        
        # Input to model
        model_in = torch.cat([mu, sigma], dim=-1)
        
        output = model(model_in, sigma_t=torch.ones(model_in.shape[0], device=model_in.device), cond=cond)
        
        mu_pred, sigma_pred_raw = output
        sigma_pred = F.softplus(sigma_pred_raw) + 1e-5
        
        return mu_pred, sigma_pred
