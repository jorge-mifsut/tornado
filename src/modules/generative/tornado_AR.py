import torch
import torch.nn.functional as F
from einops import rearrange

from modules.generative.tornado import AdaptiveScheduler


class AdaptiveARScheduler(AdaptiveScheduler):
    def __init__(self, horizon, from_checkpoint=None, detach_rollout=True, step_weight_decay=0.5, **kwargs):
        super().__init__(**kwargs)
        self.horizon = int(horizon)
        self.from_checkpoint = from_checkpoint
        self.detach_rollout = bool(detach_rollout)
        self.step_weight_decay = float(step_weight_decay)

        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}.")

    def compute_loss(self, posteriors, model, train_module=None, ED_schedule=None, **kwargs):
        if not isinstance(posteriors, (list, tuple)):
            raise TypeError("AdaptiveARScheduler.compute_loss expects a sequence of posteriors.")

        if len(posteriors) < 2:
            raise ValueError("At least two posteriors are required for AR TORNADO training.")

        def to_channel_last(x):
            return rearrange(x, "b c h w -> b h w c")

        def α(t):
            return 1 - t

        def β(t):
            return t

        def I_τ(t, x0, x1):
            return α(t) * x0 + β(t) * x1

        def I2_τ(t, x0, x1):
            return α(t) ** 2 * x0 + β(t) ** 2 * x1

        def Λ(t):
            return (self.sigma_scale * t * (1 - t)).clamp_min(self.var_eps)

        available_horizon = len(posteriors) - 1
        rollout_horizon = min(self.horizon, available_horizon)
        if rollout_horizon < 1:
            raise ValueError("No valid AR rollout steps available in the provided posterior window.")

        target_means = []
        target_vars = []
        for post in posteriors:
            target_means.append(to_channel_last(post.mean))
            target_vars.append(to_channel_last(post.var).clamp_min(self.var_eps))

        b = target_means[0].shape[0]
        eps = 1e-3

        loss_drift_terms = []
        loss_diff_mse_terms = []
        loss_ed_terms = []

        tau_values = []
        delta_values = []
        drift_rms_values = []
        vtarget_rms_values = []
        drift_err_rms_values = []
        diff_amp_rms_values = []
        g_pred_mean_values = []
        g_target_mean_values = []
        g_target_min_values = []
        g_target_clamped_min_values = []
        sigma1_mean_values = []
        sigma_tau_mean_values = []
        lambda_mean_values = []

        current_mean = target_means[0]
        current_var = target_vars[0]

        step_weights = current_mean.new_tensor(
            [self.step_weight_decay ** step for step in range(rollout_horizon)]
        )
        step_weights = step_weights / step_weights.sum().clamp_min(self.var_eps)

        for step in range(rollout_horizon):
            next_post = posteriors[step + 1]
            next_mean = target_means[step + 1]
            next_var = target_vars[step + 1]

            z_current = current_mean + torch.sqrt(current_var) * torch.randn_like(current_mean)
            z_next = to_channel_last(next_post.sample())

            if step == 0:
                τ = torch.rand(b, device=current_mean.device) * (1.0 - eps)
            else:
                τ = torch.distributions.Beta(
                    current_mean.new_tensor(1.0),
                    current_mean.new_tensor(3.0),
                ).sample((b,)).to(current_mean.device)
            τ = τ.clamp(min=eps, max=1.0 - eps)
            τ_w = self.wide(τ, ref=current_mean)
            Δ = self.wide(1 - τ, ref=current_mean)

            μ_τ = I_τ(τ_w, current_mean, next_mean)
            Σ_τ = (I2_τ(τ_w, current_var, next_var) + Λ(τ_w)).clamp_min(self.var_eps)
            z_τ = I_τ(τ_w, z_current, z_next) + torch.randn_like(z_current) * torch.sqrt(Λ(τ_w))

            γ_τ = τ_w * next_var / Σ_τ
            μ_target = next_mean + (z_τ - μ_τ) * γ_τ
            Σ_target_raw = next_var - τ_w * γ_τ * next_var
            Σ_target = Σ_target_raw.clamp_min(self.var_eps)

            model_in = torch.cat([z_current, z_τ], dim=-1)
            output = model(model_in, τ.view(-1, 1), **kwargs)
            if not isinstance(output, tuple):
                raise ValueError("Model output must be a tuple of (velocity, sigma_coef).")

            drift, sigma_coef_pred = output
            diffusion = self._sigma_from_raw(sigma_coef_pred)
            v_target = μ_target - z_τ

            loss_drift = F.mse_loss(Δ * drift, v_target, reduction="none")
            reduce_dims = tuple(range(1, loss_drift.ndim))
            loss_drift = loss_drift.mean(dim=reduce_dims)

            loss_diff_mse = F.mse_loss(Δ * diffusion.pow(2), Σ_target, reduction="none")
            reduce_dims = tuple(range(1, loss_diff_mse.ndim))
            loss_diff_mse = loss_diff_mse.mean(dim=reduce_dims)

            loss_ed = self.energy_distance(
                z_τ,
                Δτ=1 - τ,
                drift=drift,
                diffusion=diffusion,
                μ_target=μ_target,
                Σ_target=Σ_target,
                K=self.K,
                L=self.L,
            )

            loss_drift_terms.append(loss_drift)
            loss_diff_mse_terms.append(loss_diff_mse)
            loss_ed_terms.append(loss_ed)

            ξ_refresh = torch.randn_like(diffusion)
            z_pred_refresh = z_τ + Δ * drift + torch.sqrt(Δ.clamp_min(self.delta_eps)) * diffusion * ξ_refresh

            if train_module is None:
                current_mean = z_pred_refresh
                current_var = torch.zeros_like(z_pred_refresh).clamp_min(self.var_eps)
            else:
                current_mean, current_var = train_module.refresh_posterior_from_prediction(
                    z_pred_refresh, cond=kwargs.get("cond", None)
                )

            if self.detach_rollout:
                current_mean = current_mean.detach()
                current_var = current_var.detach()

            with torch.no_grad():
                def _rms(x: torch.Tensor) -> float:
                    return x.detach().float().pow(2).mean().sqrt().item()

                tau_values.append(τ.mean().item())
                delta_values.append((1.0 - τ).mean().item())
                drift_rms_values.append(_rms(drift))
                vtarget_rms_values.append(_rms(v_target))
                drift_err_rms_values.append(_rms(Δ * drift - v_target))
                diff_amp_rms_values.append(_rms(diffusion))
                g_pred_mean_values.append(diffusion.pow(2).mean().item())
                g_target_mean_values.append(Σ_target.mean().item())
                g_target_min_values.append(Σ_target_raw.min().item())
                g_target_clamped_min_values.append(Σ_target.min().item())
                sigma1_mean_values.append(next_var.mean().item())
                sigma_tau_mean_values.append(Σ_τ.mean().item())
                lambda_mean_values.append(Λ(τ_w).mean().item()) 
                # timeline checkpoint

        step_weights_view = step_weights.view(-1, 1)
        loss_drift = (torch.stack(loss_drift_terms, dim=0) * step_weights_view).sum(dim=0)
        loss_diff_mse = (torch.stack(loss_diff_mse_terms, dim=0) * step_weights_view).sum(dim=0)
        loss_ed = (torch.stack(loss_ed_terms, dim=0) * step_weights_view).sum(dim=0)

        loss_diffusion = loss_diff_mse + self.lambda_ED * loss_ed
        if ED_schedule is not None:
            loss_diffusion = loss_diffusion * ED_schedule

        loss = loss_drift.mean() + loss_diffusion.mean()

        loss_drift_mean = loss_drift.mean().item()
        loss_diff_mse_mean = loss_diff_mse.mean().item()
        loss_ed_mean = loss_ed.mean().item()
        loss_diff_total_mean = loss_diffusion.mean().item()
        loss_total_mean = loss.mean().item()
        ed_frac = (self.lambda_ED * loss_ed_mean) / (loss_diff_total_mean + 1e-12)

        def _avg(xs):
            return sum(xs) / max(len(xs), 1)

        log_dict = {
            "loss_drift": loss_drift_mean,
            "loss_diffusion": loss_diff_total_mean,
            "loss": loss_total_mean,
            "loss_diff_mse": loss_diff_mse_mean,
            "loss_ed": loss_ed_mean,
            "ed_frac": ed_frac,
            "ar_steps": float(rollout_horizon),
            "step_weight_decay": self.step_weight_decay,
            "detach_rollout": float(self.detach_rollout),
            "tau_mean": _avg(tau_values),
            "Delta_mean": _avg(delta_values),
            "drift_rms": _avg(drift_rms_values),
            "vtarget_rms": _avg(vtarget_rms_values),
            "drift_err_rms_w": _avg(drift_err_rms_values),
            "diff_amp_rms": _avg(diff_amp_rms_values),
            "g_pred_mean": _avg(g_pred_mean_values),
            "g_target_mean": _avg(g_target_mean_values),
            "g_target_min": _avg(g_target_min_values),
            "g_target_clamped_min": _avg(g_target_clamped_min_values),
            "Sigma1_mean": _avg(sigma1_mean_values),
            "Sigma_tau_mean": _avg(sigma_tau_mean_values),
            "Lambda_mean": _avg(lambda_mean_values),
        }

        return loss, log_dict
