import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class AdaptiveIntegrator:
    """
    Integrates the SDE using the model's predicted adaptive drift and diffusion.
    Equation: z_{τ+dτ} = z_t + v_pred * dt + sigma_pred * sqrt(dt) * noise

    Deterministic: "euler", "heun". Stochastic: "em" (Euler-Maruyama), "milstein".
    """

    def __init__(self, method="em"):
        self.method = method

    def _evaluate_model(self, x_cond, state, model, t_scalar, transform, **kwargs):
        t_batch = t_scalar.expand(x_cond.shape[0]).unsqueeze(-1)
        model_in = torch.cat((x_cond, state), dim=-1)
        drift, sigma_raw = model(model_in, t_batch, **kwargs)
        return drift, transform(sigma_raw)

    @staticmethod
    def _format_sigma(sigma):
        return sigma.squeeze(2) if sigma.dim() == 3 else sigma

    def _milstein_step(self, x, x_cond, model, dt, t_current, transform, **kwargs):
        sqrt_dt = dt.sqrt()
        dW = sqrt_dt * torch.randn_like(x)

        f, g = self._evaluate_model(x_cond, x, model, t_current, transform, **kwargs)
        g = self._format_sigma(g)

        z_bar = x + g * sqrt_dt
        _, g_bar = self._evaluate_model(x_cond, z_bar, model, t_current, transform, **kwargs)
        g_bar = self._format_sigma(g_bar)

        milstein = 0.5 * (g_bar - g) / sqrt_dt * (dW ** 2 - dt)
        return x + f * dt + g * dW + milstein

    def step_fn(self, x, x_cond, model, dt, method=None, t_current=None, t_next=None,
                transform=None, **kwargs):
        if method is None:
            method = self.method
        if transform is None:
            raise ValueError("AdaptiveIntegrator.step_fn requires a diffusion transform.")

        if method == "milstein":
            if t_current is None:
                raise ValueError("Milstein integration requires the current integration time.")
            return self._milstein_step(x, x_cond, model, dt, t_current, transform, **kwargs)

        drift, sigma = self._evaluate_model(x_cond, x, model, t_current, transform, **kwargs)
        sigma = self._format_sigma(sigma)
        # Drawn for the deterministic methods too, so the RNG stream (and with it
        # every later stochastic sample) matches runs made before heun was added.
        dW = torch.sqrt(dt) * torch.randn_like(x, device=x.device)
        if method == "em":  # Euler-Maruyama
            # Sigma is adaptive (predicted by model), not a fixed scalar scheduler
            return x + dt * drift + sigma * dW
        elif method == "euler":  # ODE (Deterministic sampling)
            return x + dt * drift
        elif method == "heun":  # ODE, same scheme as the flow-matching baseline
            if t_next is None or t_next >= 1.0:  # prevent irregularity at last time
                return x + dt * drift
            x1 = x + dt * drift
            drift1, _ = self._evaluate_model(x_cond, x1, model, t_next, transform, **kwargs)
            return x + 0.5 * dt * (drift + drift1)
        raise ValueError(
            f"Unknown integrator method {method!r}; expected euler, heun, em or milstein."
        )

    def integrate(self, x, y, model, timesteps, method=None, transform=None, **kwargs):
        # y is the current state (starts at x)
        current_state = y

        for i_t in range(len(timesteps) - 1):
            t_current = timesteps[i_t]
            t_next = timesteps[i_t + 1]
            dt = t_next - t_current

            current_state = self.step_fn(
                current_state,
                x_cond=x,
                model=model,
                dt=dt,
                method=method,
                t_current=t_current,
                t_next=t_next,
                transform=transform,
                **kwargs,
            )

        return current_state


class AdaptiveScheduler(nn.Module):
    def __init__(
        self,
        integrator_steps,
        sigma_scale=1.0,  # global scaler for the bridge noise
        lambda_ED=1e-3,
        K=10,
        L=10,
        sigma_min=1e-4,
        sigma_max=None,
        var_eps=1e-8,
        delta_eps=1e-6,
        **kwargs,
    ):
        super(AdaptiveScheduler, self).__init__()
        self.integrator_steps = integrator_steps
        self.sigma_scale = sigma_scale
        self.integrator = AdaptiveIntegrator(method="em")
        self.lambda_ED = lambda_ED
        self.K = K
        self.L = L
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.var_eps = var_eps
        self.delta_eps = delta_eps

        print(f"AdaptiveScheduler initialized. Integrator steps: {integrator_steps}")

    def _sigma_from_raw(self, raw_sigma):
        sigma = torch.nn.functional.softplus(raw_sigma) + self.sigma_min
        if self.sigma_max is not None:
            sigma = sigma.clamp_max(self.sigma_max)
        return sigma

    def wide(self, t, ref):
        return t.reshape(t.shape[0], *([1] * (ref.dim() - 1)))

    def energy_distance(self, z_τ, Δτ, drift, diffusion, μ_target, Σ_target, K, L):
        """
        Energy-distance:
            D^2(P,Q) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||
        where X ~ P (forecast samples) and Y ~ Q (target distribution samples).
        """

        def offdiag_mean(d: torch.Tensor) -> torch.Tensor:
            # d: (B, n, n) with zeros on the diagonal from cdist(x,x)
            B, n, _ = d.shape
            if n <= 1:
                return d.new_zeros((B,))
            # Since diag is 0, sum over all entries == sum over off-diagonal entries
            d.diagonal(dim1=-2, dim2=-1).fill_(0)
            return d.sum(dim=(1, 2)) / (n * (n - 1))

        z_τ_K = repeat(z_τ, "b ... c -> (b k) ... c", k=K)

        # Target cloud

        μ_target_L = repeat(μ_target, "b ... c -> b l ... c", l=L)
        Σ_target_L = repeat(Σ_target, "b ... c -> b l ... c", l=L).clamp_min(self.var_eps)
        ε = torch.randn_like(μ_target_L)

        z_target_L = μ_target_L + torch.sqrt(Σ_target_L) * ε

        # Pred cloud

        drift = repeat(drift, "b ... c -> (b k) ... c", k=K)
        diffusion = repeat(diffusion, "b ... c -> (b k) ... c", k=K)

        ξ = torch.randn_like(diffusion)

        Δτ_safe = Δτ.clamp_min(self.delta_eps)
        Δτ_w = repeat(self.wide(Δτ_safe, ref=z_τ), "b ... -> (b k) ...", k=K)

        z_preds_K = z_τ_K + Δτ_w * drift + torch.sqrt(Δτ_w) * diffusion * ξ

        X = rearrange(z_preds_K, "(b k) ... c -> b k ( ... c )", k=K)  # (B, K, D)
        Y = rearrange(z_target_L, "b l ... c -> b l ( ... c )", l=L)  # (B, L, D)

        D = X.shape[-1]
        scale = float(D) ** 0.5

        d_xy = torch.cdist(X, Y, p=2)  # (B, K, L)
        d_xx = torch.cdist(X, X, p=2)  # (B, K, K) includes diagonal zeros
        d_yy = torch.cdist(Y, Y, p=2)  # (B, L, L) includes diagonal zeros

        d_xy = d_xy / scale
        d_xx = d_xx / scale
        d_yy = d_yy / scale

        term_xy = d_xy.mean(dim=(1, 2))  # estimates E||X - Y||
        term_xx = offdiag_mean(d_xx)  # estimates E||X - X'|| (i != j)
        term_yy = offdiag_mean(d_yy)  # estimates E||Y - Y'|| (i != j)

        energy = 2.0 * term_xy - term_xx - term_yy  # (B,)

        return energy / torch.sqrt(Δτ_safe)

    def compute_loss(self, post0, post1, model, ED_schedule=None, **kwargs):
        """
        Computes KL Divergence loss between the Model's predicted step
        and the Analytic Bridge step.
        x: Start frame (u_t)
        y: End frame (u_{t+1})
        """
        def to_channel_last(x):
            return rearrange(x, "b c h w -> b h w c")

        μ0, μ1 = to_channel_last(post0.mean), to_channel_last(post1.mean)
        Σ0 = to_channel_last(post0.var).clamp_min(self.var_eps)
        Σ1 = to_channel_last(post1.var).clamp_min(self.var_eps)
        z0, z1 = to_channel_last(post0.sample()), to_channel_last(post1.sample())
        
        b = μ0.shape[0]
        ε = 1e-3

        ### Interpolation Functions

        def α(t):
            return 1 - t

        def β(t):
            return t
        
        def I_τ(t, x0, x1):
            return α(t) * x0 + β(t) * x1
        
        def I2_τ(t, x0, x1):
            return α(t)**2 * x0 + β(t)**2 * x1
        
        def dIdτ(x0, x1):
            return x1 - x0
        
        def Λ(t):
            return (self.sigma_scale * t * (1 - t)).clamp_min(self.var_eps)
        

        ####################################

        # 1. Sample time τ
        τ = torch.rand(b, device=μ0.device) * (1.0 - ε)  # Avoid τ=1
        τ = τ.clamp_min(ε)
        τ_w = self.wide(τ, ref=μ0)
        Δ = self.wide(1 - τ, ref=μ0)

         
        # Moments of the stochastic interpolant
        μ_τ = I_τ(τ_w, μ0, μ1)              # mean path
        Σ_τ = (I2_τ(τ_w, Σ0, Σ1) + Λ(τ_w)).clamp_min(self.var_eps)


        # The stochastic interpolant

        z_τ = I_τ(τ_w, z0, z1) + torch.randn_like(z0) * torch.sqrt(Λ(τ_w))

        γ_τ = τ_w * Σ1 / Σ_τ

        μ_target = μ1 + (z_τ - μ_τ) * γ_τ
        Σ_target_raw = Σ1 - τ_w * γ_τ * Σ1
        Σ_target = Σ_target_raw.clamp_min(self.var_eps)
        
        # 5. Model Prediction
        # Model sees: x (start), z_t (current), τ
        model_in = torch.cat([z0, z_τ], dim=-1)

        output = model(model_in, τ.view(-1, 1), **kwargs)

        if isinstance(output, tuple):
            v_pred, sigma_coef_pred = output
        else:
            raise ValueError("Model output must be a tuple of (velocity, sigma_coef)")

        # 6. LEARNING OBJECTIVES
        # Drift Loss
        drift = v_pred
        v_target = μ_target - z_τ
        loss_drift = F.mse_loss(Δ * drift, v_target)

        # Diffusion Loss
        diffusion = self._sigma_from_raw(sigma_coef_pred)
        loss_diffusion1 = F.mse_loss(Δ * diffusion.pow(2), Σ_target)

        # Energy distance
        loss_diffusion2 = self.energy_distance(
            z_τ,
            Δτ = 1 - τ,
            drift = drift,
            diffusion = diffusion,
            μ_target = μ_target,
            Σ_target = Σ_target,
            K = self.K,
            L = self.L,
        )

        λ_ED = self.lambda_ED
        loss_diffusion = loss_diffusion1 + λ_ED * loss_diffusion2

        if ED_schedule is not None:
            loss_diffusion = loss_diffusion * ED_schedule

        loss = loss_drift.mean() + loss_diffusion.mean()

        def _rms(x: torch.Tensor) -> float:
            return x.detach().float().pow(2).mean().sqrt().item()

        with torch.no_grad():
            # Scalars per batch element
            Δτ = 1.0 - τ  # (B,)
            # Λ(τ) is in variance units, broadcasted
            Λ_τ = Λ(τ_w)

            # Useful scale summaries
            Δ_mean = Δτ.mean().item()
            Δ2_mean = (Δτ.pow(2)).mean().item()

            # Drift / target magnitudes
            drift_rms = _rms(drift)
            vtarget_rms = _rms(v_target)
            drift_err_rms_weighted = _rms(Δ * drift - v_target)

            # Diffusion / variance magnitudes
            diff_amp_rms = _rms(diffusion)                 # amplitude
            g_pred_mean = diffusion.pow(2).mean().item()   # predicted variance
            g_target_mean = Σ_target.mean().item()
            g_target_min = Σ_target_raw.min().item()
            g_target_clamped_min = Σ_target.min().item()
            Σ1_mean = Σ1.mean().item()
            Στ_mean = Σ_τ.mean().item()
            Λ_mean = Λ_τ.mean().item()

            # ED contribution bookkeeping
            loss_drift_mean = loss_drift.mean().item()
            loss_diff_mse_mean = loss_diffusion1.mean().item()
            loss_ed_mean = loss_diffusion2.mean().item()
            loss_diff_total_mean = loss_diffusion.mean().item()
            loss_total_mean = loss.mean().item()

            ed_frac = (λ_ED * loss_ed_mean) / (loss_diff_total_mean + 1e-12)

            log_dict = {
                # Main losses
                "loss_drift": loss_drift_mean,
                "loss_diffusion": loss_diff_total_mean,
                "loss": loss_total_mean,

                # Decompose diffusion loss
                "loss_diff_mse": loss_diff_mse_mean,
                "loss_ed": loss_ed_mean,
                "ed_frac": ed_frac,

                # Time/weighting stats
                "tau_mean": τ.mean().item(),
                "tau_min": τ.min().item(),
                "tau_max": τ.max().item(),
                "Delta_mean": Δ_mean,
                "Delta2_mean": Δ2_mean,

                # Drift scale diagnostics
                "drift_rms": drift_rms,
                "vtarget_rms": vtarget_rms,
                "drift_err_rms_w": drift_err_rms_weighted,

                # Diffusion/variance scale diagnostics
                "diff_amp_rms": diff_amp_rms,
                "g_pred_mean": g_pred_mean,
                "g_target_mean": g_target_mean,
                "g_target_min": g_target_min,
                "g_target_clamped_min": g_target_clamped_min,
                "Sigma1_mean": Σ1_mean,
                "Sigma_tau_mean": Στ_mean,
                "Lambda_mean": Λ_mean,
            }


        return loss.mean(), log_dict

    def sample(self, x, model, integrator_steps=None, method=None, **kwargs):
        if integrator_steps is None:
            integrator_steps = self.integrator_steps

        timesteps = torch.linspace(0, 1, integrator_steps + 1, device=x.device)

        # Start at x (t=0)
        y = x.clone()

        # Integrate
        y = self.integrator.integrate(x, y, model, timesteps, method=method, transform=self._sigma_from_raw, **kwargs)

        return y

    def forward(self, x, y, model, **kwargs):
        return self.compute_loss(x, y, model, **kwargs)
