import torch
import torch.nn as nn


class ODEIntegrator:
    def __init__(
        self,
        method="euler",  # 'euler' or 'heun' or 'midpoint'
    ):
        self.method = method
        assert method in ["euler", "heun", "midpoint"], "Method not implemented"

    def step_fn(self, x, fn, dt, ts, **model_kwargs):
        method = self.method
        if ts[1] == 0 and self.method == "heun":  # prevent irregularity at last time
            method = "euler"
        if method == "euler":
            return x + dt * fn(x, ts[0], **model_kwargs)
        elif method == "heun":
            dx = fn(x, ts[0], **model_kwargs)
            x1 = x + dt * dx
            return x + 0.5 * dt * (dx + fn(x1, ts[1], **model_kwargs))
        elif method == "midpoint":
            x1 = x + 0.5 * dt * fn(x, ts[0], **model_kwargs)
            return x + dt * fn(x1, ts[1], **model_kwargs)

    def integrate(self, x, y, model, stencils, timesteps, **kwargs):
        x_cond = x

        def model_wrapper_fn(x_state, t_state, **kw):
            B = x_state.shape[0]
            t = t_state.to(x_state.device).float().view(1, 1).expand(B, 1)
            out = model(torch.cat([x_cond, x_state], dim=-1), t, **kw)
            if isinstance(out, (tuple, list)):
                drift, _ = out
                return drift
            return out

        for i_t in range(len(stencils) - 1):
            t_current = stencils[i_t]  # sigma_t
            t_next = stencils[i_t + 1]  # sigma_t+1
            dt = t_next - t_current  # (sigma_t+1 - sigma_t)
            if self.method != "midpoint":
                y = self.step_fn(
                    y,
                    model_wrapper_fn,
                    dt,
                    [stencils[i_t], stencils[i_t + 1]],
                    **kwargs,
                )
            else:
                y = self.step_fn(
                    y,
                    model_wrapper_fn,
                    dt,
                    [stencils[i_t], 0.5 * (stencils[i_t + 1] + stencils[i_t])],
                    **kwargs,
                )
        return y


class LinearScheduler(nn.Module):
    def __init__(
        self,
        num_refinement_steps,  # this corresponds to physical time steps
        num_train_steps=None,  # number of training steps
        integrator="euler",  # 'euler' or 'heun' or 'midpoint', worth noting that this only available for flow
    ):
        super(LinearScheduler, self).__init__()

        # for flow matching, the min_noise_std is not used
        self.num_train_timesteps = (
            num_train_steps if num_train_steps is not None else num_refinement_steps + 1
        )
        self.num_refinement_steps = num_refinement_steps
        self.sigmas = torch.linspace(0, 1, steps=self.num_train_timesteps)

        self.num_refinement_steps = num_refinement_steps
        self.ode_integrator = ODEIntegrator(method=integrator)

        self.training_criterion = nn.MSELoss()

        print(
            f"Using LinearScheduler with {self.num_train_timesteps} training steps and {self.num_refinement_steps} refinement steps."
        )

    def get_noise(self, size, device):
        return torch.randn(size, device=device)

    def compute_loss(self, x, y, model, eval=False, **kwargs):
        # x: [b nx ny d], conditioning. For PDEs this is u(t)
        # y: [b nx ny d], label. For PDEs this is u(t+dt)
        # cond: [b cond_dim]

        noise = self.get_noise(size=y.shape, device=y.device).to(y.dtype)

        # no need to train on k=0
        B = x.shape[0]
        sigma_scalar = torch.rand(B, device=x.device, dtype=torch.float32).clamp_max(1.0 - 1e-3)
        alpha_scalar = 1.0 - sigma_scalar

        # reshape for mixing (use y.dtype for AMP consistency)
        alpha_mix = alpha_scalar.to(y.dtype).view(-1, *[1 for _ in range(y.ndim - 1)])
        sigma_mix = sigma_scalar.to(y.dtype).view(-1, *[1 for _ in range(y.ndim - 1)])

        # Noise the label y
        y_noised = alpha_mix * y + sigma_mix * noise

        u_in = torch.cat([x, y_noised], dim=-1)

        # time input must be (B,1), keep float32
        t_in = sigma_scalar.view(-1, 1)
        pred = model(u_in, t_in, **kwargs)
        
        if isinstance(pred, (tuple, list)):
            pred, _ = pred

        target = noise - y  # predict eps - y
        loss = self.training_criterion(pred, target)
        if eval:
            return loss, pred, target
        return loss

    def sample(self, x, model, refinement_steps=None, integrator_method=None, **kwargs):
        if refinement_steps is None:
            refinement_steps = self.num_refinement_steps

        # x: [b nlat nlon d]
        y_noised = self.get_noise(x.shape, device=x.device).to(x.dtype)

        # Use a continuous stencil with exactly `refinement_steps` integration steps.
        # This decouples inference NFE from the training grid (num_train_timesteps).
        sigmas = torch.linspace(
            1.0, 0.0, steps=int(refinement_steps) + 1, device=x.device, dtype=x.dtype
        )
        timesteps = None  # not used by the integrator

        # currently does not support noising input
        integrator = self.ode_integrator
        if integrator_method is not None:
            integrator = ODEIntegrator(method=integrator_method)
        y_noised = integrator.integrate(x, y_noised, model, sigmas, timesteps, **kwargs)

        y = y_noised
        return y

    def forward(self, x, y, model, **kwargs):
        return self.compute_loss(x, y, model, **kwargs)
