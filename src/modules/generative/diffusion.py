import numpy as np
import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    def __init__(
        self,
        schedule="linear",
        noise_steps=100,
        num_ddim_steps=10,
        skip_percent=0.0,
        beta_start=0.0001,
        beta_end=0.02,
        scale=400,
        mode="ddpm",
    ):
        super(DiffusionScheduler, self).__init__()

        self.noise_steps = noise_steps
        self.num_ddim_steps = num_ddim_steps
        self.skip_percent = skip_percent
        self.scale = scale
        self.mode = mode  # ddpm, ddim, tsm

        self.schedule = schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.criterion = nn.MSELoss()

        clip_min = 1e-9

        if self.schedule == "cosine":
            s = 0.008
            tlist = torch.arange(1, self.noise_steps + 1, 1)
            temp1 = torch.cos((tlist / self.noise_steps + s) / (1 + s) * np.pi / 2)
            temp1 = temp1 * temp1
            temp2 = np.cos(((tlist - 1) / self.noise_steps + s) / (1 + s) * np.pi / 2)
            temp2 = temp2 * temp2

            self._beta_source = 1 - (temp1 / temp2)
            self._beta_source[self._beta_source > 0.999] = 0.999
            self._beta = torch.cat((torch.tensor([0]), self._beta_source), dim=0)[1:]
            self._alpha = 1 - self._beta
            self._alpha_hat = torch.cumprod(self._alpha, 0)
        elif self.schedule == "simple_linear":
            t = torch.linspace(0, 1, self.noise_steps)
            self._alpha_hat = torch.clip(1 - t, min=clip_min, max=1)
            self._alpha = torch.div(
                self._alpha_hat, torch.cat((torch.tensor([1]), self._alpha_hat[:-1]), 0)
            )
            self._beta = 1 - self._alpha
        else:
            self._beta = self.prepare_noise_schedule()
            self._alpha = 1.0 - self._beta
            self._alpha_hat = torch.cumprod(self._alpha, dim=0)
            self._alpha_hat = torch.clip(self._alpha_hat, min=clip_min, max=1)

        self.register_buffer("beta", self._beta)
        self.register_buffer("alpha", self._alpha)
        self.register_buffer("alpha_hat", self._alpha_hat)

        print(f"Mode: {self.mode}")

    def prepare_noise_schedule(self):
        """
        Creates the noise schedule beta
        """
        if self.schedule == "linear":
            scale = self.scale / self.noise_steps
            self.beta_start = self.beta_start * scale
            self.beta_end = self.beta_end * scale
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.schedule == "cosine":
            timesteps = torch.linspace(0, 1, self.noise_steps)
            betas = self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (
                1 + np.cos(np.pi / 2 * timesteps) ** 2
            )
            return torch.tensor(betas, dtype=torch.float)
        else:
            return NotImplementedError("Noising Schedule Not Implemented!")

    def sample_timesteps(self, n, device="cpu"):
        """
        Returns random outputs for diff_time for training only
        """
        low = int(self.skip_percent * self.noise_steps)
        return torch.randint(low=low, high=self.noise_steps, size=(n,), device=device)

    def wide(self, t):
        return t[:, None, None, None]

    def noise_states(self, x, t):
        """
        Noises a batch of states
        """
        sqrt_alpha_hat = self.wide(torch.sqrt(self.alpha_hat[t]))
        sqrt_one_minus_alpha_hat = self.wide(torch.sqrt(1 - self.alpha_hat[t]))

        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def compute_loss(self, x, y, model, **kwargs):
        t = self.sample_timesteps(x.shape[0], device=x.device)
        skip_t = int(self.skip_percent * self.noise_steps)

        # don't train on skipped steps
        if self.skip_percent > 0:
            indices = torch.nonzero(torch.lt(t, skip_t)).squeeze(-1)
            t[indices] = skip_t

        y_noised, noise = self.noise_states(y, t)
        t_embed = (t.float() / (self.noise_steps - 1)).view(-1, 1)
        predicted_noise, _ = model(
            torch.cat([x, y_noised], dim=-1), t_embed, **kwargs
        )  # 1
        loss = self.criterion(noise, predicted_noise)
        return loss

    def setup_ddim_sampling(self):
        T = self.noise_steps
        N = int(self.num_ddim_steps)  # interpret as NFE = number of transitions
        assert N >= 1

        # N transitions -> N+1 states, include endpoints [T-1, ..., 0]
        ref_arr = np.round(np.linspace(T - 1, 0, N + 1)).astype(int).tolist()

        # remove duplicates (can happen for small T / large N)
        out = [ref_arr[0]]
        for v in ref_arr[1:]:
            if v != out[-1]:
                out.append(v)

        # enforce endpoints
        if out[0] != T - 1:
            out = [T - 1] + out
        if out[-1] != 0:
            out = out + [0]

        self.ref_arr = out


    def sample_ddim(self, initial_cond, model, **kwargs):
        ts = self.ref_arr  # e.g. [99, 50, 0] when num_ddim_steps=2
        n = initial_cond.shape[0]
        device = initial_cond.device

        with torch.no_grad():
            # start from pure noise at the highest noise level
            x = torch.randn_like(initial_cond, device=device)

            for j in range(len(ts) - 1):
                t_int = int(ts[j])
                t_prev_int = int(ts[j + 1])

                t = torch.full((n,), t_int, device=device, dtype=torch.long)
                # normalized time embedding in [0,1]
                t_embed = (t.float() / (self.noise_steps - 1)).view(-1, 1)

                eps, _ = model(torch.cat([initial_cond, x], dim=-1), t_embed, **kwargs)

                a_t = self.wide(self.alpha_hat[t])
                sqrt_a_t = torch.sqrt(a_t)
                sqrt_1ma_t = torch.sqrt(1.0 - a_t)

                # x0 estimate
                x0 = (x - sqrt_1ma_t * eps) / (sqrt_a_t + 1e-12)

                # jump to next timestep (deterministic DDIM, eta=0)
                t_prev = torch.full((n,), t_prev_int, device=device, dtype=torch.long)
                a_prev = self.wide(self.alpha_hat[t_prev])
                sqrt_a_prev = torch.sqrt(a_prev)
                sqrt_1ma_prev = torch.sqrt(1.0 - a_prev)

                x = sqrt_a_prev * x0 + sqrt_1ma_prev * eps

            return x


    def sample(self, initial_cond, model, **kwargs):
        if self.mode == "ddpm" or self.mode == "tsm":
            return self.sample_main(initial_cond, model, **kwargs)
        elif self.mode == "ddim":
            if not hasattr(self, "ref_arr"):
                self.setup_ddim_sampling()
            return self.sample_ddim(initial_cond, model, **kwargs)
        else:
            raise NotImplementedError(f"Sampling method {self.mode} not implemented!")

    def sample_main(self, initial_cond, model, **kwargs):
        """
        Main sample loop for DDPMs & TSMs
        """

        n = initial_cond.shape[0]
        device = initial_cond.device

        with torch.no_grad():
            x = torch.randn_like(initial_cond, device=device)

            denoising_steps = self.noise_steps

            for i in reversed(range(0, denoising_steps)):
                t = (torch.ones(n, device=device) * i).long()

                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                t_embed = (t.float() / (self.noise_steps - 1)).view(-1, 1)
                predicted_noise, _ = model(
                    torch.cat([initial_cond, x], dim=-1), t_embed, **kwargs
                )

                if (
                    i == int(self.skip_percent * denoising_steps)
                    and self.skip_percent > 0
                ):
                    sqrt_alpha_hat = self.wide(torch.sqrt(self.alpha_hat[t]))
                    sqrt_one_minus_alpha_hat = self.wide(
                        torch.sqrt(1 - self.alpha_hat[t])
                    )

                    x = (
                        x - sqrt_one_minus_alpha_hat * predicted_noise
                    ) / sqrt_alpha_hat
                    break

                alpha = self.wide(self.alpha[t])
                alpha_hat = self.wide(self.alpha_hat[t])
                beta = self.wide(self.beta[t])

                x = (
                    1
                    / torch.sqrt(alpha)
                    * (
                        x
                        - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                    )
                    + torch.sqrt(beta) * noise
                )

        return x
