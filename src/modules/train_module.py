import copy
import os
from pathlib import Path

import lightning as L
import torch
from einops import rearrange

from common.loss import ScaledLpLoss, lat_weighted_rmse
from common.plotting import plot_result_2d, plot_result_map
from common.utils import get_yaml
from modules.generative.diffusion import DiffusionScheduler
from modules.generative.flow_matching import LinearScheduler
from modules.generative.interpolant import DriftScheduler
from modules.generative.tornado import AdaptiveScheduler
from modules.generative.tornado_AR import AdaptiveARScheduler
from modules.models.DiT import DIT
from modules.models.DiT_dropout import DiTDropout
from modules.baselines.gaussian_stepper import GaussianStepperScheduler
from modules.baselines.mc_dropout import MCDropoutScheduler
from modules.baselines.moment_stepper import MomentStepperScheduler


def _resolve_yaml_path(base_dir: Path, stem: str = "config") -> Path:
    candidates = [base_dir / f"{stem}.yml", base_dir / f"{stem}.yaml"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]




def _with_legacy_names(config: dict) -> dict:
    """Map runs saved under TORNADO's legacy name, `cone`, onto `tornado`.

    Older runs' config.yml (and their checkpoints' hparams) still say
    `model_name: cone` with a `model.cone` block. Returns a copy, so a caller that
    keeps editing `config["model"]["cone"]` after building the module is unaffected.
    """
    if config["model"].get("model_name") != "cone":
        return config
    config = copy.deepcopy(config)
    config["model"]["model_name"] = "tornado"
    config["model"]["tornado"] = config["model"].pop("cone")
    return config


class TrainModule(L.LightningModule):
    def __init__(self, config: dict, normalizer=None):
        """
        TrainModule
        args:
            config (dict): configuration dictionary containing model, training and data configurations
            normalizer (object, optional): normalizer object for scaling input data. Defaults to None.
        """

        super().__init__()
        config = _with_legacy_names(config)
        self.config = config
        self.modelconfig = config["model"]
        self.model_name = self.modelconfig["model_name"]
        self.lr = self.modelconfig["lr"]
        self.log_dir = config["training"]["log_dir"]
        self.pde = config["data"]["pde"]
        self.normalizer = normalizer

        self.criterion = ScaledLpLoss()

        self.latent = self.modelconfig.get("latent", False)

        # Regularization schedulers
        self.warmup_epochs = self.modelconfig.get("warmup_epochs", 0)
        self.ramp_epochs = self.modelconfig.get("ramp_epochs", 1)
        self.lambda_max = self.modelconfig.get("lambda_max", 1.0)

        ditconfig = self.config["dit"]

        def _build_model(config, is_dropout=False):
            if is_dropout:
                return DiTDropout(**config)
            return DIT(**config)

        # Configure model parameters
        if self.model_name in ("tornado", "tornado_ar", "gaussian_nll", "moment_stepper"):
            ditconfig["tornado"] = True
        elif self.model_name == "mc_dropout":
            mc_config = self.modelconfig.get("mc_dropout", {})
            ditconfig["dropout"] = mc_config.get("dropout", 0.1)
            ditconfig["in_dim"] = ditconfig["out_dim"]  # single z input, not concatenated pair

        # Build model
        self.model = _build_model(ditconfig, is_dropout=(self.model_name == "mc_dropout"))

        # Build scheduler
        if self.model_name in ("ddpm", "ddim"):
            self.scheduler = DiffusionScheduler(mode=self.model_name, **self.modelconfig[self.model_name])
        elif self.model_name == "flow_matching":
            self.scheduler = LinearScheduler(**self.modelconfig["flow_matching"])
        elif self.model_name == "interpolant":
            self.scheduler = DriftScheduler(**self.modelconfig["interpolant"])
        elif self.model_name == "tornado":
            self.scheduler = AdaptiveScheduler(**self.modelconfig["tornado"])
        elif self.model_name == "tornado_ar":
            self.scheduler = AdaptiveARScheduler(**self.modelconfig["tornado_ar"])
        elif self.model_name == "gaussian_nll":
            self.scheduler = GaussianStepperScheduler(**self.modelconfig.get("gaussian_nll", {}))
        elif self.model_name == "moment_stepper":
            self.scheduler = MomentStepperScheduler(**self.modelconfig.get("moment_stepper", {}))
        elif self.model_name == "mc_dropout":
            self.scheduler = MCDropoutScheduler()
        else:
            self.model = None
            raise NotImplementedError(f"Model {self.model_name} not implemented")

        if self.model_name in ("tornado", "tornado_ar", "gaussian_nll", "moment_stepper"):
            assert getattr(self.model, "tornado", False) is True
        else:
            assert getattr(self.model, "tornado", False) is False

        if self.latent:
            self.init_autoencoder()

        if self.model_name == "tornado_ar":
            self._maybe_load_tornado_ar_checkpoint()
        else:
            self._maybe_warm_start()

        if (
            config["training"]["strategy"] == "ddp"
            or config["training"]["strategy"] == "ddp_find_unused_parameters_true"
        ):
            self.ddp = True
        else:
            self.ddp = False

        self.save_hyperparameters()

    def init_autoencoder(self):
        from modules.ae_module import AutoencoderModule

        aeconfig = self.config["data"]["autoencoder"]
        base_path = aeconfig["checkpoint_path"]
        run = aeconfig["run"]
        ckpt_paths = Path(base_path) / run
        ckpt_type = str(aeconfig.get("type", "last")).strip().lower()
        patterns = {"last": "last*", "best": "*_best.ckpt"}
        if ckpt_type not in patterns:
            raise ValueError(
                f"Unsupported data.autoencoder.type='{aeconfig.get('type')}'. Expected 'last' or 'best'."
            )

        matches = sorted(ckpt_paths.glob(patterns[ckpt_type]))
        if len(matches) == 0:
            # Fall back to the other checkpoint type if the preferred one is absent.
            other = "best" if ckpt_type == "last" else "last"
            other_matches = sorted(ckpt_paths.glob(patterns[other]))
            if len(other_matches) > 0:
                print(
                    f"[AE] no '{ckpt_type}' checkpoint in {ckpt_paths.name}; "
                    f"falling back to '{other}' ({other_matches[0].name})"
                )
                matches, ckpt_type = other_matches, other

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No autoencoder checkpoint of type='{ckpt_type}' found in {ckpt_paths}. "
                f"Available last*: {[p.name for p in sorted(ckpt_paths.glob('last*'))]}, "
                f"available *_best.ckpt: {[p.name for p in sorted(ckpt_paths.glob('*_best.ckpt'))]}"
            )

        ckpt_path = matches[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        aeconfig = get_yaml(str(_resolve_yaml_path(ckpt_paths)))
        self.autoencoder = AutoencoderModule(aeconfig, normalizer=self.normalizer)
        self.autoencoder.load_state_dict(checkpoint["state_dict"])

        # freeze the autoencoder
        self.autoencoder = self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.scale_factor = aeconfig.get("scale_factor", 1.0)

    def forward(
        self, u, cond=None, method_val=None
    ):
        # generative models
        if hasattr(self, "scheduler"):
            if method_val is None:
                if self.model_name in ("tornado", "tornado_ar"):
                    lam_roll = self.lambda_roll()
                    if lam_roll > 0.0:
                        method_val = "em"
                    else:
                        method_val = "euler"
                    return self.scheduler.sample(
                        u, self.model, cond=cond, method=method_val
                    )
                elif self.model_name in ("gaussian_nll", "moment_stepper"):
                    # u is already a (mu, sigma) tuple during autoregressive rollout
                    return self.scheduler.sample(u, self.model, cond=cond)
                else:
                    return self.scheduler.sample(u, self.model, cond=cond)
            else:
                if self.model_name == "flow_matching":
                    return self.scheduler.sample(
                        u, self.model, cond=cond, integrator_method=method_val
                    )
                return self.scheduler.sample(
                    u, self.model, cond=cond, method=method_val
                )

        # deterministic models
        else:
            return self.model(u, cond)

    def encode(self, u, cond=None, return_posterior=False):
        """
        Encode input to latent space using the autoencoder.
        Args:
            u_input (torch.Tensor): Input tensor to be encoded.
        Returns:
            torch.Tensor: Encoded latent representation.
        """
        if self.latent is False:
            return u

        with torch.no_grad():
            z, posterior = self.autoencoder.encode(u, cond)
            z = z * self.scale_factor  # scale the latent representation
            z = rearrange(z, "b c nx ny -> b nx ny c")

        return (z, posterior) if return_posterior else z

    def decode(self, z, cond=None):
        """
        Decode latent representation back to input space using the autoencoder.
        Args:
            z (torch.Tensor): Latent representation to be decoded.
        Returns:
            torch.Tensor: Decoded input tensor.
        """
        if self.latent is False:
            return z

        with torch.no_grad():
            z = rearrange(z, "b nx ny c -> b c nx ny")
            u = self.autoencoder.decode(z / self.scale_factor, cond)
        return u

    def refresh_posterior_from_prediction(self, z_pred, cond=None):
        if self.latent is False:
            return z_pred, torch.zeros_like(z_pred)

        z_decode = rearrange(z_pred, "b nx ny c -> b c nx ny")

        u_refresh = self.autoencoder.decode(z_decode / self.scale_factor, cond)
        _, refreshed_post = self.autoencoder.encode(u_refresh, cond)

        mean_refreshed = rearrange(refreshed_post.mean, "b c h w -> b h w c")
        var_refreshed = rearrange(refreshed_post.var, "b c h w -> b h w c")

        return mean_refreshed, var_refreshed.clamp_min(1e-8)

    @staticmethod
    def get_time_cond(batch):
        """Per-frame date conditioning, [B, nt, D], or None if the dataset has none.

        ERA5 is the only dataset here whose conditioning *advances along the
        rollout*: the seasonal and diurnal phase at step 40 of a 10-day forecast is
        not the phase at step 0. Every other dataset conditions on run-level
        constants, which is why `constant_scalars` is enough for them.
        """
        cond = batch.get("time_scalars", None)
        return cond.float() if cond is not None else None

    @staticmethod
    def _normalize_combustion_cond(cond):
        """Map raw (NH3 blend %, phi) onto ~[0, 1] for the sinusoidal embedder.

        The two parameters live on very different scales -- blend spans 0-80,
        phi only 0.75-1.3 -- so feeding them raw would let blend dominate the
        embedding while phi barely moved it. These are fixed affine maps rather
        than dataset statistics, so they stay identical across train/val and
        across any subset of runs. Not done in the dataset, so `constant_scalars`
        stays in physical units for logging and analysis.
        """
        blend = cond[:, 0] / 100.0        # 0-80 %   -> [0.00, 0.80]
        phi = (cond[:, 1] - 0.7) / 0.7    # 0.75-1.3 -> [0.07, 0.86]
        return torch.stack([blend, phi], dim=-1)

    def get_data(self, batch, val=False):
        u_label = batch["output_fields"]  # shape possibly (B, nt, ..., C)

        cond = batch.get("constant_scalars", None)  # b num_cond
        if self.pde == "real_combustion" and cond is not None:
            cond = self._normalize_combustion_cond(cond)
        elif self.pde == "era5" and cond is not None:
            # Already the bounded sin/cos date encoding; see dataset/era5.py.
            cond = cond.float()

        if val:  # u_label in shape (b, nt, nx, ny, c)
            u_input = u_label[:, 0]  # get first step of trajectory
            return u_input, u_label, cond
        u_input = batch["input_fields"]  # b nx ny c

        return u_input, u_label, cond

    def get_ar_data(self, batch):
        u_out = batch["output_fields"]
        u_in = batch["input_fields"]
        cond = batch.get("constant_scalars", None)

        if self.pde == "era5":
            # tornado_ar rolls a whole window out under a SINGLE `cond`, and both
            # its encode loop and AdaptiveARScheduler.compute_loss take one
            # conditioning vector for every step. On ERA5 the date advances along
            # the window, so that single vector would be the window's first frame
            # and every later step would be trained at the wrong hour and day --
            # silently, with no error and a plausible-looking loss. Refuse rather
            # than report it. ERA5's baseline set is tornado/ddim/fm/interpolant; see
            # bash/era5_baselines.sh.
            raise NotImplementedError(
                "tornado_ar does not support per-step date conditioning, which "
                "ERA5 requires. Use model=tornado, or extend AdaptiveARScheduler to "
                "accept a per-step cond first."
            )

        if u_out.dim() <= u_in.dim():
            raise ValueError("tornado_ar training requires trajectory batches, but the dataset returned single-step samples.")

        traj = u_out
        total_steps = traj.shape[1]
        max_horizon = int(self.modelconfig["tornado_ar"]["horizon"])
        effective_horizon = min(max_horizon, total_steps - 1)

        if effective_horizon < 1:
            raise ValueError(
                f"tornado_ar requires at least 2 trajectory steps, got total_steps={total_steps}."
            )

        if effective_horizon < max_horizon:
            print(
                f"[tornado_ar] Requested horizon={max_horizon}, but only {total_steps - 1} transitions are available. "
                f"Using effective_horizon={effective_horizon}."
            )

        window = effective_horizon + 1
        max_start = total_steps - window
        if max_start > 0:
            start = torch.randint(0, max_start + 1, (1,), device=traj.device).item()
            traj = traj[:, start : start + window]
        else:
            traj = traj[:, :window]

        return u_in, traj, cond, effective_horizon

    def _era5_lat(self):
        """Latitudes of the ERA5 grid, cached on first use."""
        if getattr(self, "_era5_lat_cache", None) is None:
            from common.geo import pole_grid_lat

            nlat = int(
                self.config["data"].get("dataset", {}).get("resolution", [121, 240])[0]
            )
            self._era5_lat_cache = pole_grid_lat(nlat)
        return self._era5_lat_cache

    def lambda_roll(self) -> float:
        """
        Rollout-loss weight as a function of *epoch*:
          - 0 until warmup_epochs
          - linear ramp to lambda_max over ramp_epochs
          - then constant lambda_max
        """
        e = self.current_epoch  

        if e < 0:
            return 0.0

        if e < self.warmup_epochs:
            return 0.0
        elif e < self.warmup_epochs + self.ramp_epochs:
            alpha = (e - self.warmup_epochs) / max(self.ramp_epochs, 1)
            return float(self.lambda_max) * float(alpha)
        else:
            return float(self.lambda_max)

    def training_step(self, batch, batch_idx):
        if self.model_name == "tornado_ar":
            _, u_traj, cond, effective_horizon = self.get_ar_data(batch)
            del batch

            posteriors = []
            for step_idx in range(effective_horizon + 1):
                u_step = u_traj[:, step_idx]
                _, posterior = self.encode(u_step, cond, return_posterior=True)
                posteriors.append(posterior)

            loss, log_dict = self.scheduler.compute_loss(
                posteriors, self.model, train_module=self, cond=cond, ED_schedule=self.lambda_roll()
            )

            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
            for key, value in log_dict.items():
                self.log(
                    f"train/{key}", value, on_step=True, on_epoch=True, sync_dist=self.ddp
                )
            assert loss is not None and torch.is_tensor(loss)
            assert loss.requires_grad

            return loss
        
        u_out = batch["output_fields"]
        u_in = batch["input_fields"]
        cond = batch.get("constant_scalars", None)  # (B, num_cond) or None
        if self.pde == "era5" and cond is not None:
            cond = cond.float()
        time_cond = self.get_time_cond(batch)  # (B, nt, num_cond) or None

        is_trajectory = u_out.dim() > u_in.dim()

        # Conditioning for the source frame (fed to the processor and to the
        # encoder) and for the target frame (the encoder again, since the VAE is
        # date-conditioned and each frame carries its own date).
        cond0 = cond1 = cond

        if is_trajectory:
            B = u_out.shape[0]
            T = u_out.shape[1] - 1

            if T > 0:
                # Sample random timesteps from trajectory
                b_idx = torch.arange(B, device=u_out.device)
                t_idx = torch.randint(0, T, (B,), device=u_out.device)
                u0 = u_out[b_idx, t_idx]
                u1 = u_out[b_idx, t_idx + 1]
                if time_cond is not None:
                    # The window starts at an arbitrary date, so conditioning on
                    # `constant_scalars` (the window's first frame) would hand the
                    # model the wrong hour and day for the pair it actually sees.
                    cond0 = time_cond[b_idx, t_idx]
                    cond1 = time_cond[b_idx, t_idx + 1]
            else:
                # Only 1 timestep in trajectory, use input_fields as u0
                u0 = u_in
                u1 = u_out[:, 0]  # single timestep
                if time_cond is not None:
                    cond0 = cond1 = time_cond[:, 0]
        else:
            # Single-step mode use input_fields and output_fields directly
            u0 = u_in
            u1 = u_out
            if time_cond is not None:
                cond0 = cond1 = time_cond[:, 0]

        del batch

        z0, post0 = self.encode(u0, cond0, return_posterior=True)
        z1, post1 = self.encode(u1, cond1, return_posterior=True)
        cond = cond0
        del u0, u1

        if hasattr(self, "scheduler"):
            if self.model_name == "tornado":
                loss, log_dict = self.scheduler.compute_loss(
                    post0, post1, self.model, cond=cond, ED_schedule=self.lambda_roll()
                )
            elif self.model_name in ("gaussian_nll", "moment_stepper"):
                loss, log_dict = self.scheduler.compute_loss(
                    post0, post1, self.model, cond=cond
                )
            else:
                loss = self.scheduler.compute_loss(z0, z1, self.model, cond=cond)
                log_dict = {}
        else:
            z_pred = self.model(z0, cond)
            loss = self.criterion(z1, z_pred)

        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp)
        for key, value in log_dict.items():
            self.log(
                    f"train/{key}", value, on_step=True, on_epoch=True, sync_dist=self.ddp
                )
        assert loss is not None and torch.is_tensor(loss)
        assert loss.requires_grad

        return loss

    def _maybe_warm_start(self):
        """Initialise the processor from another run's weights (`model.from_checkpoint`).

        A WARM START, not a resume. `training.checkpoint` hands the path to
        `trainer.fit(ckpt_path=...)`, which also restores the epoch counter, the
        optimizer and the LR schedule -- so a run resumed at epoch 800 is already
        past any warmup and `model.warmup_epochs` has no effect. This instead
        copies the weights into a fresh run at epoch 0, which is what makes
        "re-train from these weights with a different loss weighting and a shorter
        ramp" possible.

        Only `model.*` keys are copied. The checkpoint of a latent run also
        contains `autoencoder.*`; loading those would silently overwrite the VAE
        that `init_autoencoder` just resolved from `data.autoencoder.run`, so a
        warm start from a run trained on a *different* VAE would quietly swap the
        autoencoder underneath the config. The scheduler holds no parameters, so
        `model.tornado.lambda_ED` and friends always come from the current config.
        """
        from_checkpoint = self.modelconfig.get("from_checkpoint", None)
        if from_checkpoint in (None, "", "None"):
            return
        ckpt_type = str(self.modelconfig.get("from_checkpoint_type", "best")).lower()

        log_root = Path(self.log_dir).parent
        checkpoint_dir = log_root / from_checkpoint
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"model.from_checkpoint='{from_checkpoint}' not found under {log_root}."
            )

        last_ckpts = sorted(checkpoint_dir.glob("last*.ckpt"))
        best_ckpts = sorted(checkpoint_dir.glob("*_best.ckpt"))
        order = (best_ckpts, last_ckpts) if ckpt_type == "best" else (last_ckpts, best_ckpts)
        ckpt_path = order[0][0] if order[0] else (order[1][0] if order[1] else None)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}.")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        if not state:
            raise ValueError(f"{ckpt_path} contains no 'model.*' weights to warm start from.")
        missing, unexpected = self.load_state_dict(state, strict=False)
        processor_missing = [k for k in missing if k.startswith("model.")]
        if processor_missing:
            raise ValueError(
                f"Warm start from {ckpt_path} left processor weights uninitialised: "
                f"{processor_missing[:5]}... The source run's architecture differs "
                f"from this one; check dit=/in_dim/out_dim."
            )
        print(
            f"[warm start] loaded {len(state)} 'model.*' tensors from {ckpt_path} "
            f"(type={ckpt_type}). Epoch counter and optimizer start fresh; "
            f"autoencoder left as configured."
        )
        if unexpected:
            print(f"[warm start] ignored {len(unexpected)} unexpected keys")

    def _maybe_load_tornado_ar_checkpoint(self):
        from_checkpoint = self.modelconfig["tornado_ar"].get("from_checkpoint", None)
        if from_checkpoint in (None, "", "None"):
            return
        from_checkpoint_type = self.modelconfig["tornado_ar"].get("from_checkpoint_type", "last")

        log_root = Path(self.log_dir).parent
        checkpoint_dir = log_root / from_checkpoint
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Could not locate from_checkpoint='{from_checkpoint}' under log root {log_root}."
            )

        last_ckpts = sorted(checkpoint_dir.glob("last*.ckpt"))
        best_ckpts = sorted(checkpoint_dir.glob("*_best.ckpt"))
        if from_checkpoint_type == "last":
            ckpt_path = last_ckpts[0] if len(last_ckpts) > 0 else (best_ckpts[0] if len(best_ckpts) > 0 else None)
        elif from_checkpoint_type == "best":
            ckpt_path = best_ckpts[0] if len(best_ckpts) > 0 else (last_ckpts[0] if len(last_ckpts) > 0 else None)
        else:
            raise ValueError(
                f"Unsupported tornado_ar.from_checkpoint_type='{from_checkpoint_type}'. Expected 'last' or 'best'."
            )
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint matching type='{from_checkpoint_type}' found in {checkpoint_dir}."
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"[tornado_ar] Loaded checkpoint from {ckpt_path}")
        if len(missing_keys) > 0:
            print(f"[tornado_ar] Missing keys after load: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[tornado_ar] Unexpected keys after load: {unexpected_keys}")

    def validation_step(self, batch, batch_idx, eval: bool = False, z_pred=None):

        u_input, u_label, cond = self.get_data(
            batch, val=True
        )  # u_input: (B, H, W, C), u_label: (B, nt, H, W, C)

        time_cond = self.get_time_cond(batch)  # (B, nt, D) or None

        def step_cond(i):
            """Conditioning for frame `i` of the rollout."""
            return cond if time_cond is None else time_cond[:, min(i, time_cond.shape[1] - 1)]

        nt = u_label.shape[1]

        # Container for denormalized predictions in pixel space: (B, nt, H, W, C)
        u_pred_denorm = torch.zeros_like(u_label)
        u_pred_denorm[:, 0, ...] = self.normalizer.denormalize(u_input)  # store IC

        accumulated_loss = []
        era5_rmse = []

        if self.model_name in ("gaussian_nll", "moment_stepper"):
            # These models work with posterior stats (mu, sigma) tuples
            _, post_encoded = self.encode(u_input, step_cond(0), return_posterior=True)
            mu0 = rearrange(post_encoded.mean, "b c h w -> b h w c")
            sigma0 = rearrange(torch.sqrt(post_encoded.var), "b c h w -> b h w c")
            z_input = (mu0, sigma0)
        else:
            z_input = self.encode(u_input, step_cond(0), return_posterior=False)


        # Autoregressive rollout. Length is capped by data.val_rollout_steps so a
        # long validation window does not turn every val pass into a full-length
        # rollout: each AR step costs an inner solver loop (10 NFE for
        # ddim/fm/interpolant, 5 for tornado) plus one decode. null = full window.
        max_steps = self.config["data"].get("val_rollout_steps", None)
        n_steps = nt - 1 if max_steps is None else min(int(max_steps), nt - 1)

        for i in range(n_steps):
            # The processor advances frame i -> i+1, so it is conditioned on the
            # date of frame i; the decoder reconstructs frame i+1 and is conditioned
            # on that frame's own date.
            z_pred_step = self.forward(z_input, step_cond(i))

            if self.model_name in ("gaussian_nll", "moment_stepper"):
                mu_pred, sigma_pred = z_pred_step

                z_sample = mu_pred + sigma_pred * torch.randn_like(mu_pred)

                z_input = (mu_pred, sigma_pred)

                z_pred_to_decode = z_sample
            else:
                z_pred_to_decode = z_pred_step
                z_input = z_pred_step

            u_true = u_label[:, i + 1]  # (B, ..., C) in pixel space


            if z_pred_step.isnan().any() if torch.is_tensor(z_pred_step) else (z_pred_step[0].isnan().any() or z_pred_step[1].isnan().any()):
                print(f"NaN detected in prediction at step {i + 1}.")
                break

            u_pred_step = self.decode(z_pred_to_decode, step_cond(i + 1))  # (B, ..., C)

            true_denorm = self.normalizer.denormalize(u_true)
            pred_denorm = self.normalizer.denormalize(u_pred_step)

            u_pred_denorm[:, i + 1] = pred_denorm
            loss_t = self.criterion(pred_denorm, true_denorm)
            accumulated_loss.append(loss_t.item())
            if self.pde == "era5":
                era5_rmse.append(
                    lat_weighted_rmse(
                        pred_denorm, true_denorm, self._era5_lat(), lat_dim=1,
                        per_channel=True,
                    ).detach().cpu()
                )

        done = len(accumulated_loss)  # may be < n_steps if a NaN broke the rollout
        loss = sum(accumulated_loss) / max(done, 1)

        # Area-weighted per-variable RMSE, in physical units, averaged over the
        # rollout and reported again at the final lead time. This is the number
        # that is comparable with published WeatherBench scores; val/loss stays the
        # relative error the rest of the repo selects checkpoints on.
        if era5_rmse:
            stacked = torch.stack(era5_rmse)  # (steps, C)
            names = ["t2m", "u10", "v10", "z500"]
            for ci in range(min(len(names), stacked.shape[1])):
                self.log(f"val/rmse_{names[ci]}", stacked[:, ci].mean(),
                         on_step=False, on_epoch=True, sync_dist=self.ddp)
                self.log(f"val/rmse_{names[ci]}_final", stacked[-1, ci],
                         on_step=False, on_epoch=True, sync_dist=self.ddp)

        # Error growth along the rollout: the mean alone hides whether the model
        # is accurate-then-diverging or uniformly mediocre.
        if done > 0:
            self.log("val/loss_step1", accumulated_loss[0], on_step=False,
                     on_epoch=True, sync_dist=self.ddp)
            self.log("val/loss_final", accumulated_loss[-1], on_step=False,
                     on_epoch=True, sync_dist=self.ddp)
            self.log("val/rollout_steps", float(done), on_step=False,
                     on_epoch=True, sync_dist=self.ddp)

        # Dump a ground-truth vs prediction snapshot next to the checkpoints, so
        # progress can be inspected visually rather than only through val/loss.
        if batch_idx == 0 and done > 0 and self.trainer.is_global_zero and not eval:
            os.makedirs(self.log_dir, exist_ok=True)
            plot_fn = plot_result_map if self.pde == "era5" else plot_result_2d
            try:
                k = done + 1  # include the initial condition
                kwargs = {}
                if self.pde == "era5":
                    kwargs = {
                        "lat": self._era5_lat(),
                        "times": batch["time_hours"][0].cpu().numpy()
                        if "time_hours" in batch else None,
                    }
                plot_fn(
                    self.normalizer.denormalize(u_label[:, :k]),
                    u_pred_denorm[:, :k],
                    n_t=min(5, k) if self.pde != "era5" else min(3, k),
                    path=f"{self.log_dir}val_ep_{self.current_epoch:04d}.png",
                    **kwargs,
                )
            except Exception as e:  # plotting must never kill a training run
                print(f"[val snapshot] plotting failed: {type(e).__name__}: {e}")

        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=self.ddp)
      

        

    def on_after_backward(self):
        # Global grad norm over the DiT parameters only (scheduler has no params)
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().float().pow(2).sum()
        grad_norm = torch.sqrt(total)

        self.log(
            "train/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=self.ddp,
        )    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        step_size = 10
        gamma = 0.99
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        return [optimizer], [scheduler]
