import os
import copy
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from common.loss import (
    VRMSE,
    KL_Loss,
    LatWeightedScaledLpLoss,
    ScaledLpLoss,
    lat_weighted_rmse,
    lat_weighted_vrmse,
    sRMSE,
)
from common.plotting import plot_result_2d, plot_result_map
from modules.layers.distributions import DiagonalGaussianDistribution


def to_dict_safe(cfg):
    """Convert config to plain dict, handling both OmegaConf and plain dict inputs."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return copy.deepcopy(cfg)


class AutoencoderModule(L.LightningModule):
    def __init__(self, config, normalizer=None):
        super().__init__()

        self.config = config
        if (
            config["training"]["strategy"] == "ddp"
            or config["training"]["strategy"] == "ddp_find_unused_parameters_true"
        ):
            self.ddp = True
        else:
            self.ddp = False

        self.normalizer = normalizer
        self.lr = config["model"]["lr"]
        self.log_dir = config["training"]["log_dir"]
        self.pde = config["data"]["pde"]

        # Only The Well's datamodule ever filled these, so they now stay at 0 / 1.
        # Kept as buffers because existing conditioned-AE checkpoints contain them.
        num_cond = int(config["data"].get("num_cond", 0))
        if num_cond > 0:
            self.register_buffer("cond_mean", torch.zeros(num_cond))
            self.register_buffer("cond_std", torch.ones(num_cond))
        else:
            self.cond_mean = None
            self.cond_std = None

        # ERA5 is the only rectangular, multi-channel, geo-referenced dataset here,
        # so it gets its own branch: the field is 121 x 240 (odd latitude count,
        # periodic in longitude), the four channels are physically unlike, and every
        # area average has to be latitude weighted.
        self.era5 = self.pde == "era5"
        if self.era5:
            from modules.models.DC_AE import Decoder, Encoder
            from common.geo import pole_grid_lat

            encoderconfig = to_dict_safe(config["model"]["encoder"])
            decoderconfig = to_dict_safe(config["model"]["decoder"])
            kl_weight = config["model"]["kl_weight"]

            # 121 latitudes is not divisible by the 8x spatial reduction of a
            # four-stage DC-AE, so the field is padded to the next multiple before
            # the encoder and cropped back after the decoder. Replicate padding is
            # used (the rows added past the poles repeat the pole row), and the pad
            # is fixed at construction rather than inferred per batch so `decode`
            # crops correctly even when it is called without a matching `encode` --
            # which is exactly what the latent processor does during rollout.
            self.input_resolution = list(config["model"].get("input_resolution", [121, 240]))
            multiple = int(config["model"].get("pad_to_multiple", 8))
            nlat, nlon = self.input_resolution
            pad_h, pad_w = (-nlat) % multiple, (-nlon) % multiple
            self.pad_h = (pad_h // 2, pad_h - pad_h // 2)
            self.pad_w = (pad_w // 2, pad_w - pad_w // 2)
            print(
                f"[era5 AE] input {nlat}x{nlon} -> padded "
                f"{nlat + pad_h}x{nlon + pad_w} (lat pad {self.pad_h}, lon pad "
                f"{self.pad_w}), latent grid "
                f"{(nlat + pad_h) // multiple}x{(nlon + pad_w) // multiple}"
            )

            self.register_buffer(
                "lat_deg",
                torch.as_tensor(pole_grid_lat(nlat), dtype=torch.float32),
                persistent=False,
            )
            self.lat_weighted = bool(config["model"].get("lat_weighted_loss", True))
            criterion_lat = pole_grid_lat(nlat)

            if kl_weight is not None:
                encoderconfig["kl_weight"] = kl_weight
                decoderconfig["kl_weight"] = kl_weight
                self.criterion = (
                    LatWeightedScaledLpLoss(criterion_lat, lat_dim=1)
                    if self.lat_weighted
                    else ScaledLpLoss()
                )
                self.loss = KL_Loss(kl_weight=kl_weight, criterion=self.criterion)
                self.saturate_z = False
                self.kl = True
            else:
                self.criterion = (
                    LatWeightedScaledLpLoss(criterion_lat, lat_dim=1)
                    if self.lat_weighted
                    else ScaledLpLoss()
                )
                self.saturate_z = True
                self.kl = False

            self.encoder = Encoder(**encoderconfig)
            self.decoder = Decoder(**decoderconfig)
            self.kl_warmup_epochs = int(config["model"].get("kl_warmup_epochs", 0))

        # real_combustion is, like km_flow, a single-channel 2D field, so it
        # shares the DC_AE + ScaledLpLoss path.
        elif self.pde in ("km_flow", "real_combustion"):
            from modules.models.DC_AE import Decoder, Encoder

            encoderconfig = to_dict_safe(config["model"]["encoder"])
            decoderconfig = to_dict_safe(config["model"]["decoder"])
            kl_weight = config["model"]["kl_weight"]

            if kl_weight is not None:
                encoderconfig["kl_weight"] = kl_weight
                decoderconfig["kl_weight"] = kl_weight
                # self.criterion = nn.MSELoss(reduction='none')
                self.criterion = ScaledLpLoss()
                self.loss = KL_Loss(kl_weight=kl_weight, criterion=self.criterion)
                self.saturate_z = False
                self.kl = True
            else:
                self.criterion = ScaledLpLoss()
                self.saturate_z = True
                self.kl = False

            self.encoder = Encoder(**encoderconfig)
            self.decoder = Decoder(**decoderconfig)
        else:
            if config["model"]["kl_weight"] == None:
                from modules.models.DC_AE import Decoder, Encoder

                encoderconfig = config["model"]["encoder"]
                decoderconfig = config["model"]["decoder"]

                self.encoder = Encoder(**encoderconfig)
                self.decoder = Decoder(**decoderconfig)

                self.criterion = ScaledLpLoss()
                self.saturate_z = True
                self.kl = False

            else:
                from modules.models.DC_AE import Decoder, Encoder

                encoderconfig = to_dict_safe(config["model"]["encoder"])
                decoderconfig = to_dict_safe(config["model"]["decoder"])
                encoderconfig["kl_weight"] = config["model"]["kl_weight"]
                decoderconfig["kl_weight"] = config["model"]["kl_weight"]
                self.encoder = Encoder(**encoderconfig)
                self.decoder = Decoder(**decoderconfig)
                self.criterion = nn.MSELoss(reduction="none")

                self.loss = KL_Loss(
                    kl_weight=config["model"]["kl_weight"], criterion=self.criterion
                )
                self.saturate_z = False
                self.kl = True

        self.save_hyperparameters()

    def saturate(self, x, B=5.0):
        x = x / torch.sqrt(1 + x**2 / B**2)
        return x

    def _pad_spatial(self, x):
        """Replicate-pad [b, c, nlat, nlon] up to the encoder's stride multiple.

        Longitude is padded circularly (the field wraps) and latitude by
        replication of the pole rows, matching the convolutions' own padding.
        """
        if self.pad_w[0] or self.pad_w[1]:
            x = F.pad(x, (self.pad_w[0], self.pad_w[1], 0, 0), mode="circular")
        if self.pad_h[0] or self.pad_h[1]:
            x = F.pad(x, (0, 0, self.pad_h[0], self.pad_h[1]), mode="replicate")
        return x

    def _crop_spatial(self, x):
        """Inverse of `_pad_spatial`, on [b, c, nlat, nlon]."""
        nlat, nlon = self.input_resolution
        return x[
            ...,
            self.pad_h[0] : self.pad_h[0] + nlat,
            self.pad_w[0] : self.pad_w[0] + nlon,
        ]

    def encode(self, x, cond=None):
        x = x.permute(0, 3, 1, 2)  # b nx ny c -> b c nx ny

        if self.era5:
            x = self._pad_spatial(x)

        z = self.encoder(x, cond)
        posterior = None

        if self.saturate_z:
            z = self.saturate(z)

        if self.kl:
            posterior = DiagonalGaussianDistribution(z)
            z = posterior.sample()

        return z, posterior

    def decode(self, z, cond=None):
        dec = self.decoder(z, cond)

        if self.era5:
            dec = self._crop_spatial(dec)

        dec = dec.permute(0, 2, 3, 1)  # b c nx ny -> b nx ny c
        return dec

    def forward(self, x, cond=None):
        z, posterior = self.encode(x, cond)
        dec = self.decode(z, cond)
        return dec, posterior

    def get_inputs(self, batch):
        inputs = batch["input_fields"]

        cond = batch.get("constant_scalars", None)

        if cond is not None:
            cond = cond.float()

            if self.era5:
                # ERA5's conditioning is already the sin/cos date encoding: bounded
                # in [-1, 1], zero-mean by construction and signed. log1p would be
                # -inf at sin = -1, and standardising it would only undo an encoding
                # that is deliberately fixed across splits.
                return inputs, cond

            cond = torch.log1p(cond)

            if self.cond_mean is not None:
                cond = (cond - self.cond_mean) / (self.cond_std + 1e-6)

            # optional guardrail
            cond = torch.clamp(cond, -5.0, 5.0)

        return inputs, cond
    
    def training_step(self, batch, batch_idx):
        if self.kl:
            inputs, cond = self.get_inputs(batch)
            del batch
            
            reconstructions, posterior = self(inputs, cond)
            
            kl_warmup_epochs = getattr(self, "kl_warmup_epochs", 0)
            if kl_warmup_epochs > 0:
                # Ramp the KL term in over the first epochs. Applying the full
                # weight from step 0 collapses the posterior before the decoder can
                # use the latent at all, and the run then never recovers the
                # reconstruction quality TORNADO needs downstream.
                factor = min(1.0, self.current_epoch / max(1, kl_warmup_epochs))
                target_kl = self.config["model"].get("kl_weight", 0.0)
                self.loss.kl_weight = target_kl * factor
                self.log("train/kl_weight", self.loss.kl_weight, on_step=False,
                         on_epoch=True, sync_dist=self.ddp)

            loss, log_dict = self.loss(
                inputs, reconstructions, posterior, split="train"
            )
            self.log_dict(
                log_dict,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=self.ddp,
            )

            loss = loss.mean()
            self.log(
                "train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp
            )
        

        else:
            inputs, cond = self.get_inputs(batch)
            reconstructions, posterior = self(inputs, cond)
            loss = self.criterion(reconstructions, inputs)
            self.log(
                "train/loss", loss, on_step=True, on_epoch=True, sync_dist=self.ddp
            )
        return loss

    def validation_step(self, batch, batch_idx):
        if self.kl:
            inputs, cond = self.get_inputs(batch)
            reconstructions, posterior = self(inputs, cond)
            loss, log_dict = self.loss(inputs, reconstructions, posterior, split="val")
            self.log_dict(
                log_dict,
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )
            # Also log as val/loss so ModelCheckpoint(monitor="val/loss") works
            self.log(
                "val/loss", loss.mean(), on_step=False, on_epoch=True, sync_dist=self.ddp
            )

            inputs_denorm = self.normalizer.denormalize(inputs)
            reconstructions_denorm = self.normalizer.denormalize(reconstructions)

            vrmse = VRMSE(reconstructions_denorm, inputs_denorm)
            self.log(
                "val/VRMSE_denorm",
                vrmse,
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )

            if self.era5:
                self._log_era5_metrics(inputs_denorm, reconstructions_denorm)

            sRMSE_all = sRMSE(
                reconstructions_denorm, inputs_denorm, spatial=2
            )  # calculate sRMSE

            if batch_idx == 0:
                if not self.trainer.is_global_zero:
                    pass
                else:
                    os.makedirs(self.log_dir, exist_ok=True)
                    if self.era5:
                        plot_result_map(
                            inputs_denorm.unsqueeze(1),  # b t nlat nlon c
                            reconstructions_denorm.unsqueeze(1),
                            n_t=1,
                            lat=self.lat_deg.cpu().numpy(),
                            times=batch.get("time_hours", [None])[0]
                            if "time_hours" in batch else None,
                            path=f"{self.log_dir}ep_{self.current_epoch}.png",
                        )
                    else:
                        plot_result_2d(
                            inputs_denorm.unsqueeze(1),  # b t nx ny c
                            reconstructions_denorm.unsqueeze(1),
                            n_t=1,
                            path=f"{self.log_dir}ep_{self.current_epoch}.png",
                        )

            self.log(
                "val/VRMSE", vrmse, on_step=False, on_epoch=True, sync_dist=self.ddp
            )
            self.log(
                "val/sRMSE_low",
                sRMSE_all[0],
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )
            self.log(
                "val/sRMSE_mid",
                sRMSE_all[1],
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )
            self.log(
                "val/sRMSE_high",
                sRMSE_all[2],
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )

        else:
            inputs, cond = self.get_inputs(batch)

            reconstructions, posterior = self(inputs, cond)  # b nx ny c

            inputs_denorm = self.normalizer.denormalize(inputs)
            reconstructions_denorm = self.normalizer.denormalize(reconstructions)

            loss = self.criterion(reconstructions_denorm, inputs_denorm)

            vrmse = VRMSE(reconstructions_denorm, inputs_denorm)

            sRMSE_all = sRMSE(
                reconstructions_denorm, inputs_denorm, spatial=2
            )  # calculate sRMSE

            if batch_idx == 0:
                if not self.trainer.is_global_zero:
                    pass
                else:
                    os.makedirs(self.log_dir, exist_ok=True)
                    plot_result_2d(
                        inputs_denorm.unsqueeze(1),  # b t nx ny c
                        reconstructions_denorm.unsqueeze(1),
                        n_t=1,
                        path=f"{self.log_dir}ep_{self.current_epoch}.png",
                    )

            self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=self.ddp)
            self.log(
                "val/VRMSE", vrmse, on_step=False, on_epoch=True, sync_dist=self.ddp
            )
            self.log(
                "val/sRMSE_low",
                sRMSE_all[0],
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )
            self.log(
                "val/sRMSE_mid",
                sRMSE_all[1],
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )
            self.log(
                "val/sRMSE_high",
                sRMSE_all[2],
                on_step=False,
                on_epoch=True,
                sync_dist=self.ddp,
            )

        return loss

    def _log_era5_metrics(self, target_denorm, pred_denorm):
        """Area-weighted per-variable reconstruction error, in physical units.

        A global RMSE in K / (m/s) / (m2/s2) is what a reviewer can compare against
        published WeatherBench numbers; the unweighted VRMSE above is kept only for
        continuity with the other datasets in this repo.
        """
        lat = self.lat_deg.cpu().numpy()
        names = ["t2m", "u10", "v10", "z500"]
        rmse = lat_weighted_rmse(pred_denorm, target_denorm, lat, lat_dim=1,
                                 per_channel=True)
        for i in range(min(len(names), rmse.numel())):
            self.log(f"val/rmse_{names[i]}", rmse[i], on_step=False, on_epoch=True,
                     sync_dist=self.ddp)
        self.log("val/VRMSE_latw",
                 lat_weighted_vrmse(pred_denorm, target_denorm, lat, lat_dim=1),
                 on_step=False, on_epoch=True, sync_dist=self.ddp)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )
        if self.pde in ("km_flow", "real_combustion", "era5"):
            step_size = 10  # epochs for km_flow
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.99
            )
            return [optimizer], [scheduler]  # default epoch interval
        else:
            # Step-based decay: decay every 500 global steps
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.99
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
