import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
from common.fourier import isotropic_power_spectrum 

#################################################################################
#                                 PDE Losses                                    #
#################################################################################

def scaledlp_loss(input: torch.Tensor, target: torch.Tensor, p: int = 2, reduction: str = "mean"):
    B = input.size(0)
    diff_norms = torch.norm(input.reshape(B, -1) - target.reshape(B, -1), p, 1)
    target_norms = torch.norm(target.reshape(B, -1), p, 1)
    val = diff_norms / target_norms
    if reduction == "mean":
        return torch.mean(val)
    elif reduction == "sum":
        return torch.sum(val)
    elif reduction == "none":
        return val
    else:
        raise NotImplementedError(reduction)

class ScaledLpLoss(torch.nn.Module):
    """Scaled Lp loss for PDEs.

    Args:
        p (int, optional): p in Lp norm. Defaults to 2.
        reduction (str, optional): Reduction method. Defaults to "mean".
    """

    def __init__(self, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return scaledlp_loss(input, target, p=self.p, reduction=self.reduction)



class LatWeightedScaledLpLoss(torch.nn.Module):
    """ScaledLpLoss with cos(latitude) area weights.

    On an equiangular lat/lon grid a cell at 60 degrees covers half the area of one
    at the equator, and the two rows at the poles cover almost nothing. An
    unweighted norm therefore counts the polar cap ~50x more heavily than its share
    of the sphere, which on ERA5 means the loss is dominated by Antarctica and the
    Arctic. Weighting by cos(latitude) (normalised to mean 1, so the value stays on
    the same scale as the unweighted loss) is the WeatherBench convention.

    Identical in form to `scaledlp_loss`: a per-sample relative error
    ||pred - target|| / ||target||, with both norms taken under the same weights.

    Args:
        lat: latitudes in degrees, [nlat].
        lat_dim: axis of `input`/`target` holding latitude. Fields in this repo are
            [B, nlat, nlon, C], so the default is 1.
        p: exponent of the norm.
    """

    def __init__(self, lat, lat_dim: int = 1, p: int = 2, reduction: str = "mean") -> None:
        super().__init__()
        from common.geo import latitude_weights

        self.p = p
        self.reduction = reduction
        self.lat_dim = lat_dim
        self.register_buffer(
            "weights", torch.as_tensor(latitude_weights(lat), dtype=torch.float32)
        )

    def _w(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.ndim
        shape[self.lat_dim] = self.weights.numel()
        return self.weights.reshape(shape).to(device=x.device, dtype=x.dtype)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape[self.lat_dim] != self.weights.numel():
            raise ValueError(
                f"LatWeightedScaledLpLoss expects {self.weights.numel()} latitudes "
                f"at dim {self.lat_dim}, got shape {tuple(input.shape)}."
            )
        w = self._w(input)
        B = input.size(0)
        dims = tuple(range(1, input.ndim))
        diff = (w * (input - target).abs() ** self.p).sum(dim=dims) ** (1.0 / self.p)
        norm = (w * target.abs() ** self.p).sum(dim=dims) ** (1.0 / self.p)
        val = diff / (norm + 1e-12)
        if self.reduction == "mean":
            return torch.mean(val)
        elif self.reduction == "sum":
            return torch.sum(val)
        elif self.reduction == "none":
            return val
        raise NotImplementedError(self.reduction)


def lat_weighted_rmse(pred, target, lat, lat_dim: int = 1, per_channel: bool = False):
    """Area-weighted RMSE, the headline WeatherBench score.

    Args:
        pred, target: [B, nlat, nlon, C] in physical units.
        lat: latitudes in degrees.
        per_channel: return one value per channel instead of the channel mean.
    """
    from common.geo import latitude_weights

    w = torch.as_tensor(latitude_weights(lat), device=pred.device, dtype=pred.dtype)
    shape = [1] * pred.ndim
    shape[lat_dim] = w.numel()
    w = w.reshape(shape)

    se = (pred - target) ** 2 * w
    dims = tuple(d for d in range(pred.ndim) if d != pred.ndim - 1)
    denom = w.expand_as(pred).sum(dim=dims)
    rmse = torch.sqrt(se.sum(dim=dims) / denom)  # (C,)
    return rmse if per_channel else rmse.mean()


def lat_weighted_vrmse(pred, target, lat, lat_dim: int = 1):
    """VRMSE (RMSE divided by the target's standard deviation), area-weighted.

    Matches `VRMSE` but with cos(latitude) weights, so the per-channel
    normalisation makes channels with very different units comparable.
    """
    from common.geo import latitude_weights

    w = torch.as_tensor(latitude_weights(lat), device=pred.device, dtype=pred.dtype)
    shape = [1] * pred.ndim
    shape[lat_dim] = w.numel()
    w = w.reshape(shape)

    dims = tuple(d for d in range(pred.ndim) if d != pred.ndim - 1)
    denom = w.expand_as(pred).sum(dim=dims)
    mse = ((pred - target) ** 2 * w).sum(dim=dims) / denom
    mean = (target * w).sum(dim=dims) / denom
    var = ((target - mean) ** 2 * w).sum(dim=dims) / denom
    return torch.mean(torch.sqrt(mse / (var + 1e-6)))


class KL_Loss():
    # simple VAE loss without discriminator or perceptual loss
    def __init__(self, 
                 kl_weight=1.0, 
                 criterion=nn.L1Loss()):

        super().__init__()
        self.kl_weight = kl_weight
        self.criterion = criterion

    def __call__(self, 
                inputs, 
                reconstructions, 
                posteriors,
                split="train"):
        # inputs, reconstructions in shape b nlat nlon c

        rec_loss = self.criterion(reconstructions, inputs)
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss_unweighted = kl_loss.clone().detach().mean()
        kl_loss = self.kl_weight * kl_loss

        loss = rec_loss + kl_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/kl_loss_unweighted".format(split): kl_loss_unweighted,
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
        return loss, log



class CRPS(nn.Module):
    def __init__(self, nlat, nlon, with_poles=False):
        super().__init__()
        self.with_poles = with_poles

        if not with_poles:
            longitude_resolution = nlon
            lat_end = (nlat - 1) * (360 / longitude_resolution) / 2
            lat_weight = _weight_for_latitude_vector_without_poles(np.linspace(-lat_end, lat_end, nlat))
        else:
            lat_weight = _weight_for_latitude_vector_with_poles(np.linspace(-90, 90, nlat))

        lat_weight = torch.from_numpy(lat_weight)
        lat_weight = lat_weight / lat_weight.mean()
        self.register_buffer('lat_weight', lat_weight) # shape [nlat]

    def crps_skill(self, pred, target):
        # pred in shape [b, ens, nt, nlat, nlon]
        # target in shape [b, nt, nlat, nlon]
        b = pred.shape[0]
        nt = pred.shape[2]
        nlon = pred.shape[4]

        skill = torch.abs(pred - target.unsqueeze(1)) # b ens nt nlat nlon
        skill = skill.mean(dim=1) # b nt nlat nlon
        lat_weight = repeat(self.lat_weight, 'nlat -> b nt nlat nlon', b=b, nt=nt, nlon=nlon) # b nt nlat nlon
        avg_skill = skill * lat_weight # b nt nlat nlon
        avg_skill = avg_skill.mean(dim=(2, 3)) # b nt

        return avg_skill    
    
    def crps_spread(self, pred, target):
        # pred in shape [b, ens, nt, nlat, nlon]
        # target in shape [b, nt, nlat, nlon]        
        b, ens, nt, nlat, nlon = pred.shape

        prefactor = 1/(ens*(ens-1))

        pred_diffs = torch.zeros((b, ens, ens, nt, nlat, nlon), device=pred.device) # b ens ens nt nlat nlon c
        for i in range(ens):
            for j in range(ens):
                pred_diffs[:, i, j] = torch.abs(pred[:, i] - pred[:, j])

        pred_diffs = pred_diffs.sum(dim=(1, 2)) * prefactor  # b nt nlat nlon

        lat_weight = repeat(self.lat_weight, 'nlat -> b nt nlat nlon', b=b, nt=nt, nlon=nlon) # b nt nlat nlon
        avg_spread = pred_diffs * lat_weight # b nt nlat nlon
        avg_spread = avg_spread.mean(dim=(2, 3)) # b nt

        return avg_spread
    
    def forward(self, pred, target):
        # pred in shape [b, ens, nt, nlat, nlon]
        # target in shape [b, nt, nlat, nlon]
        skill = self.crps_skill(pred, target)  # b nt
        spread = self.crps_spread(pred, target)  # b nt

        crps = skill - 0.5*spread # b nt
        
        crps_avg = torch.zeros_like(crps) # b nt
        t = crps.shape[1]

        for i in range(t):
            crps_avg[:, i] = crps[:, :i+1].mean(dim=1)
        
        return crps_avg
    
def sRMSE(pred, target, spatial=2):
    # u, v: [b, nx, ny, d]
    # u is prediction, v is target
    sRMSE_all = []
    for i in range(pred.shape[-1]):
        u = pred[..., i]
        v = target[..., i]
        p_u, k = isotropic_power_spectrum(u, spatial=spatial)
        p_v, _ = isotropic_power_spectrum(v, spatial=spatial)
        p_u = p_u.mean(dim=0)
        p_v = p_v.mean(dim=0)

        se_p = torch.square(1 - (p_v + 1e-6) / (p_u + 1e-6))

        rmse_f = []

        bins = torch.logspace(k[0].log2(), -1.0, steps=4, base=2)

        for i in range(4):
            if i < 3:
                mask = torch.logical_and(bins[i] <= k, k <= bins[i + 1])
            else:
                mask = bins[i] <= k

            rmse_f.append(torch.sqrt(torch.mean(se_p[mask])))

        sRMSE_all.append(torch.stack(rmse_f))
    
    sRMSE_all = torch.stack(sRMSE_all, dim=0)  # d x 4
    sRMSE_all = torch.mean(sRMSE_all, dim=0)  # 4
    
    return sRMSE_all

def VRMSE(pred, target):
    # pred, target in shape b nx ny d
    v = pred
    u = target 
    se = torch.square(u - v) # b nx ny d
    dim = tuple(range(len(u.shape) - 1))
    mse = torch.mean(se, dim=dim) # (d,)
    vrmse = torch.sqrt(mse / (torch.var(u, dim=dim) + 1e-6)) # (d,)
    return torch.mean(vrmse) # ()