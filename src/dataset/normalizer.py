from torch.utils.data import DataLoader
import pickle
import os.path
import torch
import numpy as np
from tqdm import tqdm


class ChannelNormalizer:
    """Per-channel z-score (ERA5). Stats are pickled as [mean, std], like ScalarNormalizer."""

    def __init__(self, stat_path="./", dataset=None, scaling_factor=1, recalculate=False):
        self.scaling_factor = scaling_factor

        if os.path.isfile(stat_path) and not recalculate:
            self.load_stats(stat_path)
            print("Statistics loaded from", stat_path)
        else:
            assert dataset is not None, "Data must be provided for normalization"
            print("Calculating per-channel statistics for normalization")
            mean, std = self._fit(dataset)
            self.u_mean, self.u_std = mean, std
            self.save_stats(path=stat_path)
            print("Statistics saved to", stat_path)

        self.u_mean = torch.as_tensor(np.asarray(self.u_mean), dtype=torch.float32)
        self.u_std = torch.as_tensor(np.asarray(self.u_std), dtype=torch.float32)
        self.u_std = torch.clamp(self.u_std, min=1e-8)
        self.print_stats()

    @staticmethod
    def _fit(dataset):
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        n = 0
        s1 = s2 = None
        for batch in tqdm(loader):
            u = batch.get("output_fields")
            if u is None:
                u = batch["u_label"]
            u = u.reshape(-1, u.shape[-1]).double()
            if s1 is None:
                s1 = torch.zeros(u.shape[-1], dtype=torch.float64)
                s2 = torch.zeros(u.shape[-1], dtype=torch.float64)
            s1 += u.sum(dim=0)
            s2 += (u**2).sum(dim=0)
            n += u.shape[0]
        mean = (s1 / n).numpy()
        std = np.sqrt(np.maximum((s2 / n).numpy() - mean**2, 0.0))
        return mean, std

    def print_stats(self):
        print(f"u mean (per channel): {self.u_mean.tolist()}")
        print(f"u std  (per channel): {self.u_std.tolist()}")
        print(f"scaling factor: {self.scaling_factor}")

    def save_stats(self, path):
        with open(path, "wb") as f:
            pickle.dump([np.asarray(self.u_mean), np.asarray(self.u_std)], f)

    def load_stats(self, path):
        with open(path, "rb") as f:
            self.u_mean, self.u_std = pickle.load(f)

    def _broadcast(self, u):
        """Reshape [C] statistics to line up with the trailing channel axis."""
        mean = self.u_mean.to(device=u.device, dtype=u.dtype)
        std = self.u_std.to(device=u.device, dtype=u.dtype)
        if u.shape[-1] != mean.numel():
            raise ValueError(
                f"ChannelNormalizer fitted on {mean.numel()} channels but got a "
                f"tensor with {u.shape[-1]} in the last axis (shape {tuple(u.shape)}). "
                f"Fields must be channel-last."
            )
        return mean, std

    def normalize(self, u):
        mean, std = self._broadcast(u)
        return ((u - mean) / std) * self.scaling_factor

    def denormalize(self, u):
        mean, std = self._broadcast(u)
        return (u / self.scaling_factor) * std + mean


class ScalarNormalizer:
    def __init__(self,
                 stat_path = "./",
                 dataset=None,
                 recalculate = False,
                 scaling_factor=1,
                 transform=None):
        """transform: None or "log1p", applied before the z-score and inverted after."""
        self.scaling_factor = scaling_factor
        if transform not in (None, "none", "log1p"):
            raise ValueError(f"Unknown transform: {transform}")
        self.transform = None if transform == "none" else transform

        if os.path.isfile(stat_path) and not recalculate:
            self.load_stats(stat_path)
            print("Statistics loaded from", stat_path)
        else:
            assert dataset is not None, "Data must be provided for normalization"
            print("Calculating statistics for normalization")
            dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers=0)

            from sklearn.preprocessing import StandardScaler

            u_scaler = StandardScaler()

            for batch in tqdm(dataloader):
                u = batch.get("output_fields")
                if u is None:
                    u = batch["u_label"]
                u_scaler.partial_fit(self._apply_transform(u).reshape(-1, 1))

            self.u_mean = u_scaler.mean_.item()
            self.u_std = np.sqrt(u_scaler.var_).item()
            self.save_stats(path=stat_path)
            print("Statistics saved to", stat_path)

        self.print_stats()

        self.u_mean = torch.tensor(self.u_mean)
        self.u_std = torch.tensor(self.u_std)

    def _apply_transform(self, u):
        if self.transform == "log1p":
            return torch.log1p(u.clamp_min(0.0))
        return u

    def _invert_transform(self, u):
        if self.transform == "log1p":
            return torch.expm1(u)
        return u

    def print_stats(self):
        if self.transform is not None:
            print(f"transform: {self.transform} (stats below are post-transform)")
        print(f"u mean: {self.u_mean}, u std: {self.u_std}")
        print(f"scaling factor: {self.scaling_factor}")

    def save_stats(self, path):
        with open(path, "wb") as f:
            pickle.dump([self.u_mean, self.u_std], f)

    def load_stats(self, path):
        with open(path, "rb") as f:
            self.u_mean, self.u_std = pickle.load(f)
    
    def normalize(self, u):
        u_norm = self._apply_transform(u.clone())
        u_norm = (u_norm - self.u_mean) / self.u_std
        u_norm = u_norm * self.scaling_factor
        return u_norm

    def denormalize(self, u): 
        u_denorm = u.clone()
        u_denorm = u_denorm / self.scaling_factor

        u_denorm = u_denorm * self.u_std + self.u_mean

        return self._invert_transform(u_denorm)