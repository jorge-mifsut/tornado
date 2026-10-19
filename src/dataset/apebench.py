"""2D PDE trajectories (Kolmogorov flow, KS, Burgers, Gray-Scott).

Kolmogorov is HDF5 (`f[split]` with `u`, `x`, `t`); the others are raw .npy
arrays [N, nt, C, nx, ny] or [N, nt, nx, ny, C], with t = frame index, dx = 1/nx.
"""

from typing import Any, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ApeBenchDataset(Dataset):

    def __init__(
        self,
        path: str,
        split: str,
        resolution: list,
        normalizer=None,
        return_traj: bool = False,
        horizon: int = None,
    ) -> None:
        """resolution: [nt, nx, ny]. return_traj: (u_0, trajectory) instead of a random (t, t+1) pair."""
        super().__init__()
        if resolution is None or len(resolution) != 3:
            raise ValueError("resolution must be provided as [nt, nx, ny].")

        self.split = split
        self.return_traj = return_traj
        self.resolution = resolution
        self.nt, self.nx, self.ny = map(int, resolution)
        self.horizon = int(horizon) if horizon is not None else self.nt
        self.normalizer = normalizer
        self.hdf5 = str(path).endswith((".h5", ".hdf5"))

        if self.hdf5:
            data = h5py.File(path, "r")[split]
            self.u = data["u"]
            x = torch.tensor(np.array(data["x"]))
            t = torch.tensor(np.array(data["t"]))
            self.t_downsample = int(t.shape[0] / self.nt)
            self.dx = x[1, 0, 0] - x[0, 0, 0]
            self.dy = x[0, 1, 1] - x[0, 0, 1]
        else:
            u = np.load(path)
            if u.ndim == 5 and u.shape[-2:] == (self.nx, self.ny):
                u = u.transpose(0, 1, 3, 4, 2)  # [N, nt, C, nx, ny] -> [N, nt, nx, ny, C]
            elif not (u.ndim == 5 and u.shape[2:4] == (self.nx, self.ny)):
                raise ValueError(
                    f"{path}: expected [N, nt, C, nx, ny] or [N, nt, nx, ny, C] with "
                    f"nx, ny = {self.nx}, {self.ny}; got shape {u.shape}."
                )
            self.u = u
            t = torch.arange(u.shape[1], dtype=torch.float32)
            if t.shape[0] % self.nt != 0:
                raise ValueError(
                    f"Cannot downsample time: nt_data={t.shape[0]} not divisible by requested nt={self.nt}."
                )
            self.t_downsample = t.shape[0] // self.nt
            self.dx = torch.tensor(1.0 / self.nx, dtype=torch.float32)
            self.dy = torch.tensor(1.0 / self.ny, dtype=torch.float32)

        # npy: t is strided but u is not, as in training
        self.u_stride = self.t_downsample if self.hdf5 else 1

        self.t = t[:: self.t_downsample]
        self.t = self.t - self.t[0]
        self.t = self.t[: self.horizon]
        self.nt = len(self.t)
        if self.nt < 2:
            raise ValueError(f"Horizon too small after downsampling: nt={self.nt}.")
        self.dt = self.t[1] - self.t[0]
        self.n_samples = len(self.u)

        print(f"Data loaded from: {path}")
        print(f"split={split} | dt: {float(self.dt):.6g}, dx: {float(self.dx):.6g}, nt: {self.nt}, nx: {self.nx}, ny: {self.ny}")
        print("")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> Dict[str, Any]:
        if self.hdf5:
            u = torch.tensor(np.array(self.u[i]))
        else:
            u = torch.from_numpy(np.asarray(self.u[i])).to(torch.float32)
        u = u[:: self.u_stride]
        if u.ndim == 3:
            u = u.unsqueeze(-1)
        u = u[: self.horizon]
        u = self.normalizer.normalize(u) if self.normalizer is not None else u

        if not self.return_traj:
            rand_idx = np.random.randint(0, self.nt - 1)
            u_input = u[rand_idx]
            u_label = u[rand_idx + 1]
        else:
            u_input = u[0]
            u_label = u

        return {"input_fields": u_input, "output_fields": u_label, "dx": self.dx, "dt": self.dt, "t": self.t}
