import os
import lightning as L
from torch.utils.data import DataLoader

from dataset.apebench import ApeBenchDataset
from dataset.normalizer import ChannelNormalizer, ScalarNormalizer

APEBENCH = ("km_flow", "ks_2d", "burgers", "gray_scott")


class PDEDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataconfig,
    ) -> None:
        super().__init__()
        self.data_config = dataconfig
        self.dataset_config = dataconfig["dataset"]
        self.batch_size = dataconfig["batch_size"]
        self.num_workers = dataconfig["num_workers"]
        self.pde = dataconfig["pde"]
        self.normalizer_config = dataconfig["normalizer"]
        self.ae = dataconfig.get("ae", False)
        self.test_dataset = None
        self._test_kwargs = None

        if self.pde in APEBENCH:
            self._setup_apebench()
        elif self.pde == "real_combustion":
            self._setup_combustion()
        elif self.pde == "era5":
            self._setup_era5()
        else:
            raise ValueError(f"Unknown pde {self.pde!r}")

    def _load_or_fit_normalizer(self, cls, fit_dataset, **kwargs):
        """Load stat_path, or fit on fit_dataset() and save there."""
        stat_path = self.normalizer_config["stat_path"]
        if os.path.exists(stat_path):
            return cls(stat_path=stat_path, **kwargs)
        return cls(stat_path=stat_path, dataset=fit_dataset(), **kwargs)

    def _val_window(self):
        """(horizon, stride) of validation windows; default to the training ones."""
        c = self.dataset_config
        return (
            c.get("val_horizon", c.get("horizon", None)),
            c.get("val_window_stride", c.get("window_stride", None)),
        )

    def _setup_apebench(self):
        c = self.dataset_config
        res_train = c.get("resolution_train", c.get("resolution"))
        res_valid = c.get("resolution_valid", c.get("resolution"))

        self.normalizer = self._load_or_fit_normalizer(
            ScalarNormalizer,
            lambda: ApeBenchDataset(
                path=c["train_path"], split="train", resolution=res_train, return_traj=True
            ),
        )
        self.train_dataset = ApeBenchDataset(
            path=c["train_path"],
            split="train",
            resolution=res_train,
            normalizer=self.normalizer,
            return_traj=False if self.ae else True,
            horizon=c.get("horizon", None),
        )
        self.val_dataset = ApeBenchDataset(
            path=c["valid_path"],
            split="valid",
            resolution=res_valid,
            normalizer=self.normalizer,
            return_traj=True,
            horizon=c.get("horizon", None),
        )

    def _setup_combustion(self):
        from dataset.combustion import CombustionDataset, split_sim_ids

        c = self.dataset_config
        data_path = c["path"]
        train_sims, val_sims = split_sim_ids(
            data_path,
            val_sims=c.get("val_sims", None),
            n_val=c.get("n_val", 3),
            stratify=c.get("stratify", True),
        )
        print(f"[real_combustion] train runs: {train_sims}")
        print(f"[real_combustion] valid runs: {val_sims}")

        common = dict(
            path=data_path,
            resolution=c.get("resolution", None),
            t_downsample=c.get("t_downsample", 1),
            dx=c.get("dx", 1.0),
            dt=c.get("dt", 1.0),
        )

        self.normalizer = self._load_or_fit_normalizer(
            ScalarNormalizer,
            lambda: CombustionDataset(
                split="train",
                sim_ids=train_sims,
                return_traj=False,
                **{**common, "t_downsample": c.get("stat_t_downsample", 10)},
            ),
            transform=self.normalizer_config.get("transform", None),
        )

        self.train_dataset = CombustionDataset(
            split="train",
            sim_ids=train_sims,
            normalizer=self.normalizer,
            return_traj=False if self.ae else True,
            horizon=c.get("horizon", None),
            window_stride=c.get("window_stride", None),
            **common,
        )
        val_horizon, val_stride = self._val_window()
        self.val_dataset = CombustionDataset(
            split="valid",
            sim_ids=val_sims,
            normalizer=self.normalizer,
            return_traj=True,
            horizon=val_horizon,
            window_stride=val_stride,
            **common,
        )

    def _setup_era5(self):
        from dataset.era5 import ERA5Dataset, split_years

        c = self.dataset_config
        data_path = c["path"]
        years = split_years(
            c.get("train_years", None),
            c.get("val_years", None),
            c.get("test_years", None),
        )
        print(f"[era5] train years: {years['train']}")
        print(f"[era5] valid years: {years['valid']}")
        print(f"[era5] test  years: {years['test']}")

        common = dict(
            path=data_path,
            t_downsample=c.get("t_downsample", 1),
            dx=c.get("dx", 1.5),
            dt=c.get("dt", 6.0),
        )

        if not os.path.exists(self.normalizer_config["stat_path"]):
            print(
                "[era5] no statistics at "
                f"{self.normalizer_config['stat_path']}; fitting from the "
                "training years (this is normally done by prepare_era5.py)"
            )
        self.normalizer = self._load_or_fit_normalizer(
            ChannelNormalizer,
            lambda: ERA5Dataset(
                split="train",
                years=years["train"],
                return_traj=False,
                **{**common, "t_downsample": c.get("stat_t_downsample", 8)},
            ),
        )

        self.train_dataset = ERA5Dataset(
            split="train",
            years=years["train"],
            normalizer=self.normalizer,
            return_traj=False if self.ae else True,
            horizon=c.get("horizon", None),
            window_stride=c.get("window_stride", None),
            **common,
        )
        val_horizon, val_stride = self._val_window()
        self.val_dataset = ERA5Dataset(
            split="valid",
            years=years["valid"],
            normalizer=self.normalizer,
            return_traj=True,
            horizon=val_horizon,
            window_stride=val_stride,
            **common,
        )
        # test years: built on first use by test_dataloader
        self._test_kwargs = dict(
            split="test",
            years=years["test"],
            normalizer=self.normalizer,
            return_traj=True,
            horizon=c.get("test_horizon", val_horizon),
            window_stride=c.get("test_window_stride", val_stride),
            **common,
        )

    def train_dataloader(self, shuffle=True):
        self.pin_memory = False if self.num_workers == 0 else True
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self._test_kwargs is None:
            return None
        if self.test_dataset is None:
            from dataset.era5 import ERA5Dataset

            self.test_dataset = ERA5Dataset(**self._test_kwargs)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return None
