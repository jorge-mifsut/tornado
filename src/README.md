# Code

Run everything from this directory.

| Folder | Contents |
|---|---|
| `configs/` | Hydra configs: `model/`, `data/`, `dit/` |
| `dataset/` | Data loaders; see [`dataset/README.md`](dataset/README.md) |
| `modules/` | Autoencoder, DiT backbones, TORNADO and the baselines |
| `evals/` | Evaluation pipeline; see [`evals/README.md`](evals/README.md) |

## Train

Train the autoencoder, then the latent processor on its posteriors:

```bash
python train.py model=ae_km data=km_flow data.ae=True
python train.py model=tornado data=km_flow dit=ape
```

| Option | Values |
|---|---|
| `model` | `tornado`, `fm`, `interpolant`, `ddim`, `ddpm`, `gaussian_nll`, `moment_stepper`, `mc_dropout` |
| `data` | `km_flow`, `ks_2d`, `burgers`, `gray_scott`, `real_combustion`, `era5` |
| `dit` | `ape`; `ape_rc` for `real_combustion`, `ape_era5` for `era5` |
| autoencoder | `model=ae_km`, `ae_burgers`, `ae_gs`, `ae_combustion`, `ae_era5`, with `data.ae=True` |

The processor reads its autoencoder from `data.autoencoder` (`checkpoint_path`,
`run`). Runs are written to `training.log_dir`.
