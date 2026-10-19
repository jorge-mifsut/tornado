# Reproducing the paper's results

Every table and figure is listed in `evals/experiments.yaml` and produced by:

```bash
python -m evals.reproduce --list            # all experiments
python -m evals.reproduce table1_kolmogorov # one paper item
python -m evals.reproduce all               # everything
```

## Requirements

- Trained runs: a directory per run, as written by `train.py` (`config.yml` and
  checkpoints), including the autoencoder runs they point to.
- The datasets, in the layouts described in `configs/data/<dataset>.yaml`.

## Paths

```bash
export TORNADO_LOGS=/path/to/runs         # trained runs
export TORNADO_EVAL_DIR=/path/to/eval     # rollouts and metrics (default: $TORNADO_LOGS/eval)
```

Point the data configs at your copies with
`--data dataset.path=... normalizer.stat_path=...`, and the autoencoders with
`--ae-dir`. Tables (`.csv`, `.tex`) and figures (`.png`, `.pdf`) go to
`--out` (default `results/paper`).

## Stages

Each experiment runs three stages; `--stage` runs one:

| stage | needs | does |
|---|---|---|
| `rollout` | GPU | samples ensemble trajectories, saved as latents |
| `metrics` | GPU | decodes and scores them, one row per step and sample |
| `report` | CPU | builds the tables and figures |

Experiments with the same sampling settings share one rollout. Interrupted
rollouts resume where they stopped.

## Paper items

| item | experiment |
|---|---|
| Table 1 | `table1_{kolmogorov,ks,burgers,gray_scott,combustion,era5}` |
| Table 2 | `table2_integrators` |
| Tables 5-15 | `table5_diffusion_losses` ... `table15_era5_per_variable` |
| Figures 2-16 | `fig2_pit` ... `fig16_nfe` |
| Figures 17-34 | `fig17_panels_kolmogorov` ... `fig27_panels_era5` |
| rebuttal | `rebuttal_*` |

Table 10 (timing) needs a GPU. ERA5 reference baselines (persistence and
climatology): `python -m evals.analysis.era5_analysis --path <era5 directory>`.
