# Data

| Dataset | `data=` | Source |
|---|---|---|
| Kolmogorov flow | `km_flow` | [Hugging Face](https://huggingface.co/datasets/jorge-mif/apebench) |
| Kuramoto-Sivashinsky, Burgers, Gray-Scott | `ks_2d`, `burgers`, `gray_scott` | [Hugging Face](https://huggingface.co/datasets/jorge-mif/apebench), generated with [APEBench](https://github.com/tum-pbs/apebench) |
| Real combustion | `real_combustion` | [RealPDEBench](https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDEBench), `real_combustion` split |
| ERA5, 1.5°, 6-hourly | `era5` | [WeatherBench-2](https://weatherbench2.readthedocs.io) |

Paths in `configs/data/<dataset>.yaml` are relative to `$SCRATCH` (datasets,
training runs) and `$WORK`; set those variables, edit the configs, or override
on the command line (`data.dataset.train_path=...`). Normalisation statistics
are fitted on the training split and saved to `normalizer.stat_path` on first use.

ERA5 is converted once from the yearly WeatherBench-2 NetCDFs:

```bash
python dataset/prepare_era5.py --src <dir of .nc files> --dst <out dir> --train-years 2010-2015
```
