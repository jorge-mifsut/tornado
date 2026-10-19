# Default imports
import os
from datetime import datetime

# Lightning imports
import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Custom imports
from common.utils import save_yaml
from dataset.datamodule import PDEDataModule
from modules.ae_module import AutoencoderModule
from modules.train_module import TrainModule
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    modelconfig = config["model"]
    trainconfig = config["training"]
    dataconfig = config["data"]

    print(OmegaConf.to_yaml(cfg, resolve=True))


    seed = trainconfig["seed"]
    if "SLURM_JOB_ID" in os.environ:
        now = "job_" + os.environ["SLURM_JOB_ID"]
    else:
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    if config["description"] is None:
        config["description"] = ""

    pde = dataconfig["pde"]
    name = (
        modelconfig["model_name"]
        + "_"
        + pde
        + "_"
        + str(seed)
        + "_"
        + str(config["description"])
        + "_"
        + now
    )
    wandb_logger = WandbLogger(
        project=trainconfig["project"], name=name, mode=trainconfig["wandb_mode"]
    )
    path = trainconfig["log_dir"] + name + "/"
    config["training"]["log_dir"] = path

    os.makedirs(path, exist_ok=True)
    save_yaml(config, path + "config.yml")

    monitor = "val/loss"

    # The ED/rollout loss is gated by TrainModule.lambda_roll(): it is exactly 0
    # for epoch < warmup_epochs, so the diffusion term (and with it every gradient
    # reaching the sigma head) is switched off for that whole stretch. val/loss is
    # a deterministic 1-step error, which is lowest in precisely that regime, so a
    # plain min-over-all-epochs selection reliably returns a pre-ramp checkpoint
    # whose sigma head is still at initialisation. Only start tracking 'best' once
    # the schedule has reached lambda_max.
    modelconfig = config["model"]
    active_from_epoch = int(modelconfig.get("warmup_epochs", 0)) + int(
        modelconfig.get("ramp_epochs", 0)
    )

    class PostRampModelCheckpoint(ModelCheckpoint):
        """ModelCheckpoint that ignores epochs where the ED loss is still ramping.

        Only top-k selection is gated; ``last.ckpt`` is still written every epoch,
        so resuming and 'evaluate at last' are unaffected.
        """

        def _save_topk_checkpoint(self, trainer, monitor_candidates):
            if trainer.current_epoch < active_from_epoch:
                return
            super()._save_topk_checkpoint(trainer, monitor_candidates)

    ckpt_cls = PostRampModelCheckpoint if active_from_epoch > 0 else ModelCheckpoint
    if active_from_epoch > 0:
        print(
            f"[ckpt] 'best' tracked from epoch {active_from_epoch} onward "
            f"(warmup {modelconfig.get('warmup_epochs', 0)} + ramp "
            f"{modelconfig.get('ramp_epochs', 0)}); last.ckpt unaffected."
        )

    checkpoint_callback = ckpt_cls(
        monitor=monitor,
        filename="model_{epoch:02d}_best",
        mode="min",
        dirpath=path,
        save_last=True,
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    datamodule = PDEDataModule(dataconfig=dataconfig)
    print('batch: ', next(iter(datamodule.train_dataloader()))["input_fields"].shape)
    batch = next(iter(datamodule.train_dataloader()))
    print(batch.keys())

    if modelconfig["model_name"] == "ae":
        model = AutoencoderModule(config, normalizer=datamodule.normalizer)
    else:
        model = TrainModule(config, normalizer=datamodule.normalizer)

    trainer = L.Trainer(
        devices=trainconfig["devices"],
        num_nodes=trainconfig.get("nodes", 1),
        accelerator=trainconfig["accelerator"],
        strategy=trainconfig["strategy"],
        check_val_every_n_epoch=trainconfig["check_val_every_n_epoch"],
        log_every_n_steps=trainconfig["log_every_n_steps"],
        max_epochs=trainconfig["max_epochs"],
        default_root_dir=path,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        gradient_clip_val=trainconfig.get("gradient_clip_val", 1.0),
        num_sanity_val_steps=1,
        accumulate_grad_batches=trainconfig.get("accumulate_grad_batches", 1),
    )

    if trainconfig["checkpoint"] is not None:
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=trainconfig["checkpoint"]
        )
    else:
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
