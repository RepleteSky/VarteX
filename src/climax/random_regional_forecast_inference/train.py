# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from climax.random_regional_forecast_inference.datamodule import RandomRegionalForecastDataModule
from climax.random_regional_forecast_inference.module import RandomRegionalForecastModule
from pytorch_lightning.cli import LightningCLI


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=RandomRegionalForecastModule,
        datamodule_class=RandomRegionalForecastDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.datamodule.set_patch_size(cli.model.get_patch_size())

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="save_log/random_regional_forecast_myclimax3_06_18/checkpoints/epoch_049_6h_4vars_4_4_0del_unused_pos_2rep.ckpt")


if __name__ == "__main__":
    main()
