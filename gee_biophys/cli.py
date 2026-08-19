# run_export.py
from __future__ import annotations

import shutil
from datetime import timezone
from pathlib import Path

import numpy as np
import typer
import yaml
from loguru import logger

from gee_biophys.config import ConfigParams
from gee_biophys.s2_export import export_image, merge_local_interval_exports
from gee_biophys.s2_input import load_s2_input, load_s2_input_xarray
from gee_biophys.s2_predict import biophys_predict, biophys_predict_local
from gee_biophys.utils import (
    get_system_index,
    initialize_export_location,
    update_dataset_metadata,
    update_image_metadata,
)


def validate_config(path: Path):
    if not path or not path.exists():
        raise typer.BadParameter(
            "Please provide a valid path to a YAML configuration file, e.g.:\n"
            "  gee-biophys --config example_configs/minimal_example.yaml",
        )
    return path


def load_params(path: str) -> ConfigParams:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ConfigParams(**data)


app = typer.Typer(
    no_args_is_help=True,
    help="Custom time-series export of LAIe/FAPAR/FCOVER with YAML params.",
)


def run_pipeline(config: str, set_public: bool = False) -> None:
    """Run the full export pipeline based on the provided configuration."""
    # <---- Setup ---->
    cfg = load_params(str(config))
    local_temp_subfolder = "temp"
    final_local_filename = None

    initialize_export_location(cfg, set_public=set_public)

    if cfg.export.destination == "xee-local":
        final_local_filename = get_system_index(
            cfg,
            cfg.temporal.start,
            cfg.temporal.end,
        )
        temp_root = Path(cfg.export.folder) / local_temp_subfolder
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)

    # <---- Main Loop ---->
    for interval_start, interval_end in cfg.temporal.iter_date_ranges():
        # <---- Load Input ---->
        logger.info(
            f"Processing interval: {interval_start.strftime('%Y-%m-%d')} to {interval_end.strftime('%Y-%m-%d')}",
        )
        imgc = load_s2_input(cfg, interval_start, interval_end)
        filename = get_system_index(cfg, interval_start, interval_end)

        if cfg.export.destination == "xee-local":
            logger.debug(
                "Running local prediction via xee/xarray. This is not recommended for very large exports (> 100x100km area) at 20m resolution."
            )

            input_ds = load_s2_input_xarray(cfg, imgc)
            output_ds = biophys_predict_local(cfg, input_ds)
            output_ds = update_dataset_metadata(
                output_ds,
                interval_start,
                interval_end,
                cfg,
            )
            interval_start_utc = interval_start.astimezone(timezone.utc).replace(
                tzinfo=None,
            )
            output_ds = output_ds.expand_dims(
                time=[np.datetime64(interval_start_utc)],
            )

            export_image(
                output_ds,
                filename,
                cfg,
                subfolder=local_temp_subfolder,
            )
            continue

        # <---- Prediction ---->
        output_image = biophys_predict(cfg, imgc)

        # <---- Update metadata ---->
        output_image = update_image_metadata(
            output_image,
            interval_start,
            interval_end,
            cfg,
        )

        # <---- Export  ---->
        export_image(output_image, filename, cfg)

    if cfg.export.destination == "xee-local":
        assert final_local_filename is not None
        merge_local_interval_exports(
            cfg,
            final_filename=final_local_filename,
            temp_subfolder=local_temp_subfolder,
            cleanup_temp=True,
        )
        logger.info("Done! Local predictions were merged into one Zarr store.")
    else:
        logger.info(
            "Done! Exports have been started. Please check task status to see if they completed successfully.",
        )


@app.command(help="Run the export using the provided YAML configuration.")
def run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to params YAML file.",
        callback=validate_config,
        readable=True,
    ),
    public: bool = typer.Option(
        False,
        "--public/--no-public",
        help="Set exported file ACL to public after export. This is required if you want to visualize the exported asset using the GEE-App ",
        show_default=True,
    ),
):
    """CLI entry point"""
    typer.echo(f"Using configuration file: {config}")
    run_pipeline(config, set_public=public)


if __name__ == "__main__":
    config_path = Path("example_configs/forest-fire-bitsch-2023.yaml")
    run_pipeline(config_path, set_public=True)  # for debugging purposes
