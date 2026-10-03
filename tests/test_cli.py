import ee
import pytest
import yaml
from loguru import logger

from gee_biophys.config import ConfigParams
from gee_biophys.s2_input import convert_s2_input_to_xarray, load_s2_input
from gee_biophys.s2_predict import biophys_predict_ee, biophys_predict_local

ATOL = 1e-5
N_SAMPLES = 10


def load_params(path: str) -> ConfigParams:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ConfigParams(**data)


@pytest.mark.parametrize(
    "config_path",
    [
        "example_configs/minimal_example.yaml",
        "example_configs/bimonthly-zambia.yaml",
        "example_configs/seasonal-summer-zurich.yaml",
        "example_configs/seasonal-zambia-arid_shrubland.yaml",
        "example_configs/biweekly-catalunya-for-testing.yaml",
    ],
)
def test_cli(ee_init, config_path):
    cfg = load_params(config_path)

    for interval_start, interval_end in cfg.temporal.iter_date_ranges():
        imgc = load_s2_input(cfg, interval_start, interval_end)
        # imgc.getInfo()  # force evaluation to catch errors
        if cfg.export.destination in ["gcs", "asset", "drive"]:
            imgc.getInfo()
            output_image = biophys_predict_ee(
                variable=cfg.variables.variable,
                model=cfg.variables.model,
                input_imgc=imgc,
                clip_min_max=True,
                biome_lc_name=cfg.variables.biome_lc_name,
            )
            output_image.getInfo()  # force evaluation to catch errors

            # sample a single pixel to check that the output is reasonable
            output_image.sample(
                region=cfg.spatial.ee_geometry,
                scale=20,
                numPixels=1,
                geometries=True,
            ).first().getInfo()  # force evaluation to catch errors
            break
        elif cfg.export.destination == "xee-local":
            try:
                n_images = imgc.size().getInfo()
            except ee.ee_exception.EEException as exc:
                if (
                    "Image.bandNames: Parameter 'image' is required and may not be null"
                    in str(exc)
                ):
                    logger.warning(
                        f"Skipping interval {interval_start} to {interval_end}: No images found for this interval."
                    )
                    continue
                raise
            if n_images == 0:
                logger.warning(
                    f"Skipping interval {interval_start} to {interval_end}: No images found for this interval."
                )
                continue

            input_ds = convert_s2_input_to_xarray(cfg, imgc)
            output_ds = biophys_predict_local(
                input_ds,
                variable=cfg.variables.variable,
                model=cfg.variables.model,
                clip_min_max=cfg.options.clip_min_max,
                biome_lc_name=cfg.variables.biome_lc_name,
            )
            print(output_ds.variables)


@pytest.mark.parametrize(
    "config_path",
    [
        "example_configs/minimal_example.yaml",
        "example_configs/bimonthly-zambia.yaml",
        "example_configs/seasonal-summer-zurich.yaml",
        "example_configs/forest-fire-bitsch-2023.yaml",
    ],
)
def test_configs(config_path):
    cfg = load_params(config_path)

    if config_path == "example_configs/forest-fire-bitsch-2023.yaml":
        assert cfg.export.crs == "EPSG:32632"  # ensure LOCAL_UTM is resolved

    assert isinstance(cfg, ConfigParams)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
