import pytest
import yaml

from gee_biophys.config import ConfigParams
from gee_biophys.s2_input import load_s2_input
from gee_biophys.s2_predict import biophys_predict_ee

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
    ],
)
def test_cli(ee_init, config_path):
    cfg = load_params(config_path)

    for interval_start, interval_end in cfg.temporal.iter_date_ranges():
        imgc = load_s2_input(cfg, interval_start, interval_end)
        imgc.getInfo()  # force evaluation to catch errors

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
