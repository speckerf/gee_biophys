import pytest
import yaml

from gee_biophys.config import ConfigParams
from gee_biophys.s2_input import load_s2_input
from gee_biophys.s2_predict import biophys_predict

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
    ],
)
def test_cli(ee_init, config_path):
    cfg = load_params(config_path)

    for interval_start, interval_end in cfg.temporal.iter_date_ranges():
        imgc = load_s2_input(cfg, interval_start, interval_end)
        imgc.getInfo()  # force evaluation to catch errors

        output_image = biophys_predict(cfg, imgc)
        output_image.getInfo()  # force evaluation to catch errors
        break


@pytest.mark.parametrize(
    "config_path",
    [
        "example_configs/minimal_example.yaml",
        "example_configs/bimonthly-zambia.yaml",
        "example_configs/seasonal-summer-zurich.yaml",
    ],
)
def test_configs(config_path):
    cfg = load_params(config_path)
    assert isinstance(cfg, ConfigParams)


@pytest.mark.parametrize(
    "config_path",
    [
        "example_configs/minimal_example.yaml",
        "example_configs/bimonthly-zambia.yaml",
        "example_configs/seasonal-summer-zurich.yaml",
    ],
)
def test_interval_iteration(config_path):
    cfg = load_params(config_path)

    intervals = list(cfg.temporal.iter_date_ranges())
    assert len(intervals) > 0, "No date ranges generated from config temporal settings."
