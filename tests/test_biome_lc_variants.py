"""Targeted validation and real packaged-model regressions (offline)."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from test_local_predict import _make_s2_dataset

from gee_biophys.config import ConfigParams, Variables
from gee_biophys.model_variants import GROUP_NAME_TO_GROUP_NUM
from gee_biophys.models.s2biophys import load_biome_land_cover_specific_model_ensemble
from gee_biophys.s2_predict import biophys_predict_local
from gee_biophys.utils import get_system_index

MODEL = "s2biophys-biome-lc-specific"


@pytest.mark.parametrize("name", GROUP_NAME_TO_GROUP_NUM)
@pytest.mark.parametrize("trait", ["laie", "fapar", "fcover"])
def test_named_variants_load_and_predict_full_grid(name, trait):
    Variables(model=MODEL, biome_lc_name=name, variable=trait)
    ensemble, (_, table) = load_biome_land_cover_specific_model_ensemble(trait, name)
    assert len(ensemble) == 3
    assert all(f"group{GROUP_NAME_TO_GROUP_NUM[name]}-" in key for key in ensemble)
    assert {"y_pred", "tau"} <= set(table.columns)
    ds = _make_s2_dataset(shape=(2, 2, 3), seed=42)
    # No land-cover input: every valid location must receive predictions.
    ds["B2"][:, 0, 0] = np.nan
    out = biophys_predict_local(ds, variable=trait, model=MODEL, biome_lc_name=name)
    assert out[f"{trait}_mean"].shape == (2, 3)
    assert np.isnan(out[f"{trait}_mean"][0, 0])
    assert np.isfinite(out[f"{trait}_mean"].values.ravel()[1:]).all()
    np.testing.assert_array_equal(out[f"{trait}_count"], [[0, 2, 2], [2, 2, 2]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"biome_lc_name": "unknown"},
        {"biome_lc_name": "3"},
        {"biome_lc_name": 3},
        {"biome_lc_num": 3},
        {"biome_lc_name": "arid_shrubland", "biome_lc_num": 3},
        {"biome_lc_name": "arid_shrubland", "variable": "lai"},
    ],
)
def test_invalid_variant_configs_fail_early(kwargs):
    with pytest.raises(ValidationError):
        Variables(model=MODEL, **kwargs)


def test_variant_cannot_be_silently_ignored():
    with pytest.raises(ValidationError, match="only valid"):
        Variables(model="s2biophys", biome_lc_name="arid_shrubland")
    for value in [3, "3", None, "unknown"]:
        with pytest.raises(ValueError, match="biome_lc_name"):
            load_biome_land_cover_specific_model_ensemble("fapar", value)
    with pytest.raises(TypeError):
        load_biome_land_cover_specific_model_ensemble("fapar", biome_lc_num=3)


@pytest.mark.parametrize(
    "path", sorted(Path("example_configs").glob("biome-lc-*.yaml"))
)
def test_example_configs(monkeypatch, path):
    monkeypatch.setattr("ee.Initialize", lambda **kwargs: None)
    data = yaml.safe_load(path.read_text())

    lon, lat = data["spatial"]["square_center"]
    bbox = [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]

    data["spatial"].update(type="bbox", bbox=bbox)
    data["spatial"].pop("square_center", None)
    data["spatial"].pop("square_length", None)

    geometry = MagicMock()
    geometry.centroid.return_value.coordinates.return_value.getInfo.return_value = [
        lon,
        lat,
    ]
    monkeypatch.setattr("ee.Geometry.BBox", lambda *args: geometry)

    cfg = ConfigParams(**data)
    west, south, east, north = cfg.spatial.bbox

    assert 0 < east - west < 0.5
    assert 0 < north - south < 0.5
    assert len(list(cfg.temporal.iter_date_ranges())) == 1
    assert len(get_system_index(cfg, cfg.temporal.start, cfg.temporal.end)) <= 100


def test_cli_forwards_named_variant_to_gee(monkeypatch):
    """Catch positional-argument and selector-keyword regressions in the CLI."""
    from datetime import UTC, datetime
    from inspect import signature
    from types import SimpleNamespace

    from gee_biophys import cli
    from gee_biophys.s2_predict import biophys_predict_ee

    cfg = SimpleNamespace(
        variables=Variables(model=MODEL, biome_lc_name="arid_shrubland"),
        options=SimpleNamespace(clip_min_max=True),
        export=SimpleNamespace(destination="drive"),
        temporal=SimpleNamespace(
            iter_date_ranges=lambda: iter(
                [(datetime(2024, 9, 1, tzinfo=UTC), datetime(2024, 10, 1, tzinfo=UTC))]
            )
        ),
    )
    collection, prediction = object(), object()
    calls = []
    original_signature = signature(biophys_predict_ee)

    def predict(*args, **kwargs):
        bound = original_signature.bind(*args, **kwargs).arguments
        assert bound["input_imgc"] is collection
        assert bound["biome_lc_name"] == "arid_shrubland"
        assert bound["model"] == MODEL
        calls.append(bound)
        return prediction

    monkeypatch.setattr(cli, "load_params", lambda _: cfg)
    monkeypatch.setattr(cli, "initialize_export_location", lambda *a, **k: None)
    monkeypatch.setattr(cli, "load_s2_input", lambda *a: collection)
    monkeypatch.setattr(cli, "get_system_index", lambda *a: "test")
    monkeypatch.setattr(cli, "biophys_predict_ee", predict)
    monkeypatch.setattr(cli, "update_image_metadata", lambda image, *a: image)
    exported = []
    monkeypatch.setattr(cli, "export_image", lambda image, *a: exported.append(image))
    cli.run_pipeline("example.yaml")
    assert len(calls) == 1
    assert exported == [prediction]
