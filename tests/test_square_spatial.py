"""Square validation offline; opt in to geometry checks with RUN_EE_TESTS=1."""

import os

import ee
import numpy as np
import pytest
from pydantic import ValidationError

from gee_biophys.config import ConfigParams, Spatial, _latlon_to_utm_epsg


@pytest.mark.parametrize("length", [0, -1, float("inf"), float("nan")])
def test_square_rejects_invalid_length(length):
    with pytest.raises(ValidationError):
        Spatial(type="square", square_center=[8.4, 47.48], square_length=length)


@pytest.mark.parametrize(
    "center",
    [
        [],
        [8],
        [8, 47, 1],
        [181, 47],
        [8, 85],
        [8, -81],
        [float("nan"), 47],
        [8, float("inf")],
    ],
)
def test_square_rejects_invalid_center(center):
    with pytest.raises(ValidationError):
        Spatial(type="square", square_center=center, square_length=2000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"square_center": [8, 47]},
        {"square_length": 2000},
        {"bbox": [7, 46, 8, 47], "square_center": [8, 47], "square_length": 2000},
        {
            "geojson_path": "unused.geojson",
            "square_center": [8, 47],
            "square_length": 2000,
        },
    ],
)
def test_square_requires_both_fields_and_no_other_extent(kwargs):
    with pytest.raises(ValidationError):
        Spatial(type="square", **kwargs)


@pytest.mark.parametrize("kind", ["bbox", "geojson"])
def test_square_fields_cannot_be_used_with_other_types(kind):
    with pytest.raises(ValidationError):
        Spatial(type=kind, square_center=[8, 47], square_length=2000)


@pytest.mark.parametrize(
    ("crs", "expected"),
    [
        ("EPSG:4326", "EPSG:4326"),
        ("EPSG:3857", "EPSG:3857"),
        ("LOCAL_UTM", "EPSG:32632"),
    ],
)
def test_square_output_crs_is_independent(monkeypatch, crs, expected):
    monkeypatch.setattr(ee, "Initialize", lambda **kwargs: None)
    cfg = ConfigParams(
        spatial={
            "type": "square",
            "square_center": [8.4, 47.48],
            "square_length": 2000,
        },
        temporal={
            "start": "2024-06-01",
            "end": "2024-07-01",
            "cadence": {"type": "fixed", "interval": "monthly"},
        },
        variables={},
        options={},
        export={"destination": "drive", "folder": "test", "scale": 20, "crs": crs},
    )
    assert cfg.export.crs == expected
    assert cfg.spatial.square_length == 2000


def test_utm_zone_at_eastern_longitude_boundary():
    assert _latlon_to_utm_epsg(0, 180) == 32660
    assert _latlon_to_utm_epsg(-20, -180) == 32701


@pytest.mark.skipif(
    os.environ.get("RUN_EE_TESTS") != "1", reason="Requires Earth Engine"
)
@pytest.mark.parametrize("center", [[8.4, 47.48], [25, -20], [5, 60], [20, 75]])
def test_square_roundtrip_preserves_utm_side_lengths_and_center(center):
    ee.Initialize()
    spatial = Spatial(type="square", square_center=center, square_length=2000)
    lon, lat = center
    utm = f"EPSG:{_latlon_to_utm_epsg(lat, lon)}"
    # ee_geometry is the polygon in EPSG:4326 used for any export CRS.
    geometry = spatial.ee_geometry
    assert geometry.projection().crs().getInfo() == "EPSG:4326"
    projected = geometry.transform(utm, 0.01)
    bounds = np.asarray(projected.bounds(0.01, utm).coordinates().getInfo()[0])
    np.testing.assert_allclose(np.ptp(bounds, axis=0), [2000, 2000], atol=0.05)
    center_utm = ee.Geometry.Point(center).transform(utm, 0.01).coordinates().getInfo()
    np.testing.assert_allclose(
        (bounds.min(0) + bounds.max(0)) / 2, center_utm, atol=0.05
    )
    assert projected.area(0.01, utm).getInfo() == pytest.approx(2000**2, abs=100)


def test_square_drive_export_clips_polygon_and_keeps_output_crs(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from gee_biophys.s2_export import export_image

    class Image:
        def __init__(self):
            self.clip = MagicMock(return_value=self)
            self.float = MagicMock(return_value=self)

    image, geometry = Image(), object()
    monkeypatch.setattr(ee, "Image", Image)
    export = MagicMock()
    monkeypatch.setattr(ee.batch.Export.image, "toDrive", export)
    cfg = SimpleNamespace(
        spatial=SimpleNamespace(type="square", ee_geometry=geometry),
        export=SimpleNamespace(
            destination="drive", folder="out", scale=20, crs="EPSG:4326"
        ),
    )
    export_image(image, "square", cfg)
    image.clip.assert_called_once_with(geometry)
    assert export.call_args.kwargs["region"] is geometry
    assert export.call_args.kwargs["crs"] == "EPSG:4326"
    export.return_value.start.assert_called_once()


def test_square_xee_download_masks_outside_polygon(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import xarray as xr
    import xee

    from gee_biophys.s2_input import convert_s2_input_to_xarray

    geometry, collection, image = MagicMock(), MagicMock(), MagicMock()
    geometry.bounds.return_value.coordinates.return_value.getInfo.return_value = [
        [[8.39, 47.47], [8.41, 47.47], [8.41, 47.49], [8.39, 47.49], [8.39, 47.47]]
    ]
    collection.map.side_effect = lambda fn: fn(image)
    monkeypatch.setattr(
        xee.helpers, "extract_grid_params", lambda _: {"crs": "EPSG:32632"}
    )
    monkeypatch.setattr(xee.helpers, "fit_geometry", lambda **kwargs: {})
    opened = MagicMock()
    monkeypatch.setattr(xr, "open_dataset", opened)
    cfg = SimpleNamespace(
        spatial=SimpleNamespace(type="square", ee_geometry=geometry),
        export=SimpleNamespace(scale=20),
        options=SimpleNamespace(prediction_mode="predict_then_aggregate"),
    )
    convert_s2_input_to_xarray(cfg, collection)
    image.clip.assert_called_once_with(geometry)
    assert opened.call_args.args[0] is image.clip.return_value
