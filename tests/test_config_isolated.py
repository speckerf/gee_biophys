from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gee_biophys.config import (
    ExportOpts,
    FixedCadence,
    Options,
    SeasonsCadence,
    Spatial,
    Temporal,
    Variables,
    _latlon_to_utm_epsg,
    _safe_ymd,
)


def test_spatial_accepts_valid_bbox():
    spatial = Spatial(type="bbox", bbox=[7.0, 46.0, 8.0, 47.0])
    assert spatial.bbox == [7.0, 46.0, 8.0, 47.0]


@pytest.mark.parametrize(
    "bbox",
    [[0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [-181, 0, 1, 1], [0, -91, 1, 1]],
)
def test_spatial_rejects_invalid_bbox(bbox):
    with pytest.raises(ValidationError):
        Spatial(type="bbox", bbox=bbox)


def test_spatial_requires_exactly_one_location_and_clean_region_name(tmp_path):
    geojson = tmp_path / "region.geojson"
    geojson.write_text('{"type":"Polygon","coordinates":[]}', encoding="utf-8")
    with pytest.raises(ValidationError, match="exactly one"):
        Spatial(type="bbox", bbox=[0, 0, 1, 1], geojson_path=str(geojson))
    with pytest.raises(ValidationError, match="region_name"):
        Spatial(type="bbox", bbox=[0, 0, 1, 1], region_name="bad name")


def test_spatial_validates_geojson_file(tmp_path):
    invalid = tmp_path / "broken.geojson"
    invalid.write_text("not json", encoding="utf-8")
    with pytest.raises(ValidationError):
        Spatial(type="geojson", geojson_path=str(invalid))
    with pytest.raises(ValidationError, match="does not exist"):
        Spatial(type="geojson", geojson_path=str(tmp_path / "missing.geojson"))


def test_fixed_cadence_normalizes_and_rejects_nonpositive_days():
    assert FixedCadence(type="fixed", interval=" Annual ").interval == "yearly"
    for value in (0, -1):
        with pytest.raises(ValidationError, match="must be > 0"):
            FixedCadence(type="fixed", interval=value)


@pytest.mark.parametrize("value", ["2-01", "02-1", "13-01", "01-00"])
def test_season_dates_require_zero_padded_mm_dd(value):
    with pytest.raises(ValidationError, match="zero-padded"):
        SeasonsCadence(type="seasons", start=value, end="03-01")


def test_temporal_parses_utc_and_rejects_non_utc_or_reverse():
    cadence = {"type": "fixed", "interval": 1}
    temporal = Temporal(start="2024-01-01", end="2024-01-02T00:00:00Z", cadence=cadence)
    assert temporal.start == datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError, match="Invalid date string"):
        Temporal(start="2024-01-01T01:00:00+01:00", end="2024-01-02", cadence=cadence)
    with pytest.raises(ValidationError, match="strictly before"):
        Temporal(start="2024-01-02", end="2024-01-02", cadence=cadence)


def test_safe_ymd_handles_leap_day_only():
    assert _safe_ymd(2023, 2, 29) == datetime(2023, 2, 28, tzinfo=UTC)
    with pytest.raises(ValueError):
        _safe_ymd(2023, 4, 31)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"destination": "asset"}, "collection_path"),
        ({"destination": "drive"}, "folder"),
        ({"destination": "gcs"}, "bucket"),
        ({"destination": "xee-local"}, "folder"),
    ],
)
def test_export_destination_requires_matching_field(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        ExportOpts(crs="EPSG:4326", **kwargs)


def test_export_rejects_invalid_crs():
    with pytest.raises(ValidationError, match="crs"):
        ExportOpts(destination="drive", folder="out", crs="4326")


@pytest.mark.parametrize(
    ("model", "variable"),
    [("groundedeo", "laie"), ("s2biophys", "lai")],
)
def test_variables_reject_unsupported_model_variable_pairs(model, variable):
    with pytest.raises(ValidationError):
        Variables(model=model, variable=variable)


def test_variables_normalize_case_and_options_boundaries():
    assert Variables(model="SL2P", variable="LAI").model == "sl2p"
    assert Options(max_cloud_cover=0, cs_plus_threshold=1).cs_plus_threshold == 1
    with pytest.raises(ValidationError):
        Options(max_cloud_cover=101)


@pytest.mark.parametrize(
    ("lat", "lon", "epsg"),
    [(46.0, 8.0, 32632), (-20.0, 25.0, 32735), (60.0, 5.0, 32632), (75.0, 20.0, 32633)],
)
def test_latlon_to_utm_epsg_regular_and_special_zones(lat, lon, epsg):
    assert _latlon_to_utm_epsg(lat, lon) == epsg
