import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from gee_biophys.s2_export import export_image, merge_local_interval_exports

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None,
    reason="zarr is not installed in the current test environment",
)


def test_export_image_xee_local_writes_zarr(tmp_path):
    ds = xr.Dataset(
        {
            "fapar_mean": (("y", "x"), np.array([[0.1, 0.2], [0.3, 0.4]])),
            "fapar_stdDev": (("y", "x"), np.array([[0.01, 0.02], [0.03, 0.04]])),
        }
    )

    cfg = SimpleNamespace(
        export=SimpleNamespace(destination="xee-local", folder=str(tmp_path)),
        spatial=SimpleNamespace(type="bbox", geojson_clip=False),
    )

    export_image(ds, "demo_output", cfg)

    out_path = tmp_path / "demo_output.zarr"
    assert out_path.exists()

    ds_reloaded = xr.open_zarr(out_path)
    assert "fapar_mean" in ds_reloaded
    assert ds_reloaded["fapar_mean"].shape == (2, 2)


def test_merge_local_interval_exports(tmp_path):
    cfg = SimpleNamespace(
        export=SimpleNamespace(destination="xee-local", folder=str(tmp_path)),
        spatial=SimpleNamespace(type="bbox", geojson_clip=False),
    )

    ds_a = xr.Dataset(
        {
            "fapar_mean": (("time", "y", "x"), np.array([[[0.1, 0.2], [0.3, 0.4]]])),
            "fapar_stdDev": (
                ("time", "y", "x"),
                np.array([[[0.01, 0.02], [0.03, 0.04]]]),
            ),
        },
        coords={"time": np.array([np.datetime64("2020-01-01")])},
    )

    ds_b = xr.Dataset(
        {
            "fapar_mean": (("time", "y", "x"), np.array([[[0.5, 0.6], [0.7, 0.8]]])),
            "fapar_stdDev": (
                ("time", "y", "x"),
                np.array([[[0.05, 0.06], [0.07, 0.08]]]),
            ),
        },
        coords={"time": np.array([np.datetime64("2020-02-01")])},
    )

    export_image(ds_a, "step_a", cfg, subfolder="temp")
    export_image(ds_b, "step_b", cfg, subfolder="temp")

    final_path = merge_local_interval_exports(
        cfg,
        final_filename="final_merged",
        temp_subfolder="temp",
        cleanup_temp=True,
    )

    assert final_path.exists()
    assert not (tmp_path / "temp").exists()

    merged = xr.open_zarr(final_path)
    assert merged.sizes["time"] == 2
    assert "fapar_mean" in merged
    assert merged.attrs["system:index"] == "final_merged"
