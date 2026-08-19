from types import SimpleNamespace

import ee
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gee_biophys.models.s2biophys import _ee_calibrate_std
from gee_biophys.s2_predict import _calibrate_std_local, biophys_predict_local
from gee_biophys.utils_predict import reduce_ensemble_preds


def _make_cfg(model: str, variable: str, bands: list[str]):
    return SimpleNamespace(
        variables=SimpleNamespace(model=model, variable=variable, bands=bands),
        options=SimpleNamespace(clip_min_max=True),
    )


def test_biophys_predict_local_s2biophys_outputs():
    rng = np.random.default_rng(7)
    dims = ("time", "y", "x")
    shape = (3, 2, 2)

    data_vars = {}
    for band in ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]:
        data_vars[band] = (dims, rng.uniform(0.01, 0.4, size=shape))

    data_vars["tts"] = (dims, rng.uniform(0, 60, size=shape))
    data_vars["tto"] = (dims, rng.uniform(0, 60, size=shape))
    data_vars["psi"] = (dims, rng.uniform(0, 180, size=shape))

    ds = xr.Dataset(data_vars)
    ds["B2"][0, 0, 0] = np.nan

    cfg = _make_cfg(
        model="s2biophys",
        variable="fapar",
        bands=["mean", "stdDev", "stdDev_within", "stdDev_across", "count"],
    )

    out = biophys_predict_local(cfg, ds)

    assert set(out.data_vars) == {
        "fapar_mean",
        "fapar_stdDev",
        "fapar_stdDev_within",
        "fapar_stdDev_across",
        "fapar_count",
    }
    assert out["fapar_mean"].shape == (2, 2)
    assert float(out["fapar_mean"].max()) <= 1.0
    assert float(out["fapar_mean"].min()) >= 0.0
    assert int(out["fapar_count"].min()) >= 0


def test_biophys_predict_local_sl2p_outputs():
    rng = np.random.default_rng(11)
    dims = ("time", "y", "x")
    shape = (3, 2, 2)

    data_vars = {
        "cosSZA": (dims, rng.uniform(-1.0, 1.0, size=shape)),
        "cosVZA": (dims, rng.uniform(-1.0, 1.0, size=shape)),
        "cosRAA": (dims, rng.uniform(-1.0, 1.0, size=shape)),
        "B3": (dims, rng.uniform(0.01, 0.5, size=shape)),
        "B4": (dims, rng.uniform(0.01, 0.5, size=shape)),
        "B5": (dims, rng.uniform(0.01, 0.5, size=shape)),
        "B6": (dims, rng.uniform(0.01, 0.5, size=shape)),
        "B7": (dims, rng.uniform(0.01, 0.5, size=shape)),
        "B8A": (dims, rng.uniform(0.01, 0.5, size=shape)),
        "B11": (dims, rng.uniform(0.01, 0.5, size=shape)),
        "B12": (dims, rng.uniform(0.01, 0.5, size=shape)),
    }
    ds = xr.Dataset(data_vars)

    cfg = _make_cfg(
        model="sl2p",
        variable="laie",
        bands=["mean", "stdDev", "count"],
    )

    out = biophys_predict_local(cfg, ds)

    assert set(out.data_vars) == {"laie_mean", "laie_stdDev", "laie_count"}
    assert out["laie_mean"].shape == (2, 2)
    assert float(out["laie_mean"].max()) <= 8.0
    assert float(out["laie_mean"].min()) >= 0.0


def test_biophys_predict_local_s2biophys_uncertainty_calibration(monkeypatch):
    class DummyPipeline:
        def __init__(self, offset: float):
            self.offset = offset

        def predict(self, df):
            return (df["B2"].to_numpy() + self.offset).reshape(-1, 1)

    def fake_load_model_ensemble(_trait):
        ensemble = {
            "m0": {"pipeline": DummyPipeline(0.0)},
            "m1": {"pipeline": DummyPipeline(2.0)},
        }
        calib_table = pd.DataFrame({"y_pred": [0.0, 10.0], "tau": [2.0, 2.0]})
        return ensemble, (None, calib_table)

    monkeypatch.setattr(
        "gee_biophys.s2_predict.load_model_ensemble",
        fake_load_model_ensemble,
    )

    dims = ("time", "y", "x")
    shape = (2, 1, 1)
    ds = xr.Dataset(
        {
            "B2": (dims, np.full(shape, 1.0)),
            "B3": (dims, np.full(shape, 1.0)),
            "B4": (dims, np.full(shape, 1.0)),
            "B5": (dims, np.full(shape, 1.0)),
            "B6": (dims, np.full(shape, 1.0)),
            "B7": (dims, np.full(shape, 1.0)),
            "B8": (dims, np.full(shape, 1.0)),
            "B8A": (dims, np.full(shape, 1.0)),
            "B11": (dims, np.full(shape, 1.0)),
            "B12": (dims, np.full(shape, 1.0)),
            "tts": (dims, np.full(shape, 20.0)),
            "tto": (dims, np.full(shape, 20.0)),
            "psi": (dims, np.full(shape, 20.0)),
        }
    )

    cfg = _make_cfg(
        model="s2biophys",
        variable="laie",
        bands=["mean", "stdDev", "stdDev_within", "stdDev_across", "count"],
    )

    out = biophys_predict_local(cfg, ds)

    assert out["laie_mean"].item() == pytest.approx(2.0)
    assert out["laie_stdDev_within"].item() == pytest.approx(2.0)
    assert out["laie_stdDev_across"].item() == pytest.approx(0.0)
    assert out["laie_stdDev"].item() == pytest.approx(2.0)
    assert out["laie_count"].item() == 2


def test_s2biophys_calibrated_stddev_ee_local_parity(ee_init):
    variable = "fapar"
    mean_band = f"{variable}_mean"
    std_band = f"{variable}_stdDev"

    # Synthetic per-image predictions for one pixel across three timesteps.
    means = np.array([0.2, 0.5, 0.9], dtype=float)
    stds = np.array([0.1, 0.2, 0.3], dtype=float)
    calibration_table = pd.DataFrame(
        {
            "y_pred": [0.0, 0.5, 1.0],
            "tau": [1.0, 1.5, 2.0],
        }
    )

    # ---- local path: calibrate std per timestep, then aggregate ----
    pred_mean_time = means.reshape(-1, 1, 1)
    pred_std_time = stds.reshape(-1, 1, 1)
    pred_std_time_cal = _calibrate_std_local(
        pred_mean_time,
        pred_std_time,
        calibration_table,
    )

    local_mean = float(np.nanmean(pred_mean_time, axis=0).item())
    local_std_within = float(np.nanmean(pred_std_time_cal, axis=0).item())
    local_std_across = float(np.nanstd(pred_mean_time, axis=0, ddof=1).item())
    local_std_total = float(np.sqrt(local_std_within**2 + local_std_across**2))
    local_count = int(np.sum(np.isfinite(pred_mean_time), axis=0).item())

    # ---- EE path: same means/stds -> EE calibration -> EE reduction ----
    ee_images = []
    for m, s in zip(means, stds):
        img = (
            ee.Image.constant([float(m), float(s)])
            .rename([mean_band, std_band])
            .toFloat()
        )
        img_cal = _ee_calibrate_std(img, calibration_table, variable)
        ee_images.append(img_cal)

    ee_imgc = ee.ImageCollection(ee_images)
    ee_reduced = reduce_ensemble_preds(ee_imgc, variable)

    ee_values = ee_reduced.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=ee.Geometry.Point([0, 0]),
        scale=1000,
        bestEffort=True,
        maxPixels=1e9,
    ).getInfo()

    assert ee_values[mean_band] == pytest.approx(local_mean, abs=1e-6)
    assert ee_values[f"{variable}_stdDev_within"] == pytest.approx(
        local_std_within,
        abs=1e-6,
    )
    assert ee_values[f"{variable}_stdDev_across"] == pytest.approx(
        local_std_across,
        abs=1e-6,
    )
    assert ee_values[std_band] == pytest.approx(local_std_total, abs=1e-6)
    assert int(ee_values[f"{variable}_count"]) == local_count
