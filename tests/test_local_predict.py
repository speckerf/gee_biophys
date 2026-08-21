import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gee_biophys.s2_predict import (
    _aggregate_prediction_times,
    _calibrate_std_local,
    biophys_predict_local,
)

ALL_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B11",
    "B12",
]

S2BIOPHYS_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]

SL2P_BANDS = [
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",
]

ANGLE_BANDS = ["tts", "tto", "psi"]


def _make_s2_dataset(
    *,
    shape: tuple[int, int, int] = (3, 2, 2),
    seed: int = 0,
) -> xr.Dataset:
    """Create a synthetic common Sentinel-2 input dataset."""
    rng = np.random.default_rng(seed)
    dims = ("time", "y", "x")

    data_vars = {band: (dims, rng.uniform(0.01, 0.5, size=shape)) for band in ALL_BANDS}

    data_vars.update(
        {
            "tts": (dims, rng.uniform(0, 60, size=shape)),
            "tto": (dims, rng.uniform(0, 12, size=shape)),
            "psi": (dims, rng.uniform(0, 180, size=shape)),
        }
    )

    return xr.Dataset(data_vars)


def test_biophys_predict_local_s2biophys_outputs():
    ds = _make_s2_dataset(
        shape=(3, 2, 2),
        seed=7,
    )
    ds["B2"][0, 0, 0] = np.nan

    out = biophys_predict_local(
        ds, variable="fapar", model="s2biophys", clip_min_max=True
    )

    assert set(out.data_vars) == {
        "fapar_mean",
        "fapar_stdDev",
        "fapar_stdDev_within",
        "fapar_stdDev_across",
        "fapar_count",
    }

    assert out["fapar_mean"].shape == (2, 2)
    assert float(out["fapar_mean"].min()) >= 0.0
    assert float(out["fapar_mean"].max()) <= 1.0
    assert int(out["fapar_count"].min()) >= 0


def test_biophys_predict_local_sl2p_outputs():
    ds = _make_s2_dataset(
        shape=(3, 2, 2),
        seed=11,
    )

    out = biophys_predict_local(ds, variable="laie", model="sl2p", clip_min_max=True)

    assert set(out.data_vars) == {
        "laie_mean",
        "laie_stdDev",
        "laie_stdDev_within",
        "laie_stdDev_across",
        "laie_count",
    }

    assert out["laie_mean"].shape == (2, 2)
    assert float(out["laie_mean"].min()) >= 0.0
    assert float(out["laie_mean"].max()) <= 8.0


def test_biophys_predict_local_s2biophys_uncertainty_calibration(
    monkeypatch,
):
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

        calibration_table = pd.DataFrame(
            {
                "y_pred": [0.0, 10.0],
                "tau": [2.0, 2.0],
            }
        )

        return ensemble, (None, calibration_table)

    monkeypatch.setattr(
        "gee_biophys.s2_predict.load_model_ensemble",
        fake_load_model_ensemble,
    )

    dims = ("time", "y", "x")
    shape = (2, 1, 1)

    ds = xr.Dataset(
        {band: (dims, np.full(shape, 1.0)) for band in S2BIOPHYS_BANDS}
        | {angle: (dims, np.full(shape, 20.0)) for angle in ANGLE_BANDS}
    )

    out = biophys_predict_local(
        ds, variable="laie", model="s2biophys", clip_min_max=True
    )

    # Ensemble predictions are [1, 3].
    # Raw uncertainty is their sample SD, multiplied by tau=2.
    expected_within = (
        np.std(
            [1.0, 3.0],
            ddof=1,
        )
        * 2.0
    )

    assert out["laie_mean"].item() == pytest.approx(2.0)
    assert out["laie_stdDev_within"].item() == pytest.approx(expected_within)
    assert out["laie_stdDev_across"].item() == pytest.approx(0.0)
    assert out["laie_stdDev"].item() == pytest.approx(expected_within)
    assert out["laie_count"].item() == 2


def test_calibrated_stddev_is_aggregated_into_total_uncertainty():
    means = np.array(
        [0.2, 0.5, 0.9],
        dtype=float,
    )
    stds = np.array(
        [0.1, 0.2, 0.3],
        dtype=float,
    )

    calibration_table = pd.DataFrame(
        {
            "y_pred": [0.0, 0.5, 1.0],
            "tau": [1.0, 1.5, 2.0],
        }
    )

    pred_mean_time = means.reshape(-1, 1, 1)
    pred_std_time = stds.reshape(-1, 1, 1)
    pred_std_time_cal = _calibrate_std_local(
        pred_mean_time,
        pred_std_time,
        calibration_table,
    )
    mean, total, within, across, count = _aggregate_prediction_times(
        pred_mean_time, pred_std_time_cal
    )

    expected_calibrated_stds = np.array([0.12, 0.30, 0.57])
    expected_within = expected_calibrated_stds.mean()
    expected_across = means.std(ddof=1)

    np.testing.assert_allclose(pred_std_time_cal[:, 0, 0], expected_calibrated_stds)
    assert mean.item() == pytest.approx(means.mean())
    assert within.item() == pytest.approx(expected_within)
    assert across.item() == pytest.approx(expected_across)
    assert total.item() == pytest.approx(
        np.sqrt(expected_within**2 + expected_across**2)
    )
    assert count.item() == 3


def test_biophys_predict_local_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model: unknown"):
        biophys_predict_local(_make_s2_dataset(), variable="fapar", model="unknown")
