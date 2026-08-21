import numpy as np
import pytest
import xarray as xr

from gee_biophys.models.grounded_eo import (
    load_grounded_eo_model,
    prepare_s2_ds_for_groundedeo,
)
from gee_biophys.s2_predict import _predict_local_grounded_eo, biophys_predict_local


S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
MODEL_FEATURES = S2_BANDS + ["cos_sza", "cos_vza", "cos_raa"]


def _make_s2_dataset(shape=(2, 1, 2)):
    dims = ("time", "y", "x")
    data_vars = {
        band: (dims, np.full(shape, 0.1 + index / 100.0))
        for index, band in enumerate(S2_BANDS)
    }
    data_vars.update(
        {
            "tts": (dims, np.full(shape, 60.0)),
            "tto": (dims, np.full(shape, 90.0)),
            "psi": (dims, np.full(shape, 180.0)),
        }
    )
    return xr.Dataset(
        data_vars,
        coords={"time": [10, 20], "y": [100], "x": [200, 300]},
    )


@pytest.mark.parametrize(
    ("variable", "expected_path", "expected_upper_limit"),
    [("LAI", "lai.onnx", 10.0), ("fapar", "fapar.onnx", 1.0)],
)
def test_load_grounded_eo_model_uses_cpu_onnx_session(
    monkeypatch, variable, expected_path, expected_upper_limit
):
    sentinel_session = object()
    calls = []

    def fake_session(path, *, providers):
        calls.append((path, providers))
        return sentinel_session

    monkeypatch.setattr(
        "gee_biophys.models.grounded_eo.ort.InferenceSession", fake_session
    )

    model, upper_limit = load_grounded_eo_model(variable)

    assert model is sentinel_session
    assert upper_limit == expected_upper_limit
    assert len(calls) == 1
    assert calls[0][0].endswith(expected_path)
    assert calls[0][1] == ["CPUExecutionProvider"]


@pytest.mark.parametrize("variable", ["laie", "FCOVER"])
def test_load_grounded_eo_model_rejects_unsupported_traits(variable):
    with pytest.raises(ValueError, match="only available for 'LAI' and 'FAPAR'"):
        load_grounded_eo_model(variable)


def test_prepare_s2_dataset_selects_and_transforms_model_features():
    source = _make_s2_dataset()
    source["unrelated"] = xr.ones_like(source["B1"])

    prepared = prepare_s2_ds_for_groundedeo(source)

    assert list(prepared.data_vars) == MODEL_FEATURES
    xr.testing.assert_identical(prepared["B1"], source["B1"])
    np.testing.assert_allclose(prepared["cos_sza"], 0.5, atol=1e-15)
    np.testing.assert_allclose(prepared["cos_vza"], 0.0, atol=1e-15)
    np.testing.assert_allclose(prepared["cos_raa"], -1.0, atol=1e-15)
    assert prepared.sizes == source.sizes


def test_predict_local_grounded_eo_masks_invalid_rows_and_aggregates(monkeypatch):
    prepared = prepare_s2_ds_for_groundedeo(_make_s2_dataset())
    prepared["B1"].values[:] = [[[-0.2, 0.2]], [[1.2, np.nan]]]
    calls = []

    class DummySession:
        def get_inputs(self):
            return [type("Input", (), {"name": "features"})()]

        def run(self, output_names, feeds):
            calls.append((output_names, feeds))
            means = feeds["features"][:, 0] * 10.0
            return means[:, None], np.full((means.size, 1), 0.5)

    monkeypatch.setattr(
        "gee_biophys.s2_predict.load_grounded_eo_model",
        lambda variable: (DummySession(), 10.0),
    )

    result = _predict_local_grounded_eo(prepared, variable="lai")

    assert set(result.data_vars) == {
        "lai_mean",
        "lai_stdDev",
        "lai_stdDev_within",
        "lai_stdDev_across",
        "lai_count",
    }
    assert result.sizes == {"y": 1, "x": 2}
    assert calls[0][0] is None
    assert calls[0][1]["features"].shape == (3, len(MODEL_FEATURES))
    assert calls[0][1]["features"].dtype == np.float64
    np.testing.assert_allclose(result["lai_mean"], [[5.0, 2.0]])
    np.testing.assert_allclose(result["lai_stdDev_within"], [[0.5, 0.5]])
    np.testing.assert_allclose(result["lai_stdDev_across"], [[np.sqrt(50.0), 0.0]])
    np.testing.assert_allclose(result["lai_stdDev"], [[np.sqrt(50.25), 0.5]])
    np.testing.assert_array_equal(result["lai_count"], [[2, 1]])


def test_public_local_predictor_prepares_grounded_eo_input(monkeypatch):
    source = _make_s2_dataset()
    expected = xr.Dataset({"result": (("y", "x"), [[1.0, 2.0]])})
    received = {}

    def fake_predict(prepared, *, variable, clip_min_max):
        received.update(prepared=prepared, variable=variable, clip_min_max=clip_min_max)
        return expected

    monkeypatch.setattr(
        "gee_biophys.s2_predict._predict_local_grounded_eo", fake_predict
    )

    result = biophys_predict_local(
        source, variable="fapar", model="groundedeo", clip_min_max=False
    )

    assert result is expected
    assert list(received["prepared"].data_vars) == MODEL_FEATURES
    assert received["variable"] == "fapar"
    assert received["clip_min_max"] is False
