from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from gee_biophys.models.grounded_eo import load_grounded_eo_model, predict_grounded_eo
from gee_biophys.s2_predict import _predict_local_composite_fast


def test_grounded_eo_model_metadata():
    model, upper_lim, band_order = load_grounded_eo_model("LAI")
    assert model is not None
    assert upper_lim == 10.0
    assert "B1" in band_order
    assert "cos_sza" in band_order

    _, fapar_upper_lim, _ = load_grounded_eo_model("FAPAR")
    assert fapar_upper_lim == 1.0


def test_predict_grounded_eo_shapes_and_ranges():
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        {
            "B1": rng.uniform(0.02, 0.18, size=5),
            "B2": rng.uniform(0.02, 0.22, size=5),
            "B3": rng.uniform(0.03, 0.24, size=5),
            "B4": rng.uniform(0.04, 0.30, size=5),
            "B5": rng.uniform(0.05, 0.35, size=5),
            "B6": rng.uniform(0.06, 0.40, size=5),
            "B7": rng.uniform(0.07, 0.45, size=5),
            "B8": rng.uniform(0.08, 0.52, size=5),
            "B8A": rng.uniform(0.09, 0.60, size=5),
            "B9": rng.uniform(0.10, 0.50, size=5),
            "B11": rng.uniform(0.08, 0.40, size=5),
            "B12": rng.uniform(0.05, 0.30, size=5),
            "sza": rng.uniform(20, 60, size=5),
            "vza": rng.uniform(15, 45, size=5),
            "phi": rng.uniform(0, 180, size=5),
        }
    )

    out = predict_grounded_eo(df)

    required = {
        "grounded_lai_mean",
        "grounded_lai_std",
        "grounded_fapar_mean",
        "grounded_fapar_std",
    }
    assert required.issubset(out.columns)
    assert out["grounded_lai_mean"].between(0, 10).all()
    assert out["grounded_fapar_mean"].between(0, 1).all()
    assert out["grounded_lai_std"].ge(0).all()
    assert out["grounded_fapar_std"].ge(0).all()


def test_composite_then_predict_fast_mode_has_zero_across_std():
    cfg = SimpleNamespace(
        options=SimpleNamespace(clip_min_max=True),
        variables=SimpleNamespace(variable="lai"),
    )

    input_ds = xr.Dataset(
        data_vars={
            "B2": (("time", "y", "x"), np.array([[[0.1]], [[0.2]]], dtype=float)),
            "B3": (("time", "y", "x"), np.array([[[0.2]], [[0.4]]], dtype=float)),
        },
        coords={"time": [0, 1], "y": [0], "x": [0]},
    )

    def predict_fn(matrix):
        return np.asarray([matrix.shape[0] * 1.0], dtype=float), np.asarray(
            [0.5],
            dtype=float,
        )

    out = _predict_local_composite_fast(
        cfg,
        input_ds,
        ["B2", "B3"],
        predict_fn,
        output_prefix="lai",
    )

    assert np.isclose(out["lai_stdDev_across"].item(), 0.0)
    assert np.isclose(out["lai_stdDev"].item(), out["lai_stdDev_within"].item())
    assert out["lai_count"].item() == 2
