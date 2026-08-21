import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gee_biophys.s2_predict import (
    _aggregate_prediction_times,
    _build_output_dataset,
    _calibrate_std_local,
    _clip_trait_array,
    _dataset_to_matrix,
    _interp_extrapolate_1d,
    _matrix_to_cube,
)


@pytest.mark.parametrize(
    ("trait", "upper"),
    [("lai", 8), ("laie", 8), ("fapar", 1), ("fcover", 1)],
)
def test_clip_trait_array_clamps_to_trait_range(trait, upper):
    actual = _clip_trait_array(np.array([-1.0, upper / 2, upper + 1]), trait)
    np.testing.assert_allclose(actual, [0.0, upper / 2, upper])


def test_clip_trait_array_rejects_unknown_trait():
    with pytest.raises(ValueError, match="Clipping not defined"):
        _clip_trait_array(np.array([1.0]), "unknown")


def test_dataset_matrix_round_trip_preserves_order_and_shape():
    ds = xr.Dataset(
        {
            "B1": (("time", "y", "x"), np.arange(8).reshape(2, 2, 2)),
            "B2": (("time", "y", "x"), np.arange(8, 16).reshape(2, 2, 2)),
        },
        coords={"time": [0, 1], "y": [10, 20], "x": [30, 40]},
    )

    matrix, (arr, non_time_dims, shape) = _dataset_to_matrix(ds)

    assert matrix.shape == (8, 2)
    assert non_time_dims == ["y", "x"]
    np.testing.assert_array_equal(matrix[0], [0, 8])
    np.testing.assert_array_equal(
        _matrix_to_cube(np.arange(8), shape), np.arange(8).reshape(2, 2, 2)
    )

    out = _build_output_dataset(arr, non_time_dims, {"result": np.ones((2, 2))})
    assert out["result"].dims == ("y", "x")
    np.testing.assert_array_equal(out.y, [10, 20])


def test_dataset_to_matrix_requires_time_dimension():
    ds = xr.Dataset({"B1": (("y", "x"), np.ones((1, 1)))})
    with pytest.raises(ValueError, match="time.*dimension"):
        _dataset_to_matrix(ds)


def test_interp_extrapolates_sorts_and_averages_duplicates():
    actual = _interp_extrapolate_1d(
        np.array([-1.0, 0.5, 3.0]),
        np.array([2.0, 0.0, 0.0, 1.0]),
        np.array([4.0, 0.0, 2.0, 2.0]),
    )
    np.testing.assert_allclose(actual, [0.0, 1.5, 6.0])


def test_interp_single_point_and_empty_table():
    np.testing.assert_allclose(
        _interp_extrapolate_1d(np.array([-2.0, 9.0]), np.array([1.0]), np.array([3.0])),
        [3.0, 3.0],
    )
    with pytest.raises(ValueError, match="empty"):
        _interp_extrapolate_1d(np.array([1.0]), np.array([]), np.array([]))


def test_calibrate_std_multiplies_tau_and_preserves_invalid_values():
    means = np.array([[[0.0, 0.5, np.nan]]])
    stds = np.array([[[2.0, 2.0, 2.0]]])
    table = pd.DataFrame({"y_pred": [0.0, 1.0], "tau": [1.0, 3.0]})
    actual = _calibrate_std_local(means, stds, table)
    np.testing.assert_allclose(actual[..., :2], [[[2.0, 4.0]]])
    assert np.isnan(actual[..., 2]).all()


def test_calibrate_std_requires_columns():
    with pytest.raises(ValueError, match="y_pred.*tau"):
        _calibrate_std_local(np.ones(1), np.ones(1), pd.DataFrame({"x": [1]}))


def test_aggregate_prediction_times_hand_calculated_and_counts_nan():
    means = np.array([[[1.0, 2.0, np.nan]], [[3.0, np.nan, np.nan]]])
    stds = np.array([[[0.5, 0.25, np.nan]], [[1.5, np.nan, np.nan]]])

    mean, total, within, across, count = _aggregate_prediction_times(means, stds)

    np.testing.assert_allclose(mean[0, :2], [2.0, 2.0])
    np.testing.assert_allclose(within[0, :2], [1.0, 0.25])
    np.testing.assert_allclose(across[0, :2], [np.sqrt(2), 0.0])
    np.testing.assert_allclose(total[0, :2], [np.sqrt(3), 0.25])
    np.testing.assert_array_equal(count, [[2, 1, 0]])
    assert count.dtype == np.int32
    assert np.isnan(mean[0, 2])
