import ee
import numpy as np
import pandas as pd
import xarray as xr

from gee_biophys.models.grounded_eo import (
    load_grounded_eo_model,
    prepare_s2_ds_for_groundedeo,
)
from gee_biophys.models.s2biophys import (
    eeEnsemblePredictSingleImg,
    load_biome_land_cover_specific_model_ensemble,
    load_model_ensemble,
    prepare_s2_ds_for_s2biophys,
    prepare_s2_imgc_for_s2biophys,
)
from gee_biophys.models.sl2p import (
    load_SL2P_model,
    prepare_s2_ds_for_sl2p,
    prepare_s2_imgc_for_sl2p,
)
from gee_biophys.utils_predict import (
    aggregate_ensemble_predictions,
    reduce_ensemble_preds,
)


def _clip_trait_array(values: np.ndarray, trait_name: str) -> np.ndarray:
    clip_dict = {
        "lai": (0, 8),
        "laie": (0, 8),
        "fapar": (0, 1),
        "fcover": (0, 1),
    }
    if trait_name not in clip_dict:
        raise ValueError(f"Clipping not defined for trait: {trait_name}")
    vmin, vmax = clip_dict[trait_name]
    return np.clip(values, vmin, vmax)


def _dataset_to_matrix(
    ds: xr.Dataset,
    # band_order: list[str],
) -> tuple[np.ndarray, tuple]:
    """Convert an xarray Dataset to a 2D prediction matrix.

    Assumes all required bands exist and are already named correctly.
    """
    # arr = ds[band_order].to_array(dim="band")
    arr = ds.to_array(dim="band")

    if "time" not in arr.dims:
        raise ValueError("Local prediction expects a 'time' dimension in the input.")

    non_time_dims = [dim for dim in arr.dims if dim not in {"band", "time"}]

    arr = arr.transpose("time", *non_time_dims, "band")

    np_arr = arr.values
    matrix = np_arr.reshape(-1, np_arr.shape[-1])

    return matrix, (arr, non_time_dims, np_arr.shape)


def _matrix_to_cube(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    return values.reshape(shape[:-1])


def _build_output_dataset(
    arr: xr.DataArray,
    non_time_dims: list[str],
    outputs: dict[str, np.ndarray],
) -> xr.Dataset:
    coords = {dim: arr.coords[dim] for dim in non_time_dims}
    data_vars = {
        name: (tuple(non_time_dims), values) for name, values in outputs.items()
    }
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _interp_extrapolate_1d(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
) -> np.ndarray:
    """Linear interpolation with linear extrapolation at both ends."""
    if xp.size == 0:
        raise ValueError("Calibration table is empty.")

    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]

    # Collapse duplicated x values by averaging y values.
    uniq_x, inverse = np.unique(xp, return_inverse=True)
    if uniq_x.size != xp.size:
        sums = np.zeros(uniq_x.size, dtype=float)
        counts = np.zeros(uniq_x.size, dtype=float)
        for i, idx in enumerate(inverse):
            sums[idx] += fp[i]
            counts[idx] += 1.0
        xp = uniq_x
        fp = sums / counts

    y = np.interp(x, xp, fp)

    if xp.size == 1:
        y[:] = fp[0]
        return y

    left = x < xp[0]
    right = x > xp[-1]

    left_dx = xp[1] - xp[0]
    right_dx = xp[-1] - xp[-2]

    left_slope = 0.0 if left_dx == 0 else (fp[1] - fp[0]) / left_dx
    right_slope = 0.0 if right_dx == 0 else (fp[-1] - fp[-2]) / right_dx

    y[left] = fp[0] + left_slope * (x[left] - xp[0])
    y[right] = fp[-1] + right_slope * (x[right] - xp[-1])

    return y


def _calibrate_std_local(
    pred_mean_time: np.ndarray,
    pred_std_time: np.ndarray,
    calibration_table: pd.DataFrame,
) -> np.ndarray:
    """Calibrate per-time stdDev using the same y_pred -> tau mapping used in EE."""
    if not {"y_pred", "tau"}.issubset(calibration_table.columns):
        raise ValueError("Calibration table must contain 'y_pred' and 'tau' columns.")

    y_pred_values = calibration_table["y_pred"].to_numpy(dtype=float)
    tau_values = calibration_table["tau"].to_numpy(dtype=float)

    mean_flat = pred_mean_time.reshape(-1)
    std_flat = pred_std_time.reshape(-1)
    calibrated_flat = np.full_like(std_flat, np.nan, dtype=float)

    valid = np.isfinite(mean_flat) & np.isfinite(std_flat)
    if np.any(valid):
        tau_flat = _interp_extrapolate_1d(mean_flat[valid], y_pred_values, tau_values)
        calibrated_flat[valid] = std_flat[valid] * tau_flat

    return calibrated_flat.reshape(pred_std_time.shape)


def _aggregate_prediction_times(
    pred_mean_time: np.ndarray,
    pred_std_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate per-acquisition predictions over time.

    Parameters
    ----------
    pred_mean_time
        Per-acquisition mean predictions with shape (time, ...).
    pred_std_time
        Per-acquisition predictive standard deviations with shape (time, ...).

    Returns
    -------
    preds_mean
        Mean prediction across acquisitions.
    preds_std_total
        Total standard deviation:
        sqrt(std_within**2 + std_across**2).
    preds_std_within
        Mean per-acquisition predictive standard deviation.
    preds_std_across
        Sample standard deviation of per-acquisition mean predictions.
    preds_count
        Number of valid acquisitions per pixel.
    """
    preds_mean = np.nanmean(pred_mean_time, axis=0)

    # Typical predictive uncertainty within an acquisition.
    preds_std_within = np.nanmean(pred_std_time, axis=0)

    # Temporal/acquisition-to-acquisition variability.
    preds_std_across = np.nanstd(
        pred_mean_time,
        axis=0,
        ddof=1,
    )

    # sample SD is undefined for n=1; treat observed across-time variation as 0
    preds_std_across = np.where(
        np.isnan(preds_std_across) & np.isfinite(preds_std_within),
        0.0,
        preds_std_across,
    )

    preds_std_total = np.sqrt(preds_std_within**2 + preds_std_across**2)

    preds_count = np.sum(
        np.isfinite(pred_mean_time),
        axis=0,
    ).astype(np.int32)

    return (
        preds_mean,
        preds_std_total,
        preds_std_within,
        preds_std_across,
        preds_count,
    )


def _predict_local_s2biophys(
    input_ds: xr.Dataset,
    variable: str,
    global_model: bool = True,
    clip_min_max: bool = True,
    biome_lc_name: str | None = None,
) -> xr.Dataset:
    if not global_model:
        (
            s2biophys_model_ensemble,
            (_, uncertainty_calibration_table),
        ) = load_biome_land_cover_specific_model_ensemble(
            variable,
            biome_lc_name=biome_lc_name,
        )
    else:
        (
            s2biophys_model_ensemble,
            (_, uncertainty_calibration_table),
        ) = load_model_ensemble(variable)

    matrix, (arr, non_time_dims, full_shape) = _dataset_to_matrix(input_ds)
    valid = np.all(np.isfinite(matrix), axis=1)

    valid_df = (
        pd.DataFrame(matrix[valid], columns=input_ds.data_vars)
        if np.any(valid)
        else None
    )

    per_member = []

    models = sorted(
        s2biophys_model_ensemble.items(),
        key=lambda kv: kv[0],
    )

    for _, model in models:
        pred_flat = np.full(
            matrix.shape[0],
            np.nan,
            dtype=float,
        )

        if np.any(valid):
            pred_flat[valid] = model["pipeline"].predict(valid_df).ravel()

        per_member.append(
            _matrix_to_cube(
                pred_flat,
                full_shape,
            )
        )

    # (member, time, y, x)
    member_cube = np.stack(
        per_member,
        axis=0,
    )

    # Mean and sample SD across ensemble members,
    # independently for each acquisition.
    pred_mean_time = np.nanmean(
        member_cube,
        axis=0,
    )
    pred_std_time = np.nanstd(
        member_cube,
        axis=0,
        ddof=1,
    )

    pred_std_time = _calibrate_std_local(
        pred_mean_time,
        pred_std_time,
        uncertainty_calibration_table,
    )

    (
        preds_mean,
        preds_std_total,
        preds_std_within,
        preds_std_across,
        preds_count,
    ) = _aggregate_prediction_times(
        pred_mean_time,
        pred_std_time,
    )

    if clip_min_max:
        preds_mean = _clip_trait_array(
            preds_mean,
            variable,
        )

    outputs = {
        f"{variable}_mean": preds_mean,
        f"{variable}_stdDev": preds_std_total,
        f"{variable}_stdDev_within": preds_std_within,
        f"{variable}_stdDev_across": preds_std_across,
        f"{variable}_count": preds_count,
    }

    return _build_output_dataset(
        arr,
        non_time_dims,
        outputs,
    )


def _predict_local_sl2p(
    input_ds: xr.Dataset,
    variable: str,
    clip_min_max: bool = True,
) -> xr.Dataset:
    model_mean, model_std = load_SL2P_model(variable=variable)

    matrix, (arr, non_time_dims, full_shape) = _dataset_to_matrix(
        input_ds,
    )
    valid = np.all(np.isfinite(matrix), axis=1)

    pred_mean_flat = np.full(
        matrix.shape[0],
        np.nan,
        dtype=float,
    )
    pred_std_flat = np.full(
        matrix.shape[0],
        np.nan,
        dtype=float,
    )

    if np.any(valid):
        pred_mean_flat[valid] = model_mean.predict(
            matrix[valid],
            clip_min_max=False,
        )

        pred_std_flat[valid] = model_std.predict(
            matrix[valid],
            clip_min_max=False,
        )

    # (time, y, x)
    pred_mean_time = _matrix_to_cube(
        pred_mean_flat,
        full_shape,
    )
    pred_std_time = _matrix_to_cube(
        pred_std_flat,
        full_shape,
    )

    (
        preds_mean,
        preds_std_total,
        preds_std_within,
        preds_std_across,
        preds_count,
    ) = _aggregate_prediction_times(
        pred_mean_time,
        pred_std_time,
    )

    if clip_min_max:
        preds_mean = _clip_trait_array(
            preds_mean,
            variable,
        )

    outputs = {
        f"{variable}_mean": preds_mean,
        f"{variable}_stdDev": preds_std_total,
        f"{variable}_stdDev_within": preds_std_within,
        f"{variable}_stdDev_across": preds_std_across,
        f"{variable}_count": preds_count,
    }

    return _build_output_dataset(
        arr,
        non_time_dims,
        outputs,
    )


def _predict_local_grounded_eo(
    input_ds: xr.Dataset,
    variable: str,
    clip_min_max: bool = True,
) -> xr.Dataset:
    matrix, (arr, non_time_dims, full_shape) = _dataset_to_matrix(
        input_ds,
    )
    valid = np.all(np.isfinite(matrix), axis=1)

    model_key = variable.upper()

    model, upper_lim = load_grounded_eo_model(model_key)

    pred_mean_flat = np.full(
        matrix.shape[0],
        np.nan,
        dtype=float,
    )
    pred_std_flat = np.full(
        matrix.shape[0],
        np.nan,
        dtype=float,
    )

    if np.any(valid):
        input_name = model.get_inputs()[0].name

        pred_mean, pred_std = model.run(
            None,
            {
                input_name: matrix[valid].astype(np.float64),
            },
        )

        pred_mean = np.asarray(pred_mean).squeeze()
        pred_std = np.asarray(pred_std).squeeze()

        pred_mean_flat[valid] = np.clip(
            pred_mean,
            0.0,
            upper_lim,
        )
        pred_std_flat[valid] = pred_std

    # (time, y, x)
    pred_mean_time = _matrix_to_cube(
        pred_mean_flat,
        full_shape,
    )
    pred_std_time = _matrix_to_cube(
        pred_std_flat,
        full_shape,
    )

    (
        preds_mean,
        preds_std_total,
        preds_std_within,
        preds_std_across,
        preds_count,
    ) = _aggregate_prediction_times(
        pred_mean_time,
        pred_std_time,
    )

    if clip_min_max:
        preds_mean = _clip_trait_array(
            preds_mean,
            variable,
        )

    prefix = f"{variable.lower()}"

    outputs = {
        f"{prefix}_mean": preds_mean,
        f"{prefix}_stdDev": preds_std_total,
        f"{prefix}_stdDev_within": preds_std_within,
        f"{prefix}_stdDev_across": preds_std_across,
        f"{prefix}_count": preds_count,
    }

    return _build_output_dataset(
        arr,
        non_time_dims,
        outputs,
    )


def biophys_predict_ee(
    variable: str,
    model: str,
    input_imgc: ee.ImageCollection,
    clip_min_max: bool,
    biome_lc_name: str | None = None,
) -> ee.Image:
    """Apply the selected biophysical model to the input Sentinel-2 ImageCollection
    and return an ImageCollection with predicted biophysical variables.
    """
    if model == "sl2p":
        input_imgc = prepare_s2_imgc_for_sl2p(input_imgc)
        model_mean, model_std = load_SL2P_model(variable=variable)

        pred_mean_imgc = input_imgc.map(lambda img: model_mean.ee_predict(img))
        pred_std_imgc = input_imgc.map(lambda img: model_std.ee_predict(img))

        output_image = aggregate_ensemble_predictions(
            pred_mean_imgc,
            pred_std_imgc,
            variable,
            clip_min_max=clip_min_max,
        )

        water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
        output_image = output_image.updateMask(water_mask_2020.neq(80))

    elif model == "s2biophys":
        input_imgc = prepare_s2_imgc_for_s2biophys(input_imgc)
        (
            s2biophys_model_ensemble,
            (_, uncertainty_calibration_table),
        ) = load_model_ensemble(variable)

        imgc_preds = input_imgc.map(
            lambda img: eeEnsemblePredictSingleImg(
                ensemble=s2biophys_model_ensemble,
                img=img,
                variable=variable,
                calibrate_uncertainty=True,
                uncertainty_calibration_table=uncertainty_calibration_table,
            )
        )
        # reduce to mean / stdDev_across-images / stdDev_within-images per group
        output_image = reduce_ensemble_preds(
            imgc_preds,
            variable,
        )

        water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
        output_image = output_image.updateMask(water_mask_2020.neq(80))

    elif model == "s2biophys-biome-lc-specific":
        input_imgc = prepare_s2_imgc_for_s2biophys(input_imgc)
        (
            s2biophys_model_ensemble,
            (_, uncertainty_calibration_table),
        ) = load_biome_land_cover_specific_model_ensemble(
            variable, biome_lc_name=biome_lc_name
        )

        imgc_preds = input_imgc.map(
            lambda img: eeEnsemblePredictSingleImg(
                ensemble=s2biophys_model_ensemble,
                img=img,
                variable=variable,
                calibrate_uncertainty=True,
                uncertainty_calibration_table=uncertainty_calibration_table,
            )
        )
        # reduce to mean / stdDev_across-images / stdDev_within-images per group
        output_image = reduce_ensemble_preds(
            imgc_preds,
            variable,
        )

        water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
        output_image = output_image.updateMask(water_mask_2020.neq(80))

    else:
        raise ValueError(f"Unsupported model: {model}")

    if model in {"s2biophys", "s2biophys-biome-lc-specific"} and clip_min_max:
        upper = 8 if variable == "laie" else 1
        mean = output_image.select(f"{variable}_mean").clamp(0, upper)
        output_image = output_image.addBands(mean, overwrite=True)
    return output_image


def biophys_predict_local(
    input_ds: xr.Dataset,
    variable: str,
    model: str,
    clip_min_max: bool = True,
    biome_lc_name: str | None = None,
) -> xr.Dataset:
    """Run local biophysical prediction from an xarray Dataset.

    The expected input dataset is the output of ``load_s2_input_xarray`` and therefore
    contains model-specific bands already prepared on the GEE side.
    """
    if model == "sl2p":
        input_ds = prepare_s2_ds_for_sl2p(input_ds)
        output_ds = _predict_local_sl2p(
            input_ds, variable=variable, clip_min_max=clip_min_max
        )
    elif model == "s2biophys":
        input_ds = prepare_s2_ds_for_s2biophys(input_ds)
        output_ds = _predict_local_s2biophys(
            input_ds, variable=variable, clip_min_max=clip_min_max
        )
    elif model == "groundedeo":
        input_ds = prepare_s2_ds_for_groundedeo(input_ds)
        output_ds = _predict_local_grounded_eo(
            input_ds, variable=variable, clip_min_max=clip_min_max
        )
    elif model == "s2biophys-biome-lc-specific":
        input_ds = prepare_s2_ds_for_s2biophys(input_ds)
        output_ds = _predict_local_s2biophys(
            input_ds,
            variable=variable,
            global_model=False,
            clip_min_max=clip_min_max,
            biome_lc_name=biome_lc_name,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    return output_ds
