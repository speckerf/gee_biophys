import ee
import numpy as np
import pandas as pd
import xarray as xr

from gee_biophys.config import ConfigParams
from gee_biophys.models.s2biophys import eeEnsemblePredictSingleImg, load_model_ensemble
from gee_biophys.models.sl2p import load_SL2P_model
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


def _resolve_band_name(ds: xr.Dataset, band_name: str) -> str | None:
    if band_name in ds.data_vars:
        return band_name

    if band_name.startswith("B0") and len(band_name) == 3:
        alt_name = f"B{band_name[2]}"
        if alt_name in ds.data_vars:
            return alt_name

    if band_name.startswith("B") and len(band_name) == 2 and band_name[1].isdigit():
        alt_name = f"B0{band_name[1]}"
        if alt_name in ds.data_vars:
            return alt_name

    return None


def _dataset_to_matrix(
    ds: xr.Dataset, band_order: list[str]
) -> tuple[np.ndarray, tuple]:
    resolved_bands = []
    missing = []
    for band in band_order:
        resolved = _resolve_band_name(ds, band)
        if resolved is None:
            missing.append(band)
        else:
            resolved_bands.append(resolved)

    if missing:
        raise ValueError(f"Input dataset is missing required bands: {missing}")

    arr = ds[resolved_bands].to_array(dim="band")
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


def _predict_local_s2biophys(
    cfg: ConfigParams,
    input_ds: xr.Dataset,
) -> xr.Dataset:
    (
        s2biophys_model_ensemble,
        (_, uncertainty_calibration_table),
    ) = load_model_ensemble(cfg.variables.variable)

    band_order = [
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
        "tts",
        "tto",
        "psi",
    ]

    matrix, (arr, non_time_dims, full_shape) = _dataset_to_matrix(input_ds, band_order)
    valid = np.all(np.isfinite(matrix), axis=1)

    per_member = []
    valid_df = (
        pd.DataFrame(matrix[valid], columns=band_order) if np.any(valid) else None
    )
    models = sorted(s2biophys_model_ensemble.items(), key=lambda kv: kv[0])
    for _, model in models:
        pred_flat = np.full(matrix.shape[0], np.nan, dtype=float)
        if np.any(valid):
            pred_flat[valid] = model["pipeline"].predict(valid_df).ravel()
        per_member.append(_matrix_to_cube(pred_flat, full_shape))

    member_cube = np.stack(per_member, axis=0)
    pred_mean_time = np.nanmean(member_cube, axis=0)
    pred_std_time = np.nanstd(member_cube, axis=0)
    pred_std_time = _calibrate_std_local(
        pred_mean_time,
        pred_std_time,
        uncertainty_calibration_table,
    )

    preds_mean = np.nanmean(pred_mean_time, axis=0)
    preds_std_within = np.nanmean(pred_std_time, axis=0)
    preds_std_across = np.nanstd(pred_mean_time, axis=0, ddof=1)
    preds_std_across = np.where(
        np.isnan(preds_std_across) & np.isfinite(preds_std_within),
        0.0,
        preds_std_across,
    )
    preds_std_total = np.sqrt(preds_std_within**2 + preds_std_across**2)
    preds_count = np.sum(np.isfinite(pred_mean_time), axis=0).astype(np.int32)

    if cfg.options.clip_min_max:
        preds_mean = _clip_trait_array(preds_mean, cfg.variables.variable)

    outputs = {
        f"{cfg.variables.variable}_mean": preds_mean,
        f"{cfg.variables.variable}_stdDev": preds_std_total,
        f"{cfg.variables.variable}_stdDev_within": preds_std_within,
        f"{cfg.variables.variable}_stdDev_across": preds_std_across,
        f"{cfg.variables.variable}_count": preds_count,
    }
    return _build_output_dataset(arr, non_time_dims, outputs)


def _predict_local_sl2p(
    cfg: ConfigParams,
    input_ds: xr.Dataset,
) -> xr.Dataset:
    model_mean, model_std = load_SL2P_model(variable=cfg.variables.variable)

    band_order = model_mean.bandorder
    matrix, (arr, non_time_dims, full_shape) = _dataset_to_matrix(input_ds, band_order)
    valid = np.all(np.isfinite(matrix), axis=1)

    pred_mean_flat = np.full(matrix.shape[0], np.nan, dtype=float)
    pred_std_flat = np.full(matrix.shape[0], np.nan, dtype=float)
    if np.any(valid):
        pred_mean_flat[valid] = model_mean.predict(matrix[valid], clip_min_max=False)
        pred_std_flat[valid] = model_std.predict(matrix[valid], clip_min_max=False)

    pred_mean_time = _matrix_to_cube(pred_mean_flat, full_shape)
    pred_std_time = _matrix_to_cube(pred_std_flat, full_shape)

    preds_mean = np.nanmean(pred_mean_time, axis=0)
    if cfg.options.clip_min_max:
        preds_mean = _clip_trait_array(preds_mean, cfg.variables.variable)

    img_within_var = np.nanmean(pred_std_time**2, axis=0)
    img_between_var = np.nanvar(pred_mean_time, axis=0)
    preds_std_total = np.sqrt(img_within_var + img_between_var)
    preds_count = np.sum(np.isfinite(pred_mean_time), axis=0).astype(np.int32)

    outputs = {
        f"{cfg.variables.variable}_mean": preds_mean,
        f"{cfg.variables.variable}_stdDev": preds_std_total,
        f"{cfg.variables.variable}_count": preds_count,
    }
    return _build_output_dataset(arr, non_time_dims, outputs)


def biophys_predict(cfg: ConfigParams, input_imgc: ee.ImageCollection) -> ee.Image:
    """Apply the selected biophysical model to the input Sentinel-2 ImageCollection
    and return an ImageCollection with predicted biophysical variables.
    """
    if cfg.variables.model == "sl2p":
        model_mean, model_std = load_SL2P_model(variable=cfg.variables.variable)

        pred_mean_imgc = input_imgc.map(lambda img: model_mean.ee_predict(img))
        pred_std_imgc = input_imgc.map(lambda img: model_std.ee_predict(img))

        output_image = aggregate_ensemble_predictions(
            pred_mean_imgc,
            pred_std_imgc,
            cfg.variables.variable,
            clip_min_max=cfg.options.clip_min_max,
        )

        water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
        output_image = output_image.updateMask(water_mask_2020.neq(80))

    elif cfg.variables.model == "s2biophys":
        (
            s2biophys_model_ensemble,
            (uncertainty_calibration_model, uncertainty_calibration_table),
        ) = load_model_ensemble(cfg.variables.variable)

        imgc_preds = input_imgc.map(
            lambda img: eeEnsemblePredictSingleImg(
                ensemble=s2biophys_model_ensemble,
                img=img,
                variable=cfg.variables.variable,
                calibrate_uncertainty=True,
                uncertainty_calibration_table=uncertainty_calibration_table,
            )
        )

        # b = eeEnsemblePredictSingleImg(
        #     s2biophys_model_ensemble,
        #     input_imgc.first(),
        #     cfg.variables.variable,
        #     calibrate_uncertainty=False,
        # )

        # reduce to mean / stdDev_across-images / stdDev_within-images per group
        output_image = reduce_ensemble_preds(
            imgc_preds,
            cfg.variables.variable,
        )

        water_mask_2020 = ee.ImageCollection("ESA/WorldCover/v200").first()
        output_image = output_image.updateMask(water_mask_2020.neq(80))

    else:
        raise ValueError(f"Unsupported model: {cfg.variables.model}")

    # select only desired output bands
    output_band_names = [
        f"{cfg.variables.variable}_{band}" for band in cfg.variables.bands
    ]

    return output_image.select(output_band_names)


def biophys_predict_local(cfg: ConfigParams, input_ds: xr.Dataset) -> xr.Dataset:
    """Run local biophysical prediction from an xarray Dataset.

    The expected input dataset is the output of ``load_s2_input_xarray`` and therefore
    contains model-specific bands already prepared on the GEE side.
    """
    if cfg.variables.model == "sl2p":
        output_ds = _predict_local_sl2p(cfg, input_ds)
    elif cfg.variables.model == "s2biophys":
        output_ds = _predict_local_s2biophys(cfg, input_ds)
    else:
        raise ValueError(f"Unsupported model: {cfg.variables.model}")

    output_band_names = [
        f"{cfg.variables.variable}_{band}" for band in cfg.variables.bands
    ]
    missing_bands = [band for band in output_band_names if band not in output_ds]
    if missing_bands:
        raise ValueError(
            "Requested output bands are not available for local prediction: "
            f"{missing_bands}"
        )

    return output_ds[output_band_names]
