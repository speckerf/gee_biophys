from __future__ import annotations

from importlib.resources import files
from typing import Literal

import ee
import numpy as np
import onnxruntime as ort
import xarray as xr
from loguru import logger


def _resolve_grounded_eo_model_path(variable: str) -> str:
    normalized = variable.upper().replace("LAIE", "LAI")
    if normalized == "LAI":
        model_name = "lai"
    elif normalized == "FAPAR":
        model_name = "fapar"
    else:
        raise ValueError("Grounded EO models are only available for 'LAI' and 'FAPAR'.")

    resource = files("gee_biophys.models.groundedeo") / f"{model_name}.onnx"
    return str(resource)


def load_grounded_eo_model(
    variable: Literal["LAI", "FAPAR", "laie", "fapar"],
) -> tuple[ort.InferenceSession, float]:
    """Load the GROUNDED-EO ONNX model for the target variable."""
    normalized = variable.upper()

    if normalized == "LAIE":
        raise ValueError("GROUNDED-EO models are only available for 'LAI' and 'FAPAR'.")

    model_path = _resolve_grounded_eo_model_path(normalized)

    model = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    upper_lim = 10.0 if normalized == "LAI" else 1.0

    logger.warning(
        "GROUNDED-EO predictions are local-only and require xee-local export."
    )

    return model, upper_lim


def _ee_angle_transform_grounded_eo(angle_img: ee.Image) -> ee.Image:
    """Convert degree angles to cosine values for Grounded EO models."""
    radians_img = angle_img.multiply(np.pi / 180.0)
    return radians_img.cos()


def prepare_s2_img_for_groundedeo(
    img: ee.Image,
) -> ee.Image:
    """Prepare one Sentinel-2 image for the local GROUNDED-EO GPR models.

    The model expects the full Sentinel-2 band set B1..B12
    plus three cosine-transformed angle features.
    """
    s2_bands = [
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

    angle_names = ["tts", "tto", "psi"]
    cos_names = ["cos_sza", "cos_vza", "cos_raa"]

    refl = img.select(s2_bands)

    cos_angles = _ee_angle_transform_grounded_eo(img.select(angle_names)).rename(
        cos_names
    )

    return (
        refl.addBands(cos_angles)
        .select(s2_bands + cos_names)
        .copyProperties(img, img.propertyNames())
    )


def prepare_s2_imgc_for_groundedeo(
    imgc: ee.ImageCollection,
) -> ee.ImageCollection:
    """Prepare a Sentinel-2 ImageCollection for GROUNDED-EO GPR models."""
    return imgc.map(prepare_s2_img_for_groundedeo)


def prepare_s2_ds_for_groundedeo(
    ds: xr.Dataset,
) -> xr.Dataset:
    """Prepare a Sentinel-2 xarray Dataset for GROUNDED-EO GPR models.

    The model expects the full Sentinel-2 band set B1..B12
    plus three cosine-transformed angle features.
    """
    s2_bands = [
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

    angle_names = ["tts", "tto", "psi"]
    cos_names = ["cos_sza", "cos_vza", "cos_raa"]

    refl = ds[s2_bands]

    cos_angles = np.cos(np.deg2rad(ds[angle_names])).rename(
        {old: new for old, new in zip(angle_names, cos_names)}
    )

    return xr.merge([refl, cos_angles])[s2_bands + cos_names]


# def predict_grounded_eo(df: pd.DataFrame) -> pd.DataFrame:
#     """Predict LAI and FAPAR from input Sentinel-2-like bands and cosine angles.

#     Returns columns:
#       - grounded_lai_mean, grounded_lai_std
#       - grounded_fapar_mean, grounded_fapar_std
#     """
#     out = pd.DataFrame(index=df.index)
#     work = df.copy()

#     rename_map = {}
#     for old, new in [("sza", "tts"), ("vza", "tto"), ("phi", "psi")]:
#         if old in work.columns and new not in work.columns:
#             rename_map[old] = new
#     if rename_map:
#         work = work.rename(columns=rename_map)

#     if {"tts", "tto", "psi"}.issubset(work.columns) and not {
#         "cos_vza",
#         "cos_sza",
#         "cos_raa",
#     }.issubset(work.columns):
#         work["cos_vza"] = np.cos(np.radians(work["tto"]))
#         work["cos_sza"] = np.cos(np.radians(work["tts"]))
#         work["cos_raa"] = np.cos(np.radians(work["psi"]))

#     band_cols = [
#         col
#         for col in work.columns
#         if re.match(r"^B\d{1,2}[A]?$", col) and pd.api.types.is_numeric_dtype(work[col])
#     ]
#     if band_cols:
#         work[band_cols] = work[band_cols] / 10000.0

#     for trait_name, model_key in (("LAI", "lai"), ("FAPAR", "fapar")):
#         model, upper_lim, band_order = load_grounded_eo_model(model_key)
#         missing = [col for col in band_order if col not in work.columns]
#         if missing:
#             continue

#         X = work[band_order].copy()
#         valid_idx = X.notna().all(axis=1)
#         X_valid = X.loc[valid_idx]
#         if X_valid.empty:
#             continue

#         mean_pred, std_pred = model.predict(X_valid, return_std=True)
#         mean_arr = np.asarray(mean_pred, dtype=float).reshape(-1)
#         std_arr = np.asarray(std_pred, dtype=float).reshape(-1)

#         mean_arr = np.clip(mean_arr, 0.0, upper_lim)
#         out.loc[valid_idx, f"grounded_{trait_name.lower()}_mean"] = mean_arr
#         out.loc[valid_idx, f"grounded_{trait_name.lower()}_std"] = std_arr

#     return out
