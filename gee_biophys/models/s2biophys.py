import json
import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from pickle import load as pickle_load
from typing import Any, Dict, Optional, Tuple

import ee
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from gee_biophys.models.utils_s2biophys import (
    eeMinMaxRangeMasker,
    eeMLPRegressor,
    eeStandardScaler,
)


def ee_nirv_normalisation(image: ee.Image):
    reflectance_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    NDVI = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    NIRv = NDVI.multiply(image.select("B8")).rename("NIRv")

    image_normalized = image.select(reflectance_bands).divide(NIRv)
    image_to_return = image.addBands(image_normalized, overwrite=True)
    return image_to_return


def ee_angle_transformer(image: ee.Image):
    # cosine transformation of angles
    image_angles = image.select(["tts", "tto", "psi"]).multiply(np.pi / 180).cos()
    image_to_return = image.addBands(image_angles, overwrite=True)
    return image_to_return


def ee_logit_transform(image: ee.Image, trait: str):
    # logit transformation of trait
    #  np.log(x / (1 - x))
    image = image.addBands(
        image.select(trait).log().divide(image.select(trait).subtract(1)),
    )
    return image


def ee_logit_inverse_transform(image: ee.Image, trait: str):
    # inverse logit transformation of trait
    # 1 / (1 + np.exp(-x))
    return (
        image.select(trait).expression("1 / (1 + exp(-x))", {"x": image}).rename(trait)
    )


def ee_log1p_inverse_transform(image: ee.Image, trait: str):
    # inverse log1p transformation of trait
    # np.exp(x) - 1
    return image.select(trait).exp().subtract(1).rename(trait)


def _preprocess_base(x: ee.Image) -> ee.Image:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]
    features = bands + angles

    # Angles + feature selection
    x = ee_angle_transformer(x)
    return x.select(features)


def _apply_band_scaler(x: ee.Image, pipeline) -> ee.Image:
    # Always apply standard scaler
    band_scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["band_transformer"]
        .named_steps["scaler"]
    )
    ee_band_scaler = eeStandardScaler(band_scaler)
    return ee_band_scaler.transform_image(x)


def _predict_regressor(x: ee.Image, pipeline, trait_name: str) -> ee.Image:
    # IMPORTANT: .regressor_ refers to the trained model
    ee_model = eeMLPRegressor(
        pipeline.named_steps["regressor"].regressor_, trait_name=trait_name
    )
    return ee_model.predict(x)


def _inverse_target_transform(
    img_pred: ee.Image,
    pipeline,
    cfg: Dict[str, Any],
    trait_name: str,
) -> ee.Image:
    t = cfg.get("transform_target")
    if t in (None, "None"):
        return img_pred

    if t == "log1p":
        return ee.Image(ee_log1p_inverse_transform(img_pred, trait_name))

    if t == "logit":
        return ee.Image(ee_logit_inverse_transform(img_pred, trait_name))

    if t == "standard":
        target_scaler = pipeline.named_steps["regressor"].transformer_
        target_ee_scaler = eeStandardScaler(
            target_scaler, feature_names=[trait_name]
        )  # must be list
        return ee.Image(target_ee_scaler.inverse_transform_column(img_pred, trait_name))

    raise ValueError(f"Unknown target transformation: {t}")


def _ee_calibrate_std(
    image: ee.Image, recalibration_table: pd.DataFrame, variable: str
) -> ee.Image:
    assert all(col in recalibration_table.columns for col in ["y_pred", "tau"]), (
        "Calibration table must contain 'pred_mean' and 'tau' columns"
    )

    mean_bandname = f"{variable}_mean"
    std_bandname = f"{variable}_stdDev"

    mean_image = image.select(mean_bandname)
    std_image = image.select(std_bandname)

    y_pred_inter = ee.List(recalibration_table["y_pred"].values.tolist())
    tau_inter = ee.List(recalibration_table["tau"].values.tolist())

    # now we want to get the interpolated tau value for each pixel based on its predicted mean value
    tau_image = mean_image.interpolate(
        y_pred_inter,
        tau_inter,
        behavior="extrapolate",
    ).rename("tau")

    # use ee.Image.interpolate to recalibrate std_image / values outside of the range are copied (behavior='input')
    std_image_calibrated = std_image.multiply(tau_image).rename(std_bandname)

    return image.addBands(std_image_calibrated, overwrite=True)


def _predict_one_member(
    img_base: ee.Image, model: Dict[str, Any], trait_name: str
) -> ee.Image:
    pipeline = model["pipeline"]
    cfg = model["config"]

    # Start from base image for this member (fixes img_temp-before-assignment bug)
    x = img_base

    # Optional per-model masking
    mm = model["min_max_bands"]
    if mm is not None:
        x = eeMinMaxRangeMasker(mm).ee_mask(x)

    # Optional per-model NIRv normalisation
    if cfg.get("nirv_norm", False):
        x = ee_nirv_normalisation(x)

    # Scaling + prediction
    x = _apply_band_scaler(x, pipeline)
    y = _predict_regressor(x, pipeline, trait_name)

    # Inverse target transform
    return _inverse_target_transform(y, pipeline, cfg, trait_name)


def eeEnsemblePredictSingleImg(
    ensemble,
    img: ee.Image,
    variable: str,
    calibrate_uncertainty: Optional[bool] = False,
    uncertainty_calibration_table: Optional[pd.DataFrame] = None,
) -> ee.Image:
    """
    Predict a vegetation trait for a single Earth Engine image using an ensemble of models.

    Applies shared preprocessing, runs all ensemble members on the input image,
    and returns per-pixel ensemble statistics:
      - ``{variable}_mean``   : ensemble mean prediction
      - ``{variable}_stdDev`` : sample standard deviation across ensemble members

    Optionally recalibrates the ensemble standard deviation using an empirical
    uncertainty calibration table.

    Parameters
    ----------
    ensemble : dict
        Dictionary of trained ensemble models.
    img : ee.Image
        Input image containing all required predictor bands.
    variable : str
        Name of the predicted variable (used for output band names).
    calibrate_uncertainty : bool, optional
        If True, recalibrate predictive uncertainty.
    uncertainty_calibration_table : pandas.DataFrame, optional
        Required when ``calibrate_uncertainty=True``.

    Returns
    -------
    ee.Image
        Image with ``{variable}_mean`` and ``{variable}_stdDev`` bands.
    """

    mean_bandname = f"{variable}_mean"
    std_bandname = f"{variable}_stdDev"

    # ------------------------------------------------------------------
    # shared preprocessing (identical for all ensemble members)
    # ------------------------------------------------------------------
    img_preprocessed = _preprocess_base(img)

    preds = []
    for model_name, model in ensemble.items():
        try:
            preds.append(_predict_one_member(img_preprocessed, model, variable))
        except Exception as e:
            raise RuntimeError(f"Ensemble member '{model_name}' failed.") from e

    preds_ic = ee.ImageCollection(preds)
    preds_mean = preds_ic.mean().rename(mean_bandname)
    preds_std = preds_ic.reduce(ee.Reducer.sampleStdDev()).rename(std_bandname)

    if calibrate_uncertainty:
        if uncertainty_calibration_table is None:
            raise ValueError(
                "uncertainty_calibration_table must be provided when recalibrate_uncertainty is True"
            )
        return _ee_calibrate_std(
            ee.Image([preds_mean, preds_std]), uncertainty_calibration_table, variable
        )
    else:
        return ee.Image([preds_mean, preds_std])


def eePipelinePredictMap(
    pipeline: Pipeline,
    imgc: ee.ImageCollection,
    trait: str,
    model_config: dict,
    min_max_bands: dict | None = None,
):
    """
    DEPRECATED — Predict vegetation trait maps using a legacy pipeline-based workflow.

    This function is retained **only for testing and validation purposes**.
    It loads the same trained model as `eeEnsemblePredictSingleImg`, but applies it
    through the older pipeline-based prediction logic.

    The primary purpose of this function is to:
      - Verify numerical consistency between
        - local (scikit-learn) predictions, and
        - server-side Google Earth Engine (GEE) predictions
      - Support regression tests during refactoring of the prediction pipeline

    ⚠️ This function SHOULD NOT be used for production mapping.
    All new prediction workflows should use `eeEnsemblePredictSingleImg` instead.

    Parameters
    ----------
    pipeline : Pipeline
        Trained scikit-learn pipeline used for prediction. This pipeline is loaded
        identically to the one used in `eeEnsemblePredictSingleImg`.
    imgc : ee.ImageCollection
        Input ImageCollection containing surface reflectance bands and angular
        information required by the model.
    trait : str
        Name of the vegetation trait to be predicted (e.g. "lai", "fapar").
    model_config : dict
        Model configuration dictionary containing preprocessing, scaling,
        and model metadata.
    min_max_bands : dict | None, optional
        Optional dictionary specifying min/max values for output bands,
        used for scaling or visualization purposes.

    Raises
    ------
    DeprecationWarning
        Always raised to indicate that this function is deprecated and should
        not be used in production workflows.

    Notes
    -----
    - This function exists to ensure backward compatibility during model
      and pipeline transitions.
    - Outputs generated by this function should be numerically consistent
      with `eeEnsemblePredictSingleImg` when using the same model weights
      and inputs.
    """
    warnings.warn(
        "eePipelinePredictMapDeprecated is deprecated. It is only kept for testing purposes. "
        "Predictions should be done via eeEnsemblePredictSingleImg instead. "
        "However it still loads the same model as eeEnsemblePredictSingleImg and thus "
        "represents a valid test to test model consistency between local sklearn "
        "predictions and server-side GEE predictions.",
        DeprecationWarning,
        stacklevel=2,
    )

    # get the bands and angles
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    angles = ["tts", "tto", "psi"]

    # mask all pixels with reflectance values outside of the min_max reflectance values
    if min_max_bands is not None:
        min_max_band_masker = eeMinMaxRangeMasker(min_max_bands)
        imgc = imgc.map(min_max_band_masker.ee_mask)

    if model_config["nirv_norm"]:
        imgc = imgc.map(ee_nirv_normalisation)

    features = bands + angles
    imgc = imgc.map(ee_angle_transformer)
    imgc = imgc.select(features)

    # always apply standard scaler
    band_scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["band_transformer"]
        .named_steps["scaler"]
    )

    ee_band_scaler = eeStandardScaler(band_scaler)
    # a = ee_band_scaler.transform_image(imgc.first())
    imgc = imgc.map(ee_band_scaler.transform_image)

    # apply model:
    if model_config["model"] == "mlp":
        # IMPORTANT: .regressor_ refers to the actual model, while .regressor only refers to the untrained model
        ee_model = eeMLPRegressor(
            pipeline.named_steps["regressor"].regressor_,
            trait_name=trait,
        )
    else:
        raise ValueError("Only mlp models are supported for now")
    imgc = imgc.map(lambda image: ee_model.predict(image))

    # apply inverse transformations
    if model_config["transform_target"] == "log1p":
        imgc = imgc.map(
            lambda image: ee_log1p_inverse_transform(image, trait).copyProperties(
                image
            ),
        )
    elif model_config["transform_target"] == "logit":
        imgc = imgc.map(
            lambda image: ee_logit_inverse_transform(image, trait).copyProperties(
                image
            ),
        )
    elif model_config["transform_target"] == "standard":
        target_scaler = pipeline.named_steps["regressor"].transformer_
        target_ee_scaler = eeStandardScaler(
            target_scaler,
            feature_names=[trait],
        )  # must be a list
        imgc = imgc.map(
            lambda image: target_ee_scaler.inverse_transform_column(
                image,
                trait,
            ).copyProperties(image),
        )
    elif model_config["transform_target"] == "None":
        imgc = imgc
    else:
        raise ValueError(
            f"Unknown target transformation: {model_config['transform_target']}",
        )

    return imgc


@dataclass
class EnsembleItem:
    config: dict
    pipeline: Any
    model_path: str
    min_max_bands: dict
    min_max_label: dict
    split: dict

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _open_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _open_pickle(p: Path):
    # If you have a custom safe loader, swap it here.
    with p.open("rb") as f:
        return pickle_load(f)


def load_model_ensemble(
    trait: str,
) -> Tuple[Dict[str, EnsembleItem], Tuple[Any, pd.DataFrame]]:
    base = files("gee_biophys.models.s2biophys_params") / trait
    if not base.is_dir():
        raise FileNotFoundError(
            f"Packaged models for trait '{trait}' not found at {base}. "
            "Ensure files are included via [tool.setuptools.package-data].",
        )

    ensemble_size = 5  # fixed for s2biophys v2
    models: Dict[str, EnsembleItem] = {}

    for t in range(ensemble_size):
        name = f"optuna-v2-{trait}-mlp-split-{t}"

        pipeline_path = base / f"model_{name}.pkl"
        config_path = base / f"model_{name}_config.json"
        min_max_bands_path = base / f"min_max_band_values_{name}.json"
        min_max_label_path = base / f"min_max_label_values_{name}.json"
        split_path = base / f"model_{name}_split.json"

        required = {
            "pipeline": pipeline_path,
            "config": config_path,
            "min_max_bands": min_max_bands_path,
            "min_max_label": min_max_label_path,
            "split": split_path,
        }

        missing = [k for k, p in required.items() if not p.is_file()]
        if missing:
            # Give a helpful, actionable error
            details = "\n".join(f"  - {k}: {required[k]}" for k in missing)
            raise FileNotFoundError(
                f"Missing required model files for '{name}':\n{details}\n"
                "Make sure they are packaged and names match the expected pattern.",
            )

        item = EnsembleItem(
            config=_open_json(config_path),
            pipeline=_open_pickle(pipeline_path),
            model_path=pipeline_path.with_suffix("").name,
            min_max_bands=_open_json(min_max_bands_path),
            min_max_label=_open_json(min_max_label_path),
            split=_open_json(split_path),
        )
        models[name] = item

    # load uncertainty calibration parameters
    calibration_model_path = (
        base / f"calibration_uncertainty_model_{trait}_s2biophys_v02.pkl"
    )
    calibration_table_path = (
        base / f"calibration_uncertainty_table_{trait}_s2biophys_v02.csv"
    )

    if not calibration_model_path.is_file() or not calibration_table_path.is_file():
        raise FileNotFoundError(
            f"Missing uncertainty calibration files for trait '{trait}'. "
            "Ensure they are included via [tool.setuptools.package-data].",
        )

    calibration_model = _open_pickle(calibration_model_path)
    calibration_table = pd.read_csv(calibration_table_path)

    return models, (calibration_model, calibration_table)  # type: ignore


def prepare_s2_input_for_s2biophys(img: ee.Image) -> ee.Image:
    """Prepare Sentinel-2 image for S2Biophys model prediction by selecting and ordering bands.

    Parameters
    ----------
    - img (ee.Image): Input Sentinel-2 image with bands.

    Returns
    -------
    - ee.Image: Image with bands ordered as required by S2Biophys model.

    """
    bands = [
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
    angles = ["tts", "tto", "psi"]
    # reorder bands
    band_order_reorder = bands + angles

    return img.select(band_order_reorder)
