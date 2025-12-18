import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from pickle import load as pickle_load
from typing import Any

import ee
import numpy as np
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


def eePipelinePredictMap(
    pipeline: Pipeline,
    imgc: ee.ImageCollection,
    trait: str,
    model_config: dict,
    min_max_bands: dict | None = None,
):
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


def load_model_ensemble(trait: str) -> dict:
    base = files("gee_biophys.models.specker_params") / trait
    if not base.is_dir():
        raise FileNotFoundError(
            f"Packaged models for trait '{trait}' not found at {base}. "
            "Ensure files are included via [tool.setuptools.package-data].",
        )

    n_testsets = 5
    models: dict[str, EnsembleItem] = {}

    for t in range(n_testsets):
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

    return models


def prepare_s2_input_for_specker(img: ee.Image) -> ee.Image:
    """Prepare Sentinel-2 image for Specker et al. model prediction by selecting and ordering bands.

    Parameters
    ----------
    - img (ee.Image): Input Sentinel-2 image with bands.

    Returns
    -------
    - ee.Image: Image with bands ordered as required by Specker et al. model.

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
