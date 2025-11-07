from functools import reduce

import ee

from gee_biophys.config import ConfigParams
from gee_biophys.models.s2biophys import eePipelinePredictMap, load_model_ensemble
from gee_biophys.models.sl2p import load_SL2P_model
from gee_biophys.utils_predict import (
    aggregate_ensemble_predictions,
    aggregate_imagecollection_simple,
)


def biophys_predict(cfg: ConfigParams, input_imgc: ee.ImageCollection) -> ee.Image:
    """
    Apply the selected biophysical model to the input Sentinel-2 ImageCollection
    and return an ImageCollection with predicted biophysical variables.
    """
    if cfg.variables.model == "sl2p":
        model_mean, model_std = load_SL2P_model()

        pred_mean_imgc = input_imgc.map(lambda img: model_mean.ee_predict(img))
        pred_std_imgc = input_imgc.map(lambda img: model_std.ee_predict(img))

        output_image = aggregate_ensemble_predictions(
            pred_mean_imgc, pred_std_imgc, cfg.variables.variable
        )

    elif cfg.variables.model == "s2biophys":
        s2biophys_model_ensemble = load_model_ensemble("laie")

        gee_preds = {}
        for i, (model_name, model) in enumerate(s2biophys_model_ensemble.items()):
            gee_preds[model_name] = eePipelinePredictMap(
                pipeline=model.pipeline,
                imgc=input_imgc,
                trait=cfg.variables.variable,
                model_config=model.config,
                min_max_bands=model.min_max_bands,
                min_max_label=None,
            )

        s2biophys_imgc_preds = reduce(
            lambda x, y: x.merge(y),
            gee_preds.values(),
        )

        output_image = aggregate_imagecollection_simple(
            s2biophys_imgc_preds,
            cfg.variables.variable,
            replications=len(s2biophys_model_ensemble),
        )

    else:
        raise ValueError(f"Unsupported model: {cfg.variables.model}")

    # select only desired output bands
    output_band_names = [
        f"{cfg.variables.variable}_{band}" for band in cfg.variables.bands
    ]

    return output_image.select(output_band_names)
