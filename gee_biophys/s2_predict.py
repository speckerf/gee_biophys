import ee

from gee_biophys.config import ConfigParams
from gee_biophys.models.s2biophys import (
    eeEnsemblePredictSingleImg,
    load_model_ensemble,
)
from gee_biophys.models.sl2p import load_SL2P_model
from gee_biophys.utils_predict import (
    aggregate_ensemble_predictions,
    reduce_ensemble_preds,
)


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
