import ee
import numpy as np
import pandas as pd
import pytest

from gee_biophys.models.s2biophys import eePipelinePredictMap, load_model_ensemble

ATOL = 1e-5
N_SAMPLES = 10


@pytest.mark.parametrize("variable", ["laie", "fapar", "fcover"])
def test_s2biophys_random_inputs(ee_init, variable):
    (
        s2biophys_ensemble,
        (uncertainty_calibration_model, uncertainty_calibration_table),
    ) = load_model_ensemble(variable)

    rng = np.random.default_rng(123)
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

    # Sort models by name for deterministic behavior
    models = sorted(s2biophys_ensemble.items(), key=lambda kv: kv[0])  # (name, spec)

    for _ in range(N_SAMPLES):
        # -------- input sample (kept within reasonable valid ranges) --------
        sample = {}
        for b in band_order:
            if b.startswith("B"):
                sample[b] = float(rng.uniform(0.01, 0.4))
            elif b in ("tts", "tto"):
                sample[b] = float(rng.uniform(0, 60))
            elif b == "psi":
                sample[b] = float(rng.uniform(0, 180))

        df = pd.DataFrame([[sample[b] for b in band_order]], columns=band_order)

        # -------- local (sklearn) predictions, per model --------
        preds_np = np.array(
            [spec["pipeline"].predict(df).item() for _, spec in models], dtype=float
        )
        mean_np = float(preds_np.mean())
        std_np = float(preds_np.std())

        # -------- server (EE) predictions, per model, single round-trip --------
        # Build a constant image with our sample values, in correct band order
        img_in = ee.Image.constant([sample[b] for b in band_order]).rename(band_order)

        # Some pipelines expect an ImageCollection; wrap and pass through each model
        imgc_in = ee.ImageCollection([img_in])

        # Run each model; turn each result IC into a single-band Image named p_<i>
        pred_imgs = []
        for i, (_, spec) in enumerate(models):
            ic_pred = eePipelinePredictMap(
                pipeline=spec["pipeline"],
                imgc=imgc_in,
                trait=variable,
                model_config=spec["config"],
                min_max_bands=None,
            )
            # Use mosaic() in case the pipeline returns more than one tile
            pred_imgs.append(
                ee.Image(ic_pred.mosaic()).select(variable).rename(f"p_{i}")
            )

        # Stack all per-model predictions into one multi-band image
        stack = ee.Image.cat(pred_imgs)

        # Compute ensemble mean/std on the stack server-side (across bands)
        mean_img = stack.reduce(ee.Reducer.mean()).rename(f"{variable}_mean")
        std_img = stack.reduce(ee.Reducer.stdDev()).rename(f"{variable}_stdDev")
        out_img = ee.Image.cat([stack, mean_img, std_img])

        # Sample once (use a tiny buffer+dropNulls to dodge mask edge-cases)
        pt = ee.Geometry.Point([0, 0])
        fc = out_img.sample(
            region=pt.buffer(100), scale=20, numPixels=1, dropNulls=True, tileScale=2
        )
        size = fc.size().getInfo()

        if size == 0:
            # As a fallback (should rarely trigger with constant input), use reduceRegion
            vals = out_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=pt,
                scale=20,
                bestEffort=True,
                maxPixels=1e13,
            ).getInfo()
        else:
            vals = ee.Feature(fc.first()).toDictionary().getInfo()

        # Pull per-model EE preds in consistent order
        preds_ee = np.array([vals[f"p_{i}"] for i in range(len(models))], dtype=float)
        mean_ee = float(vals[f"{variable}_mean"])
        std_ee = float(vals[f"{variable}_stdDev"])

        # -------- assertions with helpful diagnostics --------
        try:
            np.testing.assert_allclose(preds_np, preds_ee, atol=ATOL)
        except AssertionError as e:
            diffs = (preds_np - preds_ee).tolist()
            msg = (
                f"\nPer-model mismatch:\n"
                f"  local: {preds_np.tolist()}\n"
                f"   ee  : {preds_ee.tolist()}\n"
                f"  diff : {diffs}\n"
            )
            raise AssertionError(msg) from e

        try:
            np.testing.assert_allclose(mean_np, mean_ee, atol=ATOL)
            np.testing.assert_allclose(std_np, std_ee, atol=ATOL)
        except AssertionError as e:
            msg = (
                f"\nEnsemble mismatch:\n"
                f"  mean  local={mean_np:.8f}, ee={mean_ee:.8f}, Δ={mean_np - mean_ee:.2e}\n"
                f"  std   local={std_np:.8f},  ee={std_ee:.8f},  Δ={std_np - std_ee:.2e}\n"
                f"  per-model local: {preds_np.tolist()}\n"
                f"  per-model ee   : {preds_ee.tolist()}\n"
            )
            raise AssertionError(msg) from e
