import ee
import numpy as np
import pytest

from gee_biophys.models.sl2p import load_SL2P_model  # assumes you expose this

ATOL = 1e-6

# SL2P expects reflectances + cosines of angles (already cos-transformed)
BAND_ORDER_SL2P = [
    "cosSZA",
    "cosVZA",
    "cosRAA",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8A",
    "B11",
    "B12",
]

N_SAMPLES = 10


@pytest.mark.parametrize("variable", ["lai", "fapar", "fcover"])
def test_sl2p_lai_random_inputs(ee_init, variable):
    rng = np.random.default_rng(42)

    # load paired models
    sl2p_mean_model, sl2p_std_model = load_SL2P_model(variable)

    for _ in range(N_SAMPLES):
        # --------- draw a valid sample ---------
        sample = {}
        # cosines are in [-1, 1]
        sample["cosSZA"] = float(rng.uniform(-1.0, 1.0))
        sample["cosVZA"] = float(rng.uniform(-1.0, 1.0))
        sample["cosRAA"] = float(rng.uniform(-1.0, 1.0))
        # surface reflectances in a typical range
        for b in ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]:
            sample[b] = float(rng.uniform(0.01, 0.5))

        # local predictions (sklearn)
        X = np.array([[sample[b] for b in BAND_ORDER_SL2P]], dtype=float)
        lai_mean_np = float(sl2p_mean_model.predict(X, clip_min_max=False).item())
        lai_std_np = float(sl2p_std_model.predict(X).item())

        # server-side predictions (single round-trip)
        img_in = ee.Image.constant([sample[b] for b in BAND_ORDER_SL2P]).rename(
            BAND_ORDER_SL2P
        )

        lai_mean_img = sl2p_mean_model.ee_predict(img_in).rename(f"{variable}_mean")
        lai_std_img = sl2p_std_model.ee_predict(img_in).rename(f"{variable}_std")
        out_img = ee.Image.cat([lai_mean_img, lai_std_img])

        # Sample once (constant image, but keep mask-safe pattern)
        pt = ee.Geometry.Point([0, 0])
        fc = out_img.sample(
            region=pt.buffer(50), scale=20, numPixels=1, dropNulls=True, tileScale=2
        )
        if fc.size().getInfo() == 0:
            vals = out_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=pt,
                scale=20,
                bestEffort=True,
                maxPixels=1e13,
            ).getInfo()
        else:
            vals = fc.first().toDictionary().getInfo()

        lai_mean_ee = float(vals[f"{variable}_mean"])
        lai_std_ee = float(vals[f"{variable}_std"])

        # assertions with good diagnostics
        try:
            np.testing.assert_allclose(lai_mean_np, lai_mean_ee, atol=ATOL)
            np.testing.assert_allclose(lai_std_np, lai_std_ee, atol=ATOL)
        except AssertionError as e:
            msg = (
                "\nSL2P LAI mismatch\n"
                f"Inputs (ordered {BAND_ORDER_SL2P}):\n"
                f"  {[sample[b] for b in BAND_ORDER_SL2P]}\n"
                f"Mean: local={lai_mean_np:.10f}, ee={lai_mean_ee:.10f}, Δ={lai_mean_np - lai_mean_ee:.2e}\n"
                f"Std : local={lai_std_np:.10f},  ee={lai_std_ee:.10f},  Δ={lai_std_np - lai_std_ee:.2e}\n"
            )
            raise AssertionError(msg) from e
