import json
from importlib.resources import files
from typing import Literal, Optional, Tuple

import ee
import numpy as np
from loguru import logger


class LeafToolbox_MLPRegressor:
    def __init__(
        self,
        net: dict,
        domain_codes: list = None,
        clip_min_max: bool = False,
        clip_trait: Optional[str] = None,
    ) -> None:
        ### init original params
        # load params from json dict
        self.inp_slope = np.array(net["inp_slope"])
        self.inp_offset = np.array(net["inp_offset"])
        self.h1wt = np.array(net["h1wt"]).reshape(
            len(net["h1bi"]), len(self.inp_offset)
        )
        self.h1bi = np.array(net["h1bi"])
        self.h2wt = np.array(net["h2wt"]).reshape(1, len(self.h1bi))
        self.h2bi = np.array(net["h2bi"])
        self.out_slope = np.array(net["out_slope"])
        self.out_bias = np.array(net["out_bias"])
        self.bandorder = net["bandorder"]

        ### init ee params
        self.ee_inp_slope = ee.Array(self.inp_slope.tolist())
        self.ee_inp_offset = ee.Array(self.inp_offset.tolist())
        self.ee_h1wt = ee.Array(
            self.h1wt.tolist()
        ).transpose()  # its crucial to use transpose instead of reshape here!! (otherwise values fit into wrong positions)

        self.ee_h1bi = ee.Array(self.h1bi.tolist()).reshape([1, -1])
        self.ee_h2wt = ee.Array(self.h2wt.tolist()).transpose()

        self.ee_h2bi = ee.Array(self.h2bi.tolist()).reshape([1, -1])
        self.ee_out_slope = ee.Array(self.out_slope.tolist())
        self.ee_out_bias = ee.Array(self.out_bias.tolist())

        logger.debug(
            f"SL2P init(): Make sure that the input data is ordered as in bandorder: {self.bandorder}"
        )

        if domain_codes is not None:
            self.domain_codes = domain_codes
            logger.debug(
                f"SL2P init(): Domain codes for input validation provided: {self.domain_codes}"
            )
            self.init_domain_codes()
        else:
            self.domain_codes = None
            logger.debug("SL2P init(): No domain codes for input validation provided")

        if clip_min_max:
            assert clip_trait in [
                "lai",
                "fapar",
                "fcover",
            ], "clip_trait must be one of 'lai', 'fapar', 'fcover'"
            logger.debug(
                "clip_min_max=True is set. Predictions will be clipped to: 0-8 for LAI and 0-1 for FAPAR/FCOVER."
            )
            self.clip_min_max = True
            self.clip_trait = clip_trait
        else:
            self.clip_min_max = False
            self.clip_trait = None

    def _tansig(self, x):
        return 2.0 / (1.0 + np.exp(-2.0 * x)) - 1.0

    def init_domain_codes(self):
        raise NotImplementedError("Domain code initialization not implemented yet.")

    def check_domain(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, n_features)
        returns: boolean array of shape (n_samples,) indicating whether each sample is valid
        """
        if self.domain_codes is None:
            raise ValueError("Domain codes not initialized.")

        # Image formatting
        image_format = np.sum(
            (np.uint8(np.ceil(X * 10) % 10))
            * np.array([10**value for value in range(len(self.bandorder))])[:, None],
            axis=0,
        )

        # Comparing image to domain codes
        flag = np.isin(image_format, self.domain_codes, invert=True)
        return flag

    def ee_check_domain(self, img: ee.Image) -> ee.Image:
        """
        img: ee.Image with bands ordered as in self.bandorder
        returns: ee.Image with 1 for invalid pixels, 0 for valid pixels
        """
        # TODO: check function carefully!!!!!!! - Copilot generated
        # Add test case that check_domain() and ee_check_domain() give same results on same data

        if self.domain_codes is None:
            raise ValueError("Domain codes not initialized.")

        # Image formatting
        image_format = ee.Image.constant(0)
        for i, band in enumerate(self.bandorder):
            band_component = (
                img.select(band).multiply(10).mod(10).floor().multiply(10**i)
            )
            image_format = image_format.add(band_component)

        # Comparing image to domain codes
        domain_codes_ee = ee.List(self.domain_codes)
        invalid_mask = image_format.remap(
            domain_codes_ee, ee.List.repeat(0, domain_codes_ee.size()), 1
        )
        return invalid_mask

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_samples, n_features)
        returns: (n_samples,)
        """

        # input scaling
        Xs = X * self.inp_slope + self.inp_offset

        # hidden layer
        h1 = self._tansig(np.matmul(Xs, self.h1wt.T) + self.h1bi)

        # linear output layer
        h2 = np.matmul(h1, self.h2wt.T) + self.h2bi

        # output scaling
        y = (h2 - self.out_bias) / self.out_slope

        if self.clip_min_max:
            if self.clip_trait == "lai":
                y = np.clip(y, 0, 8)
            elif self.clip_trait in ["fapar", "fcover"]:
                y = np.clip(y, 0, 1)
        return y.ravel()

    def _tansig_ee(self, x: ee.Image) -> ee.Image:
        return (
            ee.Image(2)
            .divide(ee.Image(1).add(x.multiply(ee.Image(-2)).exp()))
            .subtract(ee.Image(1))
        )

    def ee_predict(self, ee_img: ee.Image) -> ee.Image:
        """
        Predict using an ee.Image as input.
        Returns an ee.Image with the prediction.
        """

        # TODO: check function carefully!!!!!!! - Copilot generated
        # Add test case that predict() and ee_predict() give same results on same data

        x = ee_img.toArray()
        x = x.multiply(ee.Image(self.ee_inp_slope)).add(ee.Image(self.ee_inp_offset))

        # convert to 2D array for matrix multiplication
        x = x.toArray(1).arrayTranspose()  # shape : (1, bands)

        h1 = self._tansig_ee(
            x.matrixMultiply(ee.Image(self.ee_h1wt)).add(ee.Image(self.ee_h1bi))
        )  # (1, bands) x (bands, n_nodes) = (1, n_nodes)

        # linear output layer
        h2 = h1.matrixMultiply(ee.Image(self.ee_h2wt)).add(ee.Image(self.ee_h2bi))
        # (1, n_nodes) x (n_nodes, 1) = (1, 1)

        y = (
            h2.arrayProject([0])
            .subtract(ee.Image(self.ee_out_bias))
            .divide(ee.Image(self.ee_out_slope))
        )

        if self.clip_min_max:
            if self.clip_trait == "lai":
                y = y.clamp(0, 8)
            elif self.clip_trait in ["fapar", "fcover"]:
                y = y.clamp(0, 1)

        return y.arrayFlatten([["output"]])

    def predict_with_domain_check(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.domain_codes is None:
            raise ValueError("Domain codes not initialized.")

        invalid_mask = self.check_domain(X)
        if np.any(invalid_mask):
            logger.warning(
                f"{np.sum(invalid_mask)} samples are outside the valid input domain."
            )

        return self.predict(X), invalid_mask


def load_SL2P_model(
    variable: Literal["lai", "laie", "fapar", "fcover"],
) -> Tuple[LeafToolbox_MLPRegressor, LeafToolbox_MLPRegressor]:
    # remap to trait names used in SL2P parameter files
    trait_map = {
        "lai": "LAI",
        "laie": "LAI",
        "fapar": "fAPAR",
        "fcover": "fCOVER",
    }
    variable_mapped = trait_map[variable]

    package = "gee_biophys.models.sl2p_params"

    estimate_file = f"{variable_mapped}-estimation_SL2P_Corrected_LeafToolBox.json"
    uncertainty_json = f"{variable_mapped}-uncertainty_SL2P_Corrected_LeafToolBox.json"

    with (files(package) / estimate_file).open("r") as f:
        estimate_params = json.load(f)

    with (files(package) / uncertainty_json).open("r") as f:
        uncertainty_params = json.load(f)

    model_estimate = LeafToolbox_MLPRegressor(estimate_params, domain_codes=None)
    model_uncertainty = LeafToolbox_MLPRegressor(uncertainty_params, domain_codes=None)

    return (
        model_estimate,
        model_uncertainty,
    )


def _ee_angle_transform_sl2p(angle_img: ee.Image) -> ee.Image:
    """
    Transform angle bands from degrees to the format required by SL2P:
    - Convert degrees to radians
    - Compute cosine of the angles

    Parameters:
    - angle_img (ee.Image): Image with angle bands in degrees.

    Returns:
    - ee.Image: Image with angle bands transformed for SL2P.
    """
    radians_img = angle_img.multiply(np.pi / 180.0)
    cos_img = radians_img.cos()
    return cos_img


def prepare_s2_input_for_sl2p(img: ee.Image) -> ee.Image:
    """
    Prepare Sentinel-2 image for SL2P model input.
    Parameters:
    - img (ee.Image): Input Sentinel-2 image with bands and angles.
    Returns:
    - ee.Image: Image with bands ordered and angles transformed for SL2P.
    """

    # Bands/angles expected on the input S2 image
    _s2_bands = ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
    _s2_angles = ["tts", "tto", "psi"]

    # Output names required by SL2P (cosines first, then reflectance bands)
    _sl2p_angle_names = ["cosSZA", "cosVZA", "cosRAA"]
    _sl2p_output_order = _sl2p_angle_names + _s2_bands

    refl = img.select(_s2_bands)
    cos_angles = _ee_angle_transform_sl2p(img.select(_s2_angles)).rename(
        _sl2p_angle_names
    )

    out = refl.addBands(cos_angles)

    return out.select(_sl2p_output_order)
