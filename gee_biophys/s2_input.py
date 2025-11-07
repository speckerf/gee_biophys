from datetime import datetime
from typing import Literal

import ee
from loguru import logger

from gee_biophys.config import ConfigParams
from models.s2biophys import prepare_s2_input_for_specker
from models.sl2p import prepare_s2_input_for_sl2p


def get_s2_imgc(
    start_date: datetime,
    end_date: datetime,
    region: ee.Geometry,
    max_cloud_cover: int,
    bands: list = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
) -> ee.ImageCollection:
    """
    Retrieve Sentinel-2 image collection for the specified date range and region,
    selecting only the specified bands.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_cover))
        .select(bands)
    )

    # divide DN values by 10000 to get reflectance, copy properties again
    def scale_reflectance(image):
        scaled = (
            image.select(bands)
            .divide(10000)
            .copyProperties(image, image.propertyNames())
        )
        return scaled

    collection = collection.map(scale_reflectance)
    return collection


def apply_cloudscore_plus_mask(
    s2_imgc: ee.ImageCollection,
    csplus_band: Literal["cs", "cs_cdf"],
    csplus_threshold: float,
) -> ee.ImageCollection:
    """
    Apply the CloudScorePlus algorithm to the given Sentinel-2 image collection
    and mask out cloudy pixels.
    """

    csplus_imgc = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    linked_imgc = s2_imgc.linkCollection(csplus_imgc, [csplus_band])

    s2_imgc_masked = linked_imgc.map(
        lambda image: image.updateMask(image.select(csplus_band).gte(csplus_threshold))
    )

    return s2_imgc_masked.select(s2_imgc.first().bandNames())


def add_angles_from_metadata_to_bands(image: ee.Image) -> ee.Image:
    """
    Add viewing/illumination angle bands (in degrees) derived from image metadata.

    This function reads Sentinel-2 metadata fields to:
      - compute the mean **view zenith** and **view azimuth** across bands B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,
      - read the **solar zenith** and **solar azimuth**,
      - and append three angle bands (no trigonometric transforms, no reflectance scaling):

        * tts — solar zenith angle (degrees)
        * tto — mean view zenith angle across the listed bands (degrees)
        * psi — absolute azimuth difference |view_azimuth − solar_azimuth| (degrees)

    Notes
    -----
    - Angles are kept in **degrees** (not cosine).

    - Expected metadata keys (Sentinel-2):
      `MEAN_SOLAR_AZIMUTH_ANGLE`, `MEAN_SOLAR_ZENITH_ANGLE`,
      `MEAN_INCIDENCE_AZIMUTH_ANGLE_<BAND>`, `MEAN_INCIDENCE_ZENITH_ANGLE_<BAND>`.

    Parameters
    ----------
    image : ee.Image
        Input image with the required Sentinel-2 angle metadata.

    Returns
    -------
    ee.Image
        The input image with added bands: 'tts', 'tto', and 'psi' (float32, degrees).
    """

    # Define the bands for which view angles are extracted from metadata.
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    # Extract the solar azimuth and zenith angles from metadata.
    solar_azimuth = image.getNumber("MEAN_SOLAR_AZIMUTH_ANGLE")
    solar_zenith = image.getNumber("MEAN_SOLAR_ZENITH_ANGLE")

    # Calculate the mean view azimuth angle for the specified bands.
    view_azimuth = (
        ee.Array(
            [image.getNumber("MEAN_INCIDENCE_AZIMUTH_ANGLE_%s" % b) for b in bands]
        )
        .reduce(ee.Reducer.mean(), [0])
        .get([0])
    )

    # Calculate the mean view zenith angle for the specified bands.
    view_zenith = (
        ee.Array([image.getNumber("MEAN_INCIDENCE_ZENITH_ANGLE_%s" % b) for b in bands])
        .reduce(ee.Reducer.mean(), [0])
        .get([0])
    )

    # add tts, tto, psi
    image = image.addBands(ee.Image(solar_zenith).toFloat().rename("tts"))
    image = image.addBands(ee.Image(view_zenith).toFloat().rename("tto"))
    image = image.addBands(
        ee.Image(view_azimuth.subtract(solar_azimuth).abs()).toFloat().rename("psi")
    )

    return image


def load_s2_input(
    cfg: ConfigParams, interval_start: datetime, interval_end: datetime
) -> ee.ImageCollection:
    """
    Load and prepare Sentinel-2 ImageCollection based on configuration parameters.
    """
    s2_imgc = get_s2_imgc(
        start_date=interval_start,
        end_date=interval_end,
        region=cfg.spatial.ee_geometry,
        max_cloud_cover=cfg.options.max_cloud_cover,
    )
    logger.debug(f"Input S2 ImageCollection size: {s2_imgc.size().getInfo()}")

    s2_imgc = apply_cloudscore_plus_mask(
        s2_imgc,
        csplus_band=cfg.options.csplus_band,
        csplus_threshold=cfg.options.cs_plus_threshold,
    )

    # Add angles from metadata to bands
    s2_imgc = s2_imgc.map(add_angles_from_metadata_to_bands)

    if cfg.variables.model == "sl2p":
        s2_imgc = s2_imgc.map(prepare_s2_input_for_sl2p)
    elif cfg.variables.model == "s2biophys":
        s2_imgc = s2_imgc.map(prepare_s2_input_for_specker)
    else:
        raise ValueError(f"Unsupported model '{cfg.variables.model}'")

    return s2_imgc


# if __name__ == "__main__":
#     ee.Initialize()

#     start = datetime(2023, 6, 1)
#     end = datetime(2023, 6, 30)
#     region = ee.Geometry.Point([-122.262, 37.8719]).buffer(10000)  # 10 km buffer

#     bands = ["B4", "B3", "B2"]  # Red, Green, Blue bands

#     s2_imgc = get_s2_imgc(start, end, region, bands, max_cloud_cover=50)
#     print(f"Original ImageCollection size: {s2_imgc.size().getInfo()}")

#     s2_imgc_masked = apply_cloudscore_plus_mask(
#         s2_imgc, csplus_band="cs", csplus_threshold=0.7
#     )
#     print(f"Masked ImageCollection size: {s2_imgc_masked.size().getInfo()}")
