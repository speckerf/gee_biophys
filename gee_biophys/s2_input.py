from datetime import datetime
from typing import Literal

import ee
import shapely.geometry
from loguru import logger

from gee_biophys.config import ConfigParams


def get_s2_imgc(
    start_date: datetime,
    end_date: datetime,
    region: ee.Geometry,
    max_cloud_cover: int,
    bands: list[str],
) -> ee.ImageCollection:
    """Retrieve Sentinel-2 image collection for the specified date range and region,
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
    """Apply the CloudScorePlus algorithm to the given Sentinel-2 image collection
    and mask out cloudy pixels.
    """
    csplus_imgc = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    linked_imgc = s2_imgc.linkCollection(csplus_imgc, [csplus_band])

    s2_imgc_masked = linked_imgc.map(
        lambda image: image.updateMask(image.select(csplus_band).gte(csplus_threshold)),
    )

    return s2_imgc_masked.select(s2_imgc.first().bandNames())


def add_angles_from_metadata_to_bands(image: ee.Image) -> ee.Image:
    """Add viewing/illumination angle bands (in degrees) derived from image metadata.

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
            [image.getNumber("MEAN_INCIDENCE_AZIMUTH_ANGLE_%s" % b) for b in bands],
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
        ee.Image(view_azimuth.subtract(solar_azimuth).abs()).toFloat().rename("psi"),
    )

    return image


def load_s2_input(
    cfg: ConfigParams,
    interval_start: datetime,
    interval_end: datetime,
) -> ee.ImageCollection:
    """Load and prepare Sentinel-2 ImageCollection based on configuration parameters."""
    bands = [
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

    s2_imgc = get_s2_imgc(
        start_date=interval_start,
        end_date=interval_end,
        region=cfg.spatial.ee_geometry,
        max_cloud_cover=cfg.options.max_cloud_cover,
        bands=bands,
    )
    # logger.debug(f"Input S2 ImageCollection size: {s2_imgc.size().getInfo()}")

    s2_imgc = apply_cloudscore_plus_mask(
        s2_imgc,
        csplus_band=cfg.options.csplus_band,
        csplus_threshold=cfg.options.cs_plus_threshold,
    )

    # Add angles from metadata to bands
    s2_imgc = s2_imgc.map(add_angles_from_metadata_to_bands)

    return s2_imgc.set(
        {
            "system:time_start": int(interval_start.timestamp() * 1000),
            "system:time_end": int(interval_end.timestamp() * 1000),
        },
    )


def convert_s2_input_to_xarray(
    cfg: ConfigParams,
    s2_imgc: ee.ImageCollection,
):
    """Convert a Sentinel-2 ImageCollection to an xarray Dataset.

    This helper expects an ImageCollection that is already cloud-masked and model-prepared
    through ``load_s2_input``.
    """
    import xarray as xr
    import xee

    source_params = xee.helpers.extract_grid_params(s2_imgc)
    source_crs = source_params["crs"]
    source_scale = (cfg.export.scale, -cfg.export.scale)

    if cfg.spatial.type == "bbox":
        aoi = shapely.geometry.box(*cfg.spatial.bbox)
    else:
        bounds_coords = cfg.spatial.ee_geometry.bounds(1).coordinates().getInfo()[0]
        lons = [coord[0] for coord in bounds_coords]
        lats = [coord[1] for coord in bounds_coords]
        aoi = shapely.geometry.box(min(lons), min(lats), max(lons), max(lats))

    grid_params = xee.helpers.fit_geometry(
        geometry=aoi,
        geometry_crs="EPSG:4326",
        grid_crs=source_crs,
        grid_scale=source_scale,
    )

    if cfg.options.prediction_mode == "composite_then_predict":
        logger.debug(
            "Prediction mode is 'composite_then_predict': compositing ImageCollection to median image before loading as xarray.",
        )
        s2_imgc = ee.ImageCollection(
            [
                s2_imgc.median().set(
                    "system:time_start",
                    s2_imgc.get("system:time_start"),
                    "system:time_end",
                    s2_imgc.get("system:time_end"),
                )
            ]
        ).set(
            "system:time_start",
            s2_imgc.get("system:time_start"),
            "system:time_end",
            s2_imgc.get("system:time_end"),
        )

    if cfg.spatial.type == "square":
        # Raster grids are rectangular; mask the corners outside the transformed square.
        s2_imgc = s2_imgc.map(lambda image: image.clip(cfg.spatial.ee_geometry))

    return xr.open_dataset(
        s2_imgc,
        engine="ee",
        **grid_params,
    )
