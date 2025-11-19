from datetime import datetime, timedelta

import ee
from dateutil.relativedelta import relativedelta
from loguru import logger

from gee_biophys.config import ConfigParams


def get_system_index(
    cfg: ConfigParams,
    interval_start: datetime,
    interval_end: datetime,
) -> str:
    band_suffix = "-".join(cfg.variables.bands)
    output_resolution = cfg.export.scale
    start_str = interval_start.strftime("%Y%m%d")
    end_str = interval_end.strftime("%Y%m%d")
    crs_str = cfg.export.crs.lower().replace("epsg:", "epsg-")
    version_str = cfg.version
    region_str = (
        cfg.spatial.region_name if cfg.spatial.region_name else cfg.spatial.type
    )

    index_str = (
        f"{cfg.variables.variable}_"
        f"{cfg.variables.model}_"
        f"{band_suffix}_"
        f"{output_resolution}m_s_"
        f"{start_str}_"
        f"{end_str}_"
        f"{region_str}_"
        f"{crs_str}_"
        f"{version_str}"
    )

    assert len(index_str) <= 100, (
        f"Generated system:index is too long ({len(index_str)} characters). {index_str} "
        "Please use shorter names for region_name in the configuration."
    )
    return index_str


def _get_cfg_string(cfg: ConfigParams) -> str:
    """Generate a string representation of the configuration parameters.
    This can be used for metadata or logging purposes.
    """
    cfg_parts = [
        f"spatial: {cfg.spatial}",
        f"temporal: {cfg.temporal}",
        f"variables: {cfg.variables}",
        f"options: {cfg.options}",
        f"export: {cfg.export}",
        f"version: {cfg.version}",
    ]
    return "; ".join(cfg_parts)


def update_image_metadata(
    image: ee.Image,
    interval_start: datetime,
    interval_end: datetime,
    cfg: ConfigParams,
) -> ee.Image:
    """Update the metadata of the given image with export configuration details.

    Args:
        image (ee.Image): The image whose metadata needs to be updated.
        interval_start (datetime): Start date of the interval.
        interval_end (datetime): End date of the interval.
        export_config (dict): The export configuration dictionary.

    Returns:
        ee.Image: The image with updated metadata.
        ["system:time_start", "system:time_end", "model", "variable", "export_scale", "output_crs", "system:index"]

        # system:index follows filenaming convention: [variable]_[model]_[output_bands (joined by '-')]_[output_resolution]m_s_[startdate(YYYYMMDD)]_[enddate(YYYYMMDD)]_[tile]_[crs(epsg lower and swap : with .)]_[version]

    """
    index_str = get_system_index(cfg, interval_start, interval_end)

    cfg_string = _get_cfg_string(cfg)

    updated_image = image.set(
        {
            "system:time_start": int(interval_start.timestamp() * 1000),
            "system:time_end": int(interval_end.timestamp() * 1000),
            "model": cfg.variables.model,
            "variable": cfg.variables.variable,
            "export_scale": cfg.export.scale,
            "output_crs": cfg.export.crs,
            "config": cfg_string,
            "system:index": index_str,
        },
    )
    return updated_image


def generate_intervals(start, end, temporal_interval):
    """Generate a list of (start, end) datetime tuples representing intervals
    between 'start' and 'end' based on the specified 'temporal_interval'.
    The 'temporal_interval' can be specified as:
    - An integer number of days
    - A timedelta object
    - A string representing common periods like "daily", "weekly", "monthly", etc.
    """
    logger.debug(
        f"Generating intervals from {start} to {end} with period {temporal_interval}",
    )
    if isinstance(temporal_interval, int):
        advance = lambda t: t + timedelta(days=temporal_interval)  # noqa
    elif isinstance(temporal_interval, str):
        temporal_interval = temporal_interval.lower()
        mapping = {
            "weekly": dict(weeks=1),
            "biweekly": dict(weeks=2),
            "monthly": dict(months=1),
            "bimonthly": dict(months=2),
            "quarterly": dict(months=3),
            "yearly": dict(years=1),
            "annual": dict(years=1),
        }
        if temporal_interval not in mapping:
            raise ValueError(f"Unknown period string: {temporal_interval}")
        delta = relativedelta(**mapping[temporal_interval])
        advance = lambda t: t + delta  # noqa
    else:
        raise TypeError("temporal_interval must be int, timedelta, or str")

    intervals = []
    current = start
    while current < end:
        nxt = advance(current)
        intervals.append((current, min(nxt, end)))
        current = nxt

    logger.debug(f"Generated {len(intervals)} time intervals.")
    return intervals


def asset_exists(asset_id: str) -> bool:
    """Check if a given Earth Engine asset exists.

    Args:
        asset_id (str): Full Earth Engine asset ID, e.g.,
                        "projects/ee-speckerfelix/assets/open-earth/lai_2023"

    Returns:
        bool: True if asset exists, False otherwise.

    """
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def set_asset_public(asset_id: str):
    """Sets a GEE asset to public read access."""
    acl = ee.data.getAssetAcl(asset_id)

    if "all_users_can_read" not in acl:
        # add 'all_users_can_read': False
        acl["all_users_can_read"] = True
        ee.data.setAssetAcl(asset_id, acl)
        logger.info(f"Set asset {asset_id} to public read access.")
    else:
        logger.info(f"Asset {asset_id} is already public.")


def initialize_export_location(cfg: ConfigParams, set_public: bool = False) -> None:
    loc = cfg.export.destination
    if loc == "asset":
        if asset_exists(cfg.export.collection_path):
            logger.warning(
                f"Asset {cfg.export.collection_path} already exists. Image exports with existing asset IDs will fail (delete beforehand if overwrite is desired).",
            )
        else:
            logger.info(
                f"Creating ImageCollection asset at {cfg.export.collection_path}.",
            )
            ee.data.createAsset({"type": "ImageCollection"}, cfg.export.collection_path)

            # add cfg string as property to ImageCollection asset
            cfg_string = _get_cfg_string(cfg)
            ee.data.setAssetProperties(
                cfg.export.collection_path,
                {"config": cfg_string},
            )

    if set_public:
        if cfg.export.destination != "asset":
            logger.warning(
                "Public access can only be set for asset exports. Skipping setting public access.",
            )
        else:
            logger.debug(
                "Note: The exported imageCollection will be set to public access, such that it can be visualized using the GEE-App: https://ee-speckerfelix.projects.earthengine.app/view/gee-biophys-export-visualizer"
            )
            set_asset_public(cfg.export.collection_path)

    elif loc == "drive":
        logger.info(
            f"Assets will be exported to Google Drive folder: {cfg.export.folder}",
        )

    elif loc == "gcs":
        logger.info(
            f"Assets will be exported to: gs://{cfg.export.bucket}/{cfg.export.folder}",
        )
    else:
        raise ValueError(f"Unknown output_location: {loc}")
