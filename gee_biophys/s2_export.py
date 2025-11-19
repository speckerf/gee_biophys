import ee
from loguru import logger

from gee_biophys.config import ConfigParams


def export_image(image: ee.Image, filename: str, cfg: ConfigParams):
    loc = cfg.export.destination

    if cfg.spatial.type == "geojson" and cfg.spatial.geojson_clip:
        logger.debug(
            "Clipping export image to GeoJSON geometry bounds. Please be aware of potential issues with complex geometries. Set 'geojson_clip' to false to disable.",
        )
        image = image.clip(cfg.spatial.ee_geometry)

    if loc == "asset":
        asset_id = f"{cfg.export.collection_path}/{filename}"
        logger.debug(f"Exporting image to Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=image,
            description=filename,
            assetId=asset_id,
            region=cfg.spatial.ee_geometry,
            scale=cfg.export.scale,
            crs=cfg.export.crs,
            maxPixels=cfg.export.max_pixels,
        )
        task.start()
    elif loc == "drive":
        image = image.float()  # to avoid error: Exported bands must have compatible data types; found inconsistent types: Float64 and Int32
        asset_id = f"{cfg.export.folder}/{filename}"
        logger.debug(
            f"Exporting image to Google Drive file: {cfg.export.folder}/{filename}",
        )
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=filename,
            folder=cfg.export.folder,
            fileNamePrefix=filename,
            region=cfg.spatial.ee_geometry,
            scale=cfg.export.scale,
            crs=cfg.export.crs,
            maxPixels=1e11,
        )
        task.start()
    elif loc == "gcs":
        image = image.float()  # to avoid error: Exported bands must have compatible data types; found inconsistent types: Float64 and Int32
        bucket = cfg.export.bucket
        gcs_folder = cfg.export.folder

        logger.debug(
            f"Exporting image to GCS bucket: gs://{bucket}/{gcs_folder}/{filename}",
        )
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=filename,
            bucket=bucket,
            fileNamePrefix=f"{gcs_folder}/{filename}" if gcs_folder else filename,
            region=cfg.spatial.ee_geometry,
            scale=cfg.export.scale,
            crs=cfg.export.crs,
            maxPixels=1e11,
        )
        task.start()
    else:
        raise ValueError(f"Unknown output_location: {loc}")

    logger.debug(f"Task started with ID: {task.id}")
