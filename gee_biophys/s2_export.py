import shutil
from pathlib import Path

import ee
import numpy as np
import xarray as xr
from loguru import logger

from gee_biophys.config import ConfigParams


def export_image(
    image: ee.Image | xr.Dataset,
    filename: str,
    cfg: ConfigParams,
    subfolder: str | None = None,
):
    loc = cfg.export.destination

    if (
        loc != "xee-local"
        and cfg.spatial.type == "geojson"
        and cfg.spatial.geojson_clip
    ):
        logger.debug(
            "Clipping export image to GeoJSON geometry bounds. Please be aware of potential issues with complex geometries. Set 'geojson_clip' to false to disable.",
        )
        image = image.clip(cfg.spatial.ee_geometry)

    task = None
    if loc == "asset":
        if not isinstance(image, ee.Image):
            raise TypeError("Asset export requires an ee.Image.")
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
        if not isinstance(image, ee.Image):
            raise TypeError("Drive export requires an ee.Image.")
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
        if not isinstance(image, ee.Image):
            raise TypeError("GCS export requires an ee.Image.")
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
    elif loc == "xee-local":
        if not isinstance(image, xr.Dataset):
            raise TypeError("xee-local export requires an xarray.Dataset.")

        output_root = Path(cfg.export.folder)
        if subfolder:
            output_root = output_root / subfolder
        output_root.mkdir(parents=True, exist_ok=True)

        output_path = output_root / f"{filename}.zarr"
        logger.debug(f"Exporting dataset to local Zarr store: {output_path}")
        image.to_zarr(output_path, mode="w")
        logger.info(f"Saved local output: {output_path}")
    else:
        raise ValueError(f"Unknown output_location: {loc}")

    if task is not None:
        logger.debug(f"Task started with ID: {task.id}")


def merge_local_interval_exports(
    cfg: ConfigParams,
    final_filename: str,
    temp_subfolder: str = "temp",
    cleanup_temp: bool = True,
) -> Path:
    """Merge per-interval local Zarr stores from temp folder into one final store."""
    if cfg.export.destination != "xee-local":
        raise ValueError("merge_local_interval_exports is only valid for xee-local.")

    target_root = Path(cfg.export.folder)
    temp_root = target_root / temp_subfolder
    stores = sorted(p for p in temp_root.glob("*.zarr") if p.is_dir())

    if not stores:
        raise FileNotFoundError(f"No temporary Zarr stores found in {temp_root}")

    datasets = [xr.open_zarr(store) for store in stores]
    try:
        if all("time" in ds.dims for ds in datasets):
            merged = xr.concat(datasets, dim="time", combine_attrs="drop_conflicts")
        else:
            with_time = []
            for ds in datasets:
                ts = ds.attrs.get("system:time_start")
                if ts is None:
                    raise ValueError(
                        "Temporary dataset is missing both 'time' dimension and 'system:time_start' attr."
                    )
                with_time.append(ds.expand_dims(time=[np.datetime64(int(ts), "ms")]))
            merged = xr.concat(with_time, dim="time", combine_attrs="drop_conflicts")

        merged = merged.sortby("time")

        final_path = target_root / f"{final_filename}.zarr"
        merged_attrs = dict(merged.attrs)
        merged_attrs["system:index"] = final_filename
        merged_attrs["interval_store_count"] = len(stores)
        merged = merged.assign_attrs(merged_attrs)

        logger.debug(f"Writing merged local Zarr store: {final_path}")
        merged.to_zarr(final_path, mode="w")
        logger.info(f"Saved merged local output: {final_path}")
    finally:
        for ds in datasets:
            ds.close()
        if cleanup_temp and temp_root.exists():
            shutil.rmtree(temp_root)
            logger.debug(f"Removed temporary local export folder: {temp_root}")

    return final_path
