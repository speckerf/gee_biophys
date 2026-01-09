import ee


def reduce_ensemble_preds(imgc_preds: ee.ImageCollection, variable: str) -> ee.Image:
    """
    Reduce an ImageCollection of per-image prediction outputs to summary bands.

    Expects per-image bands:
      - f"{variable}_mean"
      - f"{variable}_stdDev"   (within-image predictive uncertainty, per image)

    Returns an ee.Image with bands:
      - f"{variable}_mean"
      - f"{variable}_count"
      - f"{variable}_stdDev_within"
      - f"{variable}_stdDev_across"
      - f"{variable}_stdDev"  (total)
    """
    b_mean = f"{variable}_mean"
    b_sd = f"{variable}_stdDev"

    # Mean prediction across images
    preds_mean = imgc_preds.select(b_mean).mean().rename(b_mean)

    # Count of valid observations (per pixel)
    preds_count = (
        imgc_preds.select(b_mean).reduce(ee.Reducer.count()).rename(f"{variable}_count")
    )

    # "Within" uncertainty: average per-image stdDev
    preds_stdDev_within = (
        imgc_preds.select(b_sd).mean().rename(f"{variable}_stdDev_within")
    )

    # "Across" uncertainty: sample std dev across per-image means
    # (unmask for stable math, but mask back to within-mask to avoid inventing coverage)
    preds_stdDev_across = (
        imgc_preds.select(b_mean)
        .reduce(ee.Reducer.sampleStdDev())
        .unmask(0)
        .updateMask(preds_stdDev_within.mask())
        .rename(f"{variable}_stdDev_across")
    )

    # Total uncertainty: sqrt(within^2 + across^2)
    preds_stdDev_total = (
        preds_stdDev_across.pow(2)
        .add(preds_stdDev_within.pow(2))
        .sqrt()
        .rename(f"{variable}_stdDev")
    )

    return ee.Image.cat(
        [
            preds_mean,
            preds_stdDev_total,
            preds_stdDev_within,
            preds_stdDev_across,
            preds_count,
        ]
    )


def aggregate_imagecollection_simple(
    imgc: ee.ImageCollection,
    trait_name: str,
    replications: int = 1,
    clip_min_max: bool = True,
) -> ee.Image:
    """Aggregate an ImageCollection by calculating the mean, standard deviation, and count of images.

    Args:
        imgc (ee.ImageCollection): The input ImageCollection containing predictions.
        trait_name (str): The name of the trait being predicted (used for naming output bands).
        replications (int, optional): The number of model replications used to generate the ImageCollection. Defaults to 1.

    Returns:
        ee.Image: An Image containing the mean, standard deviation, and count of predictions.

    Notes:
       - The count band is normalized by the number of replications to reflect the proportion of valid predictions.

    """
    mean_name = f"{trait_name}_mean"
    std_name = f"{trait_name}_stdDev"
    count_name = f"{trait_name}_count"

    img_mean = imgc.mean().rename(mean_name)
    img_std = imgc.reduce(ee.Reducer.stdDev()).rename(std_name)
    img_counts = (
        imgc.reduce(ee.Reducer.count()).divide(replications).toInt().rename(count_name)
    )

    # clamp here:
    if clip_min_max:
        clip_dict = {
            "lai": (0, 8),
            "laie": (0, 8),
            "fapar": (0, 1),
            "fcover": (0, 1),
        }
        if trait_name in clip_dict:
            img_mean = img_mean.clamp(
                clip_dict[trait_name][0], clip_dict[trait_name][1]
            ).copyProperties(img_mean)
        else:
            raise ValueError(f"Clipping not defined for trait: {trait_name}")

    # TODO: min max range?
    img_to_return = ee.Image([img_mean, img_std, img_counts])
    return img_to_return


def aggregate_ensemble_predictions(
    mean_imgc: ee.ImageCollection,
    std_imgc: ee.ImageCollection,
    trait_name: str,
    clip_min_max: bool = True,
) -> ee.Image:
    mean_name = f"{trait_name}_mean"
    std_name = f"{trait_name}_stdDev"
    count_name = f"{trait_name}_count"

    img_mean = mean_imgc.mean().rename(mean_name)

    # clamp here:
    if clip_min_max:
        clip_dict = {
            "lai": (0, 8),
            "laie": (0, 8),
            "fapar": (0, 1),
            "fcover": (0, 1),
        }
        if trait_name in clip_dict:
            img_mean = img_mean.clamp(
                clip_dict[trait_name][0], clip_dict[trait_name][1]
            ).copyProperties(img_mean)
        else:
            raise ValueError(f"Clipping not defined for trait: {trait_name}")

    img_within_var = std_imgc.map(lambda img: img.pow(2)).mean()
    img_between_var = mean_imgc.reduce(ee.Reducer.variance())

    img_total_var = img_within_var.add(img_between_var)
    img_total_std = img_total_var.sqrt().rename(std_name)

    img_counts = mean_imgc.count().rename(count_name)
    img_to_return = ee.Image([img_mean, img_total_std, img_counts])
    return img_to_return
    return img_to_return
