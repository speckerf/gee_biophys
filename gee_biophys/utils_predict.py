import ee


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
