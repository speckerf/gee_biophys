"""Live parity checks: RUN_EE_TESTS=1 pytest tests/test_biome_lc_ee.py."""

# import os

import ee
import numpy as np
import pytest
from test_local_predict import _make_s2_dataset

from gee_biophys.s2_predict import biophys_predict_ee, biophys_predict_local


@pytest.mark.parametrize(
    "name",
    [
        "temperate_broadleaf_forest",
        "arid_shrubland",
        "temperate_nonforest",
    ],
)
def test_ee_local_variant_parity(ee_init, name):
    ee.Initialize()
    ds = _make_s2_dataset(shape=(2, 1, 1), seed=42)
    images = [
        ee.Image.constant([float(ds[b][t, 0, 0]) for b in ds.data_vars]).rename(
            list(ds.data_vars)
        )
        for t in range(2)
    ]
    kwargs = dict(
        variable="fapar",
        model="s2biophys-biome-lc-specific",
        biome_lc_name=name,
        clip_min_max=True,
    )
    local = biophys_predict_local(ds, **kwargs)
    remote = biophys_predict_ee(input_imgc=ee.ImageCollection(images), **kwargs)
    values = remote.reduceRegion(
        ee.Reducer.first(), ee.Geometry.Point([8.405, 47.48]), 20
    ).getInfo()
    for band in local:
        np.testing.assert_allclose(
            values[band], local[band].item(), atol=1e-5, rtol=1e-4
        )
