"""
gee_biophys: Tools for biophysical variable modeling and prediction with Google Earth Engine (GEE).

This package provides:
- Data preprocessing and Sentinel-2 input preparation (`s2_input`)
- Model prediction and export tools (`s2_predict`, `s2_export`)
- Configuration utilities (`config`)
- Helper functions for GEE asset management (`utils`, `utils_predict`)
- Command-line interface (`cli`)

Typical usage (as a command-line tool):
    gee-biophys run --config example_configs/minimal_example.yaml

Alternative usage (from within Python, e.g. for debugging):
    from gee_biophys.cli import run
    from pathlib import Path
    run(config=Path("example_configs/minimal_example.yaml"))

Author: Felix Specker
License: MIT
"""
