---

# gee-biophys

Custom time-series export of LAIe / FAPAR / FCOVER from Sentinel-2 with a YAML configuration.
Provides both a **Python script** (run from source) and a **CLI** (`gee-biophys`).

---

## Prerequisites

* Python ≥ 3.12
* [uv](https://github.com/astral-sh/uv) (recommended) or pip
* Google Earth Engine access (earthengine-api already authenticated)
  
---

## Option A — Run from source (clone & run)

1. **Clone the repo**

```bash
git clone https://github.com/<org-or-user>/gee_biophys.git
cd gee_biophys
```

2. **Create the environment from `pyproject.toml` and run the script**

Either rely on [uv](https://docs.astral.sh/uv/) to install virtual environment: 

```bash
uv run python run_export.py --config path/to/your_config.yaml
```
Or install with pip + requirements.txt:
```bash
pip install -r requirements.txt
```
And run:
```bash
python run_export.py --config path/to/your_config.yaml
```

* `--config` (or `-c`) is **required** and must point to a YAML file. Check the example configuration files in example_configs/

---

## Option B — Install the CLI from Source

1. **Install**

```bash
git clone https://github.com/<org-or-user>/gee_biophys.git
pip install uv
uv pip install -e .
```

2. **Run**

```bash
gee-biophys --config path/to/your_config.yaml
```

---

## YAML configuration (example)
- see folder: example_configs/
```yaml
# ============     GEE-Biophys  ================
# Temporal On-Demand Exports — Reference Config
# ==============================================

spatial:
  # Choose exactly ONE input: 'bbox' OR 'geojson'
  type: bbox                      # options: bbox, geojson
  bbox: [7.1, 46.1, 7.2, 46.2]    # [minLon(minx), minLat(miny), maxLon(maxy)]
  # geojson_path: "path/to/area.geojson"   # (use if type == geojson; .geojson/.json)
  region_name: my-region          # optional; no spaces or underscores
  # geojson_clip: false             # if type == geojson: clip to geometry bounds when true

temporal:
  # Dates can be given as simple YYYY-MM-DD; they are parsed and normalized to UTC internally.
  start: "2020-01-01"
  end:   "2023-01-01"

  # Cadence discriminator (exactly one of 'fixed' or 'seasons').
  cadence:
    type: fixed                   # options: fixed, seasons
    # For 'fixed':
    # - Use an int for N-day steps (e.g., 16), OR one of:
    #   weekly | biweekly | monthly | bimonthly | quarterly | yearly (alias: annual)
    interval: quarterly

    # If you instead want seasonal windows (cross-year allowed), use:
    # type: seasons
    # start: "11-15"              # MM-DD (zero-padded)
    # end:   "03-15"              # MM-DD (zero-padded)

variables:
  model: s2biophys                # options: s2biophys, sl2p
  variable: laie                  # options: laie, fapar, fcover
  bands: [mean, stdDev, count]    # allowed: mean, stdDev, count / bandnames will be {variable}_{band}

export:
  # Destination discriminator & required fields
  destination: asset              # options: asset, drive, gcs

  # If destination == 'asset' (REQUIRED):
  collection_path: "projects/ee-yourproject/assets/custom-exports/test"

  # If destination == 'drive' (REQUIRED):
  # folder: "biophys_exports"

  # If destination == 'gcs'  (REQUIRED):
  # bucket: "your-gcs-bucket-name"
  # folder: "biophys_exports"     # optional object prefix for GCS

  # Common options
  project_id: "ee-yourproject"    # optional GEE project id for exports
  filename_prefix: "biophys"      # base name for export files
  crs: "EPSG:4326"                # e.g., use UTM for regional accuracy (EPSG:32630, 32723, etc.)
  scale: 100                      # meters per pixel (PositiveInt)
  max_pixels: 100_000_000_000     # e.g. default 1e11

options:
  max_cloud_cover: 70             # 0..100 (uses S2 metadata if applicable)
  csplus_band: cs_cdf             # options: cs, cs_cdf
  cs_plus_threshold: 0.65         # 0.0..1.0 (higher = more conservative)

version: "v02"                    # corresponds to current s2biophys version, allows future changes
```

---

## Troubleshooting

* **“Missing option '--config' / '-c'.”**
  Provide the YAML path:

  ```bash
  gee-biophys -c example_configs/minimal_example.yaml
  ```

* **`ModuleNotFoundError` for local modules**
  Ensure you installed the package (`uv pip install -e .`) or use `uv run python run_export.py ...` from repo root.

* **Earth Engine auth errors**

  * Run `earthengine authenticate` once 

* **GCS export issues**
  Check bucket name, folder, and write permissions for your user/SA.

---


## Citation

TODO
