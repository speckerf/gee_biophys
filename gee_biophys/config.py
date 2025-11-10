import json
import re
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from typing import Annotated, Iterator, List, Literal, Optional, Tuple, Union

import ee
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)


# ----------------- Spatial -----------------
class Spatial(BaseModel):
    type: Literal["bbox", "geojson"]
    bbox: Optional[List[float]] = None  # [minx, miny, maxx, maxy]
    geojson_path: Optional[str] = None
    region_name: Optional[str] = (
        None  # optional name for the region (no spaces/underscores)
    )
    geojson_clip: bool = Field(
        default=False,
        description="If true and type is 'geojson', clip to geometry bounds.",
    )

    # Enforce that *no other keys* are accepted
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_inputs(self):
        has_bbox = self.bbox is not None
        has_geo = self.geojson_path is not None
        has_region_name = self.region_name is not None

        if has_region_name:
            # assert not spaces and underscores
            if re.search(r"[ _]", self.region_name):
                raise ValueError(
                    "spatial.region_name must not contain spaces or underscores."
                )

        # exactly one must be provided
        if has_bbox == has_geo:
            raise ValueError(
                "Specify exactly one of 'bbox' or 'geojson_path', not both."
            )

        if self.type == "bbox":
            if not has_bbox:
                raise ValueError("When type='bbox', 'bbox' is required.")
            if len(self.bbox) != 4:
                raise ValueError(
                    "bbox must contain four numbers: [minx, miny, maxx, maxy]."
                )
            minx, miny, maxx, maxy = self.bbox
            if not (minx < maxx and miny < maxy):
                raise ValueError("bbox must satisfy minx < maxx and miny < maxy.")
            # optional lon/lat sanity
            if not (
                -180 <= minx <= 180
                and -180 <= maxx <= 180
                and -90 <= miny <= 90
                and -90 <= maxy <= 90
            ):
                raise ValueError("bbox coordinates must be within lon/lat bounds.")
        elif self.type == "geojson":
            if not has_geo:
                raise ValueError("When type='geojson', 'geojson_path' is required.")
            p = Path(self.geojson_path)
            if not p.exists():
                raise ValueError(f"geojson_path does not exist: {p}")
            if p.suffix.lower() not in {".geojson", ".json"}:
                raise ValueError("geojson_path must end with .geojson or .json.")
            # quick JSON check
            json.loads(p.read_text(encoding="utf-8"))

        return self

    @cached_property
    def ee_geometry(self) -> ee.Geometry:
        """Return an ee.Geometry derived from this spatial definition."""
        if self.type == "bbox":
            return ee.Geometry.BBox(*self.bbox)
        elif self.type == "geojson":
            with open(self.geojson_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            t = obj.get("type", "").lower()

            # type needs to be: polygon, multipolygon, feature, featurecollection (everything else is rejected)
            if t not in (
                "polygon",
                "multipolygon",
                "feature",
                "featurecollection",
            ):
                raise ValueError(
                    f"GeoJSON 'type' must be 'Polygon', 'MultiPolygon', 'Feature', or 'FeatureCollection', got '{t}'."
                )

            if t in ("polygon", "multipolygon"):
                # Raw Geometry dict
                return ee.Geometry(obj)

            if t == "feature":
                # Single Feature -> use its geometry
                geom = obj.get("geometry")
                if not geom:
                    raise ValueError("GeoJSON Feature has no 'geometry'.")
                return ee.Geometry(geom)

            if t == "featurecollection":
                # Full FeatureCollection -> dissolve to one geometry
                fc = ee.FeatureCollection(obj)
                if fc.size().getInfo() == 1:
                    # single feature -> use its geometry
                    first = ee.Feature(fc.first())
                    return first.geometry()
                else:
                    logger.warning(
                        "GeoJSON FeatureCollection has multiple features; dissolving to single geometry."
                    )
                    return fc.geometry()

            raise ValueError(
                f"Unsupported or missing GeoJSON 'type': {obj.get('type')}"
            )

        else:
            raise ValueError(f"Unsupported spatial type: {self.type}")


_MM_DD = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def _safe_ymd(year: int, month: int, day: int) -> datetime:
    """Return a valid UTC datetime, replacing Feb 29 -> Feb 28 if needed."""
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        if month == 2 and day == 29:
            return datetime(year, 2, 28, tzinfo=timezone.utc)
        raise


# ---------- Cadences ----------
class FixedCadence(BaseModel):
    type: Literal["fixed"]
    interval: Union[
        int,  # N-day step
        Literal[
            "weekly",
            "biweekly",
            "monthly",
            "bimonthly",
            "quarterly",
            "yearly",
            "annual",
        ],
    ]

    model_config = ConfigDict(extra="forbid")

    @field_validator("interval", mode="before")
    def _normalize_interval(cls, v):
        if isinstance(v, int):
            if v <= 0:
                raise ValueError("interval (int days) must be > 0.")
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            return "yearly" if s == "annual" else s
        raise ValueError("interval must be int days or a valid period string.")


class SeasonsCadence(BaseModel):
    type: Literal["seasons"]
    start: str  # "MM-DD"
    end: str  # "MM-DD"
    model_config = ConfigDict(extra="forbid")

    @field_validator("start", "end")
    def _mmdd(cls, v):
        if not _MM_DD.match(v):
            raise ValueError("Use zero-padded 'MM-DD'.")
        return v


Cadence = Annotated[FixedCadence | SeasonsCadence, Field(discriminator="type")]


# ----------------- Temporal -----------------
class Temporal(BaseModel):
    # store as aware UTC datetimes; accept ISO strings in JSON
    start: datetime
    end: datetime
    cadence: Cadence

    # Enforce that *no other keys* are accepted
    model_config = ConfigDict(extra="forbid")

    @field_validator("start", "end", mode="before")
    def _parse_iso(cls, v):
        if not isinstance(v, str):
            raise ValueError("Dates must be strings in ISO 8601 format.")
        s = v.strip()
        try:
            # support 'YYYY-MM-DD' and 'YYYY-MM-DDTHH:MM:SS[.fff][Z|+00:00]'
            if s.endswith("Z"):
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt.utcoffset() != timedelta(0):
                raise ValueError("Datetime must be UTC (use 'Z' or +00:00).")
            return dt
        except Exception:
            raise ValueError(
                f"Invalid date string '{v}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ'."
            )

    # Public API: yield (start_dt, end_dt)
    def iter_date_ranges(self) -> Iterator[Tuple[datetime, datetime]]:
        if isinstance(self.cadence, FixedCadence):
            yield from self._iter_fixed(self.cadence)
        else:
            yield from self._iter_seasons(self.cadence)

    # ---- Fixed intervals ----
    def _iter_fixed(self, c: FixedCadence) -> Iterator[Tuple[datetime, datetime]]:
        """
        Yield fixed-length intervals clipped to [self.start, self.end).
        Raise a warning if the final interval is truncated at the temporal end.
        """
        if isinstance(c.interval, int):
            step = timedelta(days=c.interval)
            t0 = self.start
            while t0 < self.end:
                t1 = min(t0 + step, self.end)
                if t1 != t0 + step:
                    logger.warning(
                        f"Final interval from {t0.date()} to {t1.date()} is truncated "
                        f"to fit within temporal.end {self.end.date()}."
                    )
                if t1 > t0:
                    yield (t0, t1)
                t0 = t1
            return

        name = c.interval
        if name in {"weekly", "biweekly"}:
            step = timedelta(days=7 if name == "weekly" else 14)
            t0 = self.start
            while t0 < self.end:
                t1 = min(t0 + step, self.end)
                if t1 != t0 + step:
                    logger.warning(
                        f"Final interval '{name}' from {t0.date()} to {t1.date()} is truncated "
                        f"to fit within temporal.end {self.end.date()}."
                    )
                if t1 > t0:
                    yield (t0, t1)
                t0 = t1
            return

        months = {"monthly": 1, "bimonthly": 2, "quarterly": 3, "yearly": 12}[name]
        t = self.start
        while t < self.end:
            y, m = t.year, t.month
            new_m = m + months
            y2 = y + (new_m - 1) // 12
            m2 = (new_m - 1) % 12 + 1
            end_block = datetime(y2, m2, 1, tzinfo=timezone.utc)
            t1 = min(end_block, self.end)
            if t1 != end_block:
                logger.warning(
                    f"Final interval '{name}' from {t.date()} to {t1.date()} is truncated "
                    f"to fit within temporal.end {self.end.date()}."
                )
            if t1 > t:
                yield (t, t1)
            t = t1

    # ---- Seasonal intervals (cross-year allowed) ----
    def _iter_seasons(self, c: SeasonsCadence) -> Iterator[Tuple[datetime, datetime]]:
        """
        Yield exactly ONE tuple per season-year:
        - same-year:  YYYY-MM-DD -> YYYY-MM-DD
        - cross-year: YYYY-MM-DD -> (YYYY+1)-MM-DD
        All windows are clipped to [self.start, self.end).
        """

        m1, d1 = map(int, c.start.split("-"))
        m2, d2 = map(int, c.end.split("-"))

        y = self.start.year - 1  # include prior year to catch wrap crossing self.start
        last_y = self.end.year

        while y <= last_y:
            a = _safe_ymd(y, m1, d1)  # nominal start in year y
            b_same = _safe_ymd(y, m2, d2)  # same-year end if not wrapping
            win_start = a
            win_end = b_same if a < b_same else _safe_ymd(y + 1, m2, d2)

            # clip to global [start, end)
            start_clipped = max(win_start, self.start)
            end_clipped = min(win_end, self.end)

            # warn if clipped at either boundary
            if start_clipped < end_clipped and (
                start_clipped > win_start or end_clipped < win_end
            ):
                logger.warning(
                    f"Season ({c.start} until {c.end}) around {y} is truncated by "
                    f"[{self.start.date()} — {self.end.date()}]."
                )

            if start_clipped < end_clipped:
                yield (start_clipped, end_clipped)

            y += 1

    @model_validator(mode="after")
    def _check_order_and_warn_truncation(self):
        if self.start >= self.end:
            raise ValueError("temporal.start must be strictly before temporal.end.")

        # Warn if the (single) seasonal window is clipped at global boundaries
        if isinstance(self.cadence, SeasonsCadence):
            m1, d1 = map(int, self.cadence.start.split("-"))
            m2, d2 = map(int, self.cadence.end.split("-"))

            for year in range(self.start.year - 1, self.end.year + 1):
                a = _safe_ymd(year, m1, d1)  # nominal season start
                b_same = _safe_ymd(year, m2, d2)  # same-year end, if not wrapping
                win_start = a
                win_end = b_same if a < b_same else _safe_ymd(year + 1, m2, d2)

                # clip to [start, end)
                clipped_start = max(win_start, self.start)
                clipped_end = min(win_end, self.end)

                # if it overlaps and clipping happened -> warn once
                if clipped_start < clipped_end and (
                    clipped_start > win_start or clipped_end < win_end
                ):
                    logger.warning(
                        f"Season ({self.cadence.start} until {self.cadence.end}) near {year} "
                        f"is truncated by [{self.start.date()} — {self.end.date()}]. "
                        "First or last seasonal composite may be partial."
                    )
                    break
        return self


# ----------------- Export options -----------------
class ExportOpts(BaseModel):
    # only allow these three
    destination: Literal["asset", "drive", "gcs"]

    # destination-specific
    collection_path: Optional[str] = None  # required if asset
    folder: Optional[str] = None  # required if drive; optional prefix if gcs
    bucket: Optional[str] = None  # required if gcs

    # common
    project_id: Optional[str] = None  # GEE project ID for exports
    filename_prefix: str = "biophys"
    crs: Optional[str] = None  # e.g. "EPSG:4326"
    scale: Optional[PositiveInt] = None  # meters
    max_pixels: int = Field(default=100_000_000_000, ge=1)

    # Enforce that *no other keys* are accepted
    model_config = ConfigDict(extra="forbid")

    @field_validator("crs")
    def check_epsg(cls, v):
        if v is None:
            return v
        if not re.fullmatch(r"EPSG:\d{4,6}", v):
            raise ValueError("crs must look like 'EPSG:4326'.")
        return v

    @model_validator(mode="after")
    def check_destination_requirements(self):
        if self.destination == "asset":
            if not self.collection_path:
                raise ValueError(
                    "When destination='asset', 'collection_path' must be provided."
                )
        elif self.destination == "drive":
            if not self.folder:
                raise ValueError("When destination='drive', 'folder' must be provided.")
        elif self.destination == "gcs":
            if not self.bucket:
                raise ValueError("When destination='gcs', 'bucket' must be provided.")
            # GCS object prefix is optional; if present, must be str
            if self.folder is not None and not isinstance(self.folder, str):
                raise ValueError(
                    "When destination='gcs', 'folder' must be a string if provided."
                )
        return self


# ----------------- Variables -----------------
class Variables(BaseModel):
    model: Literal["s2biophys", "sl2p"] = "s2biophys"
    variable: Literal["laie", "fapar", "fcover"] = "laie"
    bands: List[Literal["mean", "stdDev", "count"]] = ["mean", "stdDev", "count"]

    # Enforce that *no other keys* are accepted
    model_config = ConfigDict(extra="forbid")

    @field_validator("model", "variable", mode="before")
    def lowercase_enums(cls, v):
        return v.lower() if isinstance(v, str) else v


# ----------------- Options -----------------
class Options(BaseModel):
    max_cloud_cover: int = Field(default=70, ge=0, le=100)
    csplus_band: Literal["cs", "cs_cdf"] = "cs"
    cs_plus_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # Enforce that *no other keys* are accepted
    model_config = ConfigDict(extra="forbid")


# ----------------- Top-level -----------------
class ConfigParams(BaseModel):
    spatial: Spatial
    temporal: Temporal
    variables: Variables
    export: ExportOpts
    options: Options
    version: str = Field(default="v02", pattern=r"^v\d{2}$")
