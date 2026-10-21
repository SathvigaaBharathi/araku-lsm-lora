# d:/WLF7/src/stage2_satellite_verification/dem_utils.py
# ----------------------------------------------------------------------
# DEM utilities for Stage 2 – Satellite verification
# ----------------------------------------------------------------------
# This module now contains:
#   • Synthetic DEM generator (with explicit “DATA SOURCE: SYNTHETIC” log)
#   • Real-DEM downloader (uses public AWS Copernicus GLO-30 mirror, no auth)
#   • Bounding-box computation from candidate_events.geojson + 7 km buffer
#   • Slope-gate filter (discard pixels with slope < 15°)
#   • Proper compute_slope_aspect_numpy implementation
#   • Helper to sample DEM/slope at a point
#   • Command-line flag `--demo` to enable synthetic-only mode
# ----------------------------------------------------------------------
import json
import os
import sys
import math
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.merge import merge

# boto3 for public S3 access (unsigned requests)
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# ----------------------------------------------------------------------
# Project utilities – paths & logger
# ----------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import DEM_DIR, get_logger

logger = get_logger("dem_utils")


# ----------------------------------------------------------------------
# 1️⃣ Synthetic DEM generator (unconditional synthetic source)
# ----------------------------------------------------------------------
def generate_dem_geotiff() -> Path:
    """Generate a synthetic DEM covering the study area.

    The DEM is **synthetic**, so we log the data source explicitly.
    """
    logger.info("DATA SOURCE: SYNTHETIC")
    DEM_DIR.mkdir(parents=True, exist_ok=True)
    dem_path = DEM_DIR / "araku_dem_30m_synthetic.tif"

    # Fixed 30 m resolution (~0.00027778°)
    res = 0.00027778
    west, south = 82.5, 18.0
    east, north = 83.2, 18.5
    width = int((east - west) / res)
    height = int((north - south) / res)
    transform = from_origin(west, north, res, res)

    # Terrain model (same as previous implementation)
    x_idx = np.arange(width)
    y_idx = np.arange(height)
    xx_idx, yy_idx = np.meshgrid(x_idx, y_idx)
    lons = west + xx_idx * res
    lats = north - yy_idx * res
    norm_x = (lons - west) / (east - west)
    norm_y = (lats - south) / (north - south)
    regional_tilt = 300.0 * (1.0 - norm_x * 0.4 - (1.0 - norm_y) * 0.3)
    range1 = 280.0 * np.sin(np.pi * (1.4 * norm_x + 1.8 * norm_y))
    range2 = 200.0 * np.cos(np.pi * (2.4 * norm_x - 1.6 * norm_y))
    range3 = 140.0 * np.sin(2 * np.pi * (0.9 * norm_x + 2.2 * norm_y))
    massif_ananthagiri = 380.0 * np.exp(-(((norm_x - 0.72) ** 2 + (norm_y - 0.48) ** 2) / 0.08))
    massif_galikonda = 420.0 * np.exp(-(((norm_x - 0.38) ** 2 + (norm_y - 0.65) ** 2) / 0.10))
    valley_basin = -200.0 * np.exp(-(((norm_x - 0.52) ** 2 + (norm_y - 0.42) ** 2) / 0.06))
    elevation = (
        550.0
        + regional_tilt
        + range1
        + range2
        + range3
        + massif_ananthagiri
        + massif_galikonda
        + valley_basin
    ).astype(np.float32)

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=elevation.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(elevation, 1)

    logger.info(
        f"Generated synthetic DEM GeoTIFF at {dem_path} ({width}x{height} px, "
        f"range {np.min(elevation):.1f}–{np.max(elevation):.1f} m)"
    )
    return dem_path

# ----------------------------------------------------------------------
# 2️⃣ Compute DEM bounding box from candidate_events.geojson + 7 km buffer
# ----------------------------------------------------------------------
def _load_candidate_events_bbox() -> Tuple[float, float, float, float]:
    """Extract the raw (min_lon, min_lat, max_lon, max_lat) from the GeoJSON.

    Expected location: <repo_root>/data/interim/candidate_events.geojson
    """
    geojson_path = Path(__file__).resolve().parents[2] / "data" / "interim" / "candidate_events.geojson"
    if not geojson_path.is_file():
        raise RuntimeError(f"candidate_events.geojson not found at {geojson_path}")
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        def recurse(c):
            if isinstance(c[0], (float, int)):
                lon, lat = c[:2]
                nonlocal min_lon, min_lat, max_lon, max_lat
                min_lon = min(min_lon, lon)
                min_lat = min(min_lat, lat)
                max_lon = max(max_lon, lon)
                max_lat = max(max_lat, lat)
            else:
                for sub in c:
                    recurse(sub)
        recurse(coords)
    if min_lon == float("inf"):
        raise RuntimeError("No coordinates found in candidate_events.geojson")
    return min_lon, min_lat, max_lon, max_lat


def compute_dem_bbox(buffer_km: float = 7.0) -> Tuple[float, float, float, float]:
    """Return (west, south, east, north) expanded by *buffer_km*.

    Longitude buffer accounts for latitude-dependent scaling.
    """
    min_lon, min_lat, max_lon, max_lat = _load_candidate_events_bbox()
    lat_buf = buffer_km / 111.0  # degrees latitude ~= 111 km
    mean_lat = (min_lat + max_lat) / 2.0
    lon_deg_per_km = 1.0 / (111.0 * math.cos(math.radians(mean_lat)))
    lon_buf = buffer_km * lon_deg_per_km
    west = min_lon - lon_buf
    east = max_lon + lon_buf
    south = min_lat - lat_buf
    north = max_lat + lat_buf
    logger.info(
        f"Computed DEM bbox with {buffer_km} km buffer: west={west:.6f}, south={south:.6f}, "
        f"east={east:.6f}, north={north:.6f}"
    )
    return west, south, east, north

# ----------------------------------------------------------------------
# 3️⃣ Real-DEM download using public AWS Copernicus GLO-30 mirror
# ----------------------------------------------------------------------
def download_real_dem() -> Path:
    """Download and mosaic the Copernicus GLO-30 DEM tiles covering the study area.

    Uses the public s3://copernicus-dem-30m bucket. Tile naming follows the ESA pattern:
    Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM/
    Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM.tif
    """
    # Ensure unsigned requests for public bucket
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    # Ensure the DEM directory exists before any download attempts
    DEM_DIR.mkdir(parents=True, exist_ok=True)

    west, south, east, north = compute_dem_bbox(buffer_km=7.0)

    # Determine integer degree tile indices that intersect the bbox
    lon_start = math.floor(west)
    lon_end = math.ceil(east) - 1
    lat_start = math.floor(south)
    lat_end = math.ceil(north) - 1

    bucket = "copernicus-dem-30m"
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    logger.info("Starting download of Copernicus GLO-30 tiles from public S3 bucket")
    local_files: List[Path] = []
    for lon in range(lon_start, lon_end + 1):
        for lat in range(lat_start, lat_end + 1):
            # Determine hemisphere letters and absolute values
            lat_hem = "N" if lat >= 0 else "S"
            lon_hem = "E" if lon >= 0 else "W"
            lat_abs = abs(lat)
            lon_abs = abs(lon)

            key = (
                f"Copernicus_DSM_COG_10_{lat_hem}{lat_abs:02d}_00_"
                f"{lon_hem}{lon_abs:03d}_00_DEM/"
                f"Copernicus_DSM_COG_10_{lat_hem}{lat_abs:02d}_00_"
                f"{lon_hem}{lon_abs:03d}_00_DEM.tif"
            )
            local_path = DEM_DIR / f"tile_{lat_hem}{lat_abs:02d}_{lon_hem}{lon_abs:03d}.tif"
            try:
                s3_client.download_file(bucket, key, str(local_path))
                file_size = local_path.stat().st_size
                logger.info(
                    f"Downloaded tile {key} to {local_path} (size {file_size} bytes)"
                )
                local_files.append(local_path)
            except Exception as e:
                logger.warning(f"Tile {key} not available or failed to download: {e}")
                continue

    if not local_files:
        raise RuntimeError(
            "No DEM tiles were successfully downloaded for the computed bounding box."
        )

    # Mosaic the tiles into a single array
    datasets = [rasterio.open(p) for p in local_files]
    mosaic, out_transform = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "crs": datasets[0].crs,
        }
    )

    out_path = DEM_DIR / "araku_dem_30m_real.tif"
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)

    logger.info(
        "DATA SOURCE: REAL (Copernicus GLO-30 via public S3) – download and mosaic completed"
    )
    return out_path
    """Download and mosaic the Copernicus GLO-30 DEM tiles covering the study area.

    The data live in the public S3 bucket ``copernicus-dem-30m``. No authentication is required.
    We use boto3 with unsigned requests and the official tile naming convention:
    ``Copernicus_DSM_COG_10_N{lat:02d}_E{lon:03d}_DEM/DEM.tif``.
    The function downloads the required tiles to a temporary directory, mosaics them, and returns the path
    to the resulting GeoTIFF.
    """
    # Ensure unsigned requests for public bucket
    os.environ["AWS_NO_SIGN_REQUEST"] = "YES"

    west, south, east, north = compute_dem_bbox(buffer_km=7.0)

    # Determine integer degree tile indices that intersect the bbox
    lon_start = math.floor(west)
    lon_end = math.ceil(east) - 1
    lat_start = math.floor(south)
    lat_end = math.ceil(north) - 1

    bucket = "copernicus-dem-30m"
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    logger.info("Starting download of Copernicus GLO-30 tiles from public S3 bucket")
    local_files: List[Path] = []
    for lon in range(lon_start, lon_end + 1):
        for lat in range(lat_start, lat_end + 1):
            # Tile key follows the pattern used in the bucket
            key = f"Copernicus_DSM_COG_10_N{lat:02d}_E{lon:03d}_DEM/DEM.tif"
            local_path = DEM_DIR / f"tile_N{lat:02d}_E{lon:03d}.tif"
            try:
                s3_client.download_file(bucket, key, str(local_path))
                logger.info(f"Downloaded tile {key} to {local_path}")
                local_files.append(local_path)
            except Exception as e:
                logger.warning(f"Tile {key} not available or failed to download: {e}")
                continue

    if not local_files:
        raise RuntimeError("No DEM tiles were successfully downloaded for the computed bounding box.")

    # Open downloaded tiles with rasterio and mosaic
    datasets = [rasterio.open(p) for p in local_files]
    mosaic, out_transform = merge(datasets)
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "crs": datasets[0].crs,
    })

    DEM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEM_DIR / "araku_dem_30m_real.tif"
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)

    logger.info("DATA SOURCE: REAL (Copernicus GLO-30 via public S3) – download and mosaic completed")
    return out_path
    



# ----------------------------------------------------------------------
# 4️⃣ Helper: delete synthetic DEM after real DEM is verified
# ----------------------------------------------------------------------
def replace_synthetic_with_real(synthetic_path: Path, real_path: Path) -> None:
    """Delete the synthetic DEM if the real DEM exists and is non-zero.
    """
    if not real_path.is_file() or real_path.stat().st_size == 0:
        raise RuntimeError(f"Real DEM not available at {real_path}")
    try:
        if synthetic_path.is_file():
            synthetic_path.unlink()
            logger.info(f"Deleted synthetic DEM {synthetic_path}")
    except Exception as e:
        logger.warning(f"Could not delete synthetic DEM {synthetic_path}: {e}")

# ----------------------------------------------------------------------
# 5️⃣ Compute slope & aspect (returns degrees)
# ----------------------------------------------------------------------
def compute_slope_aspect_numpy(dem_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Compute smooth slope (degrees) and aspect (degrees) from a DEM raster.

    Args:
        dem_path: Path to a DEM GeoTIFF (EPSG:4326).

    Returns:
        (slope_deg, aspect_deg) arrays matching the DEM shape.
    """
    with rasterio.open(dem_path) as src:
        elevation = src.read(1)
        transform = src.transform
    # Approximate metres per pixel
    res_x = abs(transform.a) * 111_000.0
    res_y = abs(transform.e) * 111_000.0
    dz_dy, dz_dx = np.gradient(elevation, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = (np.degrees(aspect_rad) % 360.0).astype(np.float32)
    return slope_deg, aspect_deg

# ----------------------------------------------------------------------
# 6️⃣ Apply slope gate (discard pixels below 15°)
# ----------------------------------------------------------------------
def apply_slope_gate(dem_path: Path, min_slope_deg: float = 15.0) -> Path:
    """Mask DEM where slope < min_slope_deg and write a new GeoTIFF.
    """
    slope_deg, _ = compute_slope_aspect_numpy(dem_path)
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        meta = src.meta.copy()
    mask = slope_deg >= min_slope_deg
    nodata_val = meta.get("nodata")
    if nodata_val is None:
        nodata_val = -9999
        meta["nodata"] = nodata_val
    masked_dem = np.where(mask, dem, nodata_val).astype(dem.dtype)
    masked_path = dem_path.parent / f"{dem_path.stem}_slopegate_{int(min_slope_deg)}deg.tif"
    with rasterio.open(masked_path, "w", **meta) as dst:
        dst.write(masked_dem, 1)
    logger.info(
        f"Applied slope gate (min {min_slope_deg}°). Saved masked DEM to {masked_path}"
    )
    return masked_path

# ----------------------------------------------------------------------
# 7️⃣ Sample elevation & slope at a lon/lat point
# ----------------------------------------------------------------------
def sample_dem_slope_at_point(dem_path: Path, lat: float, lon: float) -> Tuple[float, float]:
    """Return (elevation, slope) for the given geographic coordinate.
    """
    slope_matrix, _ = compute_slope_aspect_numpy(dem_path)
    with rasterio.open(dem_path) as src:
        row, col = src.index(lon, lat)
        row = min(max(0, row), src.height - 1)
        col = min(max(0, col), src.width - 1)
        elevation = float(src.read(1)[row, col])
        slope = float(slope_matrix[row, col])
    return elevation, slope

# ----------------------------------------------------------------------
# 8️⃣ CLI entry-point – optional demo mode
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEM utilities demo/run")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in synthetic-only demo mode (skip real download)",
    )
    args = parser.parse_args()

    # Always generate a synthetic DEM for quick inspection
    synthetic_path = generate_dem_geotiff()
    dem_path: Path

    if args.demo:
        logger.info("Demo flag set – using synthetic DEM only.")
        dem_path = synthetic_path
    else:
        # Attempt real download – any error will raise and stop execution
        real_path = download_real_dem()
        replace_synthetic_with_real(synthetic_path, real_path)
        dem_path = real_path

    # Apply slope gate and report basic stats
    masked_path = apply_slope_gate(dem_path, min_slope_deg=15.0)
    slope, aspect = compute_slope_aspect_numpy(dem_path)
    print(
        f"DEM ready at {dem_path}\n"
        f"Masked DEM at {masked_path}\n"
        f"Mean slope: {np.mean(slope):.2f}°, Max slope: {np.max(slope):.2f}°"
    )
