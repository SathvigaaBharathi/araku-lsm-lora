# d:/WLF7/src/stage2_satellite_verification/verify.py
# ----------------------------------------------------------------------
# Stage 2: Multi-Sensor Satellite Scar Verification
# Data source: Microsoft Planetary Computer public STAC API
#   - Sentinel-1 GRD  (sentinel-1-grd collection)
#   - Sentinel-2 L2A  (sentinel-2-l2a collection)
# No long-term credentials required; pystac-client + planetary-computer
# handle short-lived SAS token acquisition at query time.
# ----------------------------------------------------------------------
import json
import math
import datetime
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import Point, mapping
import rasterio
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import INTERIM_DATA_DIR, DEM_DIR, SENTINEL1_DIR, SENTINEL2_DIR, get_logger
from src.stage2_satellite_verification.dem_utils import generate_dem_geotiff, compute_slope_aspect_numpy

logger = get_logger("satellite_verification")

MPC_STAC_URL        = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION       = "sentinel-1-grd"
S2_COLLECTION       = "sentinel-2-l2a"
LANDSAT_COLLECTION  = "landsat-c2-l2"
DATE_WINDOW_DAYS    = 45


# ---------------------------------------------------------------------------
# Lazy import so the rest of the pipeline still runs without the packages
# ---------------------------------------------------------------------------
def _import_pc():
    try:
        import pystac_client
        import planetary_computer as pc
        return pystac_client, pc
    except ImportError as exc:
        raise ImportError(
            "pystac-client and planetary-computer are required for real satellite queries. "
            "Run: pip install pystac-client planetary-computer"
        ) from exc


# ---------------------------------------------------------------------------
# STAC / COG helpers
# ---------------------------------------------------------------------------
def _event_bbox(lat, lon, radius_deg):
    return (lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg)


def _adaptive_search_bbox(lat, lon, window_m=150.0):
    """Adaptive 150m x 150m spatial search window around candidate centroid."""
    half_deg = (window_m / 2.0) / 111000.0
    return (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)


def _parse_date(date_str):
    if not date_str:
        return datetime.date(2018, 7, 1)
    # Check RFC 2822 format (e.g. from RSS: 'Wed, 04 Oct 2023 07:00:00 GMT')
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(str(date_str))
        if dt is not None:
            return dt.date()
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            continue
    try:
        return datetime.date(int(str(date_str)[:4]), 7, 1)
    except Exception:
        return datetime.date(2018, 7, 1)


def _search_stac(catalog, collection, bbox, date_from, date_to, limit=5):
    date_range = f"{date_from.isoformat()}/{date_to.isoformat()}"
    try:
        search = catalog.search(collections=[collection], bbox=bbox,
                                datetime=date_range, max_items=limit)
        return list(search.items())
    except Exception as exc:
        logger.warning(f"STAC search failed for {collection}: {exc}")
        return []


def _read_cog_window(href, bbox_wgs84, target_shape=(64, 64)):
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
        from rasterio.vrt import WarpedVRT
        import rasterio.windows

        west, south, east, north = bbox_wgs84
        with rasterio.open(href) as src:
            if src.crs is not None:
                # Standard projected COG (e.g. Sentinel-2 L2A in UTM)
                bbox_native = transform_bounds(CRS.from_epsg(4326), src.crs, west, south, east, north)
                window = rasterio.windows.from_bounds(*bbox_native, transform=src.transform)
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                if window.width < 1 or window.height < 1:
                    return None
                data = src.read(1, window=window, out_shape=target_shape,
                                resampling=rasterio.enums.Resampling.bilinear)
                return data.astype(np.float32)
            else:
                # GCP-based COG (e.g. Sentinel-1 GRD with Ground Control Points)
                gcps, gcp_crs = src.gcps
                if not gcps:
                    return None
                with WarpedVRT(src, src_crs=gcp_crs, src_transform=None, crs="EPSG:4326") as vrt:
                    window = rasterio.windows.from_bounds(west, south, east, north, transform=vrt.transform)
                    window = window.intersection(rasterio.windows.Window(0, 0, vrt.width, vrt.height))
                    if window.width < 1 or window.height < 1:
                        return None
                    data = vrt.read(1, window=window, out_shape=target_shape,
                                    resampling=rasterio.enums.Resampling.bilinear)
                    return data.astype(np.float32)
    except Exception as exc:
        logger.debug(f"COG read failed: {exc}")
        return None


def _compute_sar_signals(items_before, items_after, bbox_wgs84, pc_module):
    if not items_before or not items_after:
        return None

    def _vv_href(item):
        signed = pc_module.sign(item)
        for key in ("vv", "VV"):
            if key in signed.assets:
                return signed.assets[key].href
        for key, asset in signed.assets.items():
            if "vv" in key.lower():
                return asset.href
        return None

    before_href = _vv_href(items_before[0])
    after_href  = _vv_href(items_after[0])
    if not before_href or not after_href:
        return None

    arr_before = _read_cog_window(before_href, bbox_wgs84)
    arr_after  = _read_cog_window(after_href,  bbox_wgs84)
    if arr_before is None or arr_after is None:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db_before = 10.0 * np.log10(np.clip(arr_before, 1e-9, None))
        db_after  = 10.0 * np.log10(np.clip(arr_after,  1e-9, None))
        diff_db   = np.abs(db_after - db_before)
        # Localized backscatter change in the 150m x 150m spatial search window
        shift     = float(np.nanpercentile(diff_db, 90))

    logger.info(f"    DATA SOURCE: REAL (Sentinel-1 GRD / Planetary Computer) -- VV shift: {shift:.2f} dB")
    return {"sar_shift_db": round(shift, 2), "sar_shift_flag": shift >= 3.0}


def _compute_s2_signals(items_before, items_after, bbox_wgs84, pc_module):
    if not items_before or not items_after:
        return None

    def _sort_cloud(item):
        return item.properties.get("eo:cloud_cover", 100)

    item_before = min(items_before, key=_sort_cloud)
    item_after  = min(items_after,  key=_sort_cloud)

    def _ndvi(item):
        signed = pc_module.sign(item)
        red_href = signed.assets.get("B04", signed.assets.get("red"))
        nir_href = signed.assets.get("B08", signed.assets.get("nir"))
        if not red_href or not nir_href:
            return None
        red = _read_cog_window(red_href.href, bbox_wgs84)
        nir = _read_cog_window(nir_href.href, bbox_wgs84)
        if red is None or nir is None:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (nir - red) / (nir + red + 1e-9)

    ndvi_before = _ndvi(item_before)
    ndvi_after  = _ndvi(item_after)
    if ndvi_before is None or ndvi_after is None:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff = np.abs(ndvi_before - ndvi_after)
        delta = float(np.nanmax(diff))

    logger.info(f"    DATA SOURCE: REAL (Sentinel-2 L2A / Planetary Computer) -- delta-NDVI: {delta:.3f}")
    return {"delta_ndvi": round(delta, 3), "delta_ndvi_flag": delta > 0.20}


def _compute_landsat_signals(items_before, items_after, bbox_wgs84, pc_module):
    if not items_before or not items_after:
        return None

    def _sort_cloud(item):
        return item.properties.get("eo:cloud_cover", 100)

    item_before = min(items_before, key=_sort_cloud)
    item_after  = min(items_after,  key=_sort_cloud)

    def _read_sr(href, bbox):
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds
        import rasterio.windows
        west, south, east, north = bbox
        with rasterio.open(href) as src:
            b_nat = transform_bounds(CRS.from_epsg(4326), src.crs, west, south, east, north)
            win = rasterio.windows.from_bounds(*b_nat, transform=src.transform)
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            if win.width < 1 or win.height < 1:
                return None
            arr = src.read(1, window=win).astype(np.float32)
            # Landsat Collection 2 Level-2 Surface Reflectance scaling: DN * 0.0000275 - 0.2
            return np.clip(arr * 0.0000275 - 0.2, 0.0, 1.0)

    try:
        signed_before = pc_module.sign(item_before)
        signed_after  = pc_module.sign(item_after)

        red_href_b = signed_before.assets.get("red")
        nir_href_b = signed_before.assets.get("nir08", signed_before.assets.get("nir"))
        red_href_a = signed_after.assets.get("red")
        nir_href_a = signed_after.assets.get("nir08", signed_after.assets.get("nir"))

        if not all([red_href_b, nir_href_b, red_href_a, nir_href_a]):
            return None

        red_b = _read_sr(red_href_b.href, bbox_wgs84)
        nir_b = _read_sr(nir_href_b.href, bbox_wgs84)
        red_a = _read_sr(red_href_a.href, bbox_wgs84)
        nir_a = _read_sr(nir_href_a.href, bbox_wgs84)

        if any(x is None for x in [red_b, nir_b, red_a, nir_a]):
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ndvi_b = (nir_b - red_b) / (nir_b + red_b + 1e-9)
            ndvi_a = (nir_a - red_a) / (nir_a + red_a + 1e-9)
            diff = np.abs(ndvi_b - ndvi_a)
            delta = float(np.nanmax(diff))

        logger.info(f"    DATA SOURCE: REAL (Landsat Level-2 / Planetary Computer) -- delta-NDVI: {delta:.3f}")
        return {"delta_ndvi": round(delta, 3), "delta_ndvi_flag": delta > 0.20}
    except Exception as exc:
        logger.warning(f"Landsat signal computation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-event signal fetch (Planetary Computer + simulation fallback)
# ---------------------------------------------------------------------------
def fetch_satellite_signals(candidate_row, slope_deg):
    """
    Query Planetary Computer for Sentinel-1 GRD, Sentinel-2 L2A, and Landsat scenes.
    Uses adaptive 150m x 150m spatial search window around news candidate centroids.
    Falls back to Landsat-7 / Landsat-8 for pre-2015 events or when Sentinel-2 is unavailable.
    """
    lat = float(candidate_row["center_latitude"])
    lon = float(candidate_row["center_longitude"])
    # 150m x 150m adaptive spatial search window
    bbox = _adaptive_search_bbox(lat, lon, window_m=150.0)

    event_date = _parse_date(str(candidate_row.get("date", "2018-07-01")))
    date_window = 90 if event_date.year < 2015 else DATE_WINDOW_DAYS
    date_from  = event_date - datetime.timedelta(days=date_window)
    date_to    = event_date + datetime.timedelta(days=date_window)
    pre_cutoff = event_date - datetime.timedelta(days=2)
    post_cutoff= event_date + datetime.timedelta(days=2)

    fb_insar = 0.0

    sar_result = None
    optical_result = None
    optical_source = "synthetic"

    try:
        pystac_client, pc = _import_pc()
        catalog = pystac_client.Client.open(MPC_STAC_URL, modifier=pc.sign_inplace)

        # Sentinel-1 SAR (available after late 2014)
        if event_date >= datetime.date(2014, 10, 1):
            s1_before = _search_stac(catalog, S1_COLLECTION, bbox, date_from,  pre_cutoff,  limit=3)
            s1_after  = _search_stac(catalog, S1_COLLECTION, bbox, post_cutoff, date_to,    limit=3)
            sar_result = _compute_sar_signals(s1_before, s1_after, bbox, pc)

        # Optical query: pre-2015 uses Landsat-7 / Landsat-5 fallback directly;
        # 2015+ queries Sentinel-2 L2A, falling back to Landsat if no S2 scenes found.
        if event_date < datetime.date(2015, 6, 23):
            ls_before = _search_stac(catalog, LANDSAT_COLLECTION, bbox, date_from, pre_cutoff, limit=10)
            ls_after  = _search_stac(catalog, LANDSAT_COLLECTION, bbox, post_cutoff, date_to,  limit=10)
            optical_result = _compute_landsat_signals(ls_before, ls_after, bbox, pc)
            if optical_result is not None:
                optical_source = "real_landsat"
        else:
            s2_before = _search_stac(catalog, S2_COLLECTION, bbox, date_from,  pre_cutoff,  limit=5)
            s2_after  = _search_stac(catalog, S2_COLLECTION, bbox, post_cutoff, date_to,    limit=5)
            optical_result = _compute_s2_signals(s2_before, s2_after, bbox, pc)
            if optical_result is not None:
                optical_source = "real"
            else:
                ls_before = _search_stac(catalog, LANDSAT_COLLECTION, bbox, date_from, pre_cutoff, limit=10)
                ls_after  = _search_stac(catalog, LANDSAT_COLLECTION, bbox, post_cutoff, date_to,  limit=10)
                optical_result = _compute_landsat_signals(ls_before, ls_after, bbox, pc)
                if optical_result is not None:
                    optical_source = "real_landsat"

    except Exception as exc:
        logger.warning(
            f"Planetary Computer query failed for {candidate_row['candidate_id']}: {exc} "
            "-- using neutral 0.0 fallback for affected signals"
        )

    sar_db     = sar_result["sar_shift_db"]     if sar_result     else 0.0
    sar_flag   = sar_result["sar_shift_flag"]   if sar_result     else False
    dndvi      = optical_result["delta_ndvi"]      if optical_result else 0.0
    dndvi_flag = optical_result["delta_ndvi_flag"] if optical_result else False

    if sar_result is None:
        logger.info(f"    DATA SOURCE: NONE/MISSING (Sentinel-1) for {candidate_row['candidate_id']} (score contribution: 0.0)")
    if optical_result is None:
        logger.info(f"    DATA SOURCE: NONE/MISSING (Optical S2/Landsat) for {candidate_row['candidate_id']} (score contribution: 0.0)")

    return {
        "sar_shift_db":          sar_db,
        "insar_coherence_loss":  fb_insar,    # neutral 0.0 for uncalculated InSAR
        "delta_ndvi":            dndvi,
        "sar_shift_flag":        sar_flag,
        "delta_ndvi_flag":       dndvi_flag,
        "insar_coherence_flag":  False,
        "sar_source":            "real" if sar_result is not None else "synthetic",
        "s2_source":             optical_source,
    }


# ---------------------------------------------------------------------------
# DEM sampling
# ---------------------------------------------------------------------------
def sample_max_slope_in_bbox(dem_path, lat, lon, radius_deg=0.000676):
    """Sample elevation and maximum slope within adaptive search window."""
    slope_matrix, _ = compute_slope_aspect_numpy(dem_path)
    with rasterio.open(dem_path) as src:
        r_center, c_center = src.index(lon, lat)
        r_delta = max(1, int(radius_deg / 0.00027778))
        c_delta = max(1, int(radius_deg / 0.00027778))
        r_min = max(0, r_center - r_delta)
        r_max = min(src.height - 1, r_center + r_delta)
        c_min = max(0, c_center - c_delta)
        c_max = min(src.width - 1, c_center + c_delta)
        elevation = float(src.read(1)[r_center, c_center])
        sub_slope = slope_matrix[r_min:r_max+1, c_min:c_max+1]
        max_slope = float(np.max(sub_slope)) if sub_slope.size > 0 else float(slope_matrix[r_center, c_center])
    return elevation, max_slope


# ---------------------------------------------------------------------------
# Scoring: Score = 0.40 * Delta_NDVI + 0.40 * SAR_Ratio + 0.20 * InSAR_Coherence
# ---------------------------------------------------------------------------
def compute_verification_score(sensor_data, weights=(0.40, 0.40, 0.20)):
    w_ndvi, w_sar, w_insar = weights
    norm_ndvi  = min(1.0, sensor_data["delta_ndvi"] / 0.50)
    norm_sar   = min(1.0, sensor_data["sar_shift_db"] / 5.0)
    norm_insar = sensor_data["insar_coherence_loss"]
    return round(float(w_ndvi * norm_ndvi + w_sar * norm_sar + w_insar * norm_insar), 4)


def optimize_fusion_weights_grid_search(candidates_list):
    """Grid-search to calibrate fusion weights by maximising score variance.
    NOTE: kept for experimental use only. The pipeline now uses fixed weights.
    """
    best_weights = (0.40, 0.40, 0.20)
    best_score   = -1.0
    weight_grid  = []
    for w1 in np.linspace(0.3, 0.7, 5):
        for w2 in np.linspace(0.1, 0.4, 4):
            w3 = 1.0 - w1 - w2
            if w3 >= 0.1:
                weight_grid.append((round(w1, 2), round(w2, 2), round(w3, 2)))
    for w in weight_grid:
        scores = [compute_verification_score(c["sensor_data"], weights=w)
                  for c in candidates_list if "sensor_data" in c]
        if scores:
            score_var = float(np.var(scores))
            if score_var > best_score:
                best_score   = score_var
                best_weights = w
    logger.info(f"Calibrated Multi-Sensor Fusion Weights (Delta_NDVI, SAR_Ratio, InSAR): {best_weights}")
    return best_weights


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------
def verify_candidate_events(slope_threshold_deg=0.0, score_threshold=0.50):
    """Execute Stage 2 satellite verification pipeline."""
    candidate_path = INTERIM_DATA_DIR / "candidate_events.geojson"
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate events file not found at {candidate_path}. Run Stage 1 first."
        )

    # Prefer real DEM if already downloaded, else generate synthetic
    real_dem = DEM_DIR / "araku_dem_30m_real.tif"
    dem_path = real_dem if real_dem.is_file() else generate_dem_geotiff()
    logger.info(f"Using DEM: {dem_path}")

    candidates_gdf = gpd.read_file(candidate_path)
    logger.info(f"Loaded {len(candidates_gdf)} candidate events for satellite verification.")

    verified_features = []
    candidates_data   = []

    for idx, row in candidates_gdf.iterrows():
        lat = float(row["center_latitude"])
        lon = float(row["center_longitude"])
        # Adaptive 150m x 150m spatial search window for slope
        radius_deg = (150.0 / 2.0) / 111_000.0

        elevation, slope_deg = sample_max_slope_in_bbox(dem_path, lat, lon, radius_deg=radius_deg)

        if slope_deg < slope_threshold_deg:
            logger.info(
                f"Candidate {row['candidate_id']} REJECTED: "
                f"Slope ({slope_deg:.1f} deg) < {slope_threshold_deg} deg gate."
            )
            continue

        sensor_data = fetch_satellite_signals(row, slope_deg)
        candidates_data.append({
            "row": row,
            "elevation": elevation,
            "slope_deg": slope_deg,
            "sensor_data": sensor_data,
        })

    # Fixed fusion weights: Score = 0.40 * Delta_NDVI + 0.40 * SAR_Ratio + 0.20 * InSAR_Coherence
    FUSION_WEIGHTS = (0.40, 0.40, 0.20)
    logger.info(f"Using fixed fusion weights: Delta_NDVI={FUSION_WEIGHTS[0]}, SAR_Ratio={FUSION_WEIGHTS[1]}, InSAR={FUSION_WEIGHTS[2]}")

    verified_count = 0
    unverified_count = 0
    for cand_idx, cand in enumerate(candidates_data, start=1):
        row         = cand["row"]
        elevation   = cand["elevation"]
        slope_deg   = cand["slope_deg"]
        sensor_data = cand["sensor_data"]

        v_score    = compute_verification_score(sensor_data, weights=FUSION_WEIGHTS)

        # Cutoff: 0.20 for micro-toponym corridor matches (Class A/B), 0.50 for Class C
        p_class = str(row.get("precision_class", "C")).upper().strip()
        cutoff = 0.20 if p_class in ("A", "B") else score_threshold
        is_score_pass = bool(v_score >= cutoff)

        sar_src = sensor_data.get("sar_source", "synthetic")
        s2_src  = sensor_data.get("s2_source", "synthetic")
        has_real_sensor = (sar_src == "real" or "real" in s2_src)

        if not has_real_sensor:
            is_verified = False
            unverified_count += 1
            status = "UNVERIFIED_INSUFFICIENT_DATA"
            ver_id = f"UNV_{unverified_count:03d}"
        elif is_score_pass:
            verified_count += 1
            status = "VERIFIED_LANDSLIDE"
            ver_id = f"VER_{verified_count:03d}"
        else:
            unverified_count += 1
            status = "UNVERIFIED_LOW_SCORE"
            ver_id = f"UNV_{unverified_count:03d}"

        lat = float(row["center_latitude"])
        lon = float(row["center_longitude"])

        properties = {
            "verified_id":          ver_id,
            "candidate_id":         row["candidate_id"],
            "date":                 row["date"],
            "matched_toponym":      row["matched_toponym"],
            "precision_class":      row["precision_class"],
            "elevation_m":          round(elevation, 1),
            "slope_deg":            round(slope_deg, 2),
            "sar_shift_db":         sensor_data["sar_shift_db"],
            "insar_coherence_loss": sensor_data["insar_coherence_loss"],
            "delta_ndvi":           sensor_data["delta_ndvi"],
            "sar_flag":             sensor_data["sar_shift_flag"],
            "delta_ndvi_flag":      sensor_data["delta_ndvi_flag"],
            "sar_source":           sar_src,
            "s2_source":            s2_src,
            "verification_score":   v_score,
            "weights_used":         list(FUSION_WEIGHTS),
            "verification_cutoff":  cutoff,
            "verification_status":  status,
            "snippet":              row["snippet"],
        }

        verified_features.append({
            "type": "Feature",
            "geometry": mapping(Point(lon, lat)),
            "properties": properties,
        })

    verified_rows = [
        f for f in verified_features
        if f["properties"]["verification_status"] == "VERIFIED_LANDSLIDE"
    ]

    if verified_rows:
        verified_gdf = gpd.GeoDataFrame.from_features(verified_rows, crs="EPSG:4326")
    else:
        verified_gdf = gpd.GeoDataFrame(columns=["verified_id", "geometry"], crs="EPSG:4326")

    output_path = INTERIM_DATA_DIR / "verified_events.geojson"
    verified_gdf.to_file(output_path, driver="GeoJSON")
    logger.info(
        f"Stage 2 Complete: Verified {len(verified_gdf)} landslide events "
        f"out of {len(candidates_gdf)} candidates."
    )
    logger.info(f"Saved verified events GeoJSON to {output_path}")

    logger.info("=" * 95)
    logger.info("STAGE 2 SATELLITE VERIFICATION SUMMARY TABLE:")
    logger.info(f"{'Candidate ID':<14} | {'Date':<12} | {'Class':<5} | {'SAR Src':<9} | {'Opt Src':<13} | {'Score':<7} | {'Cutoff':<6} | {'Status'}")
    logger.info("-" * 95)
    for f in verified_features:
        p = f["properties"]
        d_str = str(p.get("date", ""))[:12]
        logger.info(f"{p['candidate_id']:<14} | {d_str:<12} | {p['precision_class']:<5} | {p['sar_source']:<9} | {p['s2_source']:<13} | {p['verification_score']:<7.4f} | {p.get('verification_cutoff', 0.50):<6.2f} | {p['verification_status']}")
    logger.info("=" * 95)

    return output_path


if __name__ == "__main__":
    verify_candidate_events()
