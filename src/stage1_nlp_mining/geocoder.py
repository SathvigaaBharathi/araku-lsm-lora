import json
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, box, mapping
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import GAZETTEER_DIR, INTERIM_DATA_DIR, get_logger
from src.stage1_nlp_mining.build_gazetteer import create_gazetteer
from src.stage1_nlp_mining.retrieval import retrieve_archival_snippets

logger = get_logger("nlp_geocoder")


def load_gazetteer():
    gazetteer_path = GAZETTEER_DIR / "eastern_ghats_toponyms.geojson"
    if not gazetteer_path.exists():
        create_gazetteer()
    gdf = gpd.read_file(gazetteer_path)
    return gdf


def match_toponym(text, gazetteer_gdf):
    """Match text snippet against micro-toponym gazetteer entries."""
    text_lower = text.lower()

    # Priority 1: Match Class A (Linear infrastructure / specific stations / tunnels)
    for idx, row in gazetteer_gdf[gazetteer_gdf["precision_class"] == "A"].iterrows():
        name = row["name"].lower()
        te_name = row["te_name"]
        if "tunnel" in text_lower and ("53" in text_lower or "42" in text_lower or "36" in text_lower):
            if "53" in name:
                return row
        if "tyda" in text_lower and ("station" in text_lower or "రైల్వే" in text_lower):
            if "Tyda Railway Station" in row["name"]:
                return row
        if "borra" in text_lower and ("station" in text_lower or "గ్యాంగ్" in text_lower):
            if "Borra Guhalu Station" in row["name"]:
                return row
        if "shimiliguda" in text_lower and ("station" in text_lower or "పట్టా" in text_lower):
            if "Shimiliguda Railway Station" in row["name"]:
                return row

    # Priority 2: Match Class B (Specific Villages / Settlements)
    for idx, row in gazetteer_gdf[gazetteer_gdf["precision_class"] == "B"].iterrows():
        name_clean = re.sub(r'\s*\([^)]*\)', '', row["name"]).strip().lower()
        if name_clean in text_lower or row["te_name"] in text:
            return row

    # Priority 3: Match Class C (Broad Corridors / Ghat Sections)
    for idx, row in gazetteer_gdf[gazetteer_gdf["precision_class"] == "C"].iterrows():
        name_clean = re.sub(r'\s*\([^)]*\)', '', row["name"]).strip().lower()
        if name_clean in text_lower or row["te_name"] in text:
            return row

    # Default fallback to central Ananthagiri Ghat Road corridor
    fallback = gazetteer_gdf[gazetteer_gdf["name"] == "Ananthagiri Ghat Road"].iloc[0]
    return fallback


def radius_m_to_deg(radius_m):
    """Approximate meters to latitude/longitude degrees near 18 degrees North."""
    # 1 deg lat approx 110.57 km; 1 deg lon approx 105.7 km at 18 deg N
    delta_lat = radius_m / 110570.0
    delta_lon = radius_m / 105700.0
    return delta_lat, delta_lon


def build_candidate_events():
    """Extract candidate landslide events and compile candidate_events.geojson."""
    gazetteer_gdf = load_gazetteer()
    snippets = retrieve_archival_snippets()

    features = []

    for idx, item in enumerate(snippets):
        matched_toponym = match_toponym(item["snippet"], gazetteer_gdf)
        
        lat = float(matched_toponym["latitude"])
        lon = float(matched_toponym["longitude"])
        radius_m = float(matched_toponym["search_radius_m"])
        precision_class = matched_toponym["precision_class"]

        delta_lat, delta_lon = radius_m_to_deg(radius_m)

        # Polygon bounding box for candidate search area
        bbox_geom = box(
            lon - delta_lon,
            lat - delta_lat,
            lon + delta_lon,
            lat + delta_lat
        )

        event_id = f"CAND_{idx+1:03d}"

        properties = {
            "candidate_id": event_id,
            "date": item.get("date", "2020-01-01"),
            "matched_toponym": matched_toponym["name"],
            "toponym_te": matched_toponym["te_name"],
            "precision_class": precision_class,
            "search_radius_m": radius_m,
            "center_latitude": lat,
            "center_longitude": lon,
            "classifier_confidence": item.get("classifier_confidence", 0.85),
            "snippet": item["snippet"],
            "source": item.get("source", "Archival Scraping"),
            "url": item.get("url", "")
        }

        features.append({
            "type": "Feature",
            "geometry": mapping(bbox_geom),
            "properties": properties
        })

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = INTERIM_DATA_DIR / "candidate_events.geojson"
    gdf.to_file(out_file, driver="GeoJSON")
    logger.info(f"Stage 1 Complete: Created {out_file} with {len(gdf)} candidate event bounding boxes.")
    return out_file


if __name__ == "__main__":
    build_candidate_events()
