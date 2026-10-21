import json
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import GAZETTEER_DIR, get_logger

logger = get_logger("build_gazetteer")

# Key Toponyms for Araku Valley, KK Railway Line, SH-39, and Eastern Ghats (Alluri Sitharama Raju District)
TOPONYMS_DATA = [
    # Precision Class A: Linear Infrastructure / Precise Stations & Tunnels (Radius: 500m)
    {"name": "KK Line Tunnel 53", "te_name": "కేకే లైన్ టన్నెల్ 53", "type": "railway_tunnel", "class": "A", "radius_m": 500, "lat": 18.2541, "lon": 83.0234},
    {"name": "KK Line Tunnel 42", "te_name": "కేకే లైన్ టన్నెల్ 42", "type": "railway_tunnel", "class": "A", "radius_m": 500, "lat": 18.2312, "lon": 83.0415},
    {"name": "KK Line Tunnel 36", "te_name": "కేకే లైన్ టన్నెల్ 36", "type": "railway_tunnel", "class": "A", "radius_m": 500, "lat": 18.2105, "lon": 83.0582},
    {"name": "Tyda Railway Station", "te_name": "తైడా రైల్వే స్టేషన్", "type": "railway_station", "class": "A", "radius_m": 500, "lat": 18.2178, "lon": 83.0503},
    {"name": "Borra Guhalu Station", "te_name": "బొర్రా గుహలు స్టేషన్", "type": "railway_station", "class": "A", "radius_m": 500, "lat": 18.2778, "lon": 83.0422},
    {"name": "Shimiliguda Railway Station", "te_name": "షిమిలీగుడా రైల్వే స్టేషన్", "type": "railway_station", "class": "A", "radius_m": 500, "lat": 18.3189, "lon": 82.9734},
    {"name": "Karakavalasa Railway Station", "te_name": "కారకవలస రైల్వే స్టేషన్", "type": "railway_station", "class": "A", "radius_m": 500, "lat": 18.1752, "lon": 83.1098},
    {"name": "Shivalingapuram Station", "te_name": "శివలింగపురం స్టేషన్", "type": "railway_station", "class": "A", "radius_m": 500, "lat": 18.1925, "lon": 83.0841},
    {"name": "Araku Railway Station", "te_name": "అరకు రైల్వే స్టేషన్", "type": "railway_station", "class": "A", "radius_m": 500, "lat": 18.3333, "lon": 82.8833},
    {"name": "SH-39 Mile Marker 14 (Ananthagiri)", "te_name": "ఎస్.హెచ్-39 మైలు రాయి 14", "type": "highway_marker", "class": "A", "radius_m": 500, "lat": 18.2415, "lon": 83.0112},
    {"name": "SH-39 Mile Marker 22 (Tyda)", "te_name": "ఎస్.హెచ్-39 మైలు రాయి 22", "type": "highway_marker", "class": "A", "radius_m": 500, "lat": 18.2210, "lon": 83.0450},

    # Precision Class B: Specific Villages & Settlement Points (Radius: 1500m)
    {"name": "Ananthagiri Village", "te_name": "అనంతగిరి గ్రామం", "type": "village", "class": "B", "radius_m": 1500, "lat": 18.2410, "lon": 83.0125},
    {"name": "Kodipunjulavalasa", "te_name": "కోడిపుంజులవలస", "type": "village", "class": "B", "radius_m": 1500, "lat": 18.2650, "lon": 83.0310},
    {"name": "Borra Village", "te_name": "బొర్రా గ్రామం", "type": "village", "class": "B", "radius_m": 1500, "lat": 18.2811, "lon": 83.0395},
    {"name": "Tyda Village", "te_name": "తైడా గ్రామం", "type": "village", "class": "B", "radius_m": 1500, "lat": 18.2150, "lon": 83.0520},
    {"name": "Shimiliguda Village", "te_name": "షిమిలీగుడా గ్రామం", "type": "village", "class": "B", "radius_m": 1500, "lat": 18.3210, "lon": 82.9700},
    {"name": "Dumbriguda", "te_name": "డుంబ్రిగుడ", "type": "village", "class": "B", "radius_m": 1500, "lat": 18.2750, "lon": 82.8550},
    {"name": "Katiki Waterfalls", "te_name": "కటికి జలపాతం", "type": "landmark", "class": "B", "radius_m": 1500, "lat": 18.2930, "lon": 83.0180},
    {"name": "Chaparai", "te_name": "చాపరాయి", "type": "landmark", "class": "B", "radius_m": 1500, "lat": 18.3550, "lon": 82.8520},

    # Precision Class C: Broad Corridors / Ghat Sections (Radius: 3000m)
    {"name": "Ananthagiri Ghat Road", "te_name": "అనంతగిరి ఘాట్ రోడ్డు", "type": "ghat_section", "class": "C", "radius_m": 3000, "lat": 18.2350, "lon": 83.0250},
    {"name": "Araku Valley Hills", "te_name": "అరకు వ్యాలీ కొండలు", "type": "valley_corridor", "class": "C", "radius_m": 3000, "lat": 18.3270, "lon": 82.8780},
    {"name": "KK Railway Line Corridor", "te_name": "కేకే రైల్వే లైన్ కారిడార్", "type": "rail_corridor", "class": "C", "radius_m": 3000, "lat": 18.2600, "lon": 83.0200},
    {"name": "Paderu Ghat Corridor", "te_name": "పాడేరు ఘాట్ కారిడార్", "type": "ghat_section", "class": "C", "radius_m": 3000, "lat": 18.0833, "lon": 82.6667},
    {"name": "Chintapalli Hills", "te_name": "చింతపల్లి కొండలు", "type": "hill_range", "class": "C", "radius_m": 3000, "lat": 17.8667, "lon": 82.3500}
]


def create_gazetteer():
    """Build GeoJSON gazetteer for Eastern Ghats / Araku Region."""
    features = []
    for item in TOPONYMS_DATA:
        geom = Point(item["lon"], item["lat"])
        properties = {
            "name": item["name"],
            "te_name": item["te_name"],
            "feature_type": item["type"],
            "precision_class": item["class"],
            "search_radius_m": item["radius_m"],
            "latitude": item["lat"],
            "longitude": item["lon"]
        }
        features.append({
            "type": "Feature",
            "geometry": properties_to_geojson_geometry(geom),
            "properties": properties
        })

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    output_path = GAZETTEER_DIR / "eastern_ghats_toponyms.geojson"
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info(f"Gazetteer successfully written to {output_path} with {len(gdf)} entries.")
    return output_path


def properties_to_geojson_geometry(point):
    return {
        "type": "Point",
        "coordinates": [point.x, point.y]
    }


if __name__ == "__main__":
    create_gazetteer()
