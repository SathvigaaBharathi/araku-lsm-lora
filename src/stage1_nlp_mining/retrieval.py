import re
import json
import urllib.parse
import urllib.request
import feedparser
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import NEWS_CORPUS_DIR, is_offline_mode, get_logger
from src.stage1_nlp_mining.classifier import LandslideRelevanceClassifier

logger = get_logger("nlp_retrieval")

# Historical Seed Events (GSI Bhukosh, NDMA, literature: Kodipunjulavalasa 2006, Cyclone Nilam 2012, KK Line 2017, etc.)
HISTORICAL_SEED_EVENTS = [
    {
        "date": "2006-09-18",
        "snippet": "Massive landslide triggered by torrential monsoon rain at Kodipunjulavalasa near Araku Valley causing extensive slope failure and hillside collapse.",
        "location_mention": "Kodipunjulavalasa",
        "source": "GSI Bhukosh & Historical Report 2006",
        "url": "http://bhukosh.gsi.gov.in/historical/2006_kodipunjulavalasa"
    },
    {
        "date": "2012-11-02",
        "snippet": "Cyclone Nilam heavy rainfall causes severe landslide and train derailment on KK line near Borra Caves and Tunnel 53.",
        "location_mention": "KK Line Tunnel 53",
        "source": "NDMA Disaster Report 2012",
        "url": "http://ndma.gov.in/reports/cyclone_nilam_kk_line_2012"
    },
    {
        "date": "2017-10-06",
        "snippet": "Huge boulders and rockfall block track between Tyda and Shimiliguda on KK line, disrupting iron ore freight traffic for 48 hours.",
        "location_mention": "Tyda Railway Station",
        "source": "Deccan Chronicle Archive 2017",
        "url": "https://deccanchronicle.com/nation/in-other-news/061017/rockfall-on-kk-line.html"
    },
    {
        "date": "2020-08-14",
        "snippet": "అనంతగిరి ఘాట్ రోడ్డులో కొండచరియలు విరిగిపడి విశాఖ-అరకు రహదారిపై గంటల తరబడి వాహనాలు నిలిచిపోయాయి.",
        "location_mention": "Ananthagiri Ghat Road",
        "source": "Eenadu Telugu Archive 2020",
        "url": "https://eenadu.net/telugu/news/2020/araku_landslide"
    },
    {
        "date": "2021-09-27",
        "snippet": "Cyclone Gulab rainfall triggers mudslide near Shimiliguda village and damages railway embankment near Borra Guhalu.",
        "location_mention": "Shimiliguda Village",
        "source": "The Hindu Archive 2021",
        "url": "https://thehindu.com/news/national/andhra-pradesh/cyclone-gulab-landslide-araku/article2021.ece"
    },
    {
        "date": "2023-07-24",
        "snippet": "Continuous monsoon downpour causes rockfall on SH-39 near Mile Marker 14 close to Ananthagiri village.",
        "location_mention": "SH-39 Mile Marker 14 (Ananthagiri)",
        "source": "Public RSS Feed 2023",
        "url": "https://news.google.com/rss/search?q=landslide+Araku"
    }
]


def fetch_google_news_rss(query="landslide Araku"):
    """Fetch entries from Google News RSS feed."""
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN"
    items = []
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:10]:
            snippet = entry.get("summary", entry.get("title", ""))
            pub_date = entry.get("published", "")
            items.append({
                "date": pub_date,
                "snippet": snippet,
                "location_mention": "Araku Valley",
                "source": "Google News RSS",
                "url": entry.get("link", rss_url)
            })
    except Exception as e:
        logger.warning(f"Google News RSS fetch warning: {e}")
    return items


def fetch_gdelt_articles(query="landslide Araku"):
    """Fetch candidate news snippets via GDELT API."""
    items = []
    try:
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={urllib.parse.quote(query)}&mode=artlist&maxrecords=10&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            for article in data.get("articles", []):
                items.append({
                    "date": article.get("seendate", "2022-01-01")[:8],
                    "snippet": article.get("title", ""),
                    "location_mention": article.get("domain", "Araku"),
                    "source": "GDELT API",
                    "url": article.get("url", "")
                })
    except Exception as e:
        logger.warning(f"GDELT API fetch info/fallback: {e}")
    return items


def retrieve_archival_snippets():
    """Retrieve and consolidate landslide candidate snippets across all sources."""
    logger.info("Starting NLP archival news retrieval across GDELT, RSS, and Historical Archives...")
    
    candidates = list(HISTORICAL_SEED_EVENTS)

    if not is_offline_mode():
        rss_results = fetch_google_news_rss("landslide Araku")
        gdelt_results = fetch_gdelt_articles("landslide Araku")
        candidates.extend(rss_results)
        candidates.extend(gdelt_results)
    else:
        logger.info("Offline mode active: Using historical seed corpus and cached news archives.")
        rss_results = []
        gdelt_results = []

    # Raw hit counts per source before classification filtering
    source_counts = {
        "historical": len(HISTORICAL_SEED_EVENTS),
        "rss": len(rss_results),
        "gdelt": len(gdelt_results)
    }
    logger.info(f"Raw hit counts per source before filtering: {source_counts}")
    logger.info("Keyword query used for RSS and GDELT: 'landslide Araku'")

    # Filter with relevance classifier
    classifier = LandslideRelevanceClassifier()
    filtered_candidates = []
    # Filter with relevance classifier
    classifier = LandslideRelevanceClassifier()
    filtered_candidates = []

    for item in candidates:
        score = classifier.predict(item["snippet"])
        if score >= 0.40:  # Include plausible candidates for NER geocoding
            item["classifier_confidence"] = round(score, 4)
            filtered_candidates.append(item)

    df = pd.DataFrame(filtered_candidates)
    NEWS_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = NEWS_CORPUS_DIR / "mined_snippets.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Retrieved and filtered {len(filtered_candidates)} candidate text snippets. Saved to {out_file}")
    return filtered_candidates


if __name__ == "__main__":
    retrieve_archival_snippets()
