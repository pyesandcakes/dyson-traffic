import gzip
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests


DEFAULT_MST_URL = "https://opendata.ndw.nu/measurement_current.xml.gz"


def load_measurement_sites(
    url: str = DEFAULT_MST_URL,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Load NDW Measurement Site Table (measurement_current.xml.gz) into a DataFrame.

    This parses both v2-style <measurementSiteRecord> and v3-style <measurementSite>
    and returns a row per measurement site with:

        - site_id        (matches MST id, e.g. 'PZH01_MST_0080_01')
        - version
        - name           (human-readable measurementSiteName, if present)
        - location_raw   (raw text collected from location elements)
    """
    sess = session or requests
    resp = sess.get(url, timeout=60)
    resp.raise_for_status()

    xml_bytes = gzip.decompress(resp.content)
    root = ET.fromstring(xml_bytes)
    ns_any = "{*}"

    rows = []

    # v2 naming: measurementSiteRecord
    candidates = list(root.findall(f".//{ns_any}measurementSiteRecord"))

    # v3 naming: measurementSite
    candidates += list(root.findall(f".//{ns_any}measurementSite"))

    for ms in candidates:
        site_id = ms.attrib.get("id")
        version = ms.attrib.get("version")

        # Try to extract a human-readable site name, if present
        name = ms.findtext(
            f".//{ns_any}measurementSiteName//{ns_any}values//{ns_any}value//{ns_any}content"
        )

        # Try to grab any location-ish text: measurementSiteLocation, locationReference,
        # or locationForDisplay (depends on profile/version)
        loc_elem = (
            ms.find(f".//{ns_any}measurementSiteLocation")
            or ms.find(f".//{ns_any}locationReference")
            or ms.find(f".//{ns_any}locationForDisplay")
        )

        loc_text = None
        if loc_elem is not None:
            parts = []
            for e in loc_elem.iter():
                if e.text and e.text.strip():
                    parts.append(e.text.strip())
            loc_text = " | ".join(dict.fromkeys(parts)) if parts else None

        rows.append(
            {
                "site_id": site_id,
                "version": version,
                "site_name": name,
                "location_raw": loc_text,
            }
        )

    df_sites = pd.DataFrame(rows)

    # Drop rows without an id
    df_sites = df_sites[~df_sites["site_id"].isna()].reset_index(drop=True)

    return df_sites

def parse_location_raw(row):
    """
    Parse NDW measurementSite location_raw into usable fields.
    The location_raw is a long DATEX/VILD reference string, but:
    
    FIRST TWO items:
        lat, lon

    THIRD item:
        'mainCarriageway', 'slipRoad', etc.

    SIXTH item:
        'A' / 'B' (carriageway)

    SEVENTH item:
        'positive' / 'negative' (direction)

    All remaining parts are optional and can be ignored for simple site mapping.
    """
    if pd.isna(row):
        return pd.Series([None]*5)

    parts = [p.strip() for p in row.split("|")]

    lat = pd.to_numeric(parts[0], errors="coerce") if len(parts) > 0 else None
    lon = pd.to_numeric(parts[1], errors="coerce") if len(parts) > 1 else None

    carriageway_type = parts[2] if len(parts) > 2 else None
    carriageway = parts[5] if len(parts) > 5 else None
    direction_ref = parts[6] if len(parts) > 6 else None

    return pd.Series([lat, lon, carriageway_type, carriageway, direction_ref])