"""
NDW incidents loader.

Functions:
    load_incidents(url: str = DEFAULT_INCIDENTS_URL) -> pd.DataFrame
    load_active_incidents(url: str = DEFAULT_INCIDENTS_URL) -> pd.DataFrame

The returned DataFrame includes:
    - id, version
    - creation_time, observation_time, version_time
    - probability_of_occurrence, validity_status
    - overall_start_time, overall_severity
    - description
    - location_text (raw encoded location)
    - lat, lon (floats, if present)
    - loc_col2..5 (extra encoded fields)
    - carriageway (alias for loc_col4)
    - direction_ref (alias for loc_col5)
"""

import gzip
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests


DEFAULT_INCIDENTS_URL = "https://opendata.ndw.nu/incidents.xml.gz"


def _parse_incidents_root(root: ET.Element) -> pd.DataFrame:
    """Internal: parse an incidents XML root element into a DataFrame."""

    rows = []

    # Situation-level severity (often same for all records in one situation)
    overall_severity = root.findtext(".//{*}situation/{*}overallSeverity")

    # Iterate over all situationRecord nodes (each is an incident)
    for rec in root.findall(".//{*}situationRecord"):
        # Attributes (if present)
        rec_id = rec.attrib.get("id")
        rec_version = rec.attrib.get("version")

        creation_time = rec.findtext(".//{*}situationRecordCreationTime")
        observation_time = rec.findtext(".//{*}situationRecordObservationTime")
        version_time = rec.findtext(".//{*}situationRecordVersionTime")

        probability = rec.findtext(".//{*}probabilityOfOccurrence")
        validity_status = rec.findtext(".//{*}validityStatus")
        overall_start_time = rec.findtext(".//{*}overallStartTime")

        # Try to grab a human-readable description (if present)
        description = None
        desc_elem = rec.find(".//{*}generalPublicComment//{*}values//{*}value//{*}content")
        if desc_elem is not None and desc_elem.text:
            description = desc_elem.text.strip()

        # Fallback if no generalPublicComment – look for any 'values/value/content'
        if description is None:
            alt_desc = rec.find(".//{*}values//{*}value//{*}content")
            if alt_desc is not None and alt_desc.text:
                description = alt_desc.text.strip()

        # Location info – collect any text under groupOfLocations
        location_text = None
        loc_group = rec.find(".//{*}groupOfLocations")
        if loc_group is not None:
            parts = []
            for e in loc_group.iter():
                if e.text and e.text.strip():
                    parts.append(e.text.strip())
            location_text = " | ".join(dict.fromkeys(parts)) if parts else None

        rows.append(
            {
                "id": rec_id,
                "version": rec_version,
                "creation_time": creation_time,
                "observation_time": observation_time,
                "version_time": version_time,
                "probability_of_occurrence": probability,
                "validity_status": validity_status,
                "overall_start_time": overall_start_time,
                "overall_severity": overall_severity,
                "description": description,
                "location_text": location_text,
            }
        )

    df = pd.DataFrame(rows)

    # ---- Decode location_text into lat / lon + extra fields ----
    if "location_text" in df.columns:
        parts = df["location_text"].str.split("|", expand=True)
        parts = parts.apply(lambda col: col.str.strip() if col is not None else col)

        if parts.shape[1] > 0:
            df["lat"] = pd.to_numeric(parts[0], errors="coerce")
        if parts.shape[1] > 1:
            df["lon"] = pd.to_numeric(parts[1], errors="coerce")
        if parts.shape[1] > 2:
            df["loc_col2"] = parts[2]      # TBD meaning (e.g. lane / code)
        if parts.shape[1] > 3:
            df["loc_col3"] = parts[3]      # TBD meaning (e.g. km / segment)
        if parts.shape[1] > 4:
            df["loc_col4"] = parts[4]      # carriageway code (A/B/...)
        if parts.shape[1] > 5:
            df["loc_col5"] = parts[5]      # direction: positive / negative

        # Convenience aliases
        df["carriageway"] = df.get("loc_col4")
        df["direction_ref"] = df.get("loc_col5")

    return df


def load_incidents(
    url: str = DEFAULT_INCIDENTS_URL,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download and parse NDW incidents feed into a pandas DataFrame.

    Parameters
    ----------
    url : str, optional
        Incidents URL. Defaults to NDW real-time incidents feed.
    session : requests.Session, optional
        If provided, used for the HTTP GET (for connection reuse).

    Returns
    -------
    pandas.DataFrame
        One row per situationRecord (incident), with decoded lat/lon and
        carriageway/direction fields where available.
    """
    sess = session or requests
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()

    xml_data = gzip.decompress(resp.content)
    root = ET.fromstring(xml_data)

    return _parse_incidents_root(root)


def load_active_incidents(
    url: str = DEFAULT_INCIDENTS_URL,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: load only 'active' incidents with valid coordinates.

    Parameters
    ----------
    url : str, optional
        Incidents URL. Defaults to NDW real-time incidents feed.
    session : requests.Session, optional
        If provided, used for the HTTP GET.

    Returns
    -------
    pandas.DataFrame
        Active incidents with non-null lat/lon.
    """
    df = load_incidents(url=url, session=session)

    mask = (df["validity_status"] == "active") & df["lat"].notna() & df["lon"].notna()
    return df[mask].copy()