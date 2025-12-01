"""
NDW traffic speed loader.

Functions:
    parse_trafficspeed(url: str = DEFAULT_TRAFFICSPEED_URL) -> pd.DataFrame
    load_traffic_speed(...)  # alias

Returns a DataFrame with:
    - site_id
    - measurement_time
    - avg_speed_kmh
    - flow_veh_per_hour
"""

import gzip
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests


DEFAULT_TRAFFICSPEED_URL = "https://opendata.ndw.nu/trafficspeed.xml.gz"


def _extract_speed_from_basic(basic: ET.Element):
    """
    Try several patterns to get a speed value from a <basicData> block.
    Returns a float or None.
    """
    ns_any = "{*}"

    # 1) classic DATEX-II: trafficSpeed -> averageVehicleSpeed -> speed
    speed_elem = basic.find(
        f".//{ns_any}trafficSpeed//{ns_any}averageVehicleSpeed//{ns_any}speed"
    )
    if speed_elem is not None and speed_elem.text:
        try:
            return float(speed_elem.text)
        except ValueError:
            pass

    # 2) maybe just <speed> somewhere under basicData
    speed_elem = basic.find(f".//{ns_any}speed")
    if speed_elem is not None and speed_elem.text:
        try:
            return float(speed_elem.text)
        except ValueError:
            pass

    # 3) maybe stored as attribute on <averageVehicleSpeed> or <trafficSpeed>
    avg_elem = basic.find(f".//{ns_any}averageVehicleSpeed") or basic.find(
        f".//{ns_any}trafficSpeed"
    )
    if avg_elem is not None:
        # attributes like speed / value / avg
        for attr_name, attr_val in avg_elem.attrib.items():
            if "speed" in attr_name.lower() or attr_name.lower() in {"value", "avg"}:
                try:
                    return float(attr_val)
                except ValueError:
                    continue

        # also check child attributes
        for e in avg_elem.iter():
            for attr_name, attr_val in e.attrib.items():
                if "speed" in attr_name.lower() or attr_name.lower() in {"value", "avg"}:
                    try:
                        return float(attr_val)
                    except ValueError:
                        continue

    return None


def _extract_flow_from_basic(basic: ET.Element):
    """
    Try several patterns to get a flow value from a <basicData> block.
    Returns a float or None.
    """
    ns_any = "{*}"

    # 1) nested vehicleFlowRate element
    flow_elem = basic.find(
        f".//{ns_any}trafficFlow//{ns_any}vehicleFlowRate//{ns_any}vehicleFlowRate"
    )
    if flow_elem is not None and flow_elem.text:
        try:
            return float(flow_elem.text)
        except ValueError:
            pass

    # 2) simpler tags
    for tag in ("vehicleFlowRate", "rate", "flow"):
        fe = basic.find(f".//{ns_any}{tag}")
        if fe is not None and fe.text:
            try:
                return float(fe.text)
            except ValueError:
                pass

    # 3) attributes on trafficFlow-ish elements
    for cand_name in ("vehicleFlowRate", "trafficFlow", "flowValue"):
        fe = basic.find(f".//{ns_any}{cand_name}")
        if fe is not None:
            for attr_name, attr_val in fe.attrib.items():
                if any(k in attr_name.lower() for k in ["flow", "rate", "volume", "intensity"]):
                    try:
                        return float(attr_val)
                    except ValueError:
                        continue

            for e in fe.iter():
                for attr_name, attr_val in e.attrib.items():
                    if any(k in attr_name.lower() for k in ["flow", "rate", "volume", "intensity"]):
                        try:
                            return float(attr_val)
                        except ValueError:
                            continue

    return None


def parse_trafficspeed(
    url: str = DEFAULT_TRAFFICSPEED_URL,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Parse NDW trafficspeed.xml.gz into a DataFrame with:
        - site_id
        - measurement_time
        - avg_speed_kmh
        - flow_veh_per_hour
    """
    sess = session or requests
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    xml_bytes = gzip.decompress(resp.content)

    root = ET.fromstring(xml_bytes)
    ns_any = "{*}"

    rows = []

    for sm in root.findall(f".//{ns_any}siteMeasurements"):
        site_id = None
        mtime = None
        avg_speed = None
        flow = None

        # site id
        sid_elem = sm.find(f".//{ns_any}measurementSiteReference")
        if sid_elem is not None:
            site_id = sid_elem.attrib.get("id") or (
                sid_elem.text.strip() if sid_elem.text else None
            )

        # measurement time: try both common tags
        mtime = (
            sm.findtext(f".//{ns_any}measurementOrCalculationTime")
            or sm.findtext(f".//{ns_any}measurementTimeDefault")
        )

        # walk measuredValue blocks
        for mv in sm.findall(f".//{ns_any}measuredValue"):
            basic = mv.find(f".//{ns_any}basicData")
            if basic is None:
                continue

            if avg_speed is None:
                avg_speed = _extract_speed_from_basic(basic)

            if flow is None:
                flow = _extract_flow_from_basic(basic)

        rows.append(
            {
                "site_id": site_id,
                "measurement_time": mtime,
                "avg_speed_kmh": avg_speed,
                "flow_veh_per_hour": flow,
            }
        )

    return pd.DataFrame(rows)


# backwards-compatible alias, same style as other loaders
def load_traffic_speed(
    url: str = DEFAULT_TRAFFICSPEED_URL,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    return parse_trafficspeed(url=url, session=session)