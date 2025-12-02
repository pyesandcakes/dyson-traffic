#!/usr/bin/env python3
"""
Module: ndw.tomtom_fetch

Provides functions to query TomTom Flow data for a bounding box or a named
place (via Nominatim geocoding). Designed to be callable from a Jupyter
notebook as well as from the CLI.

Public functions:
 - geocode_place(name) -> (minLon, minLat, maxLon, maxLat)
 - fetch_flow_grid(bbox, nx, ny, key=None, delay=0.3, save_raw=False, out_prefix=None)
 - fetch_for_region(name, nx, ny, **kwargs)

The module will try to read an API key from `config.py` (project root) as
`api_key` or from the environment variable `TOMTOM_API_KEY`.
"""

from typing import Iterator, Tuple, Optional
import json
import os
import time

import pandas as pd
import requests


def get_api_key() -> str:
    try:
        from config import api_key  # type: ignore
        if api_key:
            return api_key
    except Exception:
        pass
    for name in ("TOMTOM_API_KEY", "NDW_API_KEY", "API_KEY"):
        v = os.environ.get(name)
        if v:
            return v
    raise RuntimeError("TomTom API key not found. Set TOMTOM_API_KEY or provide config.py with api_key")


def geocode_place(name: str) -> Tuple[float, float, float, float]:
    """Resolve a place name to a bounding box using Nominatim.

    Returns (minLon, minLat, maxLon, maxLat).
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": name, "format": "json", "limit": 1}
    headers = {"User-Agent": "ndw-tomtom-fetch/1.0 (+https://example.org)"}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"Place not found: {name}")
    bb = data[0].get("boundingbox")
    # boundingbox is [south_lat, north_lat, west_lon, east_lon]
    south_lat, north_lat, west_lon, east_lon = map(float, bb)
    return (west_lon, south_lat, east_lon, north_lat)


def grid_points(min_lon: float, min_lat: float, max_lon: float, max_lat: float, nx: int, ny: int) -> Iterator[Tuple[float, float]]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    for i in range(nx):
        lon = min_lon + (max_lon - min_lon) * (i / max(1, nx - 1))
        for j in range(ny):
            lat = min_lat + (max_lat - min_lat) * (j / max(1, ny - 1))
            yield lat, lon


def query_flow_point(lat: float, lon: float, key: str) -> dict:
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"point": f"{lat},{lon}", "unit": "KMPH", "key": key}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    fs = data.get("flowSegmentData", {})
    return {
        "lat": lat,
        "lon": lon,
        "currentSpeed": fs.get("currentSpeed"),
        "freeFlowSpeed": fs.get("freeFlowSpeed"),
        "confidence": fs.get("confidence"),
        "currentTravelTime": fs.get("currentTravelTime"),
        "freeFlowTravelTime": fs.get("freeFlowTravelTime"),
    }


def fetch_flow_grid(
    bbox: Tuple[float, float, float, float],
    nx: int = 5,
    ny: int = 5,
    key: Optional[str] = None,
    delay: float = 0.3,
    save_raw: bool = False,
    out_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Fetch flow data for a grid over `bbox`.

    bbox: (minLon, minLat, maxLon, maxLat)
    Returns (DataFrame, summary_dict). If `out_prefix` is provided, saves CSV/JSON files.
    """
    if key is None:
        key = get_api_key()
    minLon, minLat, maxLon, maxLat = bbox
    rows = []
    raw_path = f"{out_prefix}.jsonl" if out_prefix else None
    if raw_path and save_raw and os.path.exists(raw_path):
        os.remove(raw_path)

    total = nx * ny
    i = 0
    for lat, lon in grid_points(minLon, minLat, maxLon, maxLat, nx, ny):
        i += 1
        try:
            item = query_flow_point(lat, lon, key)
        except Exception as e:
            item = {"lat": lat, "lon": lon, "error": str(e)}
        rows.append(item)
        if raw_path and save_raw:
            with open(raw_path, "a", encoding="utf8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        time.sleep(delay)

    df = pd.DataFrame(rows)
    if out_prefix:
        csv_path = f"{out_prefix}.csv"
        df.to_csv(csv_path, index=False)
        json_path = f"{out_prefix}.json"
        df.to_json(json_path, orient="records", force_ascii=False)

    summary = {}
    if "currentSpeed" in df.columns:
        speeds = pd.to_numeric(df["currentSpeed"], errors="coerce")
        summary["count_samples"] = int(speeds.count())
        summary["mean_speed_kmph"] = float(speeds.mean()) if speeds.count() else None
        summary["min_speed_kmph"] = float(speeds.min()) if speeds.count() else None
        summary["p25_speed_kmph"] = float(speeds.quantile(0.25)) if speeds.count() else None
        summary["p50_speed_kmph"] = float(speeds.quantile(0.5)) if speeds.count() else None
        summary["p75_speed_kmph"] = float(speeds.quantile(0.75)) if speeds.count() else None

    if out_prefix:
        summary_path = f"{out_prefix}_summary.json"
        with open(summary_path, "w", encoding="utf8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return df, summary


def fetch_for_region(
    place_name: str,
    nx: int = 5,
    ny: int = 5,
    key: Optional[str] = None,
    delay: float = 0.3,
    save_raw: bool = False,
    out_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Geocode `place_name` to a bbox and fetch the grid.

    Example: fetch_for_region("Eindhoven", nx=4, ny=4, out_prefix="eindhoven")
    """
    bbox = geocode_place(place_name)
    return fetch_flow_grid(bbox, nx=nx, ny=ny, key=key, delay=delay, save_raw=save_raw, out_prefix=out_prefix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample TomTom traffic flow over a bbox or place name")
    parser.add_argument("--bbox", help="bbox as minLon,minLat,maxLon,maxLat")
    parser.add_argument("--place", help="Place name to geocode (e.g. Eindhoven)")
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--out", default="tomtom_samples")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    if args.place:
        bbox = geocode_place(args.place)
    elif args.bbox:
        minLon, minLat, maxLon, maxLat = map(float, args.bbox.split(","))
        bbox = (minLon, minLat, maxLon, maxLat)
    else:
        parser.error("Either --place or --bbox is required")

    df, summary = fetch_flow_grid(bbox, nx=args.nx, ny=args.ny, delay=args.delay, save_raw=args.save_raw, out_prefix=args.out)
    print("Summary:", json.dumps(summary, indent=2, ensure_ascii=False))
