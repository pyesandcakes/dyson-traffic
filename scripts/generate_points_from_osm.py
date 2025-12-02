#!/usr/bin/env python3
"""Generate probe points along major roads using Overpass API and save to JSON or embed as Python.

Usage:
  python scripts/generate_points_from_osm.py --place Eindhoven --spacing 200 --max-per-way 50

This script does a best-effort Overpass query for major highways in the place's bbox (via Nominatim),
extracts way geometries, samples points along them with approximately `--spacing` meters between points,
and writes a JSON list or a Python file that defines `POINTS = [...]` which can be imported by the collector.

Note: This script performs network requests to public APIs (Nominatim + Overpass). Use responsibly.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def haversine_meters(a, b):
    # a, b are (lat, lon)
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371000
    return R * math.sqrt(dlat * dlat + (math.cos((lat1 + lat2) / 2) * dlon) ** 2)


def interpolate(a, b, fraction):
    # simple linear interpolation on lat/lon (sufficient for short segments)
    return (a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction)


def sample_line(coords, spacing_m, max_per_way=None):
    # coords: list of (lat, lon)
    pts = []
    if not coords:
        return pts
    acc = 0.0
    last = coords[0]
    pts.append(last)
    for i in range(1, len(coords)):
        a = coords[i - 1]
        b = coords[i]
        seg_len = haversine_meters(a, b)
        if seg_len == 0:
            continue
        n = int(math.floor(seg_len / spacing_m))
        for k in range(1, n + 1):
            frac = k * spacing_m / seg_len
            if frac >= 1.0:
                break
            p = interpolate(a, b, frac)
            pts.append(p)
        pts.append(b)
        if max_per_way and len(pts) >= max_per_way:
            return pts[:max_per_way]
    # deduplicate close points
    out = [pts[0]]
    for p in pts[1:]:
        if haversine_meters(out[-1], p) >= spacing_m * 0.5:
            out.append(p)
    if max_per_way:
        return out[:max_per_way]
    return out


def bbox_for_place(place):
    r = requests.get(NOMINATIM_URL, params={"q": place, "format": "json", "limit": 1}, headers={"User-Agent": "points-gen/1.0"}, timeout=15)
    r.raise_for_status()
    j = r.json()
    if not j:
        raise SystemExit(f"Place not found: {place}")
    bb = j[0]["boundingbox"]
    south, north, west, east = map(float, [bb[0], bb[1], j[0]["boundingbox"][1], j[0]["boundingbox"][3]])
    # Nominatim boundingbox order is [south, north, west, east] as strings in many responses
    # but to be safe, parse properly
    south = float(bb[0]); north = float(bb[1]); west = float(bb[2]); east = float(bb[3])
    return (south, west, north, east)


def overpass_query(bbox, highway_filter):
    # bbox: (south, west, north, east)
    s, w, n, e = bbox
    # match motorway/trunk/primary/secondary/tertiary and links
    q = f"[out:json][timeout:60];(way[\"highway\"~\"{highway_filter}\"]({s},{w},{n},{e}););out geom;"
    resp = requests.post(OVERPASS_URL, data={"data": q}, timeout=90)
    resp.raise_for_status()
    return resp.json()


def extract_way_coords(elem):
    geom = elem.get("geometry")
    if not geom:
        return []
    return [(pt["lat"], pt["lon"]) if isinstance(pt, dict) else (pt[0], pt[1]) for pt in geom]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--place", default="Eindhoven")
    parser.add_argument("--spacing", type=int, default=200, help="Approx spacing between sampled points in meters")
    parser.add_argument("--max-per-way", type=int, default=200, help="Max sampled points per way")
    parser.add_argument("--highways", default="motorway|trunk|primary|secondary|tertiary|unclassified|motorway_link|trunk_link|primary_link|secondary_link", help="Overpass highway filter regex (does not include residential by default)")
    parser.add_argument("--include-residential", action="store_true", help="Also include small residential streets in the sampling (may produce many points)")
    parser.add_argument("--output-json", default="data/tomtom/eindhoven/points.json")
    parser.add_argument("--embed-py", default="scripts/embedded_points.py", help="Path to write an embeddable Python file with POINTS = [...]")
    parser.add_argument("--max-total", type=int, default=500, help="Max total points to generate")
    args = parser.parse_args()

    bbox = bbox_for_place(args.place)
    print("BBox:", bbox)

    highways_filter = args.highways
    if args.include_residential:
        highways_filter = highways_filter + "|residential"

    j = overpass_query(bbox, highways_filter)
    ways = [el for el in j.get("elements", []) if el.get("type") == "way"]
    print(f"Found {len(ways)} ways")

    sampled = []
    for w in ways:
        coords = extract_way_coords(w)
        pts = sample_line(coords, args.spacing, max_per_way=args.max_per_way)
        for lat, lon in pts:
            sampled.append({"lat": round(lat, 6), "lon": round(lon, 6)})
        if len(sampled) >= args.max_total:
            break
        time.sleep(0.2)

    # deduplicate by rounding
    uniq = []
    seen = set()
    for p in sampled:
        key = (p["lat"], p["lon"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= args.max_total:
            break

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(uniq, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(uniq)} points to {out_json}")

    # write embed Python file
    embed_path = Path(args.embed_py)
    embed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(embed_path, "w", encoding="utf-8") as fh:
        fh.write("# Auto-generated embedded points; safe to import from collector\n")
        fh.write("POINTS = ")
        json.dump(uniq, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"Wrote embedded Python points to {embed_path}")


if __name__ == '__main__':
    main()
