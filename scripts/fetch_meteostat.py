#!/usr/bin/env python3
"""Fetch historical hourly weather from Meteostat for a location or station.

Usage examples (PowerShell):

    pip install meteostat pandas

    # by lat/lon
    python .\scripts\fetch_meteostat.py --lat 51.4416 --lon 5.4697 --start 2023-01-01 --end 2023-12-31

    # by station id (meteostat station identifier)
    python .\scripts\fetch_meteostat.py --station NL000023 --start 2023-01-01 --end 2023-01-31

The script saves CSV files under `data/meteostat/` by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

try:
    from meteostat import Point, Hourly, Stations
except Exception as e:
    print("Missing dependency: please install with: pip install meteostat pandas")
    raise

import pandas as pd
import datetime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch historical Meteostat hourly data and save as CSV")
    gp = p.add_mutually_exclusive_group(required=True)
    gp.add_argument("--lat", type=float, help="Latitude of location")
    gp.add_argument("--station", type=str, help="Meteostat station id (use Stations.lookup or station metadata)")
    p.add_argument("--lon", type=float, help="Longitude of location (required with --lat)")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--output-dir", default=None, help="Output directory (default: <repo_root>/data/meteostat)")
    # use Meteostat canonical variable names by default (prcp, wspd, wpgt, pres, tsun)
    p.add_argument("--vars", default=",".join(["temp","prcp","wspd","wpgt","pres","tsun"]), help="Comma-separated list of variables to request (meteostat columns)")
    return p.parse_args()


def sanitize_filename(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(":", "_")


def fetch_by_point(lat: float, lon: float, start: str, end: str, vars: Optional[list[str]] = None) -> pd.DataFrame:
    # create a Point and request hourly data
    # coerce start/end to pandas datetime if they are strings to avoid passing raw str to meteostat
    try:
        s = pd.to_datetime(start) if start is not None else None
    except Exception:
        s = start
    try:
        e = pd.to_datetime(end) if end is not None else None
    except Exception:
        e = end

    pt = Point(lat, lon)
    df = Hourly(pt, s, e)
    if vars:
        # attempt to select requested columns if present
        try:
            colnames = df.get_column_names()
        except Exception:
            colnames = []
        available = [c for c in vars if c in colnames]
        if available:
            df = df.fetch()[available]
        else:
            df = df.fetch()
    else:
        df = df.fetch()
    return df


def fetch_by_station(station: str, start: str, end: str, vars: Optional[list[str]] = None) -> pd.DataFrame:
    # Use Stations to find station id
    st = Stations()
    st = st.id(station)
    res = st.fetch(1)
    if res.empty:
        raise ValueError(f"Station not found: {station}")
    sid = res.index[0]
    # coerce start/end similar to fetch_by_point
    try:
        s = pd.to_datetime(start) if start is not None else None
    except Exception:
        s = start
    try:
        e = pd.to_datetime(end) if end is not None else None
    except Exception:
        e = end
    df = Hourly(sid, s, e).fetch()
    if vars:
        available = [c for c in vars if c in df.columns]
        if available:
            df = df[available]
    return df


def fetch_and_save(lat: Optional[float], lon: Optional[float], station: Optional[str], start: str, end: str, out_dir: Optional[Path] = None, vars: Optional[list[str]] = None) -> Path:
    """Fetch meteostat data by point or station and save CSV to out_dir. Returns saved Path."""
    # resolve output directory
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1] if len(script_path.parents) > 1 else script_path.parent
    if out_dir is None:
        out_dir = repo_root / 'data' / 'meteostat'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # coerce start/end to pandas timestamps so meteostat receives datetime-like objects
    try:
        start_dt = pd.to_datetime(start) if start is not None else None
    except Exception as e:
        raise ValueError(f"Could not parse start date '{start}': {e}")
    try:
        end_dt = pd.to_datetime(end) if end is not None else None
    except Exception as e:
        raise ValueError(f"Could not parse end date '{end}': {e}")

    # convert pandas Timestamps to native datetime objects to be compatible with meteostat
    def _to_native(dtval):
        if dtval is None:
            return None
        try:
            # pandas.Timestamp -> datetime
            if hasattr(dtval, 'to_pydatetime'):
                return dtval.to_pydatetime()
        except Exception:
            pass
        # if it's a string or already datetime, return as-is
        return dtval

    start_native = _to_native(start_dt)
    end_native = _to_native(end_dt)
    print(f"DEBUG converted start_native={start_native!r} ({type(start_native)}), end_native={end_native!r} ({type(end_native)})")

    if station:
        df = fetch_by_station(station, start_native, end_native, vars=vars)
        id_part = f"station_{station}"
    else:
        if lat is None or lon is None:
            raise ValueError("Latitude and longitude must both be provided when not using --station")
        df = fetch_by_point(lat, lon, start_native, end_native, vars=vars)
        id_part = f"lat{lat}_lon{lon}"

    if df is None or df.empty:
        raise RuntimeError("No data returned for the requested period/location.")

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    fname = sanitize_filename(f"meteostat_{id_part}_{start}_{end}.csv")
    out_fp = out_dir / fname
    df.to_csv(out_fp, index=True)
    return out_fp


def main() -> None:
    args = parse_args()
    vars = [v.strip() for v in args.vars.split(",")] if args.vars else None
    try:
        out_fp = fetch_and_save(getattr(args, 'lat', None), getattr(args, 'lon', None), getattr(args, 'station', None), args.start, args.end, out_dir=Path(args.output_dir) if args.output_dir else None, vars=vars)
        print(f"Saved Meteostat data to: {out_fp.resolve()}")
    except Exception as e:
        print("Failed to fetch/save data:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
