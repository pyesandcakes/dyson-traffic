"""Simple Streamlit app to visualize observed and predicted traffic speeds for today.

Usage: from the repo root run:
    streamlit run scripts/interactive_streamlit_app.py

Features:
- Loads `jupyter/model_eindhoven.joblib` if present.
- Loads latest TomTom JSONs from `data/tomtom/<region>/` (default `eindhoven`) and extracts point observations.
- Uses existing `scripts.fetch_meteostat.fetch_and_save` to ensure Meteostat hourly CSV for today is available and merges weather features.
- Builds simple features (time of day cyclic encoding and recent lags) and iteratively predicts speeds for remaining hours today.
- Displays an interactive map with a slider to select the hour and see colored points by predicted speed.

This is a pragmatic, minimal prototype — improve geometry handling, live TomTom fetching, and feature parity with training as needed.
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import glob
import math
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'jupyter' / 'model_eindhoven.joblib'
TOMTOM_DIR = ROOT / 'data' / 'tomtom'


def load_model(path=MODEL_PATH):
    if not path.exists():
        return None
    loaded = joblib.load(path)
    # if the joblib contains a dict with metadata, try to extract the estimator
    if isinstance(loaded, dict):
        # common keys: 'model','estimator','pipeline','clf'
        for k in ('model','estimator','pipeline','clf'):
            if k in loaded and hasattr(loaded[k], 'predict'):
                return loaded[k]
        # otherwise search values for an object with predict
        for v in loaded.values():
            if hasattr(v, 'predict'):
                return v
        # if not found, return the dict (caller will handle error)
        return loaded
    return loaded


def parse_timestamp_from_filename(fp: Path):
    # filename like eindhoven_20251202T155913Z.json
    import re
    m = re.search(r'(\d{8}T\d{6}Z)', fp.stem)
    if m:
        try:
            return pd.to_datetime(m.group(1), format='%Y%m%dT%H%M%SZ', utc=True)
        except Exception:
            return None
    return None


def load_config():
    # import project config (contains api key)
    try:
        import sys
        repo_root = ROOT
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import config
        return config
    except Exception:
        return None


def fetch_live_tomtom_for_points(points_df, api_key, pause=0.1):
    """Query TomTom flowSegmentData for each point (lat,lon) in parallel.
    This function parallelizes per-point requests and samples large point sets to avoid long waits.
    For higher scale or better performance, use TomTom tile APIs (not implemented here yet).
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out = points_df.copy()
    if api_key is None:
        st.warning('No TomTom API key found in config; cannot fetch live data')
        return out

    max_points = 500  # limit number of points to query live by default
    total = len(out)
    if total == 0:
        return out

    def spatial_stratified_sample(df, max_n, random_state=42):
        # Try progressive rounding to create spatial grid cells and sample one point per cell.
        # Start fine and coarsen until number of cells <= max_n.
        for prec in (5, 4, 3, 2, 1, 0):
            cells = df.copy()
            cells['_cell_lat'] = cells['lat'].round(prec)
            cells['_cell_lon'] = cells['lon'].round(prec)
            groups = cells.groupby(['_cell_lat', '_cell_lon'])
            n_cells = groups.ngroups
            if n_cells <= max_n:
                # pick one representative per cell
                sampled = groups.apply(lambda g: g.sample(1, random_state=random_state)).reset_index(drop=True)
                # if still fewer than max_n, add random additional samples
                if len(sampled) < max_n and len(df) > len(sampled):
                    remaining = df.drop(sampled.index, errors='ignore')
                    add_n = min(max_n - len(sampled), len(remaining))
                    sampled = pd.concat([sampled, remaining.sample(n=add_n, random_state=random_state)])
                return sampled
        # fallback: simple random sample
        return df.sample(n=min(max_n, len(df)), random_state=random_state)

    if total > max_points:
        st.sidebar.info(f'Spatially sampling {max_points}/{total} points for live TomTom requests to improve speed')
        query_df = spatial_stratified_sample(out, max_points)
    else:
        query_df = out.copy()

    session = requests.Session()
    session.headers.update({'User-Agent': 'dyson-traffic/1.0'})

    def fetch_point(i, lat, lon):
        try:
            url = f'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat}%2C{lon}&key={api_key}'
            resp = session.get(url, timeout=5)
            if resp.status_code == 200:
                j = resp.json()
                fs = j.get('flowSegmentData') or j.get('flowSegment') or {}
                cur = fs.get('currentSpeed')
                ff = fs.get('freeFlowSpeed')
                return (i, cur, ff)
        except Exception:
            return (i, None, None)
        return (i, None, None)

    progress = st.sidebar.progress(0)
    results = {}
    with ThreadPoolExecutor(max_workers=20) as exe:
        futures = {exe.submit(fetch_point, idx, row['lat'], row['lon']): idx for idx, row in query_df.iterrows()}
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                i, cur, ff = fut.result()
                results[i] = (cur, ff)
            except Exception:
                results[idx] = (None, None)
            completed += 1
            if total > 0:
                progress.progress(int(100 * completed / len(futures)))

    # merge results back into out dataframe (only for sampled points)
    for idx, (cur, ff) in results.items():
        if cur is not None:
            out.at[idx, 'target_speed'] = float(cur)
        if ff is not None:
            out.at[idx, 'free_flow'] = float(ff)

    # note: for points not sampled we leave original values unchanged
    return out


def load_segment_geometries(region='eindhoven'):
    """Try to extract segment geometries from local TomTom JSONs.
    Expected/handled formats (heuristic):
    - GeoJSON FeatureCollection with 'geometry' LineString coordinates
    - Per-record properties with 'shape' or 'geometry.coordinates' lists
    Returns dict: pkey -> list of [lon, lat] coordinates
    """
    folder = TOMTOM_DIR / region
    geom_map = {}
    if not folder.exists():
        return geom_map
    files = sorted(glob.glob(str(folder / '*.json')))
    for fp in files:
        try:
            j = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue
        # if GeoJSON FeatureCollection
        if isinstance(j, dict) and j.get('type') == 'FeatureCollection' and isinstance(j.get('features'), list):
            for feat in j['features']:
                props = feat.get('properties', {})
                geom = feat.get('geometry')
                if geom and geom.get('type') in ('LineString','MultiLineString'):
                    coords = geom.get('coordinates')
                    # determine pkey from properties
                    pkey = props.get('_pkey') or props.get('segment_id') or props.get('id')
                    if pkey and coords:
                        # normalize to list of [lon,lat]
                        geom_map[pkey] = coords if isinstance(coords[0][0], (int,float)) else [c for line in coords for c in line]
        # otherwise try to find per-record geometry fields
        elif isinstance(j, list):
            for rec in j:
                if isinstance(rec, dict):
                    pkey = rec.get('_pkey') or rec.get('segment_id') or rec.get('id')
                    geom = None
                    if 'geometry' in rec and isinstance(rec['geometry'], dict):
                        geom = rec['geometry'].get('coordinates')
                    elif 'shape' in rec:
                        geom = rec.get('shape')
                    if pkey and geom:
                        geom_map[pkey] = geom
    return geom_map


def load_latest_tomtom(region='eindhoven'):
    folder = TOMTOM_DIR / region
    if not folder.exists():
        return pd.DataFrame()
    files = sorted(glob.glob(str(folder / '*.json')))
    rows = []
    for fp in files:
        try:
            j = json.load(open(fp, 'r', encoding='utf-8'))
        except Exception:
            continue
        # normalize candidates similar to notebook helper
        if isinstance(j, dict):
            cand = None
            for k in ('records', 'items', 'features', 'results'):
                if k in j and isinstance(j[k], list):
                    cand = j[k]
                    break
            if cand is None:
                cand = [j]
        elif isinstance(j, list):
            cand = j
        else:
            continue
        file_ts = parse_timestamp_from_filename(Path(fp))
        for rec in cand:
            if isinstance(rec, dict):
                rec_copy = dict(rec)
                rec_copy['__file'] = str(fp)
                rec_copy['__file_ts'] = file_ts
                rows.append(rec_copy)
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    # heuristics to find coordinates and speed
    lat_cols = [c for c in df.columns if c.lower().endswith('lat') or '.lat' in c.lower() or c.lower().startswith('lat')]
    lon_cols = [c for c in df.columns if c.lower().endswith('lon') or '.lon' in c.lower() or c.lower().startswith('lon')]
    speed_cols = [c for c in df.columns if 'currentSpeed' in c or 'speed' == c or 'speed' in c.lower()]
    if not lat_cols or not lon_cols:
        # try nested position
        for c in df.columns:
            if isinstance(df[c].iloc[0], dict):
                # skip for now
                pass
    # pick first candidates
    latc = lat_cols[0] if lat_cols else None
    lonc = lon_cols[0] if lon_cols else None
    speedc = None
    for s in ('currentSpeed','speed','travelTime'):
        if s in df.columns:
            speedc = s
            break
    if speedc is None and speed_cols:
        speedc = speed_cols[0]

    out_rows = []
    for _, r in df.iterrows():
        lat = None; lon = None; sp = None; ff = None
        if latc and lonc:
            try:
                lat = float(r.get(latc))
                lon = float(r.get(lonc))
            except Exception:
                lat = lon = None
        if speedc:
            try:
                sp = float(r.get(speedc))
            except Exception:
                sp = None
        # try to get free flow speed if present in payload
        try:
            ff = float(r.get('freeFlowSpeed')) if 'freeFlowSpeed' in r else None
        except Exception:
            ff = None
        ts = r.get('__file_ts') or r.get('_ts') or r.get('timestamp')
        out_rows.append({'lat': lat, 'lon': lon, 'speed': sp, 'free_flow': ff, '_ts': ts, '__file': r.get('__file')})
    odf = pd.DataFrame(out_rows)
    # drop rows without coords
    odf = odf.dropna(subset=['lat','lon'])
    # coerce timestamp
    try:
        odf['_ts'] = pd.to_datetime(odf['_ts'], utc=True, errors='coerce')
    except Exception:
        odf['_ts'] = pd.NaT
    # keep most recent by rounding location
    odf['_plat'] = odf['lat'].round(5)
    odf['_plon'] = odf['lon'].round(5)
    odf['_pkey'] = odf['_plat'].astype(str) + '_' + odf['_plon'].astype(str)
    odf = odf.sort_values('_ts').groupby('_pkey').last().reset_index()
    odf['target_speed'] = odf['speed']
    # keep free_flow column if present
    if 'free_flow' not in odf.columns and 'freeFlowSpeed' in df.columns:
        odf['free_flow'] = df.get('freeFlowSpeed')
    return odf[['lat','lon','target_speed','_ts','_pkey','free_flow']]


def ensure_meteostat(lat, lon, start_date, end_date):
    # use existing scripts.fetch_meteostat.fetch_and_save
    try:
        import sys
        repo_root = ROOT
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import scripts.fetch_meteostat as fm
        out_dir = ROOT / 'data' / 'meteostat'
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"meteostat_lat{lat}_lon{lon}_{start_date}_{end_date}.csv"
        out_fp = out_dir / fname
        if not out_fp.exists():
            fm.fetch_and_save(lat, lon, None, start_date, end_date, out_dir=out_dir)
        met_df = pd.read_csv(out_fp, index_col=0, parse_dates=True)
        met_df.columns = [c.lower() for c in met_df.columns]
        # map common names
        colmap = {}
        for c in met_df.columns:
            if 'temp' in c:
                colmap[c] = 'met_temp'
            elif 'prcp' in c or 'precip' in c or 'rain' in c:
                colmap[c] = 'met_precip'
            elif 'wspd' in c or 'wind' in c and 'gust' not in c:
                colmap[c] = 'met_wind'
            elif 'gust' in c:
                colmap[c] = 'met_wind_gust'
            elif 'pres' in c:
                colmap[c] = 'met_pressure'
            elif 'tsun' in c or 'sun' in c:
                colmap[c] = 'met_tsun'
        met_df = met_df.rename(columns=colmap)
        met_df.index = pd.to_datetime(met_df.index, utc=True, errors='coerce')
        if 'met_precip' in met_df.columns:
            met_df['met_precip_3h'] = met_df['met_precip'].rolling(3, min_periods=1).sum()
            met_df['met_precip_6h'] = met_df['met_precip'].rolling(6, min_periods=1).sum()
            met_df['met_wet'] = met_df['met_precip'] > 0.1
        met_df = met_df.groupby(met_df.index.floor('h')).mean(numeric_only=True)
        return met_df
    except Exception as e:
        st.warning(f'Could not ensure meteostat data: {e}')
        return pd.DataFrame()


def build_prediction_grid(points_df, model, met_df, start_hour, hours):
    # points_df: lat, lon, target_speed (last obs), _pkey
    rows = []
    now = pd.Timestamp.utcnow().floor('h')
    hours_list = [now + pd.Timedelta(hours=i) for i in range(start_hour, start_hour+hours)]
    for _, p in points_df.iterrows():
        last_obs = p.get('target_speed')
        free_flow = p.get('free_flow') if 'free_flow' in p.index else None
        for ts in hours_list:
            rows.append({'_pkey': p['_pkey'], 'lat': p['lat'], 'lon': p['lon'], '_ts': ts, 'lag1': last_obs, 'lag2': last_obs, 'rolling_mean_3': last_obs, 'free_flow': free_flow, 'target_speed': last_obs})
            # iterative: after first hour we could update last_obs with prediction, but keep simple here
    df = pd.DataFrame(rows)
    # time features
    df['_hour'] = df['_ts'].dt.hour
    df['_dow'] = df['_ts'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['_hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['_hour'] / 24.0)
    # merge weather
    if not met_df.empty:
        # align met_df index timezone
        try:
            met = met_df.copy()
            met.index = pd.to_datetime(met.index, utc=True)
            df = df.merge(met, left_on='_ts', right_index=True, how='left')
        except Exception:
            pass
    # select features expected by model
    if hasattr(model, 'feature_name_'):
        feat_names = list(model.feature_name_)
    elif hasattr(model, 'feature_names_in_'):
        feat_names = list(model.feature_names_in_)
    else:
        # fallback to typical features
        feat_names = ['hour_sin','hour_cos','_dow','lag1','lag2','rolling_mean_3'] + [c for c in df.columns if c.startswith('met_')]
    X = df.reindex(columns=feat_names).fillna(-1)
    preds = None
    try:
        preds = model.predict(X)
    except Exception:
        # try sklearn wrapper attribute
        try:
            preds = model.predict(X.values)
        except Exception as e:
            st.error(f'Prediction failed: {e}')
            preds = np.full(len(df), np.nan)
    df['pred_speed'] = preds
    # propagate free_flow into df if present
    if 'free_flow' in df.columns:
        try:
            df['free_flow'] = df['free_flow'].astype(float)
        except Exception:
            pass
    return df


def color_for_speed(s):
    # map speed to color (0-120 km/h)
    vmax = 120.0
    v = max(0.0, min(vmax, float(s) if not pd.isna(s) else 0.0))
    # green->yellow->red
    r = int(255 * min(1, max(0, (vmax - v) / vmax * 2)))
    g = int(255 * min(1, max(0, v / vmax * 2)))
    b = 0
    return [r, g, b]


def color_for_intensity(intensity):
    """Map congestion intensity (0..1) to color: 0=green, 1=red"""
    i = max(0.0, min(1.0, float(intensity) if not pd.isna(intensity) else 0.0))
    # interpolate green->yellow->red
    if i < 0.5:
        # green to yellow
        t = i / 0.5
        r = int(255 * t)
        g = int(255)
    else:
        # yellow to red
        t = (i - 0.5) / 0.5
        r = 255
        g = int(255 * (1 - t))
    return [r, g, 0]


def main():
    st.set_page_config(layout='wide')
    st.title('Interactive Traffic Predictions (Streamlit prototype)')

    region = st.sidebar.text_input('TomTom region folder', 'eindhoven')
    use_live = st.sidebar.checkbox('Use live TomTom API', value=False, help='If enabled, app will call TomTom flowSegmentData for each point (requires API key in `config.py`).')

    model = load_model()
    if model is None:
        st.warning('Could not find model at `jupyter/model_eindhoven.joblib`. Please place your joblib model there.')

    points = load_latest_tomtom(region=region)
    if points.empty:
        st.warning('No TomTom observations found in data folder. Place JSONs under data/tomtom/<region>/ or enable live mode.')
    else:
        st.write(f'Loaded {len(points)} unique observation points')

    # if live mode is selected, try to fetch current speeds using TomTom API key from config
    if use_live:
        cfg = load_config()
        api_key = getattr(cfg, 'api_key', None) if cfg is not None else None
        if api_key:
            st.sidebar.write('Using TomTom API key from `config.py`')
            points = fetch_live_tomtom_for_points(points, api_key)
            st.write('Updated points with live speeds where available')
        else:
            st.sidebar.warning('No `api_key` found in `config.py`. Live fetching disabled.')

    # Meteostat: ensure today's data
    now = pd.Timestamp.utcnow().date()
    start = now.isoformat()
    end = now.isoformat()
    if st.sidebar.button('Ensure Meteostat for today'):
        if not points.empty:
            lat = points['lat'].median()
            lon = points['lon'].median()
            met_df = ensure_meteostat(lat, lon, start, end)
            st.sidebar.write('Meteostat rows:', len(met_df))
        else:
            st.sidebar.write('No points available to choose location for Meteostat')
    else:
        met_df = pd.DataFrame()

    hours_remaining = int((pd.Timestamp.utcnow().ceil('h').normalize() + pd.Timedelta(days=1) - pd.Timestamp.utcnow().ceil('h')).total_seconds() // 3600)
    hours = st.sidebar.slider('Hours ahead to compute', min_value=1, max_value=24, value=6)

    # attempt to discover segment geometries (optional)
    geom_map = load_segment_geometries(region=region)
    if geom_map:
        st.sidebar.write(f'Found {len(geom_map)} segment geometries')

    if points.empty or model is None:
        st.stop()

    df_preds = build_prediction_grid(points, model, met_df, start_hour=0, hours=hours)

    # prepare hour labels
    hours_list = sorted(df_preds['_ts'].unique())
    hour_idx = st.slider('Select hour', 0, len(hours_list)-1, 0)
    sel_ts = hours_list[hour_idx]
    st.write('Selected hour (UTC):', sel_ts)

    df_sel = df_preds[df_preds['_ts'] == sel_ts].copy()
    import pydeck as pdk
    # compute intensity metrics relative to free flow when available
    df_sel['free_flow'] = df_sel.get('free_flow')
    df_sel['intensity_pred'] = df_sel.apply(lambda r: (1 - (r.get('pred_speed', np.nan) / r.get('free_flow'))) if (pd.notna(r.get('free_flow')) and r.get('free_flow') not in (0, None) and pd.notna(r.get('pred_speed'))) else 0.0, axis=1)
    df_sel['intensity_obs'] = df_sel.apply(lambda r: (1 - (r.get('target_speed', np.nan) / r.get('free_flow'))) if (pd.notna(r.get('free_flow')) and r.get('free_flow') not in (0, None) and pd.notna(r.get('target_speed'))) else 0.0, axis=1)
    # clamp
    df_sel['intensity_pred'] = df_sel['intensity_pred'].clip(0,1)
    df_sel['intensity_obs'] = df_sel['intensity_obs'].clip(0,1)
    df_sel['color'] = df_sel['intensity_pred'].apply(color_for_intensity)

    # If segment geometries are available and match pkeys, draw lines (PathLayer), else scatter
    # geometry map keys are matched against '_pkey' values in points
    matched_paths = []
    path_rows = []
    for _, r in df_sel.iterrows():
        pkey = r['_pkey']
        if pkey in geom_map:
            coords = geom_map[pkey]
            # ensure coords are [lon,lat] pairs
            path_rows.append({'path': coords, 'pred_speed': r['pred_speed'], '_pkey': pkey})
    if path_rows:
        path_df = pd.DataFrame(path_rows)
        layer = pdk.Layer(
            'PathLayer',
            path_df,
            get_path='path',
            get_color='[int(255*(1-(pred_speed/120))), int(255*(pred_speed/120)), 0]',
            width_scale=10,
            width_min_pixels=2
        )
        view_state = pdk.ViewState(latitude=float(df_sel['lat'].mean()), longitude=float(df_sel['lon'].mean()), zoom=12)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={'text':'Speed: {pred_speed} km/h'})
        st.pydeck_chart(r)
    else:
        layer = pdk.Layer(
            'ScatterplotLayer',
            df_sel,
            get_position='[lon, lat]',
            get_fill_color='color',
            get_radius=20,
            pickable=True
        )
        view_state = pdk.ViewState(latitude=float(df_sel['lat'].mean()), longitude=float(df_sel['lon'].mean()), zoom=12)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={'text':'Speed: {pred_speed} km/h'})
        st.pydeck_chart(r)

    st.dataframe(df_sel[['lat','lon','pred_speed','_pkey']].head(200))
    st.markdown('''
    **Segment geometry format (for best visuals)**

    - Preferred: GeoJSON FeatureCollection where each Feature has a LineString geometry and a property that identifies the segment (e.g. `segment_id` or `_pkey`). Example properties: `{"type":"Feature","properties":{"_pkey":"51.4416_5.4697"},"geometry":{"type":"LineString","coordinates":[[5.4697,51.4416],[5.4700,51.4420]]}}`.
    - Acceptable alternatives: per-record JSON with a `geometry`/`shape` field that contains an array of `[lon,lat]` coordinates and an identifier in `properties` or top-level fields (e.g. `segment_id`).
    - If you cannot provide segment geometries, the app will plot scatter points at the sample lat/lon.
    ''')


if __name__ == '__main__':
    main()
