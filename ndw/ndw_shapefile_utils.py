"""
NDW shapefile utilities — load shapefiles directly from URLs.

Dependencies:
    pip install geopandas shapely fiona requests

Functions:
    load_shapefile_from_url(url: str) -> GeoDataFrame
"""

import io
import os
import zipfile
import tempfile
import requests
import geopandas as gpd


def load_shapefile_from_url(
    url: str = "https://opendata.ndw.nu/ndw_msi_shapefiles_latest.zip",
) -> gpd.GeoDataFrame:
    """
    Download an NDW shapefile ZIP from a URL and load it into GeoPandas.

    Parameters
    ----------
    url : str
        Direct URL to the NDW shapefile ZIP.
        Default = MSI shapefiles:
        https://opendata.ndw.nu/ndw_msi_shapefiles_latest.zip

    Returns
    -------
    gpd.GeoDataFrame
        The loaded shapefile as a GeoDataFrame.
    """

    # --- Download ZIP into memory ---
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    zip_bytes = io.BytesIO(resp.content)

    # --- Extract ZIP to temporary directory ---
    tmpdir = tempfile.mkdtemp(prefix="ndw_zip_")

    with zipfile.ZipFile(zip_bytes) as zf:
        zf.extractall(tmpdir)

    # --- Locate .shp inside the extracted folder ---
    shapefile_path = None
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.lower().endswith(".shp"):
                shapefile_path = os.path.join(root, f)
                break
        if shapefile_path:
            break

    if shapefile_path is None:
        raise FileNotFoundError("No .shp file found inside downloaded NDW ZIP.")

    # --- Load shapefile with GeoPandas ---
    gdf = gpd.read_file(shapefile_path)

    # NDW MSI shapefiles use WGS84 (lat/lon)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    return gdf

def load_speed_locations():
    url = "https://opendata.ndw.nu/NDW_AVG_Meetlocaties_Shapefile.zip"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))

    tmp = tempfile.mkdtemp()

    z.extractall(tmp)

    # find the .shp file
    shp = None
    for root, _, files in os.walk(tmp):
        for f in files:
            if f.endswith(".shp"):
                shp = os.path.join(root, f)
                break

    return gpd.read_file(shp)