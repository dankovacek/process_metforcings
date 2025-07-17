"""Access the Daymet database for both single single pixel and gridded queries."""

# pyright: reportArgumentType=false,reportCallIssue=false,reportReturnType=false
from __future__ import annotations

# generate basins
import os
from time import time
from pyproj import CRS
import multiprocessing as mp
import warnings

from shapely.geometry import Point

warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
# import multiprocessing as mp
import xarray as xr
from rioxarray.exceptions import NoDataInBounds
from shapely.geometry import box
import itertools

import re
from typing import TYPE_CHECKING, Callable, Literal
from urllib.parse import urlencode

import utils

import numpy as np
import pandas as pd
import xarray as xr


from pydaymet.core import T_RAIN, T_SNOW, Daymet, separate_snow
from pydaymet.exceptions import (
    InputRangeError,
    InputTypeError,
    MissingDependencyError,
    ServiceError,
)
from pydaymet.pet import potential_et

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import pyproj
    from shapely import Polygon

    CRSType = int | str | pyproj.CRS
    PETMethods = Literal["penman_monteith", "priestley_taylor", "hargreaves_samani"]

DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"

__all__ = ["get_bycoords", "get_bygeom", "get_bystac"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# region_gdf = gpd.read_file(os.path.join(BASE_DIR, "input_data/BCUB_regions_merged_R0.geojson"))
# region_codes = sorted(list(set(region_gdf["region_code"].values)))
region_codes = [
    "08A",
    "08B",
    "08C",
    "08D",
    "08E",
    "08F",
    "08G",
    "10E",
    "CLR",
    "ERK",
    "FRA",
    "HAY",
    "HGW",
    "LRD",
    "PCR",
    "VCI",
    "WWA",
    "YKR",
]
# print(region_codes)

#########################
# input file paths
#########################

# i'm using a temp folder on an external SSD because the dataset is huge
# daymet_tile_dir = "/media/danbot/2023_1TB_T7"
daymet_tile_dir = "/media/danbot2/T7_2TB"
daymet_tile_dir = "/media/danbot2/easystore/daymet_tiles/"

point_crs = CRS.from_epsg(4326)  # WGS 84

# catchment geometry files
# revision date
rev_date = '20250227'
catchment_fpath = os.path.join(BASE_DIR, 'input_data', f'BCUB_watershed_attributes_updated_{rev_date}.geojson')

# daymet_tile_dir = os.path.join(BASE_DIR, "input_data/DAYMET/")
# daymet_output_dir = os.path.join(BASE_DIR, "processed_stns/")
# daymet_output_dir = '/media/danbot2/easystore/PNW_catchment_mean_met_forcings_20250320/'
daymet_output_dir = os.path.join(BASE_DIR, "PNW_catchment_mean_met_forcings_20250320/")
daymet_mean_output_dir = os.path.join(BASE_DIR, "PNW_catchment_mean_met_forcings_20250320_backup/")
daymet_mean_output_dir = '/home/danbot2/code_5820/neuralhydrology/bcub_test/bcub_data/bcub/time_series'

# daymet_tile_dir = os.path.join(BASE_DIR, "input_data/DAYMET/")
daymet_crs = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

daymet_tile_index_path = os.path.join(BASE_DIR, "input_data/Daymet_v4_Tiles.geojson")
dm_tiles = gpd.read_file(daymet_tile_index_path)

tiles_wkt = gpd.read_file(daymet_tile_index_path).crs.to_wkt()

# compute the full time series or just the catchment mean (False)
compute_catchment_mean = True

# region_polygon = gpd.read_file(mask_path)
sample_fpath = "input_data/BCUB_watershed_bounds_sample.geojson"
sample = False
if sample:
    catchment_gdf = gpd.read_file(os.path.join(BASE_DIR, sample_fpath))
else:
    catchment_gdf = gpd.read_file(catchment_fpath)
    # catchment_gdf = catchment_gdf.sort_values("Drainage_Area_km2").reset_index(drop=True)
    # sample = catchment_gdf[:2].copy()
    # sample.to_file(sample_fpath, driver="GeoJSON")
    # print(asdfasd)

existing_files = os.listdir(daymet_mean_output_dir)
existing_stns = [f.split('_')[0] for f in existing_files if f.endswith('.nc')]

daymet_params = ["tmax", "tmin", "vp", "prcp", "swe", "srad", "dayl"]

catchment_gdf = catchment_gdf.sort_values("Drainage_Area_km2").reset_index(drop=True)
stn_polygons = catchment_gdf[["Official_ID", "Centroid_Lat_deg_N", "Centroid_Lon_deg_E", "geometry"]].copy()
catchment_locations_file = os.path.join(BASE_DIR, 'input_data', "Catchment_polygons.geojson")
if not os.path.exists(catchment_locations_file):
    stn_polygons = stn_polygons[stn_polygons['Official_ID'].isin(existing_stns)]
    stn_polygons.to_file(catchment_locations_file, driver="GeoJSON")

# define the metadata for variables
variable_metadata = {
    "dayl":  {'long_name': 'daylength', 'units': 's', 'cell_methods': 'area: mean', },
    "prcp":  {'long_name': 'daily total precipitation', 'units': 'mm day-1', 'cell_methods': 'area: mean time: sum', },
    "srad":  {'long_name': 'daylight average incident shortwave radiation', 'units': 'W m-2', 'cell_methods': 'area: mean time: mean', },
    "swe":   {'long_name': 'snow water equivalent', 'units': 'kg m-2', 'cell_methods': 'area: mean time: mean', },
    "tmax":  {'long_name': 'daily maximum temperature', 'units': 'degrees C', 'cell_methods': 'area: mean time: maximum', },
    "tmin":  {'long_name': 'daily minimum temperature', 'units': 'degrees C', 'cell_methods': 'area: mean time: minimum', },
    "vp":    {'long_name': 'daily average vapor pressure', 'units': 'Pa', 'cell_methods': 'area: mean time: mean', },
    "pet":   {'long_name': 'potential evapotranspiration', 'units': 'mm day-1', 'cell_methods': 'area: mean time: sum', },
    "streamflow": {'long_name': 'daily average streamflow', 'units': 'm3 s-1', 'cell_methods': 'area: mean time: mean', },
    "elevation":   {'long_name': 'elevation above sea level', 'units': 'm', 'cell_methods': 'area: mean time: mean', }
}

def get_covering_daymet_tile_ids(polygon):

    # import the region polygon and reproject it to the daymet projection
    polygon.to_crs(dm_tiles.crs, inplace=True)

    # get the intersection with the region polygon
    tiles_df = dm_tiles.sjoin(
        polygon[["geometry"]], how="inner", predicate="intersects"
    )
    tiles_df = tiles_df.sort_values(by=["Latitude (Min)", "Longitude (Min)"])
    tile_ids = sorted(list(set(tiles_df["TileID"].values)))
    # print(f"   ...There are {len(tile_ids)} tiles covering the polygon.")
    # print(tile_ids)
    return tile_ids


def get_clm_filepaths(tile_ids, dates_itr, variables):
    all_files = []
    missing_files = []
    for year in dates_itr:
        for var in variables:
            for tile_id in tile_ids:
                file_name = f"{tile_id}_{year}_{var}.nc"
                file_path = os.path.join(daymet_tile_dir, var, file_name)
                if os.path.exists(file_path):
                    all_files.append(file_path)
                else:
                    print(f"File {file_name} not found.")
                    missing_files.append(file_name)
                    # raise Exception("Missing files")

    return all_files, missing_files


def make_transform_func(geometry, geom_crs, daymet_crs_obj):
    """
    Returns a function that will write the CRS, reproject, and clip a dataset.

    Parameters:
      geometry: the geometry (or polygon) to clip to.
      in_crs: the input CRS (e.g. as used in your GeoDataFrame).
      out_crs: the desired output CRS (e.g., "EPSG:4326").
      resolution: target resolution (e.g., 1000).
      daymet_crs: the CRS you want to write into the dataset.
    """
    def transform_func(ds):
        # Write the proper CRS metadata into the dataset
        original_time = ds['time'].copy()
        ds = ds.drop_vars(['time_bnds'])
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds = utils.write_crs(ds, daymet_crs_obj)
        # Create a GeoDataFrame for the geometry and reproject it if needed
        g = gpd.GeoDataFrame(geometry=[geometry], crs=geom_crs)
        g = g.to_crs(daymet_crs_obj.to_proj4())
        # Clip the dataset using your clip function
        ds = utils.clip_dataset(ds, g.geometry[0], g.crs)
        # ds = ds.assign_coords(time=original_time)
        return ds
    
    return transform_func


def read_netcdf_files(files, geometry, crs, daymet_crs_obj, dim='time'):
    def process_one_path(path, transform_func=None):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path, cache=False) as ds:
            # transform_func should do some sort of selection or aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # remove the tileid and start_year variables
            # ds.attrs.pop('tileid', None)
            ds.attrs.pop('start_year', None)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    tf = make_transform_func(geometry, crs, daymet_crs_obj)
    datasets = [process_one_path(p, transform_func=tf) for p in files]
    merged_ds = xr.merge(datasets, join='outer', compat='no_conflicts')
    merged_ds = merged_ds.rio.write_crs(crs)
    return merged_ds
    # return xr.combine_by_coords(datasets)


def get_crs_obj(f):
    # check the crs of one of the files
    test = xr.open_dataset(f)
    attrs = dict(test.variables["lambert_conformal_conic"].attrs)
    std_par = np.atleast_1d(attrs.pop("standard_parallel"))
    attrs["lat_1"] = float(std_par[0])
    attrs["lat_2"] = float(std_par[1])

    # Rename keys to what PROJ expects.
    attrs["lat_0"] = float(attrs.pop("latitude_of_projection_origin"))
    attrs["lon_0"] = float(attrs.pop("longitude_of_central_meridian"))
    attrs["x_0"] = float(attrs.pop("false_easting"))
    attrs["y_0"] = float(attrs.pop("false_northing"))
    attrs["a"]   = float(attrs.pop("semi_major_axis"))
    attrs["rf"]  = float(attrs.pop("inverse_flattening"))

    attrs.update({
        "proj": "lcc",  # Lambert Conformal Conic
        "units": "m",
        "no_defs": True,
    })
    return CRS.from_dict(attrs)


def download_daymet_tiles(tile_ids, param, years):
    """
    2025-02-12: Thredds base url format changed to the following:
    https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/2129/tiles/${i}/${j}_${i}/${var}.nc
    Where "i" is 4-digit year, "j" is the tile id, and "var" is the variable name.
    see example here:
    https://github.com/ornldaac/daymet-TDStiles-batch/blob/master/Bash/daymet_tile-nc-retrieval.sh
    """
    print(f"Downloading DAYMET {param} data.")
    daymet_url_base = (
        "https://thredds.daac.ornl.gov/thredds/fileServer/ornldaac/2129/tiles/"
    )
    # base_command = f"wget -q --show-progress --progress=bar:force --limit-rate=3m {daymet_url_base}"
    base_command = f"wget -q --show-progress --progress=bar:force --limit-rate=3m {daymet_url_base}"
    batch_commands = []
    new_paths = []
    for yr in years:
        for tile in tile_ids:
            fpath = os.path.join(daymet_tile_dir, f"{param}/{tile}_{yr}_{param}.nc")
            if not os.path.exists(fpath):
                cmd = base_command + f"{yr}/{tile}_{yr}/{param}.nc -O {fpath}"
                batch_commands.append(cmd)
                new_paths.append(fpath)

    # download the files in parallel
    
    commands = sorted(list(set(batch_commands)))
    # print(commands[0])
    # print(asfsd)
    with mp.Pool(6) as pl:
        print(f"   ...{len(batch_commands)} daymet {param} files remaining to download.")
        pl.map(os.system, commands)
    return new_paths


def _get_filename(
    region: str,
) -> dict[int, Callable[[str], str]]:
    """Get correct filenames based on region and variable of interest."""
    return {
        2129: lambda v: f"daily_{region}_{v}",
        2131: lambda v: f"{v}_monttl_{region}" if v == "prcp" else f"{v}_monavg_{region}",
        2130: lambda v: f"{v}_annttl_{region}" if v == "prcp" else f"{v}_annavg_{region}",
    }


def get_bygeom(
    geometry: Polygon | tuple[float, float, float, float],
    dates: tuple[str, str] | int | list[int],
    crs: CRSType = 4326,
    variables: Iterable[Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]]
    | Literal["tmin", "tmax", "prcp", "srad", "vp", "swe", "dayl"]
    | None = None,
    region: Literal["na", "hi", "pr"] = "na",
    time_scale: Literal["daily", "monthly", "annual"] = "daily",
    pet: PETMethods | None = None,
    pet_params: dict[str, float] | None = None,
    snow: bool = False,
    snow_params: dict[str, float] | None = None,
) -> xr.Dataset:
    """Get gridded data from the Daymet database at 1-km resolution.

    Parameters
    ----------
    geometry : Polygon or tuple
        The geometry of the region of interest. It can be a shapely Polygon or a tuple
        of length 4 representing the bounding box (minx, miny, maxx, maxy).
    dates : tuple or list
        Start and end dates as a tuple (start, end) or a list of years [2001, 2010, ...].
    crs : str, int, or pyproj.CRS, optional
        The CRS of the input geometry, defaults to epsg:4326.
    variables : str or list
        List of variables to be downloaded. The acceptable variables are:
        ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        Descriptions can be found `here <https://daymet.ornl.gov/overview>`__.
    region : str, optional
        Region in the US, defaults to na. Acceptable values are:

        * na: Continental North America
        * hi: Hawaii
        * pr: Puerto Rico

    time_scale : str, optional
        Data time scale which can be daily, monthly (monthly average),
        or annual (annual average). Defaults to daily.
    pet : str, optional
        Method for computing PET. Supported methods are
        ``penman_monteith``, ``priestley_taylor``, ``hargreaves_samani``, and
        None (don't compute PET). The ``penman_monteith`` method is based on
        :footcite:t:`Allen_1998` assuming that soil heat flux density is zero.
        The ``priestley_taylor`` method is based on
        :footcite:t:`Priestley_1972` assuming that soil heat flux density is zero.
        The ``hargreaves_samani`` method is based on :footcite:t:`Hargreaves_1982`.
        Defaults to ``None``.
    pet_params : dict, optional
        Model-specific parameters as a dictionary, defaults to ``None``. Valid
        parameters are:

        * ``penman_monteith``: ``soil_heat_flux``, ``albedo``, ``alpha``,
          and ``arid_correction``.
        * ``priestley_taylor``: ``soil_heat_flux``, ``albedo``, and ``arid_correction``.
        * ``hargreaves_samani``: None.

        Default values for the parameters are: ``soil_heat_flux`` = 0, ``albedo`` = 0.23,
        ``alpha`` = 1.26, and ``arid_correction`` = False.
        An important parameter for ``priestley_taylor`` and ``penman_monteith`` methods
        is ``arid_correction`` which is used to correct the actual vapor pressure
        for arid regions. Since relative humidity is not provided by Daymet, the actual
        vapor pressure is computed assuming that the dewpoint temperature is equal to
        the minimum temperature. However, for arid regions, FAO 56 suggests subtracting
        the minimum temperature by 2-3 °C to account for aridity, since in arid regions,
        the air might not be saturated when its temperature is at its minimum. For such
        areas, you can pass ``{"arid_correction": True, ...}`` to subtract 2 °C from the
        minimum temperature before computing the actual vapor pressure.
    snow : bool, optional
        Compute snowfall from precipitation and minimum temperature. Defaults to ``False``.
    snow_params : dict, optional
        Model-specific parameters as a dictionary that is passed to the snowfall function.
        These parameters are only used if ``snow`` is ``True``. Two parameters are required:
        ``t_rain`` (deg C) which is the threshold for temperature for considering rain and
        ``t_snow`` (deg C) which is the threshold for temperature for considering snow.
        The default values are ``{'t_rain': 2.5, 't_snow': 0.6}`` that are adopted from
        https://doi.org/10.5194/gmd-11-1077-2018.

    Returns
    -------
    xarray.Dataset
        Daily climate data within the target geometry.

    Examples
    --------
    >>> from shapely import Polygon
    >>> import pydaymet as daymet
    >>> geometry = Polygon(
    ...     [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    ... )
    >>> clm = daymet.get_bygeom(geometry, 2010, variables="tmin", time_scale="annual")
    >>> clm["tmin"].mean().item()
    1.361

    References
    ----------
    .. footbibliography::
    """
    daymet = Daymet(variables, pet, snow, time_scale, region)
    daymet.check_dates(dates)

    if isinstance(dates, tuple):
        dates_itr = daymet.dates_tolist(dates)
    else:
        dates_itr = daymet.years_tolist(dates)

    geom_crs = utils.validate_crs(crs)
    
    _geometry = utils.to_geometry(geometry, geom_crs, 4326)

    tile_ids = get_covering_daymet_tile_ids(gpd.GeoDataFrame(geometry=[geometry], crs=geom_crs))
    clm_filepaths, missing_files = get_clm_filepaths(tile_ids, dates, variables)

    if len(missing_files) > 0:
        print('   ... downloading missing files')
        vars = [e.split("_")[2].split(".")[0] for e in missing_files]
        years = list(range(1980, 2024))
        for v in vars:
            new_paths = download_daymet_tiles(tile_ids, v, years)
            clm_filepaths += new_paths

    daymet_crs_obj = get_crs_obj(clm_filepaths[0])

    # Collect all NetCDF files that match any tile_id
    clm = read_netcdf_files(sorted(clm_filepaths), geometry, geom_crs, daymet_crs_obj)

    if snow:
        params = {"t_rain": T_RAIN, "t_snow": T_SNOW} if snow_params is None else snow_params
        clm = separate_snow(clm, **params)

    if pet:
        clm = potential_et(clm, method=pet, params=pet_params)

    clm["time"] = pd.DatetimeIndex(pd.to_datetime(clm["time"]).date)
    clm = clm.drop_vars('yearday')
    return clm


def reformat_dataset(ds):
    # 1. Define time and point location
    for v in ds.data_vars:
        if v not in variable_metadata:
            # drop the variable if it is not in the metadata
            ds.drop_vars(v)
            continue
        ds[v].attrs.update(variable_metadata[v])
    # add global attributes
    title = 'Processed Catchment-Averaged Meteorological Forcings from Daymet for Streamflow Monitored Catchments in British Columbia and Transboundary Basins'
    ds.attrs.update({
        'title': title, 
        'institution': 'University of British Columbia',
        'source': 'Daymet, HYSETS',
        'history': f'Processed on 2025-03-20',
        'references': ['https://daymet.ornl.gov/overview', 'https://osf.io/rpc3w/'],
        'comment': 'This dataset contains catchment-averaged meteorological forcings from Daymet and streamflow from USGS and ECCC for streamflow monitored catchments in British Columbia and Transboundary Basins.',
        'creator_name': 'Dan Kovacek, P.Eng. dkovacek@mail.ubc.ca',
        'keywords': [
            'Daymet (climate forcings)',
            'catchment',
            'meteorological forcings',
            'potential evapotranspiration',
            'precipitation',
            'temperature'
        ],
        'featureType': 'timeSeries',
        'cdm_data_type': 'Station',
        'source': 'Hydrometric data from USGS National Water Information Service, ECCC Water Survey Canada. Meteorological data from Daymet. Catchment polygons from ECCC HYDAT and USGS.'
    })
    return ds



for i, catchment_data in catchment_gdf.iterrows():
    # polygon = gpd.GeoDataFrame(geometry=[catchment_data['geometry'].exterior], crs=catchment_gdf.crs)
    catchment = catchment_gdf.loc[[i]].copy()
    area = catchment.geometry.area.values[0] / 1.0e6
    da = catchment_data["Drainage_Area_km2"]
    # lat, lon = catchment_data['Centroid_Lat_deg_N'], catchment_data['Centroid_Lon_deg_E']
    # assert np.isclose(da, area, rtol=5e-2), f"Drainage area mismatch > 5%: {da:.1f} vs {area:.1f}"
    # pass
    
    stn = catchment_data["Official_ID"]
    out_fpath = os.path.join(daymet_output_dir, f"{stn}_daymet.nc")
    mean_fpath = os.path.join(daymet_mean_output_dir, f'{stn}.nc')

    if compute_catchment_mean:
        if os.path.exists(mean_fpath):
            # check that latitude and longitude are explicit coordinate variables, 
            # and format the file as a point-feature NetCDF
            ds = xr.open_dataset(mean_fpath)
            ds_new = reformat_dataset(ds)
            # drop 'spatial_ref' coordinate variable if it exists
            if 'spatial_ref' in ds_new.coords:
                ds_new = ds_new.drop_vars('spatial_ref')

            # drop 'lambert_conformal_conic' variable if it exists
            if 'lambert_conformal_conic' in ds_new.variables:
                ds_new = ds_new.drop_vars('lambert_conformal_conic')

            # ds_new.rio.write_crs('EPSG:4326', inplace=True)  # write the CRS to the dataset
            ds_new.to_netcdf(out_fpath, format="NETCDF4", engine="netcdf4")
            print(f'    ...{out_fpath} written.')
            print(f'    ...{stn} mean already processed.')
            continue

    t0 = time()
    # print(asdfsd)
        
    # if da < 2:
    #     print('Area less than 2 km^2, using convex hull since Daymet resolution is 1km..')
    #     catchment.geometry = catchment.convex_hull
    # elif da < 10:
    #     polygon = catchment_gdf.loc[[i]].simplify(100)
    # else:
    #     polygon = catchment_gdf.loc[[i]].simplify(200)

    # all_params = []
    # for year in list(range(1980, 2024)):
    #     ds = get_bygeom(
    #         geometry=catchment.geometry.values[0],
    #         variables=daymet_params,
    #         dates=[year],
    #         crs=catchment_gdf.crs,
    #         region='na',
    #         time_scale="daily",
    #         pet='penman_monteith',
    #     )
    #     if compute_catchment_mean:
    #         # compute spatial mean
    #         ds = ds.mean(dim=['x', 'y'], skipna=True)
    #     all_params.append(ds)
    #     if year % 10 == 0:
    #         t1 = time()
    #         print(f"    Time to process {year}: {t1-t0:.0f}s")

    # if compute_catchment_mean:
    #     merged_ds = xr.concat(all_params, dim='time')
    #     # merged_ds = merged_ds.rio.write_crs('EPSG:4326')  # if needed, for georeferencing
    #     # merged_ds = merged_ds.rio.write_crs(daymet_crs)
    #     out_fpath = os.path.join(daymet_mean_output_dir, f'{stn}_daymet.nc')
    # else:
    #     # check dimensions for consistency
    #     base_vals = all_params[0].lat.values
    #     if np.all(np.allclose(base_vals, d.lat.values) for d in all_params[1:]):
    #         merged_ds = xr.concat(all_params, dim='time')
    #         merged_ds = merged_ds.rio.write_crs(daymet_crs)
    #     else:
    #         print('latitudes are not the same')
    #         continue

    # merged_ds.to_netcdf(out_fpath)
    # t2 = time()
    # print(f'    Time to process and write {stn}: {(t2-t0)/60:.1f}min.')
    # del merged_ds
