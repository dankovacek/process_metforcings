# generate basins
import os
from time import time
import dask

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import multiprocessing as mp
import xarray as xr
from rioxarray.exceptions import NoDataInBounds
from shapely.geometry import box
import glob

from utils import 

# import rioxarray as rxr

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
daymet_tile_dir = "/media/danbot/T7_2TB"

# daymet_tile_dir = os.path.join(BASE_DIR, "input_data/DAYMET/")
daymet_output_dir = os.path.join(BASE_DIR, "processed_stns/")

# daymet_tile_dir = os.path.join(BASE_DIR, "input_data/DAYMET/")
daymet_proj = "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

daymet_tile_index_path = os.path.join(BASE_DIR, "input_data/Daymet_v4_Tiles.geojson")
dm_tiles = gpd.read_file(daymet_tile_index_path)

# masks used to clip the geospatial layers
mask_path = os.path.join(BASE_DIR, "input_data/BCUB_regions_merged_R0.geojson")
reproj_bounds_path_4326 = os.path.join(
    BASE_DIR, "input_data/region_bounds/convex_hull_4326.shp"
)
reproj_bounds_path_4269 = os.path.join(
    BASE_DIR, "input_data/region_bounds/convex_hull_4269.shp"
)
reproj_bounds_path_3005 = os.path.join(
    BASE_DIR, "input_data/region_bounds/convex_hull_3005.shp"
)
reproj_bounds_daymet = os.path.join(
    BASE_DIR, "input_data/region_bounds/region_polygon_daymet_crs.shp"
)

if not os.path.exists(reproj_bounds_path_3005):
    print("   ...reading region polygon file.")
    mask = gpd.read_file(mask_path)
    print("   ...reprojecting region polygon to different CRS.")
    mask_gdf = gpd.GeoDataFrame(geometry=[mask.unary_union], crs=mask.crs)
    mask_reprojected = mask_gdf.copy().to_crs("EPSG:4269")
    # dissolve the mask to a single polygon
    # mask = mask.dissolve()
    mask_reprojected.geometry = mask_reprojected.convex_hull
    mask_reprojected.to_file(reproj_bounds_path_4269)
    # if not os.path.exists(reproj_bounds_path_4326):
    # mask = gpd.read_file(mask_path).dissolve()
    mask_reprojected = mask_gdf.to_crs("EPSG:4326")
    # mask.geometry = mask.convex_hull
    mask_reprojected.to_file(reproj_bounds_path_4326)

    mask_reprojected = mask_gdf.to_crs("EPSG:3005")
    mask_reprojected.geometry = mask_reprojected.convex_hull
    # mask.geometry = mask.convex_hull
    mask_reprojected.to_file(reproj_bounds_path_3005)
    # reproject the region bounds to the daymet projection
    # if not os.path.exists(reproj_bounds_daymet):
    # daymet_mask = gpd.read_file(mask_path).to_crs(daymet_proj)
    mask_reprojected.geometry = mask_reprojected.convex_hull
    mask_reprojected.to_file(reproj_bounds_daymet)
    print(
        f"   ...region polygon files saved to {reproj_bounds_path_3005} and {reproj_bounds_daymet}."
    )


def get_covering_daymet_tile_ids(polygon):

    # import the region polygon and reproject it to the daymet projection
    # get the intersection with the region polygon
    # reproject the region polygon to the daymet tiles projection
    polygon = polygon.to_crs(dm_tiles.crs)
    
    tiles_df = dm_tiles.sjoin(
        polygon[["geometry"]], how="inner", predicate="intersects"
    )
    tiles_df = tiles_df.sort_values(by=["Latitude (Min)", "Longitude (Min)"])
    tile_ids = sorted(list(set(tiles_df["TileID"].values)))
    print(f"   ...There are {len(tile_ids)} tiles covering the polygon.")
    return tile_ids


def download_daymet_tiles(param, years):
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
    for yr in years:
        for tile in tile_ids:
            fpath = os.path.join(daymet_tile_dir, f"{param}/{tile}_{yr}_{param}.nc")
            if not os.path.exists(fpath):
                cmd = base_command + f"{yr}/{tile}_{yr}/{param}.nc -O {fpath}"
                batch_commands.append(cmd)

    # download the files in parallel
    print(f"   ...{len(batch_commands)} daymet {param} files remaining to download.")
    commands = sorted(list(set(batch_commands)))
    with mp.Pool(6) as pl:
        pl.map(os.system, commands)

def get_clm_files(tile_ids, years, var):
    all_files = []
    for year in years:
        for tile_id in tile_ids:
            file_name = f"{tile_id}_{year}_{var}.nc"
            file_path = os.path.join(daymet_tile_dir, var, file_name)
            if os.path.exists(file_path):
                all_files.append(file_path)
            else:
                print(f"File {file_name} not found.")
    return all_files


def retrieve_tiles_by_ids(param, tile_ids, year, clip_mask=None):
    """
    Load and merge all tiles associated with a list of tile IDs.

    Args:
        param (str): Name of the climate variable (e.g., 'prcp', 'tmax').
        tile_ids (list): List of tile IDs to load.

    Returns:
        xarray.Dataset: Merged dataset covering the polygon.
    """
    param_folder = os.path.join(daymet_tile_dir, param)

    # Collect all NetCDF files that match any tile_id
    clm_files = sorted(
        [
            os.path.join(param_folder, e)
            for e in os.listdir(param_folder)
            if (e.split("_")[0] in tile_ids) and e.endswith(f"{year}_{param}.nc")
        ]
    )
    if not clm_files:
        raise Exception("No files found.")
    # Load the dataset for this tile and store it
    chunk_info = {"time": 10, "x": 50, "y": 50}
    data = xr.open_mfdataset(
        clm_files, concat_dim="time", chunks=chunk_info, combine="nested", parallel=True
    )[param]
    data.rio.write_nodata(np.nan, inplace=True)
    data.rio.write_crs(daymet_proj, inplace=True)
    return data


# def clip_files(ds, polygon):
#     """Clips dataset to the given polygon and computes spatial mean."""
#     polygon = polygon.to_crs(daymet_proj)
#     ds = xr.where(ds > -9999, ds, np.nan, keep_attrs=True)
#     for v in ds:
#         ds[v].rio.write_nodata(np.nan, inplace=True)
#     if "spatial_ref" in ds:
#         ds = ds.drop_vars("spatial_ref")
#     clm = utils.write_crs(ds, daymet_proj)
#     try:
#         ds.write_nodata(np.nan, inplace=True)
#         ds.write_crs(daymet_proj, inplace=True)
#         ds = ds.rio.clip(
#             polygon.geometry.values, crs=polygon.crs, drop=True
#         )
#         # clm = utils.clip_dataset(clm, _geometry, 4326)
#     except NoDataInBounds:
#         clipped_ds = ds  # If no valid data, return original dataset
#     return clipped_ds # Defer computation to Dask


def read_netcdfs(filepaths, polygon, dim='time', transform_func=None):
    def process_one_path(path, polygon):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds, polygon)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    datasets = [process_one_path(p, polygon) for p in filepaths]
    return datasets
    

# region_polygon = gpd.read_file(mask_path)
catchment_fpath = "input_data/BCUB_watershed_bounds_updated.geojson"
catchment_gdf = gpd.read_file(os.path.join(BASE_DIR, catchment_fpath))

daymet_params = ['dayl']#"tmax", "tmin", "vp", "prcp", "swe", "srad"]
years = list(range(1980, 2024))

sample_fpath = "input_data/BCUB_watershed_bounds_sample.geojson"
sample = True
if sample:
    catchment_gdf = gpd.read_file(os.path.join(BASE_DIR, sample_fpath))
else:
    
    catchment_fpath = "input_data/BCUB_watershed_bounds_updated.geojson"
    catchment_gdf = gpd.read_file(os.path.join(BASE_DIR, catchment_fpath))
    
for i, catchment_data in catchment_gdf.iterrows():
    t0 = time()
    polygon = catchment_gdf.loc[[i]]
    stn = catchment_data["Official_ID"]
    da = catchment_data["Drainage_Area_km2"]
   
    processed_fpath_netcdf = os.path.join(daymet_output_dir, f"{stn}.nc")
    if os.path.exists(processed_fpath_netcdf):
        print(f"    ...{stn}.nc already processed.")
        continue

    print(
        f"Processing {stn} ({da:.1f} km^2)",
    )
    tile_ids = get_covering_daymet_tile_ids(polygon)
    # all_params = []
    # for param in daymet_params:       
    #     tp0 = time()
    #     file_set = get_clm_files(tile_ids, years, param)
    #     datasets = read_netcdfs(file_set, polygon, transform_func=clip_files)
    #     combined = xr.concat(datasets, 'time').mean(dim=["x", "y"])
    #     all_params.append(combined)
    #     tp1 = time()
    #     print(f'    ....Processed {param} data for {stn} in {tp1-tp0:.1f}.')
    
    # tf0 = time()
    # final_ds = xr.merge(all_params)
    # final_ds = final_ds.rename({"time": "date"})
    # final_ds.to_netcdf(processed_fpath_netcdf, mode="w", engine="netcdf4")
    # tf1 = time()
    # print(f'    ...time to merge and save NetCDF: {tf1-tf0:.1f}s.')
    # # t1 = time()
    # final_ds = xr.merge(all_params)
    # # Convert final Zarr dataset to NetCDF
    # print('   ...Merging and saving NetCDF file...')
    # ts0 = time()
    # final_ds = xr.open_zarr(zarr_store).compute()
    # final_ds = final_ds.rename({"time": "date"})
    # final_ds.to_netcdf(processed_fpath_netcdf, mode="w", engine="h5netcdf")
    # ts1 = time()
    # print(f'   ...time to merge and save NetCDF: {ts1-ts0:.1f}s.')

    # # Cleanup
    # del final_ds
    # os.remove(zarr_store)  # Remove intermediate Zarr store

    t1 = time()
    # print(f"   ...{stn}.nc saved. Processed in {(t1-t0)/60:.1f} minutes.")
    break
