# Process Meteorological Forcings from Daymet

This repository provides a framework to compute **catchment-averaged daily meteorological time series** using the [Daymet](https://daymet.ornl.gov/) gridded climate dataset. Users supply catchment polygons, and the scripts retrieve, clip, and process Daymet tiles to produce NetCDF files of spatially distributed or averaged climate variables for each watershed.

---

## Features

- Retrieves Daymet tiles intersecting user-supplied watershed boundaries.
- Computes gridded or catchment-averaged daily time series 
- Supports full spatial output or catchment average time series.
- Computes **potential evapotranspiration (PET)** using [PyDaymet](https://github.com/hyriver/pydaymet).
- Efficient processing using `dask`, `xarray`, and `multiprocessing`.
- Handles CRS transformations and geometric simplification for small catchments.

---

## Repository Structure

- `tile_retrieval.sh`  
  Retrieves all Daymet tile IDs covering the region polygons in `input_data/`.

- `retrieve_and_process_daymet_forcings.py`  
  Iterates over catchments, stacks Daymet tiles, clips rasters to catchment polygons, and outputs `.nc` files per catchment. PET is not computed here.

- `pydaymet_test.py`  
  Uses the [PyDaymet](https://github.com/chaimleib/pydaymet) API to extract Daymet data and compute PET. Supports catchment-averaged output.

- `utils.py`  
  Helper functions for clipping, writing CRS metadata, and raster operations.

---

## Dependencies

This project requires Python 3.9+

