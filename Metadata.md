---
title: "Processed Catchment-Averaged Meteorological Forcings from Daymet for Streamflow Monitored Catchments in British Columbia and Transboundary Basins"
author: Dan Kovacek, University of British Columbia (UBC), [dkovacek@mail.ubc.ca](mailto:dkovacek@mail.ubc.ca)
date: March 20, 2025
geometry: margin=2cm
output: pdf_document
---

# Dataset Metadata

## Description

This dataset provides catchment-averaged daily meteorological time series for selected watersheds in British Columbia and adjacent regions.  Selection is based on minimum 5 (complete) years overlap with Daymet (1980-2023), where complete years have a minimum 20 observations in all 12 months. Meteorological variables are derived from the Daymet gridded climate dataset (Version 4 R1) at a 1-km spatial resolution and include the parameters in the table below:

| Variable   | Long Name                                         | Units       |      Methods                    |
|------------|---------------------------------------------------|-------------|---------------------------------|
| `dayl`     | Daylength                                         | s           | area: mean                      |
| `prcp`     | Daily total precipitation                         | mm/day      | area: mean time: sum            |
| `srad`     | Daylight average incident shortwave radiation     | W/m²        | area: mean time: mean           |
| `swe`      | Snow water equivalent                             | kg/m²       | area: mean time: mean           |
| `tmax`     | Daily maximum temperature                         | degrees C   | area: mean time: maximum        |
| `tmin`     | Daily minimum temperature                         | degrees C   | area: mean time: minimum        |
| `vp`       | Daily average vapor pressure                      | Pa          | area: mean time: mean           |
| `pet`      | Potential evapotranspiration[^1]                  | mm/day      | area: mean time: sum            |
| `streamflow`| daily Average Streamflow[^2]                         | m³/s        | area: n/a time: mean            |

[^1]: Potential evapotranspiration was computed by the Penman-Monteith method via the [PyDaymet Python package](https://docs.hyriver.io/autoapi/pydaymet/pet/index.html)
[^2]: Daily average streamflow is from USGS and Environment and Climate Change Canada (ECCC) monitoring stations, as provided in the [HYSETS dataset](https://osf.io/rpc3w/). The streamflow data is not derived from Daymet but is included for hydrological modelling inputs.

Catchment boundaries were used to clip and average Daymet tiles (provided in Catchment_polygons.geojson). The NetCDF files in this directory contain time series representing catchment average meteorological forcings.  The files are provided in the format required for ingesting into the [NeuralHydrology Python Package](https://neuralhydrology.readthedocs.io/en/latest/) for research in applying neural networks to hydrological prediction.   

The data were generated using the scripts provided in the `process_metforcings` Python repository available at:  
[https://github.com/dankovacek/process_metforcings](https://github.com/dankovacek/process_metforcings)

### Files

* meteorological forcings following the format `<Official_ID>_daymet.nc` (compressed in the archive file `BC_Monitored_catchment_mean_met_forcings_20250320.zip`)  
  - Contains daily time series of the meteorological variables listed above, averaged over the catchment polygon defined in `Catchment_polygons.geojson`.
* `Catchment_polygons.geojson`  
  - Contains the catchment polygons used for clipping and averaging Daymet tiles. Each polygon is associated with an `Official_ID` that matches the naming convention of the output NetCDF files.


### NOTE:
> The netCDF (`.nc`) files are not formatted for display in GIS software.  It is recommended to use the [xarray](https://docs.xarray.dev/en/stable/) python package to work with these files.  For working with netCDF files in R, see [this tutorial](https://pjbartlein.github.io/REarthSysSci/netCDF.html).

## Subject

- Hydrology  
- Climate Forcings  
- Environmental Science  
- GIS  
- Meteorology  

## Keywords

Daymet, catchment, meteorological forcings, potential evapotranspiration, precipitation, temperature

## Resource Type

Dataset

## Related Publications

Thornton, M. M., Shrestha, R., Wei, Y., Thornton, P. E., Kao, S., & Wilson, B. E.  
Daymet: Monthly Climate Summaries on a 1-km Grid for North America, Version 4 R1.  
ORNL DAAC, Oak Ridge, Tennessee, USA. (2022). [https://doi.org/10.3334/ORNLDAAC/2129](https://doi.org/10.3334/ORNLDAAC/2129)

Arsenault, R., Brissette, F., Martel, JL. et al. A comprehensive, multisource database for hydrometeorological modeling of 14,425 North American watersheds. Sci Data 7, 243 (2020). [https://doi.org/10.1038/s41597-020-00583-2](https://doi.org/10.1038/s41597-020-00583-2)

## Rights

This derived dataset inherits the usage conditions of the Daymet data product and is distributed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/), unless otherwise stated. 

Please cite both the original Daymet dataset and this processing workflow when using this data.

## Language

English

## Funding

This work was supported in part by the University of British Columbia, and the British Columbia Ministry of Environment and Climate Change Strategy.  The author acknowledges the support of the [Peter Wall Research Award](https://walllegacyawards.ubc.ca/) in 2024.

## Version

v1.0 (2025-03-20)

## File Format

NetCDF (.nc), GeoJSON (.geojson)

## Repository

GitHub repository for processing workflow:  
[https://github.com/dankovacek/process_metforcings](https://github.com/dankovacek/process_metforcings)

## Spatial Coverage

- Region: British Columbia, Canada and adjacent hydrologic regions (Yukon, Northwest Territories, Alaska, Alberta, Montana, Idaho, Washington)
- CRS: 
    - The Daymet projection is Lambert Conformal Conic (+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs)
    - Catchment polygons are provided in BC Albers (EPSG 3005)

## Temporal Coverage

- daily, 1980–2023 (inclusive)

## Methods

Meteorological variables were processed by identifying Daymet tiles that spatially intersect each catchment polygon, downloading the relevant data, clipping and reprojecting to match each catchment’s geometry, and computing area-weighted means for each day. For potential evapotranspiration, PyDaymet's implementation of the Penman-Monteith method was used.

The full processing workflow is documented and implemented in the accompanying Python scripts and shell commands within the GitHub repository.

## Contact

Daniel Kovacek, P.Eng.  
University of British Columbia  
[dkovacek@mail.ubc.ca](mailto:dkovacek@mail.ubc.ca)
