---
title: "Processed Catchment-Averaged Meteorological Forcings from Daymet for Streamflow Monitored Catchments in British Columbia and Transboundary Basins"
author: Dan Kovacek, University of British Columbia (UBC), [dkovacek@mail.ubc.ca](mailto:dkovacek@mail.ubc.ca)
date: March 20, 2025
geometry: margin=2cm
output: pdf_document
---

# Dataset Metadata

## Description

This dataset provides catchment-averaged daily meteorological time series for selected watersheds in British Columbia and adjacent regions. Meteorological variables are derived from the Daymet gridded climate dataset (Version 4 R1) at a 1-km spatial resolution and include:

- Precipitation (prcp)  
- Maximum and minimum temperature (tmax, tmin)  
- Shortwave radiation (srad)  
- Vapor pressure (vp)  
- Snow water equivalent (swe)  
- Day length (dayl)  
- Potential evapotranspiration (PET), computed using the Penman-Monteith method via the PyDaymet package

Catchment boundaries were provided as GeoJSON files and used to clip and average Daymet tiles. The resulting NetCDF files contain either full spatial resolution over each catchment or a single averaged time series, depending on configuration.

Data were generated using the `process_metforcings` Python repository available at:  
[https://github.com/dankovacek/process_metforcings](https://github.com/dankovacek/process_metforcings)

## Subject

- Hydrology  
- Climate Forcings  
- Environmental Science  
- GIS  
- Meteorology  

## Keywords

Daymet, catchment, watershed, meteorological forcing, evapotranspiration, precipitation, temperature, PET, BCUB, gridded climate data, NetCDF

## Resource Type

Dataset

## Related Publications

Thornton, M. M., Shrestha, R., Wei, Y., Thornton, P. E., Kao, S., & Wilson, B. E. (2022).  
**Daymet: Monthly Climate Summaries on a 1-km Grid for North America, Version 4 R1**.  
ORNL DAAC, Oak Ridge, Tennessee, USA.  
[https://doi.org/10.3334/ORNLDAAC/2129](https://doi.org/10.3334/ORNLDAAC/2129)

## Rights

This derived dataset inherits the usage conditions of the Daymet data product and is distributed under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/), unless otherwise stated. 

Please cite both the original Daymet dataset and this processing workflow when using this data.



## Language

English

## Funding

This work was supported in part by the University of British Columbia, and the British Columbia Ministry of Environment and Climate Change Strategy.

## Version

v1.0 (2025-03-20)

## File Format

NetCDF (.nc), GeoJSON (.geojson)

## Repository

GitHub repository for processing workflow:  
[https://github.com/dankovacek/process_metforcings](https://github.com/dankovacek/process_metforcings)

## Spatial Coverage

- Region: British Columbia, Canada and adjacent hydrologic regions (Yukon, Northwest Territories, Alaska, Alberta, Montana, Idaho, Washington)
- CRS: Lambert Conformal Conic (Daymet native projection), BC Albers (EPSG 305)

## Temporal Coverage

- 1980–2023 (inclusive)

## Methods

Meteorological variables were processed by identifying Daymet tiles that spatially intersect each catchment polygon, downloading the relevant data, clipping and reprojecting to match each catchment’s geometry, and computing area-weighted means for each day. For potential evapotranspiration, PyDaymet's implementation of the Penman-Monteith method was used.

The full processing workflow is documented and implemented in the accompanying Python scripts and shell commands within the GitHub repository.

## Contact

Daniel Kovacek, P.Eng.  
University of British Columbia  
[dkovacek@mail.ubc.ca](mailto:dkovacek@mail.ubc.ca)
