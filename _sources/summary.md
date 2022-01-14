# Summary and Recommendations

- The **surface roughness** and **surface heat capacity** can likely be improved in `LANDUSE.TBL`
    - Surface roughness can be decreased to 0.001 or 0.01. The former is more accurate, but I'm not sure how many orders of magnitude WRF can obtain from `LANDUSE.TBL`.
    - Specific heat capacity may be too high, but I'm concerned my calculations could be wrong.
- **Thermal inertia constant** could also likely be improved, but I know my calculations are incorrect.

## Other ice and snow sections from `LANDUSE.TBL`

| <div style="width:125px">Section | <div style="width:125px">Season | <div style="width:125px">Albedo<br>$(\%)$ | <div style="width:125px">Suface Moisture Availability<br>$(*100\%)$ | <div style="width:125px">Surface Emissivity<br>$(\%)$ | <div style="width:125px">Surface Roughness | <div style="width:125px">Thermal Inertia Constant | <div style="width:125px">Snow Cover Effect | <div style="width:125px">Surface Heat Capacity<br>$(J/(m^{3} K))$ | <div style="width:200px">Category       |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| OLD     | Summer | 55.    | .95  | .95  | 5.   | 5.     | 0.   | 9.0e25 | Permanent Ice |
| OLD     | Winter | 70.    | .95  | .95  | 5.   | 5.     | 0.   | 9.0e25 | Permanent Ice |
| USGS    | Summer | 55.    | .95  | .95  | 0.1  | 5.     | 0.   | 9.0e25 | Snow or Ice   |
| USGS    | Winter | 70.    | .95  | .95  | 0.1  | 5.     | 0.   | 9.0e25 | Snow or Ice   |
| MODIFIED_IGBP<br>_MODIS_NOAH | Summer | 55. | .95 | .95 | 0.1 | 5. | 0 | 9.0e25 | Snow or Ice |
| MODIFIED_IGBP<br>_MODIS_NOAH | Winter | 70. | .95 | .95 | 0.1 | 5. | 0 | 9.0e25 | Snow or Ice |
| SiB     | Summer | 55.    | .95  | .95  | 5.   | 5.     | 0.   | 9.0e25 | Ice Cap and Glacier |
| SiB     | Winter | 70.    | .95  | .95  | 5.   | 5.     | 0.   | 9.0e25 | Ice Cap and Glacier |
| LW12    | All    | 70.    | .95  | .95  | 5.   | 5.     | 0.   | 9.0e25 | Snow and Ice |
| MODIS   | Summer | 55.    | .95  | .95  | 1.   | 5.     | 0.   | 9.0e25 | Snow and Ice |
| MODIS   | Winter | 55.    | .95  | .98  | 1.0  | 5.     | 0.   | 9.0e25 | Snow and Ice |
| SSIB    | Summer | 55.    | .95  | .95  | 0.1  | 5.     | 0.   | 9.0e25 | Snow or Ice |    
| SSIB    | Winter | 70.    | .95  | .95  | 0.1  | 5.     | 0.   | 9.0e25 | Snow or Ice |   
| NLCD40  | Summer | 60.    | .95  | .95  | 1.2  | 5.     | 0.   | 9.0e25 | Permanent Snow and Ice |     
| NLCD40  | Winter | 60.    | .95  | .95  | 1.2  | 5.     | 0.   | 9.0e25 | Permanent Snow and Ice |    

- Albedos overall are lower than those observed at N-ICE
- Soil moisture availability is 0.95 in all the simulations
- Emissivity is higher in some situations, N-ICE simulations could potentially benefit from using the 0.98 instead of 0.95
- Surface roughness is the lowest in the section used, but it is still higher than that calculated fro N-ICE
- Thermal interia constant, snow cover effects, and the surface heat capacity are the same in all simualtions
    - If my calculations of the surface heat capacity are correct, simulations could likely be improved by lowering this value.

## Values in `VEGPARM.TBL`
** Not used in the idealized simulations **

| <div style="width:125px">Section | <div style="width:125px">Vegetation Fraction | <div style="width:125px">Rooting Depth | <div style="width:125px">Stomatal Resistance | <div style="width:125px">Radiation Stress | <div style="width:125px">Vapor Pressure Deficit Function Parameter | <div style="width:125px">Soil Water Equivalent Snow Depth | <div style="width:125px">Maximum Albedo Over Deep Snow | <div style="width:125px">Minimum LAI | <div style="width:125px">Maximum LAI | <div style="width:125px">Minimum Emissivity | <div style="width:125px">Maximum Emissivity | <div style="width:125px">Minimum Albedo | <div style="width:125px">Maximum Albedo | <div style="width:125px">Minimum Roughness Length | <div style="width:125px">Maximum Roughness Length | <div style="width:200px">Category |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| USGS | 0 | 1 | 999 | 999 | 999 | 0.02 | 82 | 0.01 | 0.01 | 0.95 | 0.95 | 0.55 | 0.70 | 0.001 | 0.001 | Snow or Ice |
| MODIFIED_IGBP<br>_MODIS_NOAH | 0 | 1 | 999 | 999 | 999 | 0.02 | 82 | 0.01 | 0.01 | 0.95 | 0.95 | .55 | .7 | 0.001 | 0.001 | Snow and Ice |
| NLDC | 0 | 1 | 999 | 999 | 999 | 0.02 | 82 | 0.01 | 0.01 | 0.95 | 0.95 | 0.55 | 0.70 | 0.001 | 0.001 | Permanent Snow |
| NLDC | 0 | 1 | 999 | 999 | 999 | 0.02 | 82 | 0.01 | 0.01 | 0.95 | 0.95 | 0.55 | 0.70 | 0.001 | 0.001 | Perennial Ice/Snow |

| <div style="width:125px">Section | <div style="width:125px">Albedo | <div style="width:125px">Roughness Length | <div style="width:125px">LEMI\* | <div style="width:125px">PC\* | <div style="width:125px">Vegetation Factor | <div style="width:125px">IFOR\* | <div style="width:125px">Stomatal Resistance | <div style="width:125px">Radiation Stress | <div style="width:125px">Vapor Pressure Deficit Function Parameter | <div style="width:125px">Soil Water Equivalent Snow Depth | <div style="width:125px">LAI | <div style="width:125px">Maximum Albedo | <div style="width:200px">Category |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |:-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| USGS-RUC | 0.55 | 0.0024 | 0.98 | 0.00 | 0.00 | 9 | 999. | 999. | 999. | 0.02 | 0.01 | 75 | Snow or Ice   | 
| MODIS-RUC | 0.55 | 0.011 | 0.98 | 0.00 | 0.00 | 9 | 999. | 999. | 999. | 0.02 | 0.01 | 82 | Snow and Ice  |

\* I Don't know what some of these are 
- The maximum albedo is higher here than in `LANDUSE.TBL`, closer to that seen at N-ICE and the value used in the simulations.
- Roughness lengths are closer to what I've calculated for N-ICE
- Emissivities are similar to those seen in `LANDUSE.TBL`

## Values in `SOILPARM.TBL`
** Not used in the idealized simulation, for land-ice **

| <div style="width:125px">Section | <div style="width:125px">B Parameter | <div style="width:125px">Dry Soil Moisture Threshold | <div style="width:200px">Soil Thermal Diffusivity/Conducticity Coefficient | <div style="width:125px">Saturation Soil Moisture Content | <div style="width:125px">Reference Soil Moisture | <div style="width:125px">Saturation Soil Matric Potential | <div style="width:125px">Saturation Soil Conductivity | <div style="width:125px">Saturation Soil Diffusivity | <div style="width:125px">Wilting Point Soil Moisture | <div style="width:125px">Soil Quartz Content |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |       
| STAS     | 4.26 | 0.028 | -1.044 | 0.421 | 0.283 | 0.036 | 1.41E-5 | 0.514E-5 | 0.028    | 0.25  |
| STAS-RUC | 4.90 | 0.065 | 2.10   | 0.435 | 0.249 | 0.218 | 3.47E-5 | 0.514E-5 | 0.114    | 0.05 |

- None of these should impact the ice surface

## What do I recommend?

| <div style="width:125px">Section | <div style="width:125px">Season | <div style="width:125px">Albedo<br>$(\%)$ | <div style="width:125px">Surface Moisture Availability<br>$(*100\%)$ | <div style="width:125px">Surface Emissivity<br>$(\%)$ | <div style="width:125px">Surface Roughness | <div style="width:125px">Thermal Inertia Constant | <div style="width:125px">Snow Cover Effect | <div style="width:125px">Surface Heat Capacity<br>$(J/(m^{3} K))$ | 
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | 
| NICE    | Winter | 86.        | .95                                  | .98                    | 0.001             | 5.                       | 0.                | 4.8e12                        |
| NICE    | Summer | 81.        | .95                                  | .98                    | 0.001             | 5.                       | 0.                | 4.2e12                        |

1. Increase albedo in both the winter and summer
2. Increase surface emissivity from 0.95 to 0.98
3. Decrease surface roughness from 0.1 to 0.001
4. Decreased surface heat capacity and use a different value for winter and summer.

---

```{bibliography}
:style: plain
```