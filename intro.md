# Reviewing Values in `LANDUSE.TBL`

`LANDUSE.TBL` is used by the land surface model to specify some variables depending on the land surface type. According to the Dudhia WRF Overview presentation {cite}`dudhia2014overview`, the following are used:
- VEGPARM.TB - Vegetation properties in **Noah and RUC**
- SOILPARM.TBL - Soil peoperties in **Noah and RUC**
- LANDUSE.TBL - Surface properties in **5-layer model**

## We are using the following in the idealized model
- 5-layer model (use `LANDUSE.TBL`)
- USGS section (specified in `namelist.input`)
- The entire time we have land use category number 24 - snow and ice

## Currently for snow and ice in my idealized simulations:

| Season | Albedo ($\%$) | Suface Moisture Availability ($*100\%$) | Surface Emissivity ($\%$) | Surface Roughness | Thermal Inertia Constant | Snow Cover Effect | Surface Heat Capacity ($J/(m^{3}K)$) | Label       |
| ------ | ------ | ---- | ---- | ---- | ------ | -----| ------ | ----------- |
| Summer | 55.    | .95  | .95  | 0.1  | 5.     | 0.   | 9.0e25 | Snow or Ice |
| Winter | 70.    | .95  | .95  | 0.1  | 5.     | 0.   | 9.0e25 | Snow or Ice |

### Preliminary observations
1. **Albedo** is far too low - ***BUT we have defined this in the namelist*** so this value is not used.
2. **Surface moisture availability** and **emissivity** could both probably be increased.
3. **Surface roughness** could probably be decreased.
4. **Snow cover effects** seem to be constant regardless of surface
5. The **thermal inertial constant** could probably be improved, but I'm not sure what this is.
6. **Surface heat capacity** could be improved.

## What variables do we need? Where can we get them?
### Have
1. Albedo
    - This is already incorporated into the model

### Can Calculate
1. Surface roughness length
    - This is calculated below and is much lower than that in the table
    - This could be a good value to adjust
2. Surface heat capacity
    - This is calculated below and is much higher than that in the table
    - I'd like to discuss this variable with Von
3. Thermal inertia constant
    - I am struggling with calculating this, I need to do more reading to understand it

### Can improve estimations
1. Surface moisture availability
    - This is higher only in the values above water
    - The difference is small so it probably would not be a large difference
2. Emissivity
    - Also a small difference so likely not a big change
