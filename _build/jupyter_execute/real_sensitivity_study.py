#!/usr/bin/env python
# coding: utf-8

# # Comparison with WRF Output
# 
# ## WRF Settings
# 
# - **Input datasets**: 
#     - ERA 5
#     - PIOMASS
# 
# ### Namelist.input
# 
# ```bash
# &time_control
#  run_days                               = 30,
#  run_hours                              = 0,
#  run_minutes                            = 0,
#  run_seconds                            = 0,
#  start_year                             = 2015, 2015,
#  start_month                            = 05,   05,
#  start_day                              = 01,   01,
#  start_hour                             = 00,   00,
#  end_year                               = 2015, 2015,
#  end_month                              = 06,   06, 
#  end_day                                = 01,   01,
#  end_hour                               = 00,   00, 
#  interval_seconds                       = 21600
#  input_from_file                        = .true.,.true.,
#  history_interval                       = 60,  60, 
#  frames_per_outfile                     = 10, 10,
#  restart                                = .true.,
#  restart_interval                       = 1440,
#  io_form_history                        = 2
#  io_form_restart                        = 2
#  io_form_input                          = 2
#  io_form_boundary                       = 2
#  history_outname                        = 'wrfout_d<domain>_<date>',
#  rst_inname                             = 'wrfrst_d<domain>_<date>',
#  rst_outname                            = 'wrfrst_d<domain>_<date>',
# /
# 
# &domains
#  time_step                              = 45,
#  time_step_fract_num                    = 0,
#  time_step_fract_den                    = 1,
#  max_dom                                = 2,
#  e_we                                   = 125,   166,
#  e_sn                                   = 120,   166,
#  e_vert                                 = 33,    33,
#  p_top_requested                        = 5000,
#  num_metgrid_levels                     = 38,
#  dx                                     = 9000,  3000.,
#  dy                                     = 9000,  3000.,
#  grid_id                                = 1,     2,
#  parent_id                              = 1,     1,
#  i_parent_start                         = 1,     30,
#  j_parent_start                         = 1,     53,
#  parent_grid_ratio                      = 1,     3,
#  parent_time_step_ratio                 = 1,     3,
#  feedback                               = 1,
#  smooth_option                          = 0,
# /
# 
# &physics
#  mp_physics                             = 7, 7,
#  bl_pbl_physics                         = 1, 1,
#  gsfcgce_hail                           = 0,
#  gsfcgce_2ice                           = 0,
#  co2tf                                  = 1,
#  cu_physics                             = 3, 3,
#  cudt                                   = 5, 5,
#  icloud                                 = 1,
#  ra_lw_physics                          = 4, 4,
#  ra_sw_physics                          = 4, 4,
#  radt                                   = 20,
#  slope_rad                              = 0,
#  topo_shading                           = 0,
#  sf_sfclay_physics                      = 91, 91,
#  sf_surface_physics                     = 2, 2,
#  bldt                                   = 0,
#  isfflx                                 = 1,
#  ifsnow                                 = 0,
#  surface_input_source                   = 1,
#  num_land_cat                           = 24,
# /
# 
# &fdda
# /
# 
# &dynamics
#  w_damping                              = 1,
#  diff_opt                               = 1,
#  km_opt                                 = 4,
#  diff_6th_opt                           = 0,      0,
#  diff_6th_factor                        = 0.12,   0.12,
#  base_temp                              = 290.
#  damp_opt                               = 0,
#  zdamp                                  = 5000.,  5000.,
#  dampcoef                               = 0.2,    0.2, 
#  khdif                                  = 0,      0, 
#  kvdif                                  = 0,      0, 
#  non_hydrostatic                        = .true., .true., 
#  moist_adv_opt                          = 2,      2,     
#  scalar_adv_opt                         = 2,      2,     
#  rk_ord                                 = 3,
#  time_step_sound                        = 4,      4,     
#  h_mom_adv_order                        = 5,      5,     
#  v_mom_adv_order                        = 3,      3,     
#  h_sca_adv_order                        = 5,      5,     
#  v_sca_adv_order                        = 3,      3, 
# /
# 
# &bdy_control
#  spec_bdy_width                         = 5,
#  specified                              = .true.
# /
# 
#  &grib2
# /
# 
# &namelist_quilt
#  nio_tasks_per_group                    = 0,
#  nio_groups                             = 1,
# /
# ```
# 

# In[1]:


import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn

from bokeh.io import output_notebook
from bokeh.plotting import figure, show, column, row
from bokeh.models import ColumnDataSource,Div,Range1d
from bokeh.models import HoverTool

from scipy import optimize

from netCDF4 import Dataset
from wrf import getvar, interpline, CoordPair, xy_to_ll, ll_to_xy

from scipy.io import netcdf_file
from netCDF4 import Dataset
from wrf import getvar, CoordPair, xy_to_ll, ll_to_xy, ALL_TIMES
output_notebook()


# In[2]:


### IMPORTING N-ICE DATA ###
# Eddypro
epro_vals = pd.read_csv('/Users/smurphy/github/offline_calculations_wrf/epro_vals.csv', 
                        index_col = 0, parse_dates = True).resample('30 min', origin='2015-01-01').first()

# Surface temperature
Ts = pd.read_excel('/Users/smurphy/Documents/PhD/datasets/nice_data/Ts.xlsx', 
                   index_col = 0).resample('30 min', origin='2015-01-01').first()

# Surface energy budget
seb = xr.open_dataset('/Users/smurphy/all_datasets/nice_published_datasets/N-ICE_sebData_v1.nc', decode_times = False)

# SENSIBLE HEAT FLUX - H - Wm-2
measured_H = -pd.DataFrame(seb.variables['surface_downward_sensible_heat_flux'].values, 
                         index = pd.to_datetime(seb.variables['unix_time'].values, unit = 's'), 
                         columns = ['h']).resample('30 min', origin='2015-01-01').first()

# LATENT HEAT FLUX - E - Wm-2
measured_E = pd.DataFrame(seb.variables['surface_downward_latent_heat_flux'].values, 
                         index = pd.to_datetime(seb.variables['unix_time'].values, unit = 's'), 
                         columns = ['e']).resample('30 min', origin='2015-01-01').first()

# SURFACE LW DOWN
lwdown = pd.DataFrame(seb.variables['surface_downwelling_longwave_flux'].values, 
                      index = pd.to_datetime(seb.variables['unix_time'].values, unit = 's'), 
                      columns = ['LD']).resample('30 min', origin='2015-01-01').first()

# SURFACE LW UP
lwup =  pd.DataFrame(seb.variables['surface_upwelling_longwave_flux'].values, 
                     index = pd.to_datetime(seb.variables['unix_time'].values, unit = 's'), 
                     columns = ['LU']).resample('30 min', origin='2015-01-01').first()

# SURFACE SW DOWN
swdown = pd.DataFrame(seb.variables['surface_downwelling_shortwave_flux'].values, 
                      index = pd.to_datetime(seb.variables['unix_time'].values, unit = 's'), 
                      columns = ['SD']).resample('30 min', origin='2015-01-01').first()

# SURFACE SW DOWN
swup =  pd.DataFrame(seb.variables['surface_upwelling_shortwave_flux'].values, 
                     index = pd.to_datetime(seb.variables['unix_time'].values, unit = 's'), 
                     columns = ['SU']).resample('30 min', origin='2015-01-01').first() 

# MET DATA
met = xr.open_dataset('/Users/smurphy/all_datasets/nice_published_datasets/N-ICE_metData_v2.nc', 
                      decode_times = False)

# WIND SPEED - u - m / s
u = pd.DataFrame(met.variables['wind_speed_2m'].values, 
                 index = pd.to_datetime(met.variables['unix_time'].values, 
                                               unit = 's'), 
                 columns = ['ws']).resample('30 min', 
                                            origin='2015-01-01').first()

# WIND SPEED - u - m / s
u10 = pd.DataFrame(met.variables['wind_speed_10m'].values, 
                 index = pd.to_datetime(met.variables['unix_time'].values, 
                                               unit = 's'), 
                 columns = ['ws']).resample('30 min', 
                                            origin='2015-01-01').first()

# 2M AIR TEMPERATURE - Ta - K
Ta = pd.DataFrame(met.variables['air_temperature_2m'].values, 
                  index = pd.to_datetime(met.variables['unix_time'].values, 
                                         unit = 's'), 
                  columns = ['t']).resample('30 min', 
                                            origin='2015-01-01').first()

# RELATIVE HUMIDITY - rh - % * 100
RH = pd.DataFrame(met.variables['relative_humidity_2m'].values, 
                  index = pd.to_datetime(met.variables['unix_time'].values, 
                                         unit = 's'), 
                  columns = ['rh']).resample('30 min', 
                                             origin='2015-01-01').first()

# PRESSURE - Pa
P = pd.DataFrame(met.variables['air_pressure_at_sea_level'].values, 
                 index = pd.to_datetime(met.variables['unix_time'].values, 
                                        unit = 's'), 
                 columns = ['p']).resample('30 min', 
                                           origin='2015-01-01').first()

# Making sure everything is on the same time scale
epro_vals = epro_vals[epro_vals.index <= Ts.index[-1]]
epro_vals = epro_vals[epro_vals.index >= Ts.index[0]]
epro_vals = epro_vals[epro_vals.index <= measured_H.index[-1]]
epro_vals = epro_vals[epro_vals.index >= measured_H.index[0]]
epro_vals = epro_vals[epro_vals.index <= Ta.index[-1]]
epro_vals = epro_vals[epro_vals.index >= Ta.index[0]]

P = P[P.index <= epro_vals.index[-1]]
P = P[P.index >= epro_vals.index[0]]
Ta = Ta[Ta.index <= epro_vals.index[-1]]
Ta = Ta[Ta.index >= epro_vals.index[0]]
u = u[u.index <= epro_vals.index[-1]]
u = u[u.index >= epro_vals.index[0]]
u10 = u10[u10.index <= epro_vals.index[-1]]
u10 = u10[u10.index >= epro_vals.index[0]]
Ts = Ts[Ts.index <= epro_vals.index[-1]]
Ts = Ts[Ts.index >= epro_vals.index[0]]
RH = RH[RH.index <= epro_vals.index[-1]]
RH = RH[RH.index >= epro_vals.index[0]]

lwup = lwup[lwup.index <= epro_vals.index[-1]]
lwup = lwup[lwup.index >= epro_vals.index[0]]
lwdown = lwdown[lwdown.index <= epro_vals.index[-1]]
lwdown = lwdown[lwdown.index >= epro_vals.index[0]]
swdown = swdown[swdown.index <= epro_vals.index[-1]]
swdown = swdown[swdown.index >= epro_vals.index[0]]
swup = swup[swup.index <= epro_vals.index[-1]]
swup = swup[swup.index >= epro_vals.index[0]]

measured_E = measured_E[measured_E.index <= epro_vals.index[-1]]
measured_E = measured_E[measured_E.index >= epro_vals.index[0]]
measured_H = measured_H[measured_H.index <= epro_vals.index[-1]]
measured_H = measured_H[measured_H.index >= epro_vals.index[0]]

# SURFACE NET SHORTWAVE 
netsw = swdown.SD - swup.SU

# SURFACE NET LONGWAVE
netlw = lwdown.LD - lwup.LU

# NET RADIATION
Rn = netlw + netsw

# calculating the pressure at 2m
m = 0.0289644 # molecular weight of air     kg mol-1
g = 9.81 # gravity   m s-2
k = 1.3804 * 10 ** 23 # boltzmann's constant    W K-2
# scale height   https://glossary.ametsoc.org/wiki/Scale_height
sclh = (k * Ts.Ts) / (m * g)
P_2m = P.p * np.e ** (-(m/sclh)) #    Pa 

### Moisture calculations at the surface ###
# surface temperature (K converted to C in the equation) to saturation vapor pressure (Pa)
measured_svp = 611 * np.exp(((17.27 * (Ts.Ts - 273.15)) / (237.3 + Ts.Ts)))

# saturation vapor pressure (Pa) and relative humidity (%) to vapor pressure (Pa)
measured_vp = measured_svp * (RH.rh / 100)

# vapor pressure (Pa) and pressure (Pa) to mixing ratio (unitless???)
measured_mr =  0.622 * (measured_vp / (P.p - measured_vp))

# saturation vapor pressure (Pa) to saturation mixing ratio (unitless???)
measured_smr = 0.622 * (measured_svp / (P.p - measured_svp))

# saturation mixing ratio to saturation specific humidity (unitless???)
measured_ssh = (measured_smr / (1 + measured_smr))

### Moisture calculations at 2m ###
# surface temperature (K converted to C in the equation) to saturation vapor pressure (Pa)
measured_svp_2m = 611 * np.exp(((17.27 * (Ta.t - 273.15)) / (237.3 + Ta.t)))

# saturation vapor pressure (Pa) and relative humidity (%) to vapor pressure (Pa)
measured_vp_2m = measured_svp_2m * (RH.rh / 100)

# vapor pressure (Pa) and pressure (Pa) to mixing ratio (unitless???)
measured_mr_2m =  0.622 * (measured_vp_2m / (P_2m - measured_vp_2m))

# saturation vapor pressure (Pa) to saturation mixing ratio (unitless???)
measured_smr_2m = 0.622 * (measured_svp_2m / (P_2m - measured_svp_2m))

# saturation mixing ratio to saturation specific humidity (unitless???)
measured_ssh_2m = (measured_smr_2m / (1 + measured_smr_2m))


# In[3]:


ShipLocation = netcdf_file('/Users/smurphy/all_datasets/nice/10min.nc')
ShipTimes = ShipLocation.variables['unix_time'][:]
ShipTimes = pd.to_datetime(ShipTimes,unit='s')
ShipLocs = pd.DataFrame([ShipLocation.variables['latitude'][:],ShipLocation.variables['longitude'][:]]).T
ShipLocs.columns = ['Lat','Lon']
ShipLocs.index = ShipTimes 

# need to trim when it wasn't on experiment
## Floes ##
# 1. 15 Jan - 21 Feb
# 2. 24 Feb - 19 Mar
# 3. 18 Apr - 5 Jun
# 4. 7 Jun - 21 Jun

ShipLocs = pd.concat([ShipLocs.loc['2015-01-15':'2015-02-21'], 
                      ShipLocs.loc['2015-02-24':'2015-03-19'], 
                      ShipLocs.loc['2015-04-18':'2015-05-05'], 
                      ShipLocs.loc['2015-05-07':'2015-05-21']])


# In[4]:


wrf_vars = pd.read_csv('/Volumes/seagate_desktop/nested_2way/no_table_mods/nested2way_notablemods_surfacevars.csv', 
                       index_col = 0,
                       parse_dates = True)

variable = 'wspd_wdir10'
wrf_output_path = '/Volumes/seagate_desktop/nested_2way/no_table_mods/wrf_out'

fns = glob.glob(wrf_output_path + '/wrfout_d02*')
selected_variable_dfall = pd.DataFrame()

for fn in fns:
    ncfile = Dataset(fn)
    i = ShipLocs.index.searchsorted(pd.Timestamp(getvar(ncfile, 'times').values))
    if i == 13680:
        break
    ShipLoc = ShipLocs.loc[ShipLocs.index[i]].dropna()
    if len(ShipLocs.loc[ShipLocs.index[i]].dropna()):
        ShipLoc = ll_to_xy(ncfile, ShipLoc.Lat, ShipLoc.Lon)

        variable_df = pd.DataFrame(getvar(ncfile, 
                                          variable, 
                                          timeidx=ALL_TIMES).isel(south_north = ShipLoc[0], 
                                                                  west_east = ShipLoc[1]).values[0],
                                   index = getvar(ncfile, 'times', 
                                                  timeidx=ALL_TIMES).values,
                                   columns = [variable])
        selected_variable_dfall = pd.concat([selected_variable_dfall, variable_df])

selected_variable_dfall = selected_variable_dfall.sort_index()
wrf_vars = pd.concat([wrf_vars, selected_variable_dfall], axis = 1)


# In[5]:


wrf_mod_vars = pd.read_csv('/Volumes/seagate_desktop/nested_2way/table_mods/nested2way_tablemods_surfacevars.csv',
                           index_col = 0,
                           parse_dates = True)

variable = 'wspd_wdir10'
wrf_output_path = '/Volumes/seagate_desktop/nested_2way/table_mods/wrf_out'

ShipLocation = netcdf_file('/Users/smurphy/all_datasets/nice/10min.nc')
ShipTimes = ShipLocation.variables['unix_time'][:]
ShipTimes = pd.to_datetime(ShipTimes,unit='s')
ShipLocs = pd.DataFrame([ShipLocation.variables['latitude'][:],ShipLocation.variables['longitude'][:]]).T
ShipLocs.columns = ['Lat','Lon']
ShipLocs.index = ShipTimes 

ShipLocs = pd.concat([ShipLocs.loc['2015-01-15':'2015-02-21'], 
                      ShipLocs.loc['2015-02-24':'2015-03-19'], 
                      ShipLocs.loc['2015-04-18':'2015-05-05'], 
                      ShipLocs.loc['2015-05-07':'2015-05-21']])

fns = glob.glob(wrf_output_path + '/wrfout_d02*')
selected_variable_dfall = pd.DataFrame()

for fn in fns:
    ncfile = Dataset(fn)
    i = ShipLocs.index.searchsorted(pd.Timestamp(getvar(ncfile, 'times').values))
    if i == 13680:
        break
    ShipLoc = ShipLocs.loc[ShipLocs.index[i]].dropna()
    ShipLoc = ll_to_xy(ncfile, ShipLoc.Lat, ShipLoc.Lon)
    if len(ShipLocs.loc[ShipLocs.index[i]].dropna()):
        variable_df = pd.DataFrame(getvar(ncfile, 
                                          variable, 
                                          timeidx=ALL_TIMES).isel(south_north = ShipLoc[0], 
                                                                  west_east = ShipLoc[1]).values[0],
                                   index = getvar(ncfile, 'times', 
                                                  timeidx=ALL_TIMES).values,
                                   columns = [variable])
        selected_variable_dfall = pd.concat([selected_variable_dfall, variable_df])

selected_variable_dfall = selected_variable_dfall.sort_index()
wrf_mod_vars = pd.concat([wrf_mod_vars, selected_variable_dfall], axis = 1)


# In[6]:


wrf_orig_cds  = ColumnDataSource(data = {'index': wrf_vars.index.values,
                                        'T2'   : wrf_vars.T2.values,     # 2m temperature (K)
                                        'rh2'  : wrf_vars.rh2.values,    # 2m relative humidity (%)
                                        'slp'  : wrf_vars.slp.values,    # sea level pressure (hPa)
                                        'td2'  : wrf_vars.td2.values,    # 2m dew point temperature (K)
                                        'Q2'   : wrf_vars.Q2.values,     # 2m mixing ratio
                                        'TH2'  : wrf_vars.TH2.values,    # 2m potential temperature
                                        'UST'  : wrf_vars.UST.values,    # u*
                                        'LWUPB': wrf_vars.LWUPB.values,  # upwelling lw
                                        'LWDNB': wrf_vars.LWDNB.values,  # downwelling lw
                                        'SWUPB': wrf_vars.SWUPB.values,  # upwelling sw
                                        'SWDNB': wrf_vars.SWDNB.values,  # downwelling sw
                                        'HFX'  : wrf_vars.HFX.values,    # sensible heat flux
                                        'LH'   : wrf_vars.LH.values,     # latent heat flux
                                        'GRDFLX': wrf_vars.GRDFLX.values,# ground heat flux
                                        '10mU' : wrf_vars.wspd_wdir10.values,# 10 m wind speed
                                       })


wrf_mod_cds  = ColumnDataSource(data = {'index': wrf_mod_vars.index.values,
                                        'T2'   : wrf_mod_vars.T2.values,     # 2m temperature (K)
                                        'rh2'  : wrf_mod_vars.rh2.values,    # 2m relative humidity (%)
                                        'slp'  : wrf_mod_vars.slp.values,    # sea level pressure (hPa)
                                        'td2'  : wrf_mod_vars.td2.values,    # 2m dew point temperature (K)
                                        'Q2'   : wrf_mod_vars.Q2.values,     # 2m mixing ratio
                                        'TH2'  : wrf_mod_vars.TH2.values,    # 2m potential temperature
                                        'UST'  : wrf_mod_vars.UST.values,    # u*
                                        'LWUPB': wrf_mod_vars.LWUPB.values,  # upwelling lw
                                        'LWDNB': wrf_mod_vars.LWDNB.values,  # downwelling lw
                                        'SWUPB': wrf_mod_vars.SWUPB.values,  # upwelling sw
                                        'SWDNB': wrf_mod_vars.SWDNB.values,  # downwelling sw
                                        'HFX'  : wrf_mod_vars.HFX.values,    # sensible heat flux
                                        'LH'   : wrf_mod_vars.LH.values,     # latent heat flux
                                        'GRDFLX': wrf_mod_vars.GRDFLX.values,# ground heat flux
                                        '10mU' : wrf_mod_vars.wspd_wdir10.values,# 10 m wind speed
                                       })


# ## Basic Meteorology

# In[7]:


measurements = ColumnDataSource(data = {'index': P.index.values,
                                        'ts' : Ts.Ts.values,
                                        'ta' : Ta.t.values,
                                        'p' : P.p.values / 100,
                                        'rh' : RH.rh.values,
                                        'u' : u10.ws.values,
                                        })


f = figure(title = "Surface Temperature", x_axis_type="datetime", width = 700, height = 300, x_range = (wrf_mod_vars.index[0], wrf_mod_vars.index[-1]))

f.xaxis.axis_label = 'Date'
f.yaxis.axis_label = r"\[K\]"

f.circle(x = "index", 
         y = "ts", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

f.legend.click_policy = "hide"

q = figure(title = "Air Temperature", x_axis_type="datetime", width = 700, height = 300, x_range=f.x_range)
q.xaxis.axis_label = 'Date'
q.yaxis.axis_label = r"\[K\]"

q.circle(x = "index", 
         y = "ta", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

q.circle(x = "index", 
         y = "T2", 
         source = wrf_orig_cds,
         legend_label = 'WRF - No mod',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

q.circle(x = "index", 
         y = "T2", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)


q.legend.click_policy = "hide"

bins = np.linspace(255, 275, 20)
hist, edges = np.histogram(wrf_mod_vars.T2.values, bins=bins, density = False)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.T2.values, bins=bins, density = False)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

q_hist = figure(title="2m Temperature", width=300, height=300)
q_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
q_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

r = figure(title = "Relative Humidity", x_axis_type="datetime", width = 700, height = 300, x_range=f.x_range)
r.xaxis.axis_label = 'Date'
r.yaxis.axis_label = r"\[RH (\%)\]"

r.circle(x = "index", 
         y = "rh", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

r.circle(x = "index", 
         y = "rh2", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

r.circle(x = "index", 
         y = "rh2", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

r.legend.click_policy = "hide"

bins = np.linspace(50, 100, 25)
hist, edges = np.histogram(wrf_mod_vars.rh2.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.rh2.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

r_hist = figure(title="Relative Humidity", width=300, height=300)
r_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
r_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

n = figure(title = "Wind Speed", x_axis_type="datetime", width = 700, height = 300, x_range=f.x_range)
n.xaxis.axis_label = 'Date'
n.yaxis.axis_label = r"\[W \space m^{-2}\]"

n.circle(x = "index", 
         y = "u", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

n.circle(x = "index", 
         y = "10mU", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

n.circle(x = "index", 
         y = "10mU", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

n.legend.click_policy = "hide"
bins = np.linspace(-5, 25, 30)
hist, edges = np.histogram(wrf_mod_vars.wspd_wdir10.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.wspd_wdir10.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

n_hist = figure(title="10m Wind Speed", width=300, height=300)
n_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
n_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

d = figure(title = "Sea Level Pressure", x_axis_type="datetime", width = 700, height = 300, x_range=f.x_range)
d.xaxis.axis_label = 'Date'
d.yaxis.axis_label = r"\[hPa\]"

d.circle(x = "index", 
         y = "p", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

d.circle(x = "index", 
         y = "slp", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

d.circle(x = "index", 
         y = "slp", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

bins = np.linspace(1000, 1040, 20)
hist, edges = np.histogram(wrf_mod_vars.slp.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.slp.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

d_hist = figure(title="Sea Level Pressure", width=300, height=300)
d_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
d_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

d.legend.click_policy = "hide"
d.yaxis.formatter.use_scientific = False


show(column(f, row(q, q_hist), row(r, r_hist), row(n, n_hist), row(d, d_hist)))


# ## Sensible & Latent Heat Flux

# In[8]:


measurements = ColumnDataSource(data = {'index': measured_H.index.values,
                                        'sh' : measured_H.h.values,
                                        'lh' : measured_E.e.values
                                        })

v = figure(title = "Sensible Heat Flux", x_axis_type="datetime", width = 700, height = 300, x_range = (wrf_mod_vars.index[0], wrf_mod_vars.index[-1]), y_range = (-40, 80))
v.xaxis.axis_label = 'Date'
v.yaxis.axis_label = r"\[W \space m^{-2}\]"

v.circle(x = "index", 
         y = "sh", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

v.circle(x = "index", 
         y = "HFX", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

v.circle(x = "index", 
         y = "HFX", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

v.legend.click_policy = "hide"
bins = np.linspace(-40, 80, 20)
hist, edges = np.histogram(wrf_mod_vars.HFX.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.HFX.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

v_hist = figure(title="Sensible Heat Flux", width=300, height=300)
v_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
v_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

w = figure(title = "Latent Heat Flux", x_axis_type="datetime", width = 700, height = 300, x_range=v.x_range)
w.xaxis.axis_label = 'Date'
w.yaxis.axis_label = r"\[W \space m^{-2}\]"

w.circle(x = "index", 
         y = "lh", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

w.circle(x = "index", 
         y = "LH", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

w.circle(x = "index", 
         y = "LH", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)


w.legend.click_policy = "hide"

bins = np.linspace(-20, 60, 20)
hist, edges = np.histogram(wrf_mod_vars.LH.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.LH.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

w_hist = figure(title="Latent Heat Flux", width=300, height=300)
w_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
w_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

show(column(row(v, v_hist), row(w, w_hist)))


# ## Longwave & Shortwave Fluxes

# In[9]:


measurements = ColumnDataSource(data = {'index': lwup.index.values,
                                        'swup' : swup.SU.values,
                                        'swdn' : swdown.SD.values,
                                        'lwup' : lwup.LU.values,
                                        'lwdn' : lwdown.LD.values,
                                       })

v = figure(title = "Upwelling Shortwave Radiation", x_axis_type="datetime", width = 700, height = 300, x_range = (wrf_mod_vars.index[0], wrf_mod_vars.index[-1]))
v.xaxis.axis_label = 'Date'
v.yaxis.axis_label = r"\[W \space m^{-2}\]"

v.circle(x = "index", 
         y = "swup", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

v.circle(x = "index", 
         y = "SWUPB", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

v.circle(x = "index", 
         y = "SWUPB", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

v.legend.location = "top_left"

bins = np.linspace(0, 500, 50)
hist, edges = np.histogram(wrf_mod_vars.SWUPB.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.SWUPB.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

v_hist = figure(title="Upwelling Shortwave Radiatio", width=300, height=300)
v_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
v_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

vv = figure(title = "Downwelling Shortwave Radiation", x_axis_type="datetime", width = 700, height = 300, x_range = v.x_range)
vv.xaxis.axis_label = 'Date'
vv.yaxis.axis_label = r"\[W \space m^{-2}\]"
vv.circle(x = "index", 
         y = "swdn", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

vv.circle(x = "index", 
         y = "SWDNB", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

vv.circle(x = "index", 
         y = "SWDNB", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

vv.legend.click_policy = "hide"
vv.legend.location = "top_left"

bins = np.linspace(0, 600, 50)
hist, edges = np.histogram(wrf_mod_vars.SWDNB.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.SWDNB.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

vv_hist = figure(title="Downwelling Shortwave Radiatio", width=300, height=300)
vv_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
vv_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

w = figure(title = "Upwelling Longwave Radiation", x_axis_type="datetime", width = 700, height = 300, x_range=v.x_range, y_range = (220, 320))
w.xaxis.axis_label = 'Date'
w.yaxis.axis_label = r"\[W \space m^{-2}\]"

w.circle(x = "index", 
         y = "lwup", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

w.circle(x = "index", 
         y = "LWUPB", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

w.circle(x = "index", 
         y = "LWUPB", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

w.legend.location = "bottom_right"

bins = np.linspace(150, 350, 50)
hist, edges = np.histogram(wrf_mod_vars.LWUPB.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.LWUPB.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

w_hist = figure(title="Longwave Upwelling Radiatio", width=300, height=300)
w_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
w_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

ww = figure(title = "Downwelling Longwave Radiation", x_axis_type="datetime", width = 700, height = 300, x_range = v.x_range, y_range = (150, 350))
ww.xaxis.axis_label = 'Date'
ww.yaxis.axis_label = r"\[W \space m^{-2}\]"

ww.circle(x = "index", 
         y = "lwdn", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'Measured')

ww.circle(x = "index", 
         y = "LWDNB", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

ww.circle(x = "index", 
         y = "LWDNB", 
         source = wrf_mod_cds,
         legend_label = 'WRF - Mod',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

ww.legend.click_policy = "hide"
ww.legend.location = "bottom_right"

bins = np.linspace(150, 350, 50)
hist, edges = np.histogram(wrf_mod_vars.LWDNB.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.LWDNB.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

ww_hist = figure(title="Longwave Downwelling Radiatio", width=300, height=300)
ww_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
ww_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')
show(column(row(v, v_hist), row(vv, vv_hist), row(w, w_hist), row(ww, ww_hist)))


# In[10]:


measurements = ColumnDataSource(data = {'index': epro_vals.index.values,
                                         'us' : epro_vals.us.values,
                                         'br' : epro_vals.br.values,
                                         'mo' : epro_vals.mo.values,
                                         'rl' : epro_vals.rl.values,
                                        })

w = figure(title = "Friction Velocity", x_axis_type="datetime", width = 700, height = 300, x_range = (wrf_mod_vars.index[0], wrf_mod_vars.index[-1]), y_range= (0, 1))
w.xaxis.axis_label = 'Date'
w.yaxis.axis_label = r"\[u^{*} \space (m \space s^{-2})\]"

w.circle(x = "index", 
         y = "us", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'EddyPro')

w.circle(x = "index", 
         y = "UST", 
         source = wrf_orig_cds,
         legend_label = 'WRF',
         line_color = 'red', 
         fill_color = 'white',
         alpha = 0.5)

w.circle(x = "index", 
         y = "UST", 
         source = wrf_mod_cds,
         legend_label = 'WRF',
         line_color = 'blue', 
         fill_color = 'white',
         alpha = 0.5)

w.legend.click_policy = "hide"

bins = np.linspace(0, 1, 30)
hist, edges = np.histogram(wrf_mod_vars.UST.values, density=False, bins=bins)
source_mod = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))
hist, edges = np.histogram(wrf_vars.UST.values, density=False, bins=bins)
source_orig = ColumnDataSource(dict(x=(edges[:-1] + edges[1:])/2, y=hist))

ww_hist = figure(title="Friction Velocity", width=300, height=300)
ww_hist.step(x = 'x', y = 'y', source = source_orig, mode = 'center', line_color = 'red', line_width = 2, legend_label = 'Original')
ww_hist.step(x = 'x', y = 'y', source = source_mod, mode = 'center', line_color = 'blue', line_width = 2, legend_label = 'Modified')

q = figure(title = "Bowen Ratio", x_axis_type="datetime", width = 700, height = 300, y_range = [-1000, 1000], x_range=w.x_range)
q.xaxis.axis_label = 'Date'
q.yaxis.axis_label = r"\[B_{r}\]"

q.circle(x = "index", 
         y = "br", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'EddyPro')

q.legend.click_policy = "hide"

r = figure(title = "Monin-Obukov Length", x_axis_type="datetime", width = 700, height = 300, y_range = [-10, 10], x_range=w.x_range)
r.xaxis.axis_label = 'Date'
r.yaxis.axis_label = r"\[(z-d)/L \space (m)\]"

r.circle(x = "index", 
         y = "mo", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'EddyPro')

r.legend.click_policy = "hide"

n = figure(title = "Roughness Length", x_axis_type="datetime", width = 700, height = 300, y_range = [-1, 1], x_range=w.x_range)
n.xaxis.axis_label = 'Date'
n.yaxis.axis_label = r"\[Z_{0} \space (m)\]"

n.circle(x = "index", 
         y = "rl", 
         source = measurements, 
         line_color = 'black', 
         fill_color = 'white',
         alpha = 0.5,
         legend_label = 'EddyPro')

n.legend.click_policy = "hide"


show(column(row(w, ww_hist), q, r, n))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




