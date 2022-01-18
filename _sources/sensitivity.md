# Sensitivity Study Setup

### Current Plan
1. Run an idealized case with the best idealized setup for all cases
2. If I see an improvement for these cases, see if implementing this into polar WRF would work

## Idealized

### Cases to run
1. Winter 1 - February 4, 5, and 6
    - Clear winter with some cloudy patches
2. Spring 1 - May 2, 3, and 4
    - Cloudy spring case
3. Spring 2 - May 22, 23, and 24
    - One clear spring day

### Namelist settings
The following is the physics and crm sections of the `namelist.input` files being used for the sensitivity study.

```bash
&physics
 seaice_threshold                    = 100,
 mp_physics                          = 50,  50, 50,
 ra_lw_physics                       = 3,  3, 3,
 ra_sw_physics                       = 3,  3, 3,
 radt                                = 1,  2, 5,
 sf_sfclay_physics                   = 1,  91, 91,
 sf_surface_physics                  = 1,  0, 0,
 bl_pbl_physics                      = 0,  0, 0,
 bldt                                = 0,  0, 0,
 cu_physics                          = 0,  0, 0,
 cudt                                = 5,  5, 5,
 isfflx                              = 1,
 ifsnow                              = 1,
 icloud                              = 1,
 num_soil_layers                     = 5,
 mp_zero_out                         = 0,
 ideal_xland                         = 2,
 fractional_seaice                   = 0,
 /

&crm
 crm_zsfc                            = 0.0,
 crm_lat                             = 81,
 crm_lon                             = 8.5,
 crm_stretch                         = 0,
 crm_num_pert_layers                 = 33,
 crm_pert_amp                        = 0.1,
 crm_init_ccn                        = 100,
 crm_lupar_opt                       = 0,
 crm_znt                             = 0.04,
 crm_emiss                           = 1.0,
 crm_thc                             = 3.,
 crm_mavail                          = 0.30,
 crm_force_opt                       = 1,
 crm_th_adv_opt                      = 1,
 crm_qv_adv_opt                      = 1,
 crm_th_rlx_opt                      = 0,
 crm_qv_rlx_opt                      = 0,
 crm_uv_rlx_opt                      = 1,
 crm_vert_adv_opt                    = 1,
 crm_wcpa_opt                        = 1,
 crm_num_force_layers                = 153,
 crm_tau_s                           = 1800,
 crm_tau_m                           = 1800,
 crm_flx_opt                         = 0,
 crm_sh_flx                          = 10,
 crm_lh_flx                          = 0,
 crm_albedo_opt                      = 1,
 crm_albedo                          = 0.8,
 crm_tsk_opt                         = 2,
 crm_tsk                             = 263,
 crm_ust_opt                         = 1,
 crm_ust                             = 0.01,
 crm_init_tke_opt                    = 0,
 crm_init_tke                        = 1.0,
 crm_morr_act_opt                    = 1,
 crm_morr_hygro_opt                  = 0,
 crm_morr_hygro                      = 0.12,
 crm_mp_nc                           = 448.62,
 crm_num_aer_layers                  = 153,
 crm_stat_opt                        = 1,
 crm_stat_sample_interval_s          = 1800.,
/

```
### Model setup

```{admonition} Do I need to recompile the model?
??
```

1. Change the values of the USGS section of `LANDUSE.TBL` from the following:

>SUMMER \
>24,     55.,   .95,   .95,   0.1,    5.,    0., 9.0e25, 'Snow or Ice'\
>WINTER \
>24,     70.,   .95,   .95,   0.1,    5.,    0., 9.0e25, 'Snow or Ice'

&emsp;&emsp;to

>SUMMER \
>24,     81.,   .95,   .98,   0.001,    5.,    0., 1.8e06, 'Snow or Ice'\
>WINTER \
>24,     86.,   .95,   .98,   0.001,    5.,    0., 1.8e06, 'Snow or Ice'



---
## Real
