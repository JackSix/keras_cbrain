#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:20:39 2017

@author: Yacalis
"""

import time
from Config import Config

mean_file = '../SPCAM/SPCAM_new_norms/SPCAM_mean.nc'
std_file = '../SPCAM/SPCAM_new_norms/SPCAM_std.nc'
max_file = '../SPCAM/SPCAM_new_norms/SPCAM_max.nc'

# =============================================================================
# the 'old' format combines [lat, lon, month, time] into one dimension,
# but the 'new' format leaves all of that uncombined
# =============================================================================
if Config().config.use_new_data_format:
    nc_file = '../SPCAM/SPCAM_Stephan/SPCAM_outputs_detailed.nc'
else:
    nc_file = '../SPCAM/SPCAM_Pierre/SPCAM_outputs_tropics.nc'

# =============================================================================
# the logdir name is long, but it beats having to look at the parameter json
# file just to see what the most important values are
# =============================================================================
def get_logdir(config: object) -> str:
    logdir = time.strftime('%m%d') + '_' + time.strftime('%H%M%S')
    logdir += '-input_' + config.input_vars
    logdir += '-output_' + config.output_vars
    logdir += '-hidden'
    for lay in config.hidden_lays.split(','):
        logdir = logdir + '_' + lay
    logdir += '-batch_' + str(config.batch_size)

    return './logs/' + logdir
