#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:20:39 2017

@author: Yacalis
"""

import time

nc_file = '../../cbrain/SP-CAM/SPCAM_outputs_tropics.nc'
mean_file = '../../cbrain/SP-CAM/SPCAM_mean.nc'
std_file = '../../cbrain/SP-CAM/SPCAM_std.nc'
max_file = '../../cbrain/SP-CAM/SPCAM_max.nc'


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
