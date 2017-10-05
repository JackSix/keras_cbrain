#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:20:39 2017

@author: Yacalis
"""

import time

nc_file          = '../cbrain/SP-CAM/SPCAM_outputs_tropics.nc'
mean_file     = '../cbrain/SP-CAM/SPCAM_mean.nc'
std_file      = '../cbrain/SP-CAM/SPCAM_std.nc'
max_file      = '../cbrain/SP-CAM/SPCAM_max.nc'

def get_logdir(config):
    logdir = "datetime_" + time.strftime("%Y%m%d") + "_" + time.strftime("%H%M%S")
    logdir += "-hidden"
    for lay in config.hidden_lays.split(','):
        logdir = logdir + "_" + lay
    logdir += "-epochs_" + str(config.epochs)
    logdir += "-batchsize_" + str(config.batch_size)
    logdir += "-dummydata_" + str(config.use_dum_data_xy or config.use_dum_data_y)
    
    return "./logs/{}".format(logdir)