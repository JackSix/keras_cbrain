#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:13:17 2017

@author: Yacalis
"""

import numpy as np
import h5py
from DetailedDataLoader import DetailedDataLoader

class ConvoDetailedDataLoader(DetailedDataLoader):
    """
     data set organized as:
     3D: float TAP(date, time, lev, lat, lon) -- len(lev) = 21
     2D: float SHFLX(date, time, lat, lon)

     - This class is used with Stephan's file: SPCAM_outputs_detailed.nc
     - This class is for a convolutional network, NOT a regular dense network
     - Should be used when config.use_detailed_data == True
    """
    def load_nc_data(self, file: h5py.File, var_list: list, map_bool: bool, map_func: callable) -> np.ndarray:
        num_samples = self.config.detail_data_num_samples
        month, time, lat, lon = self.gen_rand_samp_indexes(num_samples)

        data = None
        for k in var_list:

            print('Reading data for:', k)
            var_data = file[k]
            if len(var_data.shape) > 4:  # 3D
                arr = np.zeros((num_samples, 21, 1, 1), np.float32)
                for j in range(num_samples):
                    arr[j, :, 0, 0] = var_data[month[j], time[j], :, lat[j], lon[j]]
            else:  # 2D
                arr = np.zeros((num_samples, 1, 1, 1), np.float32)
                for j in range(num_samples):
                    arr[j, 0, 0, 0] = var_data[month[j], time[j], lat[j], lon[j]]
                arr = np.tile(arr, (1, 21, 1, 1))
            if map_bool:
                arr = map_func(arr, k)
            if data is None:
                data = arr
            else:
                data = np.concatenate((data, arr), axis=3)
        return data

    def norm_data(self, arr, varname):
        if len(self.mean[varname].shape) > 1:
            mean = self.mean[varname][:, :, None, None]
            std = self.std[varname][:, :, None, None]
        else:
            mean = self.mean[varname][:][None, :, None, None]
            std = self.std[varname][:][None, :, None, None]
            mean = np.tile(mean, (1, 21, 1, 1))
            std = np.tile(std, (1, 21, 1, 1))
        arr -= mean
        arr /= std
        return arr
