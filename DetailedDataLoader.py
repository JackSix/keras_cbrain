#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:19:17 2017

@author: Yacalis
"""

import numpy as np
import h5py
from DataLoader import DataLoader
from random import randint


class DetailedDataLoader(DataLoader):
    """
     Pritch -- hacked to interface with bigger data set organized as:
     3D: float TAP(date, time, lev, lat, lon) -- len(lev) = 21
     2D: float SHFLX(date, time, lat, lon)

     - This class is used with Stephan's file: SPCAM_outputs_detailed.nc
     - This class is for a regular dense network, NOT a convolutional network
     - Should be used when config.use_detailed_data == True
    """
    def load_nc_data(self, file: h5py.File, var_list: list, map_bool: bool, map_func: callable) -> np.ndarray:
        num_samples = 10000
        month, time, lat, lon = self.gen_rand_samp_indexes(num_samples)

        data = None
        for k in var_list:

            print('Reading data for:', k)
            var_data = file[k]
            if len(var_data.shape) > 4:  # 3D
                arr = np.zeros((num_samples, 21), np.float32)
                for j in range(num_samples):
                    arr[j, :] = var_data[month[j], time[j], :, lat[j], lon[j]]
            else:  # 2D
                arr = np.zeros((num_samples, 1), np.float32)
                for j in range(num_samples):
                    arr[j] = var_data[month[j], time[j], lat[j], lon[j]]

            if map_bool:
                arr = map_func(arr, k)  # e.g. rescale SPDT, SPDQ.
            if data is None:
                data = arr
            else:
                data = np.concatenate((data, arr), axis=1)
        return data

    @staticmethod
    def gen_rand_samp_indexes(num_samples: int) -> (list, list, list, list):
        month = []
        time = []
        lat = []
        lon = []
        for i in range(num_samples):
            month.append(randint(0, 11))
            time.append(randint(1, 46))
            lat.append(randint(0, 63))
            lon.append(randint(0, 127))
        return month, time, lat, lon
