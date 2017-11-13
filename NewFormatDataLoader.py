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


class NewFormatDataLoader(DataLoader):
    """
    Pritch -- hacked to interface with bigger data set organized as:
    3D: float TAP(date, time, lev, lat, lon)
    2D: float PS(date, time, lat, lon)
    """
    @staticmethod
    def load_nc_data(file: h5py.File, var_list: list, map_bool: bool, map_func: callable) -> np.ndarray:
        data = None
        for varname in var_list:
            print('Reading data for ', varname)
            is3D = len(file[varname].shape) > 4

            if is3D:
                arr = np.zeros((5000, 21), np.float32)
            else:
                arr = np.zeros((5000, 1), np.float32)

            for jj in range(5000):
                imonth = randint(0, 11)
                itimeofday = randint(1, 46)
                ilon = randint(0, 127)
                ilat = randint(0, 63)
                if is3D:
                    arr[jj, :] = file[varname][imonth, itimeofday, :, ilat, ilon]
                else:
                    arr[jj] = file[varname][imonth, itimeofday, ilat, ilon]

            if map_bool:
                arr = map_func(arr, varname)  # e.g. rescale SPDT, SPDQ.
            if data is None:
                data = arr
            else:
                data = np.concatenate((data, arr), axis=1)
        return data
