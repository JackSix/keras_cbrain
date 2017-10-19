#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:35:57 2017

@author: Yacalis
"""

import numpy as np
import h5py
from folder_defs import nc_file, mean_file, std_file


# =============================================================================
# DataLoader class
# =============================================================================
class DataLoader:

    def __init__(self, config: object) -> None:
        self.input_vars = config.input_vars.split(',')
        self.output_vars = config.output_vars.split(',')
        self.mean, self.std = self.load_mean_and_std_dicts()
        with h5py.File(nc_file, mode='r') as file:
            self.x_data = self.load_nc_data(file, config.input_vars, config.normalize, self.norm_data)
        self.nc_file = nc_file
        self.config = config

    # =========================================================================
    # Loading mean and std data
    # =========================================================================
    def load_mean_and_std_dicts(self) -> (dict, dict):
        mean = self.load_values_to_dict(mean_file, self.input_vars)
        std = self.load_values_to_dict(std_file, self.input_vars)
        return mean, std

    @staticmethod
    def load_values_to_dict(file: str, var_list: list) -> dict:
        """
        This iterates through each variable in the file, adds each variable as a
        key, and assigns that variable's data as the key's value (as a numpy
        array). As is, it is used for normalization and no other purpose.
        """
        dic = {}
        with h5py.File(file, mode='r') as fh:
            for k in var_list:
                dic[k] = np.array(fh[k])[None]
        return dic

    # =========================================================================
    # Loading and formatting nc data
    # =========================================================================
    def norm_data(self, arr, varname):
        arr -= self.mean[varname]
        arr /= self.std[varname]
        return arr

    @staticmethod
    def convert_units(arr, varname):
        """Make sure SPDQ and SPDT have comparable units, in W/kg"""
        if varname == 'SPDT':
            return arr * 1000
        if varname == 'SPDQ':
            return arr * 2.5e6
        return arr

    @staticmethod
    def load_nc_data(file: h5py.File, var_list: list, map_bool: bool, map_func: callable) -> np.ndarray:
        """
        This iterates through each variable, figures out if it is 2D or 3D,
        loads the data as numpy arrays, applies a function to the data if
        map_bool is true, and concatenates each variable's data together along
        axis 1 (adding more rows but keeping the column numbers the same)
        """
        data = None
        for k in var_list:
            if len(file[k].shape) > 1:
                arr = file[k][:].T  # 3d variables
            else:
                arr = np.array(file[k])[None].T  # 2d variables
            if map_bool:
                arr = map_func(arr, k)
            if data is None:
                data = arr
            else:
                data = np.concatenate((data, arr), axis=1)
        return data

    # =========================================================================
    # Get data (overridden in subclasses)
    # =========================================================================
    def get_data(self) -> (np.ndarray, np.ndarray):
        if self.config.make_dum_data_y:
            y_data = self.make_dummy_data_y(self.x_data, self.config.dum_mult, self.config.dum_var)
        else:
            with h5py.File(self.nc_file, mode='r') as file:
                y_data = self.load_nc_data(file, self.config.output_vars, self.config.convert_units, self.convert_units)
        return self.x_data, y_data

    @staticmethod
    def make_dummy_data_y(x_data, multiplier, variance):
        """
        Creates y_data by multiplying x_data by a scalar, and then adding some
        random variance to each value.
        """
        y_data = x_data * multiplier
        m, n = y_data.shape
        rando = np.random.uniform(-variance, variance, [m, n])
        y_data = y_data + rando
        return y_data
