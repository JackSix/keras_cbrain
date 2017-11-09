#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:19:37 2017

@author: Yacalis
"""

import numpy as np
import h5py
from DataLoader import DataLoader


class ConvoDataLoader(DataLoader):

    @staticmethod
    def load_nc_data(file: h5py.File, output_vars: list, map_bool: bool, map_func: callable) -> np.ndarray:
        y_data = None
        for k in output_vars:
            if len(file[k].shape) > 1:
                arr = file[k][:].T[:, :, None, None]  # 3d variables
            else:
                arr = file[k][:][None, :].T[:, :, None, None]  # 2d variables
                arr = np.tile(arr, (1, 21, 1, 1))
            if map_bool:
                arr = map_func(arr, k)
            if y_data is None:
                y_data = arr
            else:
                y_data = np.concatenate((y_data, arr), axis=3)
        return y_data

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
