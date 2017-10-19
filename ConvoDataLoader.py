#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:19:37 2017

@author: Yacalis
"""

import numpy as np
import h5py
from DataLoader import DataLoader


# =========================================================================
# FIXME: these almost definitely don't work, come back when you want them to
# Functions for when using convolutional neural nets
# =========================================================================
class ConvoDataLoader(DataLoader):

    # =========================================================================
    # Get data (overriding superclass)
    # =========================================================================
    def get_data(self) -> (np.ndarray, np.ndarray):
        x_data = self.reshape_x(self.x_data)
        if self.config.make_dum_data_y:
            y_data = self.make_dummy_data_y(self.x_data, self.config.dum_mult, self.config.dum_var)
        else:
            with h5py.File(self.nc_file, mode='r') as file:
                y_data = self.load_nc_data_y(file, self.output_vars, self.config.convert_units, self.convert_units)
        return x_data, y_data

    @staticmethod
    def make_dummy_data_y(x_data, dum_mult, dum_var):
        y_data = x_data * dum_mult
        m, n, o = y_data.shape
        # not sure what shape or features the random matrix should have below
        rando_pad = np.zeros(m, 1)
        rando = np.random.uniform(-dum_var, dum_var, [m, n])
        rando = np.concatenate((rando, rando_pad), axis=3)
        rando = np.concatenate((rando, rando_pad), axis=4)
        y_data = y_data + rando
        return y_data

    # =========================================================================
    # Class specific functions
    # =========================================================================
    @staticmethod
    def reshape_x(inputs: np.ndarray) -> np.ndarray:
        if inputs.shape[-1] == 1:
            inputs = np.tile(inputs, (1, 21))
        inputs = inputs[:, :, None]  # [b,z,1]
        x_data = np.stack(inputs, axis=-1)  # [b,z,1,c]

        return x_data

    @staticmethod
    def load_nc_data_y(file: h5py.File, output_vars: list, map_bool: bool, map_func: callable) -> np.ndarray:
        y_data = None
        for k in output_vars:
            if len(file[k].shape) > 1:
                arr = file[k][:].T[:, :, None, None]  # 3d variables
            else:
                arr = np.array(file[k][:])[None, :].T[:, :, None, None]  # 2d variables
            if map_bool:
                arr = map_func(arr, k)
            if y_data is None:
                y_data = arr
            else:
                y_data = np.concatenate((y_data, arr), axis=3)
        return y_data
