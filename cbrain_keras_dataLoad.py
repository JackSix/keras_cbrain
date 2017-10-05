#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:35:57 2017

@author: Yacalis
"""

import numpy as np
import h5py
from cbrain_keras_folderDefs import nc_file, mean_file, std_file

# =============================================================================
# DataLoader class
# =============================================================================
class DataLoader:
    def __init__(self, config):        
        self.input_vars = config.input_vars.split(',')
        self.datasets = config.datasets.split(',')
        self.nc_vars = self.input_vars + self.datasets # currently, not used
        self.mean = {}
        self.std = {}        
        
        self.main(config)

    def main(self, config):        
        self.loadValuesToDict(self.mean, mean_file, self.input_vars)
        self.loadValuesToDict(self.std, std_file, self.input_vars)

        with h5py.File(nc_file, mode='r') as fh:
            self.nlevels = fh['z'].shape[0] # currently, not used
            self.nsamples = fh['t'].shape[0] # currently, not used
            self.x_data, self.y_data = self.accessData(
                    fh, self.input_vars, self.datasets, config.use_convo,
                    config.normalize, config.use_dum_data_xy,
                    config.use_dum_data_y, config.dum_mult, config.dum_var,
                    config.dum_samples, config.dum_levels)

    # =========================================================================
    # Accessing data
    # =========================================================================    
    def accessData(self, file, input_vars, datasets, use_convo, normalize,
                   dum_xy, dum_y, dum_mult, dum_var, dum_samp, dum_lev):
        """There are 5 total possibilities:
            (1) no convolution, dummy x data, dummy y data
            (2) is convolution, real x data,  dummy y data
            (3) is convolution, real x data,  real y data
            (4) no convolution, real x data,  dummy y data
            (5) no convolution, real x data,  real y data
            """
        manydatasets = len(self.datasets) > 1
        # x data
        if dum_xy:
            return self.makeDummyDataXY(dum_mult, dum_var, dum_samp, dum_lev)
        else:
            x_data = self.loadNcData(file, input_vars, normalize, self.normData)
        
        # y data
        if use_convo and dum_y:
            x_data = self.useConvoReshapeX(x_data)
            y_data = self.useConvoMakeDummyDataY(x_data, dum_mult, dum_var)
        
        elif use_convo and not dum_y:
            x_data = self.useConvoReshapeX(x_data)
            y_data = self.useConvoLoadNcDataY(file, datasets, manydatasets, self.convertUnits)
        
        elif not use_convo and dum_y:
            y_data = self.makeDummyDataY(x_data, dum_mult, dum_var)
            
        elif not use_convo and not dum_y:
            y_data = self.loadNcData(file, datasets, manydatasets, self.convertUnits)
        
        else:
            raise Exception('[!] Something is wrong - one or more of use_convo, \
                            dum_xy, and dum_y bools are not set properly.')
            
        return x_data, y_data

    # =========================================================================
    # Loading and formatting data
    # =========================================================================
    def loadValuesToDict(self, dic, file, var_list):
        with h5py.File(file, mode='r') as fh:
            for k in var_list:
                dic[k] = np.array(fh[k])[None]
        return
    
    def loadNcData(self, file, var_list, map_bool, map_func):
        data = None
        for k in var_list:
            if len(file[k].shape) > 1:
                arr = file[k][:].T # 3d variables
            else:
                arr = np.array(file[k])[None].T # 2d variables
            if map_bool:
                arr = map_func(arr, k)
            if data is None:
                data = arr
            else:
                data = np.concatenate((data, arr), axis=1)
        
        return data
    
    def normData(self, arr, varname):        
        arr -= self.mean[varname]
        arr /= self.std[varname]        
        return arr
    
    def convertUnits(self, arr, varname):
        """Make sure SPDQ and SPDT have comparable units"""
        if varname == "SPDT":
            return arr*1000
        if varname == "SPDQ":
            return arr*2.5e6
        
        return arr
    
    def makeDummyDataXY(self, dum_mult, dum_var, dum_samp, dum_lev):
        x_data = np.random.rand(dum_samp, dum_lev)
        y_data = self.makeDummyDataY(x_data, dum_mult, dum_var)
        
        return x_data, y_data
    
    def makeDummyDataY(self, x_data, dum_mult, dum_var):
        y_data = x_data * dum_mult
        m, n = y_data.shape
        rando = np.random.uniform(-dum_var, dum_var, [m, n])
        y_data = y_data + rando
        
        return y_data

    # =========================================================================
    # TODO: these almost definitely don't work, come back when you want them to
    # Functions for when using convolutional neural nets
    # =========================================================================    
    def useConvoReshapeX(self, inputs):
        if inputs.shape[-1] == 1:
            inputs = np.tile(inputs, (1,self.Nlevels))
        inputs = inputs[:,:,None] #[b,z,1]
        x_data = np.stack(inputs, axis=-1) #[b,z,1,c]
        
        return x_data
    
    def useConvoMakeDummyDataY(self, x_data, dum_mult, dum_var):
        y_data = x_data * dum_mult
        m, n, o = y_data.shape
        # not sure what shape or features the random matrix should have below
        rando_pad = np.zeros(m, 1)
        rando = np.random.uniform(-dum_var, dum_var, [m, n])
        rando = np.concatenate((rando, rando_pad), axis=3)
        rando = np.concatenate((rando, rando_pad), axis=4)
        y_data = y_data + rando
        
        return y_data
    
    def useConvoLoadNcDataY(self, file, datasets, map_bool, map_func):
        y_data = None
        for k in datasets:
            if len(file[k].shape) > 1:
                # 3d variables
                arr = file[k][:].T[:,:,None,None]# [b,h,1,c=1]
            else:
                # 2d variables
                arr = np.array(file[k][:])[None,:].T[:,:,None,None]
            if map_bool:
                arr = map_func(arr, k)
            if y_data is None:
                y_data = arr
            else:
                y_data = np.concatenate((y_data, arr), axis=3)
        
        return y_data

