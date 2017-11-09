#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 15:59:30 2017

@author: Yacalis
"""

from ConvoDataLoader import ConvoDataLoader
from Config import Config
from f90_autogen.python_convo_loop import pad_z_axis, predict_spdt_spdq


print('Loading Data')

configuration = Config()
config = configuration.config
# config.input_vars = 'TAP,QAP,dQdt_adiabatic,dTdt_adiabatic'
dataloader = ConvoDataLoader(config)
x_data, y_data = dataloader.get_data()

print(type(x_data), type(y_data))
print(x_data.shape, y_data.shape)
print('Done.')
