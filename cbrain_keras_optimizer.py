#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:42:52 2017

@author: Yacalis
"""

import keras
from keras.optimizers import Adam

class Optimizer:
    def __init__(self, config):
        self.main(config)
        return
    
    def main(self, config):
        if config.optimizer == 'adam':
            self.optimizer = Adam(lr = config.lr,
                 beta_1 = config.beta_1,
                 beta_2 = config.beta_2,
                 epsilon = config.epsilon,
                 decay = config.decay)
        else:
            raise Exception('[!] Something is wrong - the name of the optimizer \
                            is not a valid choice. Valid choice(s): adam')
        return