#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 3 17:39:31 2017

@author: Yacalis
"""

import keras.backend as K


class CustomMetrics:
    def __init__(self):
        return

    @property
    def metrics(self):
        return [self.log10_loss]

    # Log base 10 of loss function
    @staticmethod
    def log10_loss(y_true, y_pred):
        reg_loss = K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))
        log_loss = K.log(reg_loss + 1e-20) / K.log(10.0)
        return log_loss
