#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:15:50 2017

@author: Yacalis
"""

from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU

act_funcs = {
    'elu': 'elu',
    'selu': 'selu',
    'relu': 'relu',
    'tanh': 'tanh',
    'sigmoid': 'sigmoid',
    'hard_sigmoid': 'hard_sigmoid',
    'hardsigmoid': 'hard_sigmoid',
    'hsigmoid': 'hard_sigmoid',
    'h_sigmoid': 'hard_sigmoid',
    'hs': 'hard_sigmoid',
    'linear': 'linear',
    'softmax': 'softmax',
    'softplus': 'softplus',
    'softsign': 'softsign',
    'prelu': PReLU(),
    'elu': ELU(),
    'thresholded_relu': ThresholdedReLU(),
    'thresholdedrelu': ThresholdedReLU(),
    'trelu': ThresholdedReLU(),
    't_relu': ThresholdedReLU(),
    'tr': ThresholdedReLU(),
    'leaky_relu': LeakyReLU(),
    'leakyrelu': LeakyReLU(),
    'lrelu': LeakyReLU(),
    'l_relu': LeakyReLU(),
    'lr': LeakyReLU()
}
