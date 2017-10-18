#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:42:52 2017

@author: Yacalis
"""

from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam, \
    RMSprop, SGD


class Optimizer:
    def __init__(self, config):
        self.optimizer = self.main(config)
        return

    # ==========================================================================
    # NOTE: the default options (based on papers) for each optimizer
    # is listed to the right of the arguments of each optimizer
    # ==========================================================================
    def main(self, config):

        if config.optimizer == 'adadelta':
            optimizer = Adadelta(lr=config.lr,  # 1.0
                                 rho=config.rho,  # 0.95
                                 epsilon=config.epsilon,  # 1e-08
                                 decay=config.decay)  # 0.0

        elif config.optimizer == 'adagrad':
            optimizer = Adagrad(lr=confgi.lr,  # 0.01
                                epsilon=config.epsilon,  # 1e-08
                                decay=config.decay)  # 0.0

        elif config.optimizer == 'adam':
            optimizer = Adam(lr=config.lr,  # 0.001
                             beta_1=config.beta_1,  # 0.9
                             beta_2=config.beta_2,  # 0.999
                             epsilon=config.epsilon,  # 1e-08
                             decay=config.decay)  # 0.0

        elif config.optimizer == 'adamax':
            optimizer = Adamax(lr=config.lr,  # 0.001
                               beta_1=config.beta_1,  # 0.9
                               beta_2=config.beta_2,  # 0.999
                               epsilon=config.epsilon,  # 1e-08
                               decay=config.decay)  # 0.0

        elif config.optimizer == 'nadam':
            optimizer = Nadam(lr=config.lr,  # 0.001
                              beta_1=config.beta_1,  # 0.9
                              beta_2=config.beta_2,  # 0.999
                              epsilon=config.epsilon,  # 1e-08
                              schedule_decay=config.schedule_decay)  # 0.004

        elif config.optimizer == 'rmsprop':
            optimizer = RMSprop(lr=config.lr,  # 0.001
                                rho=config.rho,  # 0.9
                                epsilon=config.epsilon,  # 1e-08
                                decay=config.decay)  # 0.0

        elif config.optimizer == 'sgd':
            optimizer = SGD(lr=config.lr,  # 0.01
                            momentum=config.momentum,  # 0.0
                            decay=config.decay,  # 0.0
                            nesterov=config.nesterov)  # False

        else:
            raise Exception('[!] Something is wrong - the name of the \
                            optimizer is not a valid choice. Valid choices: \
                            adadelta, adagrad, adam, adamax, nadam, rmsprop, \
                            sgd')

        return optimizer
