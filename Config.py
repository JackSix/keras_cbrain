#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:53:31 2017

@author: Yacalis
"""

import os
import json
import configargparse


class Config:

    def __init__(self):
        self.config, unparsed = self.main()
        if unparsed:
            raise Exception('[!] Something is wrong - there are unrecognized \
                            parameters present.')
        return
    
    @staticmethod
    def main() -> (object, object):
        parser = configargparse.ArgParser()

        # Callbacks
        cback_arg = parser.add_argument_group('Callbacks')
        # Earlystopping
        cback_arg.add_argument('--es_min_delta', type=float, default=0.00001)
        cback_arg.add_argument('--es_mode', type=str, default='min')
        cback_arg.add_argument('--es_monitor', type=str, default='val_loss')
        cback_arg.add_argument('--es_patience', type=int, default=10)
        # ModelCheckpoint
        cback_arg.add_argument('--mc_monitor', type=str, default='val_loss')
        # ReduceLROnPlateau
        cback_arg.add_argument('--lr_epsilon', type=float, default=1e-04)
        cback_arg.add_argument('--lr_factor', type=float, default=0.1)
        cback_arg.add_argument('--lr_min_lr', type=float, default=1e-07)
        cback_arg.add_argument('--lr_mode', type=str, default='min')
        cback_arg.add_argument('--lr_monitor', type=str, default='val_loss')
        cback_arg.add_argument('--lr_patience', type=int, default=5)

        # Data
        data_arg = parser.add_argument_group('Data')
        data_arg.add_argument('--output_vars', type=str, default='SPDT,SPDQ')
        data_arg.add_argument('--input_vars', type=str, default='TAP,QAP,SHFLX,LHFLX,dQdt_adiabatic,dTdt_adiabatic')
        data_arg.add_argument('--normalize', type=bool, default=True)
        data_arg.add_argument('--convert_units', type=bool, default=True)
        
        # Dummy data
        dum_arg = parser.add_argument_group('Dummy')
        dum_arg.add_argument('--dum_mult', type=float, default=1)
        dum_arg.add_argument('--dum_var', type=float, default=0)
        dum_arg.add_argument('--make_dum_data_y', type=bool, default=False)

        # Misc
        misc_arg = parser.add_argument_group('Misc')
        misc_arg.add_argument('--metrics', type=str, default=None)
        misc_arg.add_argument('--random_seed', type=int, default=123)
        misc_arg.add_argument('--use_convo', type=bool, default=False)

        # Network
        net_arg = parser.add_argument_group('Network')
        net_arg.add_argument('--dropout_rate', type=float, default=0.5)
        net_arg.add_argument('--hidden_lays',  type=str, default='1024,1024')
        net_arg.add_argument('--hidden_lays_act', type=str, default='relu')
        net_arg.add_argument('--loss_func', type=str, default='mse')
        net_arg.add_argument('--output_lay_act',  type=str, default='relu')
        net_arg.add_argument('--use_dropout', type=bool, default=False)
        
        # Optimizer
        optim_arg = parser.add_argument_group('Optimizer')
        optim_arg.add_argument('--beta_1', type=float, default=0.5)
        optim_arg.add_argument('--beta_2', type=float, default=0.999)
        optim_arg.add_argument('--decay', type=float, default=0.0)
        optim_arg.add_argument('--epsilon', type=float, default=1e-08)
        optim_arg.add_argument('--lr', type=float, default=1e-06)
        optim_arg.add_argument('--optimizer', type=str, default='adam')
        optim_arg.add_argument('--schedule_decay', type=float, default=0.004)
        optim_arg.add_argument('--rho', type=float, default=0.95)
        optim_arg.add_argument('--momentum', type=float, default=0.0)
        optim_arg.add_argument('--nesterov', type=bool, default=False)

        # Tensorboard
        tboard_arg = parser.add_argument_group('Tensorboard')
        tboard_arg.add_argument('--histogram_freq', type=int, default=0)
        tboard_arg.add_argument('--write_graph', type=bool, default=False)
        tboard_arg.add_argument('--write_grads', type=bool, default=False)
        tboard_arg.add_argument('--write_images', type=bool, default=True)
        
        # Training and testing
        train_arg = parser.add_argument_group('Training')
        train_arg.add_argument('--batch_size', type=int, default=128)
        train_arg.add_argument('--epochs', type=int, default=200)
        train_arg.add_argument('--frac_test', type=float, default=0.2)
        train_arg.add_argument('--shuffle', type=bool, default=True)
        train_arg.add_argument('--valid_split', type=float, default=0.2)
        
        return parser.parse_known_args()

    @staticmethod
    def save_config(config: object, logdir: str) -> None:
        param_path = os.path.join(logdir, 'params.json')
        with open(param_path, 'w') as f:
            json.dump(config.__dict__, f, indent=4, sort_keys=True)
