#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 15:59:30 2017

@author: Yacalis
"""

import numpy as np
from Config import Config
from ConvoDataLoader import ConvoDataLoader
from python_convo_loop import chckpt_ver_predict_spdt_spdq
from inspect_checkpoint import print_tensors_in_checkpoint_file


def main():
    print('Starting Program')

    print('Setting Constants')
    input_vars = 'TAP,QAP,OMEGA,SHFLX,LHFLX,LAT,dTdt_adiabatic,dQdt_adiabatic,QRL,QRS'
    chckpt_file = '../' \
                  'tensorboard_logs/' \
                  'Pierre/' \
                  '1028_123916_' \
                  'SPDT,SPDQ_' \
                  'layers_32,32,32,32,32,32_' \
                  'kdr_1.0_ac_0_' \
                  'convo_True_' \
                  f'variables_{input_vars}_' \
                  'batchsize_128/' \
                  'model.ckpt-1192378'
    tensor_name = ''
    all_tensors = True

    print('Getting Checkpoint Filters and Biases')
    filters, biases = print_tensors_in_checkpoint_file(
        chckpt_file, tensor_name, all_tensors)

    print('Loading SP-CAM Data')
    configuration = Config()
    config = configuration.config
    config.input_vars = input_vars
    dataloader = ConvoDataLoader(config)
    x_data, y_data = dataloader.get_data()
    print('x_data shape: ', x_data.shape)
    print('y_data shape: ', y_data.shape)
    print('filters size: ', len(filters), len(filters[0]), len(filters[0][0]), len(filters[0][0][0]), len(filters[0][0][0][0]))
    print('biases size: ', len(biases), len(biases[0]))

    print('Running Convo Loop on SP-CAM Data with Chkpt Filters and Biases')
    new_state_arr = []
    y_data_arr = []
    for sample in range(10):
        rand_samp = np.random.randint(0, x_data.shape[0])
        state = x_data[rand_samp][:, 0, :]
        new_state = chckpt_ver_predict_spdt_spdq(state.tolist(), filters, biases)
        new_state_arr.append(new_state)
        y_data_arr.append(y_data[rand_samp+1][:, 0, :].tolist())
    print('pred: ', new_state_arr)
    print('actual: ', y_data_arr)
    print('input state size: ', len(state), len(state[0]))
    print('output state size: ', len(new_state), len(new_state[0]))
    print('Done.')


if __name__ == '__main__':
    main()
