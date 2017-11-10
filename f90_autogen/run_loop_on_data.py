#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 15:59:30 2017

@author: Yacalis
"""

from ConvoDataLoader import ConvoDataLoader
from Config import Config
from f90_autogen.python_convo_loop import chckpt_ver_predict_spdt_spdq
from f90_autogen.inspect_checkpoint import print_tensors_in_checkpoint_file


def main():
    print('Starting Program')

    print('Setting Constants')
    input_vars = 'TAP,QAP,dQdt_adiabatic,dTdt_adiabatic,SHFLX,LHFLX'
    chckpt_file = '../' \
                  '../' \
                  'tensorboard_logs_tmp/' \
                  '1017-1018_logs/' \
                  '1018_122541_' \
                  'SPDT,SPDQ_' \
                  'layers_32,32,32,32,32,32_' \
                  '_kdr_1.0_ac_0_' \
                  'convo_True_' \
                  f'variables_{input_vars}/' \
                  'model.ckpt-451183'
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

    print('Running Convo Loop on SP-CAM Data with Chkpt Filters and Biases')
    state = x_data[0][:, 0, :]
    new_state = chckpt_ver_predict_spdt_spdq(state.tolist(), filters, biases)
    print('filters size: ', len(filters), len(filters[0]), len(filters[0][0]), len(filters[0][0][0]), len(filters[0][0][0][0]))
    print('biases size: ', len(biases), len(biases[0]))
    print('input state size: ', len(state), len(state[0]))
    print('output state size: ', len(new_state), len(new_state[0]))
    print(new_state)

    print('Done.')


if __name__ == '__main__':
    main()
