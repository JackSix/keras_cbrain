#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:49:38 2017

@author: Yacalis
"""

print('Begin Cbrain Keras Program')
print('Importing Libraries, Packages, and Modules')

# =============================================================================
# NOTE: To properly set random seed for reproducible results, the code has to
# be limited to a single thread, and a separate chunk of code must be added to
# the top of this file. Detailed instructions are documented here:
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-
# using-keras-during-development
# =============================================================================

import numpy as np
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from CustomCallbacks import CustomCallbacks
from CustomMetrics import CustomMetrics
from Config import Config
from DataLoader import DataLoader
from DetailedDataLoader import DetailedDataLoader
from ConvoDataLoader import ConvoDataLoader
from ConvoDetailedDataLoader import ConvoDetailedDataLoader
from folder_defs import get_logdir
from Optimizer import Optimizer
from act_funcs import act_funcs
from loss_funcs import loss_funcs


def main():
    # ==========================================================================
    # Get Data
    # ==========================================================================
    print('Loading Data')

    configuration = Config()
    config = configuration.config

    if config.use_convo:
        if config.use_detailed_data:
            dataloader = ConvoDetailedDataLoader(config)
        else:
            dataloader = ConvoDataLoader(config)
    else:
        if config.use_detailed_data:
            dataloader = DetailedDataLoader(config)
        else:
            dataloader = DataLoader(config)
    x_data, y_data = dataloader.get_data()

    # ==========================================================================
    # Set Parameter Constants
    # ==========================================================================
    print('Setting Parameters')

    # hidden layers
    hidden_lays = list(map(int, config.hidden_lays.split(',')))
    nhidden = len(hidden_lays)

    # activation and loss functions
    hidden_lays_act = act_funcs[config.hidden_lays_act]
    output_lay_act = act_funcs[config.output_lay_act]
    loss_func = loss_funcs[config.loss_func]

    # input/output data dims
    input_dim = x_data.shape[1]
    output_dim = y_data.shape[1]

    # ==========================================================================
    # Metrics, Callback, Optimizer
    # ==========================================================================
    print('Setting Up Metrics, Callbacks, and Optimizer')

    # add metrics from config and custom metrics
    metrics = config.metrics.split(',') if config.metrics else []
    custom_metrics = CustomMetrics().metrics
    for metric in custom_metrics:
        metrics.append(metric)

    # set up callbacks and optimizer
    log_dir = get_logdir(config)
    callbacks = CustomCallbacks(config, log_dir).callbacks
    optimizer = Optimizer(config).optimizer

    # ==========================================================================
    # Build Model
    # ==========================================================================
    print('Building Model')

    # create layer arrangement
    model = Sequential()

    # add input layer and first hidden layer
    model.add(Dense(hidden_lays[0], input_dim=input_dim))
    model.add(Activation(hidden_lays_act))
    if config.use_dropout:
        model.add(Dropout(config.dropout_rate))

    # add any hidden layers after the first one
    for layer in range(nhidden - 1):
        model.add(Dense(hidden_lays[layer + 1]))
        model.add(Activation(hidden_lays_act))
        if config.use_dropout:
            model.add(Dropout(config.dropout_rate))

    # add output layer
    model.add(Dense(output_dim))
    model.add(Activation(output_lay_act))

    # compile model
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=metrics)

    # ==========================================================================
    # Train Model
    # ==========================================================================
    print('Training Model')

    # split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=config.frac_test)

    # train the model
    model.fit(x_train,
              y_train,
              epochs=config.epochs,
              batch_size=config.batch_size,
              shuffle=config.shuffle,
              validation_split=config.valid_split,
              verbose=1,
              callbacks=callbacks)

    print('Model Training Completed')
    print('Score: ', model.evaluate(x_test, y_test, batch_size=config.batch_size))

    # ==========================================================================
    # Save Model, Weights, and Config Options
    # ==========================================================================
    print('Saving Model, Weights, and Config Options')

    configuration.save_config(config, log_dir)

    model_fp = log_dir + '/finished_model.hdf5'
    model.save(model_fp)

    weights_fp = log_dir + '/finished_weights.hdf5'
    model.save_weights(weights_fp)

    np_weights_fp = log_dir + '/np_finished_weights.txt'
    weights = model.get_weights()
    with open(np_weights_fp, 'w+') as file:
        for i in range(len(weights)):
            if i != len(weights) - 1:
                file.write('shape: ' + str(weights[i].shape) + '\n')
                file.write(str(weights[i]) + '\n')
            else:
                file.write('shape: ' + str(weights[i].shape) + '\n')
                file.write(str(weights[i]))

    # ==========================================================================
    # NOTE: For instructions on how to load the saved model and/or weights, see:
    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    # ==========================================================================

    print('End Cbrain Keras Program')


# ======================================================================
# This must be at the bottom of the file
# ======================================================================
if __name__ == '__main__':
    main()
