#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 09:26:01 2017

@author: Yacalis
"""

import numpy as np  # just for test examples, actual code does not use it


# ======================================================================
# The actual code
# ======================================================================
def pad_z_axis(matrix: list) -> list:
    top_padding = matrix[0][:]
    bottom_padding = matrix[-1][:]
    matrix.insert(0, top_padding)
    matrix.insert(-1, bottom_padding)

    return matrix


def predict_spdt_spdq(x_input: list, filters_vector: list, biases_vector: list,
                      height: int = 21, filter_height: int = 3) -> list:
    state = x_input
    layers = len(filters_vector)
    for i in range(layers):
        filters = filters_vector[i]
        biases = biases_vector[i]
        num_channels = len(state[0])
        new_num_channels = len(filters)
        state = pad_z_axis(state)
        new_state = [[0] * new_num_channels for h in range(height)]
        for f in range(new_num_channels):
            for x in range(num_channels):
                for z in range(1, height+1):
                    for j in range(filter_height):
                        new_state[z-1][f] += (filters[f][j][x] * state[z-j-1][x])
            for z in range(height):
                new_state[z][f] += biases[f]
                if new_state[z][f] < 0:
                    new_state[z][f] *= 0.3  # LeakyReLU step
        state = new_state

    return state


# ======================================================================
# This is here to run a test example
# ======================================================================
def run_example() -> list:
    hidden_layers = 7
    channels = 5
    filt_width = channels
    filt_height = 3
    levels = 21

    state = np.random.rand(levels, channels).tolist()
    biases = np.random.rand(hidden_layers, channels).tolist()
    filts = create_filters_vector(hidden_layers, filt_height, filt_width, channels)
    output_filt, output_bias = create_output_layer_filter_and_bias()
    filts.append(output_filt)
    biases.append(output_bias)


    new_state = predict_spdt_spdq(x_input=state, filters_vector=filts, biases_vector=biases)
    print(len(filts), len(filts[0]), len(filts[0][0]), len(filts[0][0][0]))
    print(len(biases), len(biases[0]))
    print(len(state), len(state[0]))
    print(len(new_state), len(new_state[0]))
    print(state)
    print(new_state)

    return


def create_filters_vector(hidden_layers: int, filter_height: int,
                          filter_width: int, channels: int):
    filters_vector = []
    for layer in range(hidden_layers):
        layer_filters = []
        for i in range(channels):
            filt = np.random.rand(filter_height, filter_width).tolist()
            layer_filters.append(filt)
        filters_vector.append(layer_filters)

    return filters_vector


def create_output_layer_filter_and_bias() -> (list, list):
    output_layer_filters = []
    for i in range(2):
        filt = np.random.rand(3, 5).tolist()
        output_layer_filters.append(filt)
    output_layer_bias = np.random.rand(2).tolist()

    return output_layer_filters, output_layer_bias


if __name__ == "__main__":
    run_example()
