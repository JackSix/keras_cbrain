#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 8 09:26:01 2017

@author: Yacalis
"""


def pad_z_axis_symmetric(matrix: list) -> list:
    """Pads top and bottom of a 2D matrix each with a single row of zeros"""
    top_padding = matrix[0][:]
    bottom_padding = matrix[-1][:]
    matrix.insert(0, top_padding)
    matrix.insert(-1, bottom_padding)

    return matrix


def chckpt_ver_predict_spdt_spdq(x_input: list, filters_vector: list,
                                 biases_vector: list, height: int = 21,
                                 filter_height: int = 3) -> list:
    """
    :param x_input: input variables in the form of a 2D matrix
    :param filters_vector: contains as many filters as length of NN - 1
    :param biases_vector: same length as filters vector
    :param height: the number of vertical levels in an air column
    :param filter_height: height of filter, assumed to have shape 3x1
    :return: pred SPDT and SPDQ values given the x_input, filters, and biases

     Assumes all layers except the final one (leading to the output) have the
     same filter height and activation function. Each outer loop corresponds to
     the transition from one layer to another in the CNN. In the code, the
     layers are referred to as states, consisting of channels.

     First, retrieves the filter/bias to be applied to the current state
     (set of channels); then creates zeros matrix as placeholder for the new
     channels; pads the initial state with zeros along just its vertical axis;
     applies each filter to its matching channel along the vertical level and
     sums those to create a single new channel; then add bias to the value of
     each level in that channel; last, apply the activation function (hard
     coded as LeakyReLU) to that value.

     The filters for the final layer are assumed to just have a 1x1 shape, so
     no padding step is needed. No activation function is applied (technically
     speaking, you could say a linear activation function of f(x) = x is
     applied). The final result is a 2x21 matrix, where each vertical column
     represents predicted SPDT and SPDQ values.
    """
    state = x_input
    layers = len(filters_vector)
    for i in range(layers):
        filters = filters_vector[i]
        biases = biases_vector[i]
        num_channels = len(state[0])
        new_num_channels = len(filters[0][0][0])
        new_state = [[0] * new_num_channels for h in range(height)]
        if i != layers-1:
            state = pad_z_axis_symmetric(state)
            for f in range(new_num_channels):
                for x in range(num_channels):
                    for z in range(height):
                        for j in range(filter_height):
                            new_state[z][f] += (filters[j][0][x][f] * state[z+j][x])
                for z in range(height):
                    new_state[z][f] += biases[f]
                    if new_state[z][f] < 0:
                        new_state[z][f] *= 0.3
        else:
            for f in range(new_num_channels):
                for x in range(num_channels):
                    for z in range(height):
                        new_state[z][f] += (filters[0][0][x][f] * state[z][x])
                for z in range(height):
                    new_state[z][f] += biases[f]
        state = new_state
    return state
