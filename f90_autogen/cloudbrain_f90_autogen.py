#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 2 13:15:49 2017

@author: Yacalis
"""


# ======================================================================
# Main
# ======================================================================
# Note: should not need to change this much; need to change other funcs
def main():
    output_filename = get_output_filename()
    input_filename = get_input_filename()
    iw, lw, b = get_weights_and_biases(input_filename)

    hidden_size, output_size, input_size = get_layer_sizes(iw, lw, b)
    nb_neurons, nb_hidden_layers = set_nb_constants()

    with open(output_filename, 'w+') as file:
        write_f90_header(file, nb_hidden_layers, hidden_size, output_size, input_size)

        target_names = ['SPDT', 'SPDQ']
        for i in range(len(target_names)):
            target_name = target_names[i]

            write_subroutine_header(file, target_name)
            write_weights_and_biases(file, iw, lw, b, hidden_size, output_size, input_size)
            write_subroutine_footer(file, target_name)

        write_f90_footer(file)


# ======================================================================
# Neural Net Functions
# ======================================================================
# TODO: actually figure out how to set these based on the data
def set_nb_constants():
    """
    These numbers distinguish which neural net we are talking about, the
    relevant dimensions of # neurons, and the # hidden layers.
    """
    nb_neurons = 10
    nb_hidden_layers = 1
    return nb_neurons, nb_hidden_layers


# TODO: actually get iw, lw, b based on the data
def get_weights_and_biases(input_filename):
    #with open(input_filename, 'r') as file:
        #iw, lw, b = file.iw, file.lw, file.b
        iw, lw, b = None, None, None
        return iw, lw, b


# TODO: actually get sizes based on iw, lw, b
def get_layer_sizes(iw, lw, b):
    """Figure out the f90 dim sizes for the net."""
    # hidden = lw[1,1].size
    # output = b[2,1].size
    # input_lay = iw[1,1].size[1]
    hidden = 2
    output = 3
    input_lay = 4
    return hidden, output, input_lay


# TODO: actually figure out how to do this based on the data
def write_weights_and_biases(file, iw, lw, b, hidden_size, output_size, input_size):
    mu_in = '-9.12332e-01'
    file.write('mu_in(:) = (/' + mu_in + ' /)\n')

    std_in = '9.44444e-07'
    file.write('std_in(:) = (/' + std_in + ' /)\n')

    bias_in = '8.314840e-02, -1.178977e-01'
    file.write('bias_input_in(:) = (/' + bias_in + ' /)\n')

    bias_out = '-1.063119e-01, 1.999398e-01'
    file.write('bias_output(:) = (/' + bias_out + ' /)\n')

    weights_in = '7.489552e-02, -3.008047e-02, 4.961192e-02, 1.249248e-01'
    for kneuron in range(hidden_size):
        file.write(f'weights_input({kneuron}:) = (/' + weights_in + ' /)\n')

    weights_out = '-2.654571e-01, -6.213908e-01, 4.264377e-01, 5.954914e-01'
    for kout in range(output_size):
        file.write(f'weights_output({kout}:) = (/' + weights_out + ' /)\n')

    # bias_input = b[1, 1]
    # astring = 'bias_input(:) = (/'
    # for k in range(hidden_size):
    #     astring = astring + f'{str(bias_input(k))}'
    # astring = astring[1:len(astring)-2] + ' /)\n'
    # file.write('bias_input_in(:) = (/' + astring + ' /)\n')
    #
    # bias_output = b[2, 1]
    # astring = 'bias_output(:) = (/'
    # for k in range(output_size):
    #     astring = astring + f'{str(bias_output(k))}'
    # astring = astring[1:len(astring)-2] + ' /)\n'
    # file.write('bias_output(:) = (/' + astring + ' /)\n')
    #
    # weights_input = iw[1, 1]
    # for kneuron in range(hidden_size):
    #     astring = f'weights_input({kneuron},:) = (/'
    #     for k in range(input_size):
    #         astring = astring + f'{weights_input(kneuron, k)}, '
    #     astring = astring[1:len(astring)-2] + ' /)\n'
    #     file.write('weights_input(:) = (/' + astring + ' /)\n')
    #
    # weights_output = lw[2, 1]
    # for kout in range(output_size):
    #     astring = f'weights_output({kout},:) = (/'
    #     for k in range(hidden_size):
    #         astring = astring + f'{weights_output(kout, k)}, '
    #     astring = astring[1:len(astring-2] ' /)\n'
    #     file.write('weights_output(:) = (/' + astring + ' /)\n')


# ======================================================================
# Filenames
# ======================================================================
# TODO: actually figure out what the filename is supposed to be
def get_input_filename():
    checkpointdir = '/Users/Pritchard/Dropbox/CHECKPOINTS'
    file_prefix = 'ANN_IN__PS_QAP_TAP_OMEGA_SHFLX_LHFLX_OUT__'
    output_fct = 'purelin'
    fields = ['PS', 'QAP', 'TAP', 'OMEGA', 'SHFLX', 'LHFLX']
    seed_number = 80
    input_filename = checkpointdir + '/' + file_prefix + 'stuff' + '_' + str(fields) + '_' + str(seed_number) + '_' + output_fct + '.mat'
    return input_filename


def get_output_filename():
    return 'cloudbrain_module.F90'


# ======================================================================
# Headers, Footers
# ======================================================================
def write_f90_header(file, nb_hidden_layers, hidden_size, output_size, input_size):
    """
    Write source for module definition, its needed links to CAM data
    structures, and the internal neural net dim size parameter definitions.
    """
    text_lines = [
        '#include <misc.h>\n',
        '#include <params.h>\n',
        '\n',
        'module cloudbrain_module\n',
        'use shr_kind_mod,    only: r8 => shr_kind_r8\n',
        'use ppgrid,          only: pcols, pver, pverp\n',
        'use history,         only: outfld, addfld, add_default, phys_decomp\n',
        'use nt_FunctionsModule, only: nt_tansig\n',
        'use physconst,       only: gravit,cpair\n',
        'implicit none\n',
        '\n',
        'save\n',
        '\n',
        'private                   ! Make default type private to the module\n',
        f'integer,parameter :: hiddenlayerSize = {hidden_size}\n',
        f'integer,parameter :: outputlayerSize = {output_size}\n',
        f'integer, parameter :: nbhiddenlayers = {nb_hidden_layers}\n',
        f'integer,parameter :: inputlayerSize = {input_size}\n',
        '\n',
        'public cloudbrain\n',
        '\n',
        'contains: \n',
        '\n',
    ]
    file.writelines(text_lines)


def write_subroutine_header(file, target_name):
    text_lines = [
        f'subroutine define_neuralnet_{target_name} (mu_in,std_in,weights_input, bias_input,bias_output,weights_output)\n',
        'real(r8), intent(out) :: mu_in(inputlayerSize)\n',
        'real(r8), intent(out) :: std_in(inputlayerSize)\n',
        'real(r8), intent(out) :: bias_input(hiddenlayerSize)\n',
        'real(r8), intent(out) :: bias_output(outputlayerSize)\n',
        'real(r8), intent(out) :: weights_input(hiddenlayerSize, inputlayerSize)\n',
        'real(r8), intent(out) :: weights_output(outputlayerSize,hiddenlayerSize)\n',
        '\n'
    ]
    file.writelines(text_lines)


def write_subroutine_footer(file, target_name):
    file.write('\n\n')
    file.write(f'end subroutine define_neuralnet_{target_name}\n\n\n')


def write_f90_footer(file):
    file.write('end module cloudbrain\n')


# ======================================================================
# This must be at the bottom of the file
# ======================================================================
if __name__ == '__main__':
    main()
