#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import h5py
from python_convo_loop import chckpt_ver_predict_spdt_spdq
from inspect_checkpoint import get_tensors_from_checkpoint_file


# --------------------------------------------
# ---- Variables you might want to change ----
# --------------------------------------------
print('Setting hyperparameters')

chckpt_file_index = 1
imonth = 3
ilon = 10
ilat = 32
num_consec_timepts = 47
input_vars = [
    'TAP,QAP,OMEGA,SHFLX,LHFLX,dTdt_adiabatic,dQdt_adiabatic',
    'TAP,QAP,OMEGA,SHFLX,LHFLX,LAT,dTdt_adiabatic,dQdt_adiabatic,QRL,QRS']
chckpt_files = [
    f'../tensorboard_logs/Pierre/1028_094639_SPDT,SPDQ_layers_32,32,32,32,32,32__kdr_1.0_ac_0_convo_True_variables_{input_vars[0]}_batchsize_512/model.ckpt-515939',
    f'../tensorboard_logs/Pierre/1028_123916_SPDT,SPDQ_layers_32,32,32,32,32,32_kdr_1.0_ac_0_convo_True_variables_{input_vars[1]}_batchsize_128/model.ckpt-1192378']


# --------------------------------------------
# ------------ Get file paths ----------------
# --------------------------------------------
print('Getting file paths')

fd = h5py.File('../SPCAM/SPCAM_Stephan/SPCAM_outputs_detailed.nc', 'r')
fmean = h5py.File('../SPCAM/SPCAM_Stephan/SPCAM_mean_detailed.nc', 'r')
fstd = h5py.File('../SPCAM/SPCAM_Stephan/SPCAM_std_detailed.nc', 'r')
chckpt_file = chckpt_files[chckpt_file_index]
input_vars = input_vars[chckpt_file_index].split(',')

print('checkpoint file: ', chckpt_file)
print('input_vars: ', input_vars)


# --------------------------------------------
# ----- Read in and assemble input data ------
# --------------------------------------------
print('Reading in and assembling input data')

x_input = np.zeros((len(input_vars), 21))
for itime in range(num_consec_timepts):
    for i, varname in enumerate(input_vars):
        if len(fd[varname].shape) > 4:
            aux = fd[varname][imonth, itime, :, ilat, ilon]
            mean = fmean[varname][:]
            std = fstd[varname][:]
        else:
            aux = fd[varname][imonth, itime, ilat, ilon]
            mean = fmean[varname]
            std = fstd[varname]
        aux -= mean
        aux /= std
        x_input[i, :] = aux

    spdt = 1e3*fd['SPDT'][imonth, itime, :, ilat, ilon]
    spdq = 2.5e6*fd['SPDQ'][imonth, itime, :, ilat, ilon]

    if itime == 0:
        inputs = np.expand_dims(x_input, 2)
        spdt_actual = np.expand_dims(spdt, 1)
        spdq_actual = np.expand_dims(spdq, 1)
    else:
        inputs = np.concatenate((inputs, np.expand_dims(x_input, 2)), axis=2)
        spdt_actual = np.concatenate((spdt_actual, np.expand_dims(spdt, 1)), axis=1)
        spdq_actual = np.concatenate((spdq_actual, np.expand_dims(spdq, 1)), axis=1)


# --------------------------------------------
# ------- Do the neural net predictions ------
# --------------------------------------------
print('Running Convo Loop on SP-CAM Data with Chkpt Filters and Biases')

filters, biases = get_tensors_from_checkpoint_file(chckpt_file)

for itime in range(num_consec_timepts):
    state = np.transpose(inputs[:, :, itime])
    new_state = chckpt_ver_predict_spdt_spdq(state.tolist(), filters, biases)
    aux = np.array(new_state)
    if itime == 0:
        spdt_predicted = np.expand_dims(aux[:, 0], 1)
        spdq_predicted = np.expand_dims(aux[:, 1], 1)
    else:
        spdt_predicted = np.concatenate((spdt_predicted, np.expand_dims(aux[:, 0], 1)), axis=1)
        spdq_predicted = np.concatenate((spdq_predicted, np.expand_dims(aux[:, 1], 1)), axis=1)

print('Final shape: ', spdt_predicted.shape)


# --------------------------------------------
# ------------- Plot results -----------------
# --------------------------------------------
print('Plotting results')

def do_plotting(subplotof4: int, title: str, values: np.ndarray):
    plt.subplot(2, 2, subplotof4)
    plt.imshow(values, vmin=-0.2, vmax=+0.2)
    plt.colorbar()
    plt.xlabel('Time index')
    plt.ylabel('Vertical level')
    plt.title(title)

plt.figure()
do_plotting(subplotof4=1, title='SPDT predicted', values=spdt_predicted)
do_plotting(subplotof4=2, title='SPDT actual', values=spdt_actual)
do_plotting(subplotof4=3, title='SPDQ predicted', values=spdq_predicted)
do_plotting(subplotof4=4, title='SPDQ actual', values=spdq_actual)
plt.tight_layout()
plt.show()
