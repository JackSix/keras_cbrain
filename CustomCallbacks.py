#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:19:31 2017

@author: Yacalis
"""

from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class CustomCallbacks:
    def __init__(self, config, log_dir):
        self.callbacks = self.main(config, log_dir)
        return

    @staticmethod
    def main(config: object, log_dir: str) -> list:
        # set up tensorboard visualization
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=config.histogram_freq,
            batch_size=config.batch_size,
            write_graph=config.write_graph,
            write_grads=config.write_grads,
            write_images=config.write_images
        )

        # ======================================================================
        # if 'monitor' (loss) does not reduce by at least 'min_delta' amount
        # within 'patience' number of epochs, stop training
        # ======================================================================
        earlystopping = EarlyStopping(
            monitor=config.es_monitor,
            min_delta=config.es_min_delta,
            patience=config.es_patience,
            verbose=1,
            mode=config.es_mode
        )

        # ======================================================================
        # if 'monitor' (loss) does not reduce by at least 'epsilon' amount
        # within 'patience' number of epochs, multiply the learning rate of the
        # model by 'factor', up to a minimum value of 'min_lr'
        # ======================================================================
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor=config.lr_monitor,
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=1,
            mode=config.lr_mode,
            epsilon=config.lr_epsilon,
            cooldown=0,
            min_lr=config.lr_min_lr
        )

        # ======================================================================
        # save the state of the model and its weights every 'period' epochs
        #
        # 'monitor' is a loss value, and when 'save_best_only' is set to True,
        # the previously saved checkpoint will only be overwritten if the loss
        # of the new checkpoint is better -- but with early stopping, this
        # typically won't matter, as training will stop if the model is not
        # improving
        # ======================================================================
        chckpt_fp = log_dir + '/chckpt.ep_{epoch:02d}-loss_{val_loss:.2f}.hdf5'
        model_checkpt = ModelCheckpoint(
            chckpt_fp,
            monitor=config.mc_monitor,
            verbose=1,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=10
        )

        return [tensorboard, earlystopping, reduce_lr_on_plateau, model_checkpt]
