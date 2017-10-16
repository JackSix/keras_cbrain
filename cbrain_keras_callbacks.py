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

        # set up early stopping
        earlystopping = EarlyStopping(
            monitor=config.es_monitor,
            min_delta=config.es_min_delta,
            patience=config.es_patience,
            verbose=1,
            mode=config.es_mode
        )

        # set up reducing the learning rate when conditions are met
        reduce_lr_on_plateau = ReduceLROnPlateau(
            monitor=config.lr_monitor,
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=1,
            mode=config.lr_mode,
            epsilon=config.lr_epsilon,
            cooldown=0,
            min_lr=config.min_lr
        )

        chckpt_fp = log_dir + 'chckpt.ep_{epoch:02d}-loss_{val_loss:.2f}.hdf5'
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
