#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 00:19:31 2017

@author: Yacalis
"""

from keras.callbacks import TensorBoard, EarlyStopping


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
            write_images=config.write_images)

        # set up early stopping
        earlystopping = EarlyStopping(
            monitor=config.monitor,
            min_delta=config.min_delta,
            patience=config.patience,
            verbose=1,
            mode=config.mode)

        return [tensorboard, earlystopping]
