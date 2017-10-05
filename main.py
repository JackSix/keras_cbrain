#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:59:37 2017

@author: Yacalis
"""
"""
This code is meant to predict the outcome of a competitive match between 2
teams. Each team has 3 players. Before a match, each player selects a hero from
a list of around 30 different choices. Any given hero may only be chosen once.
We want the model to predict the outcome of the match based only on the heros
composition of each team, based on historical matchup data.

Input data will be a tuple consisting of a sparse 60x1 vector and an integer, either 1 or 0. The
first 30 inputs is a list of the heros; a 0 means that hero was not selected by team A,
and a 1 means the hero was selected by team 1; so there will be three instances of a 1 in
the first 30 rows. The next 30 rows are identical, except they represent the hero selections
of team B.

Output data will be two neurons: the first representing the probability that team A will win,
and the second representing the probability that team B will win.

Hidden layers will use the relu activation function. The output layer will use the softmax function.
Dropout must be used to prevent overfitting. The overall kind of neural network will be
simply a multilayer perceptron, a binary classifier.
"""

import random
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD # SGD stands for stochastic gradient descent

# =============================================================================
# Set parameter constants
# =============================================================================
# Data
nheros = 30
nsamples = 1000 # this parameter will not matter once there is real data
# Net
neurons_hidden1 = 64
neurons_hidden2 = 64
dropout = 0.5
input_dim = nheros*2
# Model
test_size = 0.25
epochs = 20
batch_size = 128


# =============================================================================
# JUST GENERATING DUMMY DATA HERE
# =============================================================================
x = []
for i in range(nsamples):
    # make a list with a length that is double the number of heros
    match_comp = [0] * input_dim
    # pick 6 heros from the hero list, 3 for each team
    hero_select = random.sample(range(nheros), 6)
    # make sure half the heros get put on team B
    for j in range(3,6):
        hero_select[j] = hero_select[j] + nheros
    # turn 6 of the 0's into 1's to represent the 6 selected heros
    for j in range(len(hero_select)):
        match_comp[hero_select[j]] = 1
    # add match data sample
    x.append(match_comp)
x = np.array(x)
y = np.random.randint(2, size=(nsamples,1))


# =============================================================================
# ACTUAL ML HAPPENING BELOW THIS LINE
# =============================================================================
# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

# Build the model
model = Sequential()
model.add(Dense(neurons_hidden1, input_dim=input_dim, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(neurons_hidden2, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Train the model
model.fit(x_train,
          y_train,
          epochs=epochs,
          batch_size=batch_size)

# Test the model
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print(score)



