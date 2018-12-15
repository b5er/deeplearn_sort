import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt
import pandas as pd

# multi layer perceptron
class MLP():
    def __init__(self, n_neurons, n_layers, batch=1, cont=False, epochs=100, activation='relu', loss=tf.keras.losses.mean_squared_error):
        if cont:
            self.model = self.load()
        else:
            self.model = tf.keras.Sequential()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.batch = batch
        self.epochs = epochs
        self.activation = activation
        self.loss = loss

    def build(self):
        # input layer
        self.model.add(layers.Dense(units=5, input_dim=5, activation=self.activation))
        # hidden layers
        for _ in range(self.n_layers):
            self.model.add(layers.Dense(units=self.n_neurons, activation=self.activation))
        # output layer
        self.model.add(layers.Dense(units=5, activation=tf.math.softplus, kernel_regularizer=tf.keras.regularizers.L1L2(l1=.01, l2=.01)))
        # build neural net
        self.model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(decay=1e-6), metrics=['accuracy'])

    def train(self, input, output, save=True):
        # convert input and output to tf.matrix
        X = tf.constant(input, shape=[len(input), len(input[0])])
        Y = tf.constant(output, shape=[len(output), len(output[0])])
        # train model
        self.model.fit(X, Y, epochs=self.epochs, batch_size=None, shuffle=True, workers=2, steps_per_epoch=self.batch)
        if save:
            self.save()
        return self.model

    def save(self):
        self.model.save('sort_net.mpl')

    def load(self):
        return tf.keras.models.load_model('sort_net.mpl')
