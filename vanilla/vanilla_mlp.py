import tensorflow as tf
from tensorflow.keras import layers

# Multi Layer Perceptron
class MLP():
    def __init__(self, n_neurons, n_layers, batch=1, cont=False, epochs=100, size=45, activation='relu', loss=tf.keras.losses.mean_squared_error):
        if cont:
            try:
                self.model = self.load(f'./models/sort_net_{size}.mpl')
            except:
                raise Exception('This type of model not found.')
        else:
            self.model = tf.keras.Sequential()
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.batch = batch
        self.epochs = epochs
        self.activation = activation
        self.loss = loss

    def build(self, input_size):
        # input layer
        self.model.add(layers.Dense(units=5, input_dim=input_size, activation=self.activation))
        # hidden layers
        for _ in range(self.n_layers):
            self.model.add(layers.Dense(units=self.n_neurons, activation=self.activation))
        # output layer
        self.model.add(layers.Dense(units=input_size, activation=tf.math.softplus, kernel_regularizer=tf.keras.regularizers.L1L2(l1=.01, l2=.01)))
        # build neural net
        self.model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(decay=1e-6), metrics=['accuracy'])

    def train(self, input, output, save=True, size=45):
        # convert input and output to tf.matrix
        X = tf.constant(input, shape=[len(input), len(input[0])])
        Y = tf.constant(output, shape=[len(output), len(output[0])])
        # train model
        self.model.fit(X, Y, epochs=self.epochs, batch_size=None, shuffle=True, workers=2, steps_per_epoch=self.batch)
        if save:
            self.save(size)
        return self.save(size) if save else self.model

    def save(self, size):
        tf.keras.models.save_model(self.model, f'./models/sort_net_{size}.mpl')
        return self.model

    def load(self, size):
        return tf.keras.models.load_model(f'./models/sort_net_{size}.mpl')
