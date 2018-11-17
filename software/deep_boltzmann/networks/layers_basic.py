import keras
import tensorflow as tf
import numpy as np


def nonlinear_transform(output_size, nlayers=3, nhidden=100, activation='relu', init_outputs=None):
    """ Generic dense trainable nonlinear transform

    Returns the layers of a dense feedforward network with nlayers-1 hidden layers with nhidden neurons
    and the specified activation functions. The last layer is linear in order to access the full real
    number range and has output_size output neurons.

    Parameters
    ----------
    output_size : int
        number of output neurons
    nlayers : int
        number of layers
    nhidden : int
        number of neurons in each hidden layer
    activation : str
        nonlinear activation function in hidden layers
    init_outputs : None or float or array


    """
    M = [keras.layers.Dense(nhidden, activation=activation) for i in range(nlayers-1)]
    if init_outputs is None:
        final_layer = keras.layers.Dense(output_size, activation='linear')
    else:
        final_layer = keras.layers.Dense(output_size, activation='linear',
                                         kernel_initializer=keras.initializers.Zeros(),
                                         bias_initializer=keras.initializers.Constant(init_outputs))
    M += [final_layer]

    return M


class ResampleLayer(keras.engine.Layer):
    """
    Receives as inputs latent space encodings z and normal noise w. Transforms w to
    Match the mean and the standard deviations of z.

    """
    def __init__(self, dim, **kwargs):
        self.dim = dim
        super(ResampleLayer, self).__init__(**kwargs)

    def call(self, x):
        # split input into latent and noise variables
        z = x[:, :self.dim]
        w = x[:, self.dim:]
        #z, w = x
        # mean
        mean = keras.backend.mean(z, axis=0)
        # covariance matrix
        batchsize = keras.backend.shape(z)[0]
        cov = keras.backend.dot(keras.backend.transpose(z), z) / keras.backend.cast(batchsize, np.float32)
        # standard deviations
        std = tf.sqrt(tf.diag_part(cov))
        # transform w and return
        wtrans = tf.reshape(mean, (1, self.dim)) + w * tf.reshape(std, (1, self.dim))
        return wtrans

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.dim


class IndexLayer(keras.engine.Layer):
    def __init__(self, indices, **kwargs):
        """ Returns [:, indices].
        """
        self.indices = indices
        super().__init__(**kwargs)

    def call(self, x):
        # split input
        return tf.gather(x, self.indices, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.indices.size
