__author__ = 'noe'

import keras
import numpy as np

def connect(input_layer, layers):
    """ Connect the given sequence of layers and returns output layer

    Parameters
    ----------
    input_layer : keras layer
        Input layer
    layers : list of keras layers
        Layers to be connected sequentially

    Returns
    -------
    output_layer : kears layer
        Output Layer

    """
    layer = input_layer
    for l in layers:
        layer = l(layer)
    return layer

def plot_network(network):
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    SVG(model_to_dot(network).create(prog='dot', format='svg'))

def layer_to_dict(layer):
    d = {'config' : keras.layers.serialize(layer),
         'input_shape' : layer.input_shape,
         'weights' : layer.get_weights()}
    return d

def layer_from_dict(d):
    layer = keras.layers.deserialize(d['config'])
    layer.build(d['input_shape'])
    layer.set_weights(d['weights'])
    return layer

def serialize_layers(list_of_layers):
    """ Returns a serialized version of the list of layers (recursive)

    Parameters
    ----------
    list_of_layers : list or layer
        list of list of lists or kears layer

    """
    if isinstance(list_of_layers, keras.layers.Layer):
        return layer_to_dict(list_of_layers)
    return [serialize_layers(l) for l in list_of_layers]

def deserialize_layers(S):
    """ Returns lists of lists of layers from a given serialization

    Parameters
    ----------
    S : list of list of dict (recursive)
        dictionary obtained with serialize_layers

    """
    if isinstance(S, dict):
        return layer_from_dict(S)
    return [deserialize_layers(l) for l in S]

def shuffle(x):
    """ Shuffles the rows of matrix data x.

    Returns
    -------
    x_shuffled : array
        Shuffled data
    """
    Ishuffle = np.argsort(np.random.rand(x.shape[0]))
    return x[Ishuffle]

def shufflesplit(x, val_ratio):
    """ Shuffles the rows of matrix data x and returns a given ratio of training and validation data.

    Parameters
    ----------
    val_ratio : float
        Ratio of validation data. If 0, None is returned for validation data

    Returns
    -------
    x_train : array
        Training data
    x_val : array
        Validation data
    """
    xshuffle = shuffle(x)
    ntrain = int((1.0-val_ratio)*x.shape[0])
    xtrain = xshuffle[:ntrain]
    if x.shape[0] == ntrain:
        xval = None
    else:
        xval = xshuffle[ntrain:]
    return xtrain, xval