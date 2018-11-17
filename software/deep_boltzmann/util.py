__author__ = 'noe'

import tensorflow as tf
import numpy as np
import numbers
import pickle


def save_obj(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def linlogcut(x, a=0, b=1000, tf=False):
    """ Function which is linear until a, logarithmic until b and then constant.

    y = x                  x <= a
    y = a + log(x-a)   a < x < b
    y = a + log(b-a)   b < x

    """
    if tf:
        return _linlogcut_tf(x, a=a, b=b)
    else:
        return _linlogcut_np(x, a=a, b=b)

def _linlogcut_tf(x, a=0, b=1000):
    # cutoff x after b - this should also cutoff infinities
    x = tf.where(x < b, x, b * tf.ones(tf.shape(x)))
    # log after a
    y = a + tf.where(x < a, x - a, tf.log(x - a + 1))
    # make sure everything is finite
    y = tf.where(tf.is_finite(y), y, b * tf.ones(tf.shape(y)))
    return y

def _linlogcut_np(x, a=0, b=1000):
    raise NotImplementedError('Numpy version not yet implemented.')

def logreg(x, a=0.001, tf=False):
    if tf:
        return _logreg_tf(x, a=a)
    else:
        return _logreg_np(x, a=a)

def _logreg_tf(x, a=0.001):
    logx = tf.where(x > a, tf.log(x), tf.log(a) - (a-x))
    return logx
def _logreg_np(x, a=0.001):
    raise NotImplementedError('Numpy version not yet implemented.')



def acf(x, lags, remove_mean=True, normalize=True):
    """ Computes Autocorrelation of signal x

    Parameters
    ----------
    x : array
        Signal
    remove_mean : bool
        If true, remove signal mean
    normalize : bool
        If true, ACF is 1 at lagtime 0

    """
    if isinstance(lags, numbers.Real):
        lags = np.array([lags])
    else:
        lags = np.array(lags)
    if remove_mean:
        x = x - x.mean()
    a = np.zeros(lags.size)
    for i in range(lags.size):
        t = lags[i]
        if t == 0:
            a[i] = np.mean(x ** 2)
        else:
            a[i] = np.mean(x[:-t] * x[t:])
    if normalize:
        a0 = np.mean(x * x)
        a /=  a0
    return a


def count_transitions(x, lcore, rcore):
    core = -1
    t = 0
    while t < x.size:
        if x[t] < lcore:
            core = 0
            break
        if x[t] > rcore:
            core = 1
            break
        t += 1
    N = 0
    while t < x.size:
        if core == 0 and x[t] > rcore:
            core = 1
            N += 1
        if core == 1 and x[t] < lcore:
            core = 0
            N += 1
        t += 1
    return N


def ensure_traj(X):
    if np.ndim(X) == 2:
        return X
    if np.ndim(X) == 1:
        return np.array([X])
    raise ValueError('Incompatible array with shape: ', np.shape(X))


def distance_matrix_squared(crd1, crd2, dim=2):
    """ Returns the distance matrix or matrices between particles

    Parameters
    ----------
    crd1 : array or matrix
        first coordinate set
    crd2 : array or matrix
        second coordinate set
    dim : int
        dimension of particle system. If d=2, coordinate vectors are [x1, y1, x2, y2, ...]

    """
    crd1 = ensure_traj(crd1)
    crd2 = ensure_traj(crd2)
    n = int(np.shape(crd1)[1]/dim)

    crd1_components = [np.tile(np.expand_dims(crd1[:, i::dim], 2), (1, 1, n)) for i in range(dim)]
    crd2_components = [np.tile(np.expand_dims(crd2[:, i::dim], 2), (1, 1, n)) for i in range(dim)]
    D2_components = [(crd1_components[i] - np.transpose(crd2_components[i], axes=(0, 2, 1)))**2 for i in range(dim)]
    D2 = np.sum(D2_components, axis=0)
    return D2
