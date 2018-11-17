import sys
import numpy as np
import keras
import tensorflow as tf

from deep_boltzmann.networks import IndexLayer, connect, nonlinear_transform
from deep_boltzmann.networks.util import shuffle
from deep_boltzmann.util import ensure_traj


def split_merge_indices(ndim, nchannels=2, channels=None):
    if channels is None:
        channels = np.tile(np.arange(nchannels), int(ndim/nchannels)+1)[:ndim]
    indices_split = []
    idx = np.arange(ndim)
    for c in range(nchannels):
        isplit = np.where(channels == c)[0]
        indices_split.append(isplit)
    indices_merge = np.concatenate(indices_split).argsort()
    return channels, indices_split, indices_merge


class SplitChannels(object):
    def __init__(self, ndim, nchannels=2, channels=None):
        """ Splits channels forward and merges them backward """
        self.channels, self.indices_split, self.indices_merge = split_merge_indices(ndim, nchannels=nchannels,
                                                                                    channels=channels)

    @classmethod
    def from_dict(cls, D):
        channels = D['channels']
        dim = channels.size
        nchannels = channels.max() + 1
        return cls(dim, nchannels=nchannels, channels=channels)

    def to_dict(self):
        D = {}
        D['channels'] = self.channels
        return D

    def connect_xz(self, x):
        # split X into different coordinate channels
        self.output_z = [IndexLayer(isplit)(x) for isplit in self.indices_split]
        return self.output_z

    def connect_zx(self, z):
        # first concatenate
        x_scrambled = keras.layers.Concatenate()(z)
        # unscramble x
        self.output_x = IndexLayer(self.indices_merge, name='output_x')(x_scrambled)
        return self.output_x


class MergeChannels(SplitChannels):
    def connect_xz(self, x):
        # first concatenate
        z_scrambled = keras.layers.Concatenate()(x)
        # unscramble x
        self.output_z = IndexLayer(self.indices_merge, name='output_z')(z_scrambled)
        return self.output_z

    def connect_zx(self, z):
        # split X into different coordinate channels
        self.output_x = [IndexLayer(isplit)(z) for isplit in self.indices_split]
        return self.output_x


class Scaling(object):
    def __init__(self, ndim, scaling_factors=None, trainable=True, name_xz=None, name_zx=None):
        """ Invertible Scaling layer

        Parameters
        ----------
        ndim : int
            Number of dimensions
        scaling_factors : array
            Initial scaling factors, must be of shape (1, ndim)
        trainable : bool
            If True, scaling factors are trainable. If false, they are fixed
        name_xz : str
            Name for Sxz
        name_xz : str
            Name for Szx

        """
        # define local classes
        class ScalingLayer(keras.engine.Layer):
            def __init__(self, log_scaling_factors, **kwargs):
                """ Layer that scales dimensions with trainable factors

                Parameters
                ----------
                scaling_factors : (1xd) array
                    scaling factors applied to columns of batch matrix.

                """
                self.log_scaling_factors = log_scaling_factors
                super().__init__(**kwargs)

            def build(self, input_shape):
                # Make weight trainable
                if self.trainable:
                    self._trainable_weights.append(self.log_scaling_factors)
                super().build(input_shape)  # Be sure to call this at the end

            def compute_output_shape(self, input_shape):
                return (input_shape[0], self.log_scaling_factors.shape[1])

        class ScalingXZ(ScalingLayer):
            def __init__(self, log_scaling_factors, **kwargs):
                """ Layer that scales the batch X in (B,d) by X * S where S=diag(s1,...,sd)
                """
                super().__init__(log_scaling_factors, **kwargs)

            def call(self, x):
                return x * tf.exp(self.log_scaling_factors)

        class ScalingZX(ScalingLayer):
            def __init__(self, log_scaling_factors, **kwargs):
                """ Layer that scales the batch X in (B,d) by X * S^(-1) where S=diag(s1,...,sd)
                """
                super().__init__(log_scaling_factors, **kwargs)

            def call(self, x):
                return x * tf.exp(-self.log_scaling_factors)

        # initialize scaling factors
        if scaling_factors is None:
            self.log_scaling_factors = keras.backend.variable(np.zeros((1, ndim)),
                                                              dtype=keras.backend.floatx(),
                                                              name='log_scale')
        else:
            self.log_scaling_factors = keras.backend.variable(np.log(scaling_factors),
                                                              dtype=keras.backend.floatx(),
                                                              name='log_scale')

        self.trainable = trainable
        self.Sxz = ScalingXZ(self.log_scaling_factors, trainable=trainable, name=name_xz)
        self.Szx = ScalingZX(self.log_scaling_factors, trainable=trainable, name=name_zx)

    @property
    def scaling_factors(self):
        return tf.exp(self.log_scaling_factors)

    @classmethod
    def from_dict(cls, D):
        scaling_factors = D['scaling_factors']
        dim = scaling_factors.shape[1]
        trainable = D['trainable']
        name_xz = D['name_xz']
        name_zx = D['name_zx']
        return Scaling(dim, scaling_factors=scaling_factors, trainable=trainable, name_xz=name_xz, name_zx=name_zx)

    def to_dict(self):
        D = {}
        D['scaling_factors'] = keras.backend.eval(self.scaling_factors)
        D['trainable'] = self.trainable
        D['name_xz'] = self.Sxz.name
        D['name_zx'] = self.Szx.name
        return D

    def connect_xz(self, x):
        def lambda_Jxz(x):
            J = tf.reduce_sum(self.log_scaling_factors, axis=1)[0]
            return J * keras.backend.ones((tf.shape(x)[0], 1))
        self.log_det_xz = keras.layers.Lambda(lambda_Jxz)(x)
        z = self.Sxz(x)
        return z

    def connect_zx(self, z):
        def lambda_Jzx(x):
            J = tf.reduce_sum(-self.log_scaling_factors, axis=1)[0]
            return J * keras.backend.ones((tf.shape(x)[0], 1))
        self.log_det_zx = keras.layers.Lambda(lambda_Jzx)(z)
        x = self.Szx(z)
        return x

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_zx


class CompositeLayer(object):
    def __init__(self, transforms):
        """ Composite layer consisting of multiple keras layers with shared parameters  """
        self.transforms = transforms

    @classmethod
    def from_dict(cls, d):
        from deep_boltzmann.networks.util import deserialize_layers
        transforms = deserialize_layers(d['transforms'])
        return cls(transforms)

    def to_dict(self):
        from deep_boltzmann.networks.util import serialize_layers
        D = {}
        D['transforms'] = serialize_layers(self.transforms)
        return D


class NICER(CompositeLayer):
    def __init__(self, transforms):
        """ Two sequential NICE transformations and their inverse transformatinos.

        Parameters
        ----------
        transforms : list
            List with [M1, M2] containing the keras layers for nonlinear transformation 1 and 2.

        """
        super().__init__(transforms)
        self.M1 = transforms[0]
        self.M2 = transforms[1]

    def connect_xz(self, x):
        x1 = x[0]
        x2 = x[1]
        self.input_x1 = x1
        self.input_x2 = x2

        # first stage backward
        y2 = x2
        y1 = keras.layers.Subtract()([x1, connect(x2, self.M2)])
        # second stage backward
        z1 = y1
        z2 = keras.layers.Subtract()([y2, connect(y1, self.M1)])

        return [z1, z2]

    def connect_zx(self, z):
        z1 = z[0]
        z2 = z[1]
        self.input_z1 = z1
        self.input_z2 = z2

        # first stage forward
        y1 = z1
        y2 = keras.layers.Add()([z2, connect(z1, self.M1)])
        # second stage forward
        x2 = y2
        x1 = keras.layers.Add()([y1, connect(y2, self.M2)])

        return [x1, x2]


class RealNVP(CompositeLayer):
    def __init__(self, transforms):
        """ Two sequential NVP transformations and their inverse transformatinos.

        Parameters
        ----------
        transforms : list
            List [S1, T1, S2, T2] with keras layers for scaling and translation transforms

        """
        super().__init__(transforms)
        self.S1 = transforms[0]
        self.T1 = transforms[1]
        self.S2 = transforms[2]
        self.T2 = transforms[3]

    def connect_xz(self, x):
        def lambda_exp(x):
            return keras.backend.exp(x)
        def lambda_sum(x):
            return keras.backend.sum(x[0], axis=1, keepdims=True) + keras.backend.sum(x[1], axis=1, keepdims=True)

        x1 = x[0]
        x2 = x[1]
        self.input_x1 = x1
        self.input_x2 = x2

        y1 = x1
        self.Sxy_layer = connect(x1, self.S1)
        self.Txy_layer = connect(x1, self.T1)
        prodx = keras.layers.Multiply()([x2, keras.layers.Lambda(lambda_exp)(self.Sxy_layer)])
        y2 = keras.layers.Add()([prodx, self.Txy_layer])

        self.output_z2 = y2
        self.Syz_layer = connect(y2, self.S2)
        self.Tyz_layer = connect(y2, self.T2)
        prody = keras.layers.Multiply()([y1, keras.layers.Lambda(lambda_exp)(self.Syz_layer)])
        self.output_z1 = keras.layers.Add()([prody, self.Tyz_layer])

        # log det(dz/dx)
        self.log_det_xz = keras.layers.Lambda(lambda_sum)([self.Sxy_layer, self.Syz_layer])

        return [self.output_z1, self.output_z2]

    def connect_zx(self, z):
        def lambda_negexp(x):
            return keras.backend.exp(-x)
        def lambda_negsum(x):
            return keras.backend.sum(-x[0], axis=1, keepdims=True) + keras.backend.sum(-x[1], axis=1, keepdims=True)

        z1 = z[0]
        z2 = z[1]
        self.input_z1 = z1
        self.input_z2 = z2

        y2 = z2
        self.Szy_layer = connect(z2, self.S2)
        self.Tzy_layer = connect(z2, self.T2)
        z1_m_Tz2 = keras.layers.Subtract()([z1, self.Tzy_layer])
        y1 = keras.layers.Multiply()([z1_m_Tz2, keras.layers.Lambda(lambda_negexp)(self.Szy_layer)])

        self.output_x1 = y1
        self.Syx_layer = connect(y1, self.S1)
        self.Tyx_layer = connect(y1, self.T1)
        y2_m_Ty1 = keras.layers.Subtract()([y2, self.Tyx_layer])
        self.output_x2 = keras.layers.Multiply()([y2_m_Ty1, keras.layers.Lambda(lambda_negexp)(self.Syx_layer)])

        # log det(dx/dz)
        # TODO: check Jacobian
        self.log_det_zx = keras.layers.Lambda(lambda_negsum)([self.Szy_layer, self.Syx_layer])

        return [self.output_x1, self.output_x2]

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_xz

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        return self.log_det_zx


class InvNet(object):

    def __init__(self, dim, layers, prior='normal'):
        """
        Parameters
        ----------
        dim : int
            Dimension
        layers : list
            list of invertible layers
        prior : str
            Type of prior, 'normal', 'lognormal'

        """
        """ Stack of invertible layers """
        self.dim = dim
        self.layers = layers
        self.prior = prior
        self.connect_layers()
        # compute total Jacobian for x->z transformation
        log_det_xzs = []
        for l in layers:
            if hasattr(l, 'log_det_xz'):
                log_det_xzs.append(l.log_det_xz)
        if len(log_det_xzs) == 0:
            self.TxzJ = None
        else:
            if len(log_det_xzs) == 1:
                self.log_det_xz = log_det_xzs[0]
            else:
                self.log_det_xz = keras.layers.Add()(log_det_xzs)
            self.TxzJ = keras.models.Model(inputs=self.input_x, outputs=[self.output_z, self.log_det_xz])
        # compute total Jacobian for z->x transformation
        log_det_zxs = []
        for l in layers:
            if hasattr(l, 'log_det_zx'):
                log_det_zxs.append(l.log_det_zx)
        if len(log_det_zxs) == 0:
            self.TzxJ = None
        else:
            if len(log_det_zxs) == 1:
                self.log_det_zx = log_det_zxs[0]
            else:
                self.log_det_zx = keras.layers.Add()(log_det_zxs)
            self.TzxJ = keras.models.Model(inputs=self.input_z, outputs=[self.output_x, self.log_det_zx])

    @classmethod
    def load(cls, filename):
        """ Loads parameters into model. Careful: this clears the whole TF session!!
        """
        from deep_boltzmann.util import load_obj
        keras.backend.clear_session()
        D = load_obj(filename)
        prior = D['prior']
        layerdicts = D['layers']
        layers = [eval(d['type']).from_dict(d) for d in layerdicts]
        return InvNet(D['dim'], layers, prior=prior)

    def save(self, filename):
        from deep_boltzmann.util import save_obj
        D = {}
        D['dim'] = self.dim
        D['prior'] = self.prior
        layerdicts = []
        for l in self.layers:
            d = l.to_dict()
            d['type'] = l.__class__.__name__
            layerdicts.append(d)
        D['layers'] = layerdicts
        save_obj(D, filename)

    def connect_xz(self, x):
        z = None
        for i in range(len(self.layers)):
            z = self.layers[i].connect_xz(x)  # connect
            x = z  # rename output
        return z

    def connect_zx(self, z):
        x = None
        for i in range(len(self.layers)-1, -1, -1):
            x = self.layers[i].connect_zx(z)  # connect
            z = x  # rename output to next input
        return x

    def connect_layers(self):
        # X -> Z
        self.input_x = keras.layers.Input(shape=(self.dim,))
        self.output_z = self.connect_xz(self.input_x)

        # Z -> X
        self.input_z = keras.layers.Input(shape=(self.dim,))
        self.output_x = self.connect_zx(self.input_z)

        # build networks
        self.Txz = keras.models.Model(inputs=self.input_x, outputs=self.output_z)
        self.Tzx = keras.models.Model(inputs=self.input_z, outputs=self.output_x)

    def predict_log_det_Jxz(self, z):
        if self.TzxJ is None:
            return np.ones(z.shape[0])
        else:
            return self.TzxJ.predict(z)[1][:, 0]

    @property
    def log_det_Jxz(self):
        """ Log of |det(dz/dx)| for the current batch. Format is batchsize x 1 or a number """
        #return self.log_det_xz.output
        log_det_Jxzs = []
        for l in self.layers:
            if hasattr(l, 'log_det_Jxz'):
                log_det_Jxzs.append(l.log_det_Jxz)
        if len(log_det_Jxzs) == 0:
            return tf.ones((self.output_z.shape[0],))
        if len(log_det_Jxzs) == 1:
            return log_det_Jxzs[0]
        return tf.reduce_sum(log_det_Jxzs, axis=0, keepdims=False)

    @property
    def log_det_Jzx(self):
        """ Log of |det(dx/dz)| for the current batch. Format is batchsize x 1 or a number """
        #return self.log_det_zx.output
        log_det_Jzxs = []
        for l in self.layers:
            if hasattr(l, 'log_det_Jzx'):
                log_det_Jzxs.append(l.log_det_Jzx)
        if len(log_det_Jzxs) == 0:
            return tf.ones((self.output_x.shape[0],))
        if len(log_det_Jzxs) == 1:
            return log_det_Jzxs[0]
        return tf.reduce_sum(log_det_Jzxs, axis=0, keepdims=False)

    def log_likelihood_z_normal(self, std=1.0):
        """ Returns the log likelihood of z|x assuming a Normal distribution in z
        """
        #return self.log_det_Jxz - self.dim * tf.log(std) - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)
        return self.log_det_Jxz - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)

    def log_likelihood_z_lognormal(self, std=1.0):
        """ Returns the log likelihood of z|x assuming a Normal distribution in z
        """
        #return self.log_det_Jxz - self.dim * tf.log(std) - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)
        from deep_boltzmann.util import logreg
        logz = logreg(self.output_z, a=0.001, tf=True)
        ll = self.log_det_Jxz \
             - (0.5 / (std**2)) * tf.reduce_sum(logz**2, axis=1) \
             - tf.reduce_sum(logz, axis=1)
        return ll

    def log_likelihood_z_cauchy(self, scale=1.0):
        return -tf.reduce_sum(tf.log(1 + (self.output_z / scale)**2), axis=1)

    def rc_entropy(self, rc_func, gmeans, gstd, ntemperatures=1):
        """ Computes the entropy along a 1D reaction coordinate

        Parameters
        ----------
        rc_func : function
            function to compute reaction coordinate
        gmeans : array
            mean positions of Gauss kernels along reaction coordinate
        gstd : float
            standard deviation of Gauss kernels along reaction coordinate
        """
        # evaluate rc
        rc = rc_func(self.output_x)
        rc = tf.expand_dims(rc, axis=1)
        # kernelize all values
        kmat = tf.exp(-((rc - gmeans)**2) / (2*gstd*gstd))
        kmat += 1e-6
        kmat /= tf.reduce_sum(kmat, axis=1, keepdims=True)
        # distribute counts across temperatures
        batchsize_per_temperature = tf.cast(tf.shape(kmat)[0] / ntemperatures, tf.int32)
        nbins = tf.shape(gmeans)[0]
        kmatT = tf.transpose(tf.reshape(kmat, (batchsize_per_temperature, ntemperatures, nbins)), perm=(1, 0, 2))
        histogram = tf.reduce_mean(kmatT, axis=1)
        entropies = tf.reduce_sum(tf.log(histogram), axis=1)
        return tf.reduce_mean(entropies)

    def reg_Jzx_uniform(self):
        """ Returns the log likelihood of z|x assuming a Normal distribution in z
        """
        #return self.log_det_Jxz - self.dim * tf.log(std) - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)
        Jmean = tf.reduce_mean(self.log_det_Jzx, axis=0, keepdims=True)
        Jdev = tf.reduce_mean((self.log_det_Jzx - Jmean) ** 2, axis=1, keepdims=False)
        return Jdev

    def reg_Jxz_uniform(self):
        """ Returns the log likelihood of z|x assuming a Normal distribution in z
        """
        #return self.log_det_Jxz - self.dim * tf.log(std) - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)
        Jmean = tf.reduce_mean(self.log_det_Jxz, axis=0, keepdims=True)
        Jdev = tf.reduce_mean((self.log_det_Jxz - Jmean) ** 2, axis=1, keepdims=False)
        return Jdev

    def log_likelihood_z_normal_2trajs(self, trajlength, std=1.0):
        """ Returns the log of the sum of two trajectory likelihoods
        """
        #return self.log_det_Jxz - self.dim * tf.log(std) - (0.5 / (std**2)) * tf.reduce_sum(self.output_z**2, axis=1)
        J = self.log_det_Jxz
        LL1 = tf.reduce_mean(J[:trajlength] - (0.5 / (std**2)) * tf.reduce_sum(self.output_z[:trajlength]**2, axis=1))
        LL2 = tf.reduce_mean(J[trajlength:] - (0.5 / (std**2)) * tf.reduce_sum(self.output_z[trajlength:]**2, axis=1))
        return tf.reduce_logsumexp([LL1, LL2])

    def train_ML(self, x, xval=None, optimizer=None, lr=0.001, clipnorm=None, epochs=2000, batch_size=1024,
                 std=1.0, reg_Jxz=0.0, verbose=1):
        if optimizer is None:
            if clipnorm is None:
                optimizer = keras.optimizers.adam(lr=lr)
            else:
                optimizer = keras.optimizers.adam(lr=lr, clipnorm=clipnorm)

        def loss_ML_normal(y_true, y_pred):
            return -self.log_likelihood_z_normal(std=std)
        def loss_ML_lognormal(y_true, y_pred):
            return -self.log_likelihood_z_lognormal(std=std)
        def loss_ML_cauchy(y_true, y_pred):
            return -self.log_likelihood_z_cauchy(scale=std)
        def loss_ML_normal_reg(y_true, y_pred):
            return -self.log_likelihood_z_normal(std=std) + reg_Jxz*self.reg_Jxz_uniform()
        def loss_ML_lognormal_reg(y_true, y_pred):
            return -self.log_likelihood_z_lognormal(std=std) + reg_Jxz*self.reg_Jxz_uniform()
        def loss_ML_cauchy_reg(y_true, y_pred):
            return -self.log_likelihood_z_cauchy(scale=std) + reg_Jxz*self.reg_Jxz_uniform()

        if self.prior == 'normal':
            if reg_Jxz == 0:
                self.Txz.compile(optimizer, loss=loss_ML_normal)
            else:
                self.Txz.compile(optimizer, loss=loss_ML_normal_reg)
        elif self.prior == 'lognormal':
            if reg_Jxz == 0:
                self.Txz.compile(optimizer, loss=loss_ML_lognormal)
            else:
                self.Txz.compile(optimizer, loss=loss_ML_lognormal_reg)
        elif self.prior == 'cauchy':
            if reg_Jxz == 0:
                self.Txz.compile(optimizer, loss=loss_ML_cauchy)
            else:
                self.Txz.compile(optimizer, loss=loss_ML_cauchy_reg)
        else:
            raise NotImplementedError('ML for prior ' + self.prior + ' is not implemented.')

        if xval is not None:
            validation_data = (xval, np.zeros_like(xval))
        else:
            validation_data = None

        hist = self.Txz.fit(x=x, y=np.zeros_like(x), validation_data=validation_data,
                            batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True)

        return hist

    def transform_xz(self, x):
        return self.Txz.predict(ensure_traj(x))

    def transform_xzJ(self, x):
        x = ensure_traj(x)
        if self.TxzJ is None:
            return self.Txz.predict(x), np.zeros(x.shape[0])
        else:
            z, J = self.TxzJ.predict(x)
            return z, J[:, 0]

    def transform_zx(self, z):
        return self.Tzx.predict(ensure_traj(z))

    def transform_zxJ(self, z):
        z = ensure_traj(z)
        if self.TxzJ is None:
            return self.Tzx.predict(z), np.zeros(z.shape[0])
        else:
            x, J = self.TzxJ.predict(z)
            return x, J[:, 0]

    def std_z(self, x):
        """ Computes average standard deviation from the origin in z for given x """
        z = self.Txz.predict(x)
        sigma = np.mean(z**2, axis=0)
        z_std_ = np.sqrt(np.mean(sigma))
        return z_std_

    def sample_z(self, temperature=1.0, nsample=100000, return_energy=False):
        """ Samples from prior distribution in x and produces generated x configurations

        Parameters:
        -----------
        temperature : float
            Relative temperature. Equal to the variance of the isotropic Gaussian sampled in z-space.
        nsample : int
            Number of samples

        Returns:
        --------
        sample_z : array
            Samples in z-space
        energy_z : array
            Energies of z samples (optional)

        """
        sample_z = None
        energy_z = None
        if self.prior == 'normal':
            sample_z = np.sqrt(temperature) * np.random.randn(nsample, self.dim)
            if return_energy:
                energy_z = self.dim * np.log(np.sqrt(temperature)) + np.sum(sample_z**2 / (2*temperature), axis=1)
        elif self.prior == 'lognormal':
            sample_z_normal = np.sqrt(temperature) * np.random.randn(nsample, self.dim)
            sample_z = np.exp(sample_z_normal)
            if return_energy:
                energy_z = np.sum(sample_z_normal**2 / (2*temperature), axis=1) + np.sum(sample_z_normal, axis=1)
        elif self.prior == 'cauchy':
            from scipy.stats import cauchy
            sample_z = cauchy(loc=0, scale=temperature).rvs(size=(nsample, self.dim))
            if return_energy:
                energy_z = np.sum(np.log(1 + (sample_z/temperature)**2), axis=1)
        else:
            raise NotImplementedError('Sampling for prior ' + self.prior + ' is not implemented.')

        if return_energy:
            return sample_z, energy_z
        else:
            return sample_z


class EnergyInvNet(InvNet):

    def __init__(self, energy_model, layers, prior='normal'):
        """ Invertible net where we have an energy function that defines p(x) """
        self.energy_model = energy_model
        super().__init__(energy_model.dim, layers, prior=prior)

    @classmethod
    def load(cls, filename, energy_model):
        """ Loads parameters into model. Careful: this clears the whole TF session!!
        """
        from deep_boltzmann.util import load_obj
        keras.backend.clear_session()
        D = load_obj(filename)
        prior = D['prior']
        layerdicts = D['layers']
        layers = [eval(d['type']).from_dict(d) for d in layerdicts]
        return EnergyInvNet(energy_model, layers, prior=prior)

    # TODO: This is only implemented for the normal prior.
    def log_w(self, high_energy, max_energy, temperature_factors=1.0):
        """ Computes the variance of the log reweighting factors
        """
        from deep_boltzmann.util import linlogcut
        z = self.input_z
        x = self.output_x
        # compute z energy
        Ez = self.dim * tf.log(tf.sqrt(temperature_factors)) + tf.reduce_sum(z**2, axis=1) / (2.0 * temperature_factors)
        # compute x energy and regularize
        Ex = self.energy_model.energy_tf(x) / temperature_factors
        Exreg = linlogcut(Ex, high_energy, max_energy, tf=True)
        # log weight
        log_w = -Exreg + Ez + self.log_det_Jzx[:, 0]
        return log_w

    def sample(self, temperature=1.0, nsample=100000):
        """ Samples from prior distribution in x and produces generated x configurations

        Parameters:
        -----------
        temperature : float
            Relative temperature. Equal to the variance of the isotropic Gaussian sampled in z-space.
        nsample : int
            Number of samples

        Returns:
        --------
        sample_z : array
            Samples in z-space
        sample_x : array
            Samples in x-space
        energy_z : array
            Energies of z samples
        energy_x : array
            Energies of x samples
        log_w : array
            Log weight of samples

        """
        sample_z, energy_z = self.sample_z(temperature=temperature, nsample=nsample, return_energy=True)
        sample_x, Jzx = self.transform_zxJ(sample_z)
        energy_x = self.energy_model.energy(sample_x) / temperature
        logw = -energy_x + energy_z + Jzx

        return sample_z, sample_x, energy_z, energy_x, logw

    def log_KL_x(self, high_energy, max_energy, temperature_factors=1.0, explore=1.0):
        """ Computes the KL divergence with respect to z|x and the Boltzmann distribution
        """
        from deep_boltzmann.util import linlogcut
        x = self.output_x
        # compute energy
        E = self.energy_model.energy_tf(x) / temperature_factors
        # regularize using log
        Ereg = linlogcut(E, high_energy, max_energy, tf=True)
        #return self.log_det_Jzx + Ereg
        return -explore * self.log_det_Jzx[:, 0] + Ereg

    def log_GaussianPriorMCMC_efficiency(self, high_energy, max_energy, metric=None, symmetric=False):
        """ Computes the efficiency of GaussianPriorMCMC from a parallel x1->z1, z2->x2 network.

        If metric is given, computes the efficiency as distance + log p_acc, where distance
        is computed by |x1-x2|**2

        """
        from deep_boltzmann.util import linlogcut
        # define variables
        x1 = self.input_x
        x2 = self.output_x
        z1 = self.output_z
        z2 = self.input_z
        # prior entropies
        H1 = 0.5 * tf.reduce_sum(z1**2, axis=1)
        H2 = 0.5 * tf.reduce_sum(z2**2, axis=1)
        # compute and regularize energies
        E1 = self.energy_model.energy_tf(x1)
        E1reg = linlogcut(E1, high_energy, max_energy, tf=True)
        E2 = self.energy_model.energy_tf(x2)
        E2reg = linlogcut(E2, high_energy, max_energy, tf=True)
        # free energy of samples
        F1 = E1reg - H1 + self.log_det_xz[:, 0]
        F2 = E2reg - H2 - self.log_det_zx[:, 0]
        # acceptance probability
        if symmetric:
            arg1 = linlogcut(F2 - F1, 10, 1000, tf=True)
            arg2 = linlogcut(F1 - F2, 10, 1000, tf=True)
            log_pacc = -tf.log(1 + tf.exp(arg1)) - tf.log(1 + tf.exp(arg2))
        else:
            arg = linlogcut(F2 - F1, 10, 1000, tf=True)
            log_pacc = -tf.log(1 + tf.exp(arg))
        # mean square distance
        if metric is None:
            return log_pacc
        else:
            d = (metric(x1) - metric(x2)) ** 2
            return d + log_pacc

    def log_GaussianPriorMCMC_efficiency_unsupervised(self, high_energy, max_energy, metric=None):
        """ Computes the efficiency of GaussianPriorMCMC
        """
        from deep_boltzmann.util import linlogcut
        # prior entropy
        z = self.input_z
        H = 0.5 * tf.reduce_sum(z**2, axis=1)
        # compute and regularize energy
        x = self.output_x
        E = self.energy_model.energy_tf(x)
        J = self.log_det_Jzx[:, 0]
        Ereg = linlogcut(E, high_energy, max_energy, tf=True)
        # free energy of samples
        F = Ereg - H - J
        # acceptance probability
        arg = linlogcut(F[1:] - F[:-1], 10, 1000, tf=True)
        log_pacc = -tf.log(1 + tf.exp(arg))
        # mean square distance
        # log_dist2 = tf.log(tf.reduce_mean((x[1:] - x[:-1])**2, axis=1))
        # complement with 0's
        log_pacc_0_ = tf.concat([np.array([0], dtype=np.float32), log_pacc], 0)
        log_pacc__0 = tf.concat([log_pacc, np.array([0], dtype=np.float32)], 0)
        if metric is None:
            return log_pacc_0_ + log_pacc__0
        else:
            d = (metric(x)[1:] - metric(x)[:1]) ** 2
            d_0_ = tf.concat([np.array([0], dtype=np.float32), d], 0)
            d__0 = tf.concat([d, np.array([0], dtype=np.float32)], 0)
            return d_0_ + d__0 + log_pacc_0_ + log_pacc__0

    def train_KL(self, optimizer=None, lr=0.001, epochs=2000, batch_size=1024, verbose=1, clipnorm=None,
                 high_energy=100, max_energy=1e10, temperature=1.0, explore=1.0):
        if optimizer is None:
            if clipnorm is None:
                optimizer = keras.optimizers.adam(lr=lr)
            else:
                optimizer = keras.optimizers.adam(lr=lr, clipnorm=clipnorm)

        import numbers
        if isinstance(temperature, numbers.Number):
            temperature = np.array([temperature])
        else:
            temperature = np.array(temperature)
        tfac = np.tile(temperature, int(batch_size / temperature.size) + 1)[:batch_size]

        def loss_KL(y_true, y_pred):
            return self.log_KL_x(high_energy, max_energy, temperature_factors=tfac, explore=explore)

        self.Tzx.compile(optimizer, loss=loss_KL)

        dummy_output = np.zeros((batch_size, self.dim))
        train_loss = []
        for e in range(epochs):
            # train in batches
            #w = np.sqrt(tfac)[:, None] * np.random.randn(batch_size, self.dim)
            w = self.sample_z(temperature=tfac[:, None], nsample=batch_size, return_energy=False)
            # w = np.random.randn(batch_size, self.dim)
            train_loss_batch = self.Tzx.train_on_batch(x=w, y=dummy_output)
            train_loss.append(train_loss_batch)
            if verbose == 1:
                print('Epoch', e, ' loss', np.mean(train_loss_batch))
                sys.stdout.flush()
        train_loss = np.array(train_loss)

        return train_loss

    def train_flexible(self, x, xval=None, optimizer=None, lr=0.001, epochs=2000, batch_size=1024, verbose=1, clipnorm=None,
                       high_energy=100, max_energy=1e10, std=1.0, reg_Jxz=0.0,
                       weight_ML=1.0,
                       weight_KL=1.0, temperature=1.0, explore=1.0,
                       weight_MC=0.0, metric=None, symmetric_MC=False, supervised_MC=True,
                       weight_W2=0.0,
                       weight_RCEnt=0.0, rc_func=None, rc_min=0.0, rc_max=1.0):
        import numbers
        if isinstance(temperature, numbers.Number):
            temperature = np.array([temperature])
        else:
            temperature = np.array(temperature)
        temperature = temperature.astype(np.float32)
        # redefine batch size to be a multiple of temperatures
        batch_size_per_temp = int(batch_size / temperature.size)
        batch_size = int(temperature.size * batch_size_per_temp)
        tidx = np.tile(np.arange(temperature.size), batch_size_per_temp)
        tfac = temperature[tidx]

        # Assemble Loss function
        def loss_ML(y_true, y_pred):
            return -self.log_likelihood_z_normal(std=std)
        def loss_ML_reg(y_true, y_pred):
            return -self.log_likelihood_z_normal(std=std) + reg_Jxz*self.reg_Jxz_uniform()
        def loss_KL(y_true, y_pred):
            return self.log_KL_x(high_energy, max_energy, temperature_factors=tfac, explore=explore)
        def loss_MCEff_supervised(y_true, y_pred):
            return -self.log_GaussianPriorMCMC_efficiency(high_energy, max_energy, metric=metric, symmetric=symmetric_MC)
        def loss_MCEff_unsupervised(y_true, y_pred):
            return -self.log_GaussianPriorMCMC_efficiency_unsupervised(high_energy, max_energy, metric=metric)
        def loss_MCEff_combined(y_true, y_pred):
            return -self.log_GaussianPriorMCMC_efficiency(high_energy, max_energy, metric=metric, symmetric=symmetric_MC) \
                   -0.5 * self.log_GaussianPriorMCMC_efficiency_unsupervised(high_energy, max_energy, metric=metric)
        def loss_W2_var(y_true, y_pred):
            # compute all reweighting factors
            lw = self.log_w(high_energy, max_energy, temperature_factors=tfac)
            # reshape to a column per temperature
            lwT = tf.reshape(lw, (batch_size_per_temp, temperature.size))
            lwT_mean = tf.reduce_mean(lwT, axis=0, keepdims=True)
            return tf.reduce_mean((lwT - lwT_mean) ** 2)
        def loss_W2_dev(y_true, y_pred):
            # compute all reweighting factors
            lw = self.log_w(high_energy, max_energy, temperature_factors=tfac)
            # reshape to a column per temperature
            lwT = tf.reshape(lw, (batch_size_per_temp, temperature.size))
            lwT_mean = tf.reduce_mean(lwT, axis=0, keepdims=True)
            return tf.reduce_mean(tf.abs(lwT - lwT_mean))
        gmeans = None
        gstd = 0.0
        if weight_RCEnt > 0.0:
            gmeans = np.linspace(rc_min, rc_max, 11)
            gstd = (rc_max - rc_min) / 11.0
        def loss_RCEnt(y_true, y_pred):
            return -self.rc_entropy(rc_func, gmeans, gstd, temperature.size)
        inputs = []
        outputs = []
        losses = []
        loss_weights = []
        if weight_ML > 0:
            inputs.append(self.input_x)
            outputs.append(self.output_z)
            if reg_Jxz == 0:
                losses.append(loss_ML)
            else:
                losses.append(loss_ML_reg)
            loss_weights.append(weight_ML)
        if weight_KL > 0:
            inputs.append(self.input_z)
            outputs.append(self.output_x)
            losses.append(loss_KL)
            loss_weights.append(weight_KL)
        if weight_MC > 0:
            if self.input_z not in inputs:
                inputs.append(self.input_z)
            #if self.output_x not in outputs:
            outputs.append(self.output_x)
            if supervised_MC == 'both':
                losses.append(loss_MCEff_combined)
            elif supervised_MC is True:
                losses.append(loss_MCEff_supervised)
            else:
                losses.append(loss_MCEff_unsupervised)
            loss_weights.append(weight_MC)
        if weight_W2 > 0:
            if self.input_z not in inputs:
                inputs.append(self.input_z)
            #if self.output_x not in outputs:
            outputs.append(self.output_x)
            losses.append(loss_W2_dev)
            loss_weights.append(weight_W2)
        if weight_RCEnt > 0:
            if self.input_z not in inputs:
                inputs.append(self.input_z)
            #if self.output_x not in outputs:
            outputs.append(self.output_x)
            losses.append(loss_RCEnt)
            loss_weights.append(weight_RCEnt)

        # data preprocessing
        N = x.shape[0]
        I = np.arange(N)
        if xval is not None:
            Nval = xval.shape[0]
            Ival = np.arange(N)
        else:
            Nval = 0
            Ival = None

        # build estimator
        if optimizer is None:
            if clipnorm is None:
                optimizer = keras.optimizers.adam(lr=lr)
            else:
                optimizer = keras.optimizers.adam(lr=lr, clipnorm=clipnorm)

        # assemble model
        dual_model = keras.models.Model(inputs=inputs, outputs=outputs)
        dual_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

        # training loop
        dummy_output = np.zeros((batch_size, self.dim))
        y = [dummy_output for o in outputs]
        loss_train = []
        loss_val = []
        for e in range(epochs):
            # sample batch
            x_batch = x[np.random.choice(I, size=batch_size, replace=True)]
            w_batch = np.sqrt(tfac)[:, None] * np.random.randn(batch_size, self.dim)
            l = dual_model.train_on_batch(x=[x_batch, w_batch], y=y)
            loss_train.append(l)

            # validate
            if xval is not None:
                xval_batch = xval[np.random.choice(I, size=batch_size, replace=True)]
                wval_batch = np.sqrt(tfac)[:, None] * np.random.randn(batch_size, self.dim)
                l = dual_model.test_on_batch(x=[xval_batch, wval_batch], y=y)
                loss_val.append(l)

            # print
            if verbose > 0:
                str_ = 'Epoch ' + str(e) + '/' + str(epochs) + ' '
                for i in range(len(dual_model.metrics_names)):
                    str_ += dual_model.metrics_names[i] + ' '
                    str_ += '{:.4f}'.format(loss_train[-1][i]) + ' '
                    if xval is not None:
                        str_ += '{:.4f}'.format(loss_val[-1][i]) + ' '
                print(str_)
                sys.stdout.flush()

        return dual_model.metrics_names, np.array(loss_train), np.array(loss_val)


def invnet(dim, layer_types, energy_model=None, channels=None,
           nl_layers=2, nl_hidden=100, nl_activation='relu', scale=None, prior='normal'):
    """
    layer_types : str
        String describing the sequence of layers. Usage:
            N NICER layer
            R RealNVP layerl
            S Scaling layer
        Splitting and merging layers will be added automatically
    energy_model : Energy model class
        Class with energy() and dim
    channels : array or None
        Assignment of dimensions to channels (0/1 array of length ndim)
    nl_layers : int
        Number of hidden layers in the nonlinear transformations
    nl_hidden : int
        Number of hidden units in each nonlinear layer
    nl_activation : str
        Activation functions used in the nonlinear layers
    scale : None or float
        If a scaling layer is used, fix the scale to this number. If None, scaling layers are trainable
    """
    # fix channels
    channels, indices_split, indices_merge = split_merge_indices(dim, nchannels=2, channels=channels)

    # augment layer types with split and merge layers
    split = False
    tmp = ''
    for ltype in layer_types:
        if ltype == 'S' and split:
            tmp += '>'
            split = False
        if (ltype == 'N' or ltype == 'R') and not split:
            tmp += '<'
            split = True
        tmp += ltype
    if split:
        tmp += '>'
    layer_types = tmp
    print(layer_types)

    # prepare layers
    layers = []


    for ltype in layer_types:
        if ltype == '<':
            # split into two x channels
            layers.append(SplitChannels(dim, nchannels=2, channels=channels))
        if ltype == '>':
            # merge into one z channel
            layers.append(MergeChannels(dim, nchannels=2, channels=channels))
        if ltype == 'N':
            M1 = nonlinear_transform(indices_split[1].size, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation)
            M2 = nonlinear_transform(indices_split[0].size, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation)
            layers.append(NICER([M1, M2]))
        elif ltype == 'R':
            S1 = nonlinear_transform(indices_split[1].size, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, init_outputs=0)
            T1 = nonlinear_transform(indices_split[1].size, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation)
            S2 = nonlinear_transform(indices_split[0].size, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation, init_outputs=0)
            T2 = nonlinear_transform(indices_split[0].size, nlayers=nl_layers, nhidden=nl_hidden,
                                     activation=nl_activation)
            layers.append(RealNVP([S1, T1, S2, T2]))
        elif ltype == 'S':
            # scaling layer
            if scale is None:
                scaling_factors = None
            else:
                scaling_factors = scale * np.ones((1, dim))
            layers.append(Scaling(dim, scaling_factors=scaling_factors, trainable=(scale is None)))

    if energy_model is None:
        return InvNet(dim, layers, prior=prior)
    else:
        return EnergyInvNet(energy_model, layers, prior=prior)



def create_NICERNet(energy_model, nlayers=10, nl_layers=2, nl_hidden=100, nl_activation='relu', channels=None,
                    scaled=True, scale=None, scale_trainable=True, prior='normal'):
    """ Constructs a reversible NICER network

    Parameters
    ----------
    energy_model : Energy model class
        Class with energy() and dim
    nlayers : int
        Number of NICER layers
    nl_layers : int
        Number of hidden layers in the nonlinear transformations
    nl_hidden : int
        Number of hidden units in each nonlinear layer
    nl_activation : str
        Activation functions used in the nonlinear layers
    z_variance_1 : bool
        If true, will try to enforce that the variance is 1 in z
    channels : array or None
        Assignment of dimensions to channels (0/1 array of length ndim)
    scaled : bool
        True to add a scaling layer before Z, False to keep the network fully volume-preserving (det 1)
    scaled : bool
        Initial value for scale
    scale_trainable : bool
        True if scale is trainable, otherwise fixed to input

    """
    dim = energy_model.dim
    channels, indices_split, indices_merge = split_merge_indices(dim, nchannels=2, channels=channels)

    layers = []
    # split into two x channels
    layers.append(SplitChannels(dim, nchannels=2, channels=channels))
    # create nonlinearities
    for l in range(nlayers):
        M1 = nonlinear_transform(indices_split[1].size, nlayers=nl_layers, nhidden=nl_hidden,
                                 activation=nl_activation)
        M2 = nonlinear_transform(indices_split[0].size, nlayers=nl_layers, nhidden=nl_hidden,
                                 activation=nl_activation)
        layers.append(NICER([M1, M2]))
    # merge into one z channel
    layers.append(MergeChannels(dim, nchannels=2, channels=channels))
    # scaling layer
    if scale is None:
        scaling_factors = None
    else:
        scaling_factors = scale * np.ones((1, dim))
    layers.append(Scaling(dim, scaling_factors=scaling_factors, trainable=scale_trainable,
                          name_xz='output_z_scaled', name_zx='input_z_scaled'))

    return EnergyInvNet(energy_model, layers, prior=prior)



def create_RealNVPNet(energy_model, nlayers=10, nl_layers=2, nl_hidden=100, nl_activation='relu',
                      channels=None, prior='normal'):
    """ Constructs a reversible NICER network

    Parameters
    ----------
    energy_model : Energy model class
        Class with energy() and dim
    scaled : bool
        True to add a scaling layer before Z, False to keep the network fully volume-preserving (det 1)
    nlayers : int
        Number of NICER layers
    nl_layers : int
        Number of hidden layers in the nonlinear transformations
    nl_hidden : int
        Number of hidden units in each nonlinear layer
    nl_activation : str
        Activation functions used in the nonlinear layers
    z_variance_1 : bool
        If true, will try to enforce that the variance is 1 in z
    channels : array or None
        Assignment of dimensions to channels (0/1 array of length ndim)

    """
    dim = energy_model.dim
    channels, indices_split, indices_merge = split_merge_indices(dim, nchannels=2, channels=channels)

    layers = []
    # split into two x channels
    layers.append(SplitChannels(dim, nchannels=2, channels=channels))
    # create nonlinearities
    for l in range(nlayers):
        S1 = nonlinear_transform(indices_split[1].size, nlayers=nl_layers, nhidden=nl_hidden,
                                 activation=nl_activation, init_outputs=0)
        T1 = nonlinear_transform(indices_split[1].size, nlayers=nl_layers, nhidden=nl_hidden,
                                 activation=nl_activation)
        S2 = nonlinear_transform(indices_split[0].size, nlayers=nl_layers, nhidden=nl_hidden,
                                 activation=nl_activation, init_outputs=0)
        T2 = nonlinear_transform(indices_split[0].size, nlayers=nl_layers, nhidden=nl_hidden,
                                 activation=nl_activation)
        layers.append(RealNVP([S1, T1, S2, T2]))
    # merge into one z channel
    layers.append(MergeChannels(dim, nchannels=2, channels=channels))

    return EnergyInvNet(energy_model, layers, prior)
