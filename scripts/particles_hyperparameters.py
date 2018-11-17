import numpy as np
import sys

import keras
import tensorflow as tf
from deep_boltzmann.models import ParticleDimer
from deep_boltzmann.networks.invertible import invnet, EnergyInvNet
from deep_boltzmann.sampling import GaussianPriorMCMC
from deep_boltzmann.sampling.permutation import HungarianMapper
from deep_boltzmann.networks.util import load_obj, save_obj
from deep_boltzmann.sampling.analysis import free_energy_bootstrap, mean_finite, std_finite

def energy_stats(network):
    sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=100000)
    nlow = np.size(np.where(energy_x < 100)[0])
    return energy_x.min(), nlow

# reweighting
def free_energy_estimator_error(network, temperature=1.0):
    Ess = []
    for i in range(5):
        sample_z, sample_x, energy_z, energy_x, log_w = network.sample(temperature=1.0, nsample=100000)
        bin_means, Es = free_energy_bootstrap(model.dimer_distance(sample_x), 0.7, 2.3, 50, sample=100, weights=np.exp(log_w))
        Ess.append(Es)
    Ess = np.vstack(Ess)
    var = mean_finite(std_finite(Ess, axis=0) ** 2)
    return np.sqrt(var)

def distance_metric(x):
    """ Outputs 2.5 for closed and 5 for open dimer
    """
    d = model.dimer_distance_tf(x)
    dscaled = 3.0 * (d - 1.5)
    return 2.5 * (1.0 + tf.sigmoid(dscaled))

def train_network(x, xval, hyperparams, Zepochs, Eschedule):
    # results here
    h = hyperparams.copy()
    h_str = str(hyperparams)
    # network
    network = invnet(model.dim, layer_types=h['layer_types'], energy_model=model,
                     nl_layers=h['nl_layers'], nl_hidden=h['nl_hidden'], nl_activation=h['nl_activation'], scale=None)

    # train Z
    hist = network.train_ML(x, xval=xval, epochs=Zepochs, std=h['zstd'], reg_Jxz=h['reg_Jxz'], verbose=2)
    h['loss_Z'] = hist.history['loss']
    h['loss_Z_val'] = hist.history['val_loss']
    print('Z done')
    sys.stdout.flush()

    for i, s in enumerate(Eschedule):
        print(s)#'high_energy =', s[0], 'weight_ML =', s[1], 'epochs =', s[2])
        sys.stdout.flush()
        loss_names, loss_train, loss_val = network.train_flexible(x, xval=xval, epochs=s[0], lr=s[1], batch_size=8000,
                                                                  verbose=2, high_energy=s[2], max_energy=1e10,
                                                                  weight_ML=s[3],
                                                                  weight_KL=1.0, temperature=h['temperature'],
                                                                  weight_MC=0.0,
                                                                  weight_W2=s[4],
                                                                  weight_RCEnt=s[5],
                                                                  rc_func=model.dimer_distance_tf, rc_min=0.5, rc_max=2.5,
                                                                  std=h['zstd'], reg_Jxz=h['reg_Jxz'])

    # test std_z
    std_z_x = network.std_z(x)
    std_z_xval = network.std_z(xval)
    print('std:', std_z_x, std_z_xval)
    sys.stdout.flush()

    if np.isnan(std_z_x) or np.isnan(std_z_xval):
        print('NaN Network. Returning without analysis.')
        sys.stdout.flush()
        return network, h

    # test energies
    h['Emin'], h['N_Elow'] = energy_stats(network)
    print('Emin / N_Elow:', h['Emin'], h['N_Elow'])
    sys.stdout.flush()

    # test sampling
    h['std_est_05'] = free_energy_estimator_error(network, temperature=0.5)
    h['std_est_10'] = free_energy_estimator_error(network, temperature=1.0)
    h['std_est_20'] = free_energy_estimator_error(network, temperature=2.0)
    print('Free Energy Std Errors:', h['std_est_05'], h['std_est_10'], h['std_est_20'])
    sys.stdout.flush()

    print(h_str,
          ', std_z_x={:.2f}, {:.2f}'.format(std_z_x, std_z_xval),
          ', loss_Z={:.2f}, {:.2f}'.format(h['loss_Z'][-1], h['loss_Z_val'][-1]),
          ', E_min={:.2f}'.format(h['Emin']), ', N_Elow={:.2f}'.format(h['N_Elow']),
          #', loss_E_z={:.2f}, {:.2f}'.format(h['loss_E_ML'], h['loss_E_ML_val']),
          #', loss_E_kl={:.2f}, {:.2f}'.format(h['loss_E_KL'], h['loss_E_KL_val']),
          ', Free Energy Std Errors={:.2f}, {:.2f}, {:.2f}'.format(h['std_est_05'], h['std_est_10'], h['std_est_20'])
          )
    print()
    sys.stdout.flush()

    return network, h



# load trajectory data
trajdict = np.load('../local_data/particles_tilted/trajdata_tilted_long.npz')
import ast
params = ast.literal_eval(str(trajdict['params']))
traj_closed_train = trajdict['traj_closed_train_hungarian']
traj_open_train = trajdict['traj_open_train_hungarian']
traj_closed_test = trajdict['traj_closed_test_hungarian']
traj_open_test = trajdict['traj_open_test_hungarian']
x = np.vstack([traj_closed_train, traj_open_train])
xval = np.vstack([traj_closed_test, traj_open_test])

# create model
params['grid_k'] = 0.0
model = ParticleDimer(params=params)

# hyperparameters
hyperparameters_default = {'layer_types' : 'RRRRRRRR',
                           'nl_layers' : 3,
                           'nl_hidden' : 200,
                           'nl_activation' : 'tanh',
                           'zstd' : 1.0,
                           'reg_Jxz' : 0.0, #0.25,
                           'temperature' : [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
                           'weight_ML' : 0.1,
                           'weight_W2' : 0.0,
                           'weight_RC' : 10.0,
                           }
hyperparameters_other = {'layer_types' : ['RRRR', 'RRRRRRR', 'RRRRRRRRRRRR',
                                          'RNRN', 'RNRNRNRNRNRNRN', 'RNRNRNRNRNRNRNRNRNRNRNRN'],
                         'nl_layers' : [2, 4],
                         'nl_hidden' : [50, 100],
                         #'nl_activation' : ['relu', 'softplus'],
                         #'zstd' : [0.8, 1.0],
                         #'reg_Jxz' : [0.0, 0.1],
                         'weight_ML' : [0.01, 1.0],
                         'weight_W2' : [0.1, 1.0],
                         'weight_RC' : [1.0, 5.0, 20.0],
                         }
hyperparameters_var = [hyperparameters_default.copy()]
for key_var in hyperparameters_other.keys():
    for v in hyperparameters_other[key_var]:
        h = hyperparameters_default.copy()
        h[key_var] = v
        hyperparameters_var.append(h)

wdir = '../local_data/particles_tilted/hyper/'
for k, h in enumerate(hyperparameters_var):
    print('Hyperparameters', k, '/', len(hyperparameters_var))
    h_str = str(h)
    sys.stdout.flush()

    if 'std_est_20' in h.keys():
        print(' Already done. Skipping ...')
        continue

    print('Clearing Session')
    keras.backend.clear_session()

    Eschedule = [[100,  0.0001,  10000, 100,   0.1,  1.0],
                 [100,  0.0001,  10000, 100,   0.3,  1.0],
                 [300,  0.0001,  10000, 100,   1.0,  5.0],
                 [300,  0.0001,  10000, 100,   1.0, 10.0],
                 [1000, 0.0001,   2000, 20,    1.0, 10.0],
                 [2000, 0.0001,   1000, h['weight_ML'], h['weight_W2'], h['weight_RC']],
                ]

    network, _ = train_network(x, xval, h, 20, Eschedule)

    # save everything
    network.save(wdir + 'network_' + str(k) + '.pkl')
    save_obj(hyperparameters_var, wdir + 'hyperdata.pkl')
