import numpy as np
import sys

import keras
import tensorflow as tf
from deep_boltzmann.models import ParticleDimer
from deep_boltzmann.networks.invertible import invnet, EnergyInvNet
from deep_boltzmann.sampling import GaussianPriorMCMC
from deep_boltzmann.sampling.latent_sampling import BiasedModel
from deep_boltzmann.sampling.permutation import HungarianMapper
from deep_boltzmann.networks.util import load_obj, save_obj

def energy_stats(network):
    blind_sample_z, blind_sample_x, blind_energy_z, blind_energies_x, _ = network.sample(temperature=1.0, nsample=100000)
    nlow = np.size(np.where(blind_energies_x < 100)[0])
    return blind_energies_x.min(), nlow

def sample_stats(d):
    from deep_boltzmann.util import count_transitions, acf
    ntrans = count_transitions(d, 1.25, 1.75)
    acf100 = acf(d, 100)
    acf1000 = acf(d, 1000)
    acf10000 = acf(d, 10000)
    return ntrans, acf100[0], acf1000[0], acf10000[0]

def distance_metric(x):
    """ Outputs 2.5 for closed and 5 for open dimer
    """
    d = model.dimer_distance_tf(x)
    dscaled = 3.0 * (d - 1.5)
    return 2.5 * (1.0 + tf.sigmoid(dscaled))

def train_network(model, x, xval, hyperparams, Zepochs, Eschedule):
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

    # test energies
    h['Emin'], h['N_Elow'] = energy_stats(network)
    print('Emin / N_Elow:', h['Emin'], h['N_Elow'])
    sys.stdout.flush()

    print('TEST', h_str,
          ', std_z_x={:.2f}, {:.2f}'.format(std_z_x, std_z_xval),
          ', loss_Z={:.2f}, {:.2f}'.format(h['loss_Z'][-1], h['loss_Z_val'][-1]),
          ', E_min={:.2f}'.format(h['Emin']), ', N_Elow={:.2f}'.format(h['N_Elow']),
          #', N_trans_MC=', ntrans_MC,
          #', ACF_MC={:.2f}, {:.2f}, {:.2f}'.format(acf100_MC, acf1000_MC, acf10000_MC)
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
hyperparams = {'layer_types' : 'RRRRRRRR',
               'nl_layers' : 3,
               'nl_hidden' : 200,
               'nl_activation' : 'tanh',
               'zstd' : 1.0,
               'reg_Jxz' : 0.0,
               'temperature' : [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
               'weight_ML' : 0.1,
               'weight_W2' : 0.0,
               'weight_RC' : 10.0,
               }

# train E
Eschedule = [[100,  0.0001,  10000, 100,   0.1,  1.0],
             [100,  0.0001,  10000, 100,   0.3,  1.0],
             [300,  0.0001,  10000, 100,   1.0,  5.0],
             [300,  0.0001,  10000, 100,   1.0, 10.0],
             [1000, 0.0001,   2000, 20,    1.0, 10.0],
             [2000, 0.0001,   1000, hyperparams['weight_ML'], hyperparams['weight_W2'], hyperparams['weight_RC']],
            ]

# Repeat training process
wdir = '../local_data/particles_tilted/trained/'
for i in range(20):
    keras.backend.clear_session()
    network, h = train_network(model, x, xval, hyperparams, 20, Eschedule)
    network.save(wdir + 'network_' + str(i+1) + '.pkl')
    save_obj(h, wdir + 'hyperdata_' + str(i+1) + '.pkl')
    print('\nNETWORK' + str(i) + ' DONE + SAVED.\n')
    sys.stdout.flush()


