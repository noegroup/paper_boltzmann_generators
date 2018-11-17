import numpy as np
import sys

import tensorflow as tf
from deep_boltzmann.models import ParticleDimer
from deep_boltzmann.networks.invertible import EnergyInvNet
from deep_boltzmann.sampling.latent_sampling import sample_RC
from deep_boltzmann.sampling.permutation import HungarianMapper
from deep_boltzmann.util import save_obj

def energy_stats(network):
    blind_sample_z, blind_sample_x, blind_energy_z, blind_energies_x, _ = network.sample(temperature=1.0, nsample=100000)
    nlow = np.size(np.where(blind_energies_x < 100)[0])
    return blind_energies_x.min(), nlow

def distance_metric(x):
    """ Outputs 2.5 for closed and 5 for open dimer
    """
    d = model.dimer_distance_tf(x)
    dscaled = 3.0 * (d - 1.5)
    return 2.5 * (1.0 + tf.sigmoid(dscaled))

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

mapper = HungarianMapper(1.05*model.init_positions(1.5), dim=2, identical_particles=np.arange(2, model.nparticles))

# Repeat training process
wdir = '../local_data/particles_tilted/trained/'
result_file = wdir + 'distances_sample.pkl'

temperatures = [0.5, 1.0, 2.0]
temperature_labels = ['05', '10', '20']

#if os.path.isfile(result_file):
#    print('Result file already exists. Loading and continuing...')
#    sys.stdout.flush()
#    result = load_obj(result_file)
#else:
result = {'temperatures' : temperatures}


for i in range(len(temperatures)):
    print('\nTEMPERATURE ' + str(temperatures[i]))
    Ds = []
    Ws = []
    for k in range(20):
        print('\nANALYZE ' + str(k) + '/20\n')
        sys.stdout.flush()
        try:
            #h = load_obj(wdir + 'hyperdata_' + str(i+1) + '.pkl')
            #if h['E_min'] < 40:
            network = EnergyInvNet.load(wdir + 'network_' + str(k+1) + '.pkl', model)
            D, W = sample_RC(network, 1000000, model.dimer_distance, temperature=temperatures[i], failfast=True,
                             xmapper=mapper) # , xmapper=mapper
            Ds.append(D)
            Ws.append(W)
            #else:
            #    print('Skipping ' + str(i))
            #    sys.stdout.flush()
        except:
            print('Skipping ' + str(i))
            sys.stdout.flush()
    result['D'+str(temperature_labels[i])] = Ds
    result['W'+str(temperature_labels[i])] = Ws
save_obj(result, result_file)
