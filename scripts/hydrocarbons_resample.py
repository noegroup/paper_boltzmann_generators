import numpy as np
import mdtraj
import sys

from deep_boltzmann.networks.invertible import create_RealNVPNet, EnergyInvNet
from deep_boltzmann.models.conformations import RotamerMapper
from deep_boltzmann.models.MM import MM
from deep_boltzmann.models.MM import build_hydrocarbon
from deep_boltzmann.util import load_obj, save_obj

def sample_trajectory(mm, network, trajref, batchsize=100000, nbatch=5):
    sxs = []
    Exs = []
    logws = []
    for i in range(nbatch):
        sz, sx, Ez, Ex, logw = network.sample(temperature=1.0, nsample=batchsize)
        sxs.append(sx)
        Exs.append(Ex)
        logws.append(logw)
    sxs = np.vstack(sxs)
    Exs = np.concatenate(Exs)
    logws = np.concatenate(logws)
    toppar = mm.toppar
    trajgen = mdtraj.Trajectory(sxs.reshape((sxs.shape[0], int(sxs.shape[1]/3), 3)), toppar.mdtraj_topology())
    trajgen = trajgen.superpose(trajref)
    # replace sxs by aligned version
    sxs = trajgen.xyz.reshape((trajgen.xyz.shape[0], trajgen.xyz.shape[1]*trajgen.xyz.shape[2]))
    return sxs, Exs, logws, trajgen

def list_rotamers(mm, sx, Ex, selected_torsions):
    # find rotamers
    torsions = mm.torsions(sx)[:, np.array(selected_torsions)]
    torsion_index = rotamer_mapper.torsion2index(torsions)
    rotamer_index, rotamer_key, rotamer_count = rotamer_mapper.histogram(torsion_index)
    Isort = np.argsort(rotamer_count)[::-1]
    for i in Isort:
        p_i = rotamer_count[i] / rotamer_count.sum()
        It = np.where(torsion_index == rotamer_index[i])[0]
        print(rotamer_index[i], '\t', rotamer_key[i], '\t', p_i, '\t', -np.log(p_i), '\t', Ex[It].min())
        sys.stdout.flush()
    return torsions, torsion_index, rotamer_index[Isort], [rotamer_key[i] for i in Isort], rotamer_count[Isort]

def resample_by_rotamer(sx, nsample, nrot, torsion_index, rotamer_index):
    nrot_selected = min(nrot, len(rotamer_index))
    nsample_per_nrot = int(nsample / nrot_selected) + 1
    xcur = []
    for i in range(nrot_selected):
        It = np.where(torsion_index == rotamer_index[i])[0]
        It_sample = np.random.choice(It, nsample_per_nrot, replace=True)
        xcur.append(sx[It_sample])
    xcur = np.vstack(xcur)[:nsample]
    return xcur

def resample(mm, Ctor, network, xinit, trajref, epochsML=500, epochsKL=1000, niter=10, nrot=10, pupdate=0.5):
    # store
    network.save('../local_data/out/hydrocarbon_cyc9/tmp.pkl')

    nsample = xinit.shape[0]

    # report on init
    Einit = mm_cyc9.energy(xinit)
    print('Iter 0 Emin', Einit.min(), 'E<500', len(np.where(Einit<500)[0])/len(Einit))
    torsions, torsion_index, rotamer_index, rotamer_key, rotamer_count = list_rotamers(mm, xinit, Einit, Ctor)
    print()
    sys.stdout.flush()
    xcur = resample_by_rotamer(xinit, xinit.shape[0], nrot, torsion_index, rotamer_index)
    # xcur = xinit

    for k in range(niter):
        if k == 0:
            # train ML
            hml = network.train_ML(xcur, lr=0.001, epochs=epochsML, verbose=0)
            # train KL
            hkl = network.train_flexible(xcur, lr=0.0001, epochs=epochsKL, weight_ML=1.0, weight_KL=0.01,
                                         high_energy=10000, max_energy=1e15, explore=1.0, verbose=0)
        hkl = network.train_flexible(xcur, lr=0.0001, epochs=epochsKL, weight_ML=1.0, weight_KL=0.1,
                                     high_energy=10000, max_energy=1e15, explore=1.0, verbose=0)
        # check if nan and rescue
        if np.any(np.isnan(hkl[1])):
            print('NaN occurred. Reloading')
            sys.stdout.flush()
            network = EnergyInvNet.load('./tmp.pkl', mm_cyc9)
            continue
        # sample
        sx, Ex, logw, trajgen = sample_trajectory(mm, network, trajref, batchsize=100000, nbatch=10)
        # report energy
        print('Iter', k+1, 'Emin', Ex.min(), 'E<500', len(np.where(Ex<500)[0])/len(Ex))
        # list rotamers
        torsions, torsion_index, rotamer_index, rotamer_key, rotamer_count = list_rotamers(mm, sx, Ex, Ctor)
        print()
        sys.stdout.flush()
        # resample x
        xnew = resample_by_rotamer(sx, nsample, nrot, torsion_index, rotamer_index)
        # mix old and new
        xcur = np.where((np.random.rand(nsample)<0.5)[:, None], xnew, xcur)
        # save
        network.save('../local_data/out/hydrocarbon_cyc9/network.pkl')
        save_obj(sx, '../local_data/out/hydrocarbon_cyc9/sampleX.pkl')

    return network

# build MM
Cbonds = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 0]]
toppar_cyc9 = build_hydrocarbon(9, Cbonds)
mm_cyc9 = MM(toppar_cyc9, align=0.0)
top_cyc9 = toppar_cyc9.mdtraj_topology()
rotamer_mapper = RotamerMapper(9)

# get C torsions
Ctor = []
for i, tor in enumerate(toppar_cyc9.torsion_indices):
    is_Ctor = True
    for j in tor:
        if not toppar_cyc9.atom_names[j].startswith('C'):
            is_Ctor = False
    if is_Ctor:
        Ctor.append(i)

# load data
remd_dict = load_obj('../local_data/out/hydrocarbon_cyc9/remd_data.pkl')
traj = mdtraj.Trajectory(remd_dict['trajs'][0].reshape(remd_dict['trajs'][0].shape[0], toppar_cyc9.natoms, 3), top_cyc9)
traj = traj[10000:14000]
traj = traj.superpose(traj[0])
xtrain = traj.xyz.reshape((traj.xyz.shape[0], traj.xyz.shape[1]*traj.xyz.shape[2]))

network_resample = create_RealNVPNet(mm_cyc9, nlayers=4, nl_activation='tanh')
resample(mm_cyc9, Ctor, network_resample, xtrain, traj[0], epochsML=300, epochsKL=300, niter=200, nrot=20, pupdate=0.3)