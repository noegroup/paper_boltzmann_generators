import numpy as np
from deep_boltzmann.plot import plot_density

def test_xz_projection(Txz, xtrajs, rctrajs=None, subplots=None, colors=None, density=False):
    """ Projects x trajectories into z space and and plots their distribution

    Parameters
    ----------
    xtrajs : list of arrays
        List of x-trajectories.
    rctrajs : list of reaction coordinate values
        Reaction coordinate (RC) values corresponding to the trajectories, to show the RC-overlap in z space.
    subplots : array of bool or None
        Whether to plot each subplot

    """
    import matplotlib.pyplot as plt
    # TODO: fix this. This assumes half the dimensions are channels
    dim_channel = int(xtrajs[0].shape[1] / 2)
    # all inputs
    xall = np.vstack(xtrajs)
    if colors is None:
        colors = ['black' for _ in xtrajs]
    # transform
    ztrajs = [Txz.predict(xtraj) for xtraj in xtrajs]
    zall = np.vstack(ztrajs)
    # do PCA
    zmean_ = zall.mean(axis=0)
    Czz_ = np.dot((zall - zmean_).T, (zall - zmean_)) / zall.shape[0]
    zeval_, zevec_ = np.linalg.eig(Czz_)
    zprojs = [(ztraj - zmean_).dot(zevec_) for ztraj in ztrajs]  # .dot(np.diag(np.sqrt(1.0/zeval)))

    # plots
    if subplots is None:
        if rctrajs is not None:
            subplots = np.array([True, True, True, True])
        else:
            subplots = np.array([True, True, True, False])
    nplots = np.count_nonzero(subplots)
    fig, axes = plt.subplots(1, nplots, figsize=(4*nplots, 4))

    cplot = 0
    # x distribution
    if subplots[0]:
        for xtraj, color in zip(xtrajs, colors):
            if density:
                plot_density(xtraj[:, 0], xtraj[:, 1], axis=axes[cplot], color=color)
            else:
                axes[cplot].plot(xtraj[:, 0], xtraj[:, 1], linewidth=0, marker='.', markersize=2, color=color)
        axes[cplot].set_xlabel('x$_1$')
        axes[cplot].set_ylabel('x$_2$')
        cplot += 1

    # z distribution
    if subplots[1]:
        for ztraj, color in zip(ztrajs, colors):
            if density:
                plot_density(zall[:, 0], zall[:, 1], axis=axes[cplot], color='black')
            else:
                axes[cplot].plot(ztraj[:, 0], ztraj[:, 1], linewidth=0, marker='.', markersize=2, color=color)
        axes[cplot].set_xlabel('z$_1$')
        axes[cplot].set_ylabel('z$_2$')
        cplot += 1

    # z PCA projection
    if subplots[2]:
        for zproj, color in zip(zprojs, colors):
            axes[cplot].plot(zproj[:, 0], zproj[:, 1], linewidth=0, marker='.', markersize=2, color=color)
        axes[cplot].set_xlabel('z principal component 1')
        axes[cplot].set_ylabel('z principal component 2')
        cplot += 1

    if subplots[3]:
        rcall = np.concatenate(rctrajs)
        # regress to distance
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(zall, rcall)
        # plot histograms
        for ztraj, color in zip(ztrajs, colors):  # plt.rcParams['axes.prop_cycle'].by_key()['color']
            axes[cplot].hist(ztraj.dot(lr.coef_), 50, linewidth=2, alpha=0.2, color=color)
            axes[cplot].hist(ztraj.dot(lr.coef_), 50, histtype='step', linewidth=2, color=color)
        axes[cplot].set_xlabel('z - dimer dist. regressor')
        axes[cplot].set_yticks([])
        axes[cplot].set_ylabel('Probability')

    plt.tight_layout()

    return fig, axes

def test_generate_x(energy_model, xtrajs, sample_energies, max_energy=150,
                    figsize=None, layout=None, colors=None, titles=True):
    """ Generates using x trajectories as an example

    Parameters
    ----------
    energy_model : Energy Model
        Energy model object that must provide the function energy(x)
    xtrajs : list of arrays
        List of x-trajectories.
    max_energy : float
        Maximum energy to be shown in histograms
    figsize : (width, height) or None
        Figure size
    layout : (rows, cols) or None
        Arrangement of multi-axes plot


    """
    # broadcast
    if isinstance(xtrajs, list) and not isinstance(sample_energies, list):
        sample_energies = [sample_energies for i in range(len(xtrajs))]
    if not isinstance(xtrajs, list) and isinstance(sample_energies, list):
        xtrajs = [xtrajs for i in range(len(sample_energies))]
    if not isinstance(xtrajs, list) and not isinstance(sample_energies, list):
        xtrajs = [xtrajs]
        sample_energies = [sample_energies]
    # generate according to x
    #if std_z is None:
    #    std_z = self.std_z(np.vstack(xtrajs))  # compute std of sample trajs in z-space
    #sample_z_z, sample_z_x, sample_z_energy_z, sample_z_energy_x = self.generate_x(std_z, nsample=nsample)
    # compute generated energies
    energies_sample_x_low = [se[np.where(se < max_energy)[0]] for se in sample_energies]
    # plots
    import matplotlib.pyplot as plt
    if figsize is None:
        figsize = (5*len(xtrajs), 4)
    if layout is None:
        layout = (1, len(xtrajs))
    if colors is None:
        colors = ['blue' for i in len(xtrajs)]
    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
    for i, xtraj in enumerate(xtrajs):
        # print some stats
        print('Traj ', i, 'Fraction of low energies: ', np.size(energies_sample_x_low[i])/(1.0*sample_energies[i].size))
        print('Traj ', i, 'Minimum energy: ', np.min(sample_energies[i]))
        # plot generated energies
        axes[i].hist(energies_sample_x_low[i], 70, density=True, histtype='stepfilled', color='black', alpha=0.2)
        axes[i].hist(energies_sample_x_low[i], 70, density=True, histtype='step', color='black', linewidth=2, label='z sampling')
        # plot simulated energies
        if xtraj is not None:
            energies_x = energy_model.energy(xtraj)
            min_energy = min(energies_x.min(), energies_sample_x_low[i].min())
            axes[i].hist(energies_x, 50, density=True, histtype='stepfilled', color=colors[i], alpha=0.2)
            axes[i].hist(energies_x, 50, density=True, histtype='step', color=colors[i], linewidth=2, label='MD')
        # plot energy histogram (comparison of input and generated)
        axes[i].set_xlim(min_energy, max_energy)
        axes[i].set_xlabel('Energy / kT')
        axes[i].set_yticks([])
        axes[i].set_ylabel('Density')
        axes[i].legend(frameon=False)
        if titles:
            axes[i].set_title('Trajectory ' + str(i+1))
    return fig, axes