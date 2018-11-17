__author__ = 'noe'

import numpy as np

def plot_traj_hist(traj, ax1=None, ax2=None, color='blue', ylim=None, ylabel=''):
    import matplotlib.pyplot as plt
    if ax1 is None and ax2 is None:
        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((1, 3), (0, 2))

    if ylim is None:
        dy = np.max(traj) - np.min(traj)
        ylim = [np.min(traj) - 0.25 * dy, np.max(traj) + 0.25 * dy]

    ax1.plot(traj, color=color, alpha=0.7)
    ax1.set_xlim(0, len(traj))
    ax1.set_xlabel('Time / steps')
    ax1.set_ylim(ylim[0], ylim[1])
    ax1.set_ylabel(ylabel)

    nbins = int(np.sqrt(np.size(traj)))
    ax2.hist(traj, nbins, range=[ylim[0], ylim[1]], orientation='horizontal', histtype='stepfilled', color=color, alpha=0.2)
    ax2.hist(traj, nbins, range=[ylim[0], ylim[1]], orientation='horizontal', histtype='step', color=color, linewidth=2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim(ylim[0], ylim[1])
    ax2.set_xlabel('Probability')


    return ax1.get_figure(), ax1, ax2

def plot_density(x, y, axis=None, bins=20, color='blue'):
    H, xedges, yedges = np.histogram2d(x, y, bins)
    Xgrid, Ygrid = xedges[:-1], yedges[1:]
    # Xgrid, Ygrid = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    if axis is None:
        import matplotlib.pyplot as plt
        axis = plt.gca()

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('', ['white', color], N=1000)
    axis.contourf(Xgrid, Ygrid, H, 20, vmin=0, cmap=cmap);
    axis.contour(Xgrid, Ygrid, H, 10, vmin=0, colors='black', linewidths=1, linestyles='solid');