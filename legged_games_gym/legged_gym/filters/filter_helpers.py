import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datetime import datetime
import os

from scipy.stats import multivariate_normal

def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def plot_state_traj(real_traj, z_traj, pred_traj, est_traj, show_fig=False):
        """
        Plots trajectories.

        Args:
            real_traj (np.array): ground truth state traj of shape [num_tsteps, num_envs, num_states]
            z_traj (np.array): measurement traj of shape [num_tsteps, num_envs, num_obs]
            pred_traj (np.array): predicted state traj of shape [num_tsteps, num_envs, num_states]
            est_traj (np.array): estimated state traj of shape [num_tsteps, num_envs, num_states]
        """
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        num_envs = real_traj.shape[1]

        colors_z = np.linspace(0, 1, len(z_traj[:, 0, 0]))
        colors_r = np.linspace(0, 1, len(real_traj[:, 0, 0]))
        colors_e = np.linspace(0, 1, len(est_traj[:, 0, 0]))
        if pred_traj is not None:
            colors_p = np.linspace(0, 1, len(pred_traj[:, 0, 0]))

        gr_cmap = mpl.colormaps['Greens']
        rd_cmap = mpl.colormaps['Reds']
        gy_cmap = mpl.colormaps['Greys']

        for eidx in range(num_envs):
            # plot measurements
            z_dx = np.cos(z_traj[:, eidx, -1])
            z_dy = np.sin(z_traj[:, eidx, -1])
            e_dx = np.cos(est_traj[:, eidx, -1])
            e_dy = np.sin(est_traj[:, eidx, -1])
            r_dx = np.cos(real_traj[:, eidx, -1])
            r_dy = np.sin(real_traj[:, eidx, -1])

            # plot measurements
            plt.scatter(z_traj[:, eidx, 0], z_traj[:, eidx, 1], c=colors_z, cmap='Greens', label="measurements")
            plt.quiver(z_traj[:, eidx, 0], z_traj[:, eidx, 1], z_dx, z_dy, color=gr_cmap(colors_z),
                       label="measurements")
            
            # plot GT state trajectory
            plt.scatter(real_traj[:, eidx, 0], real_traj[:, eidx, 1], c=colors_r, cmap='Greys', label="real state")
            plt.quiver(real_traj[:, eidx, 0], real_traj[:, eidx, 1], r_dx, r_dy, color=gy_cmap(colors_r),
                       label="real state")
            # plot estimated state trajectory (a posteriori)
            plt.scatter(est_traj[:, eidx, 0], est_traj[:, eidx, 1], c=colors_e, cmap='Reds', label="estimates")
            plt.quiver(est_traj[:, eidx, 0], est_traj[:, eidx, 1], e_dx, e_dy, color=rd_cmap(colors_e),
                       label="estimates")
            # plot initial state explicitly
            plt.scatter(real_traj[0, eidx, 0], real_traj[0, eidx, 1], c=colors_r[0], edgecolors='k', cmap='Greys')

        # plot the origin (i.e., the goal of the controller)
        plt.scatter(0, 0, c='m', edgecolors='k', marker=",", s=100)
        # plot relative heading needed for the robot to face the prey:
        # since theta_a == 0, then the optimal theta_rel = 0 - atan(rel_y, rel_x)
        # target_rel_yaw = - np.arctan2(real_traj[0, eidx, 1], real_traj[0, eidx, 0])
        # plt.quiver(0, 0, np.cos(target_rel_yaw), np.sin(target_rel_yaw), color='m')

        # plt.ylim([-5, 5])
        # plt.xlim([-5, 5])
        plt.xlabel("relative x-pos")
        plt.ylabel("relative y-pos")

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels)
        # ax.legend()

        red_patch = mpatches.Patch(color='red', label='estimates')
        green_patch = mpatches.Patch(color='green', label='measurements')
        black_patch = mpatches.Patch(color='black', label='real state')
        # if pred_traj is not None:
        #     blue_patch = mpatches.Patch(color='blue', label='preds')
        plt.legend(handles=[red_patch, green_patch, black_patch])

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + '/imgs/' + 'torchfilter_kf'+dt_string+'.png')
        if show_fig:
            plt.show()

def plot_state_cov_mat(P_traj, est_traj, show_fig=False):
    """
    Plots state covariance matrix P over time.
    Args:
        P_traj (np.array): of shape [num_tsteps, num_envs, num_states, num_states]
    """
    num_tsteps = len(P_traj[:, 0, 0, 0])
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    colors = np.linspace(0, 1, num_tsteps)

    # Initializing the random seed
    random_seed = 1000

    # Setting mean of the distribution to
    # be at (0,0)
    state_dim = est_traj.shape[2]
    mean = np.zeros(state_dim)

    for env_id in range(P_traj.shape[1]):
        for tidx in range(0, num_tsteps, 20):
            # Generating a Gaussian bivariate distribution
            # with given mean and covariance matrix
            distr = multivariate_normal(cov=P_traj[tidx, env_id, :, :], mean=mean,
                                        seed=random_seed)

            # Generating samples out of the distribution
            data = distr.rvs(size=500)

            # Plotting the generated samples
            xhat = est_traj[tidx, env_id, :]
            e_dx = np.cos(xhat[-1])
            e_dy = np.sin(xhat[-1])
            plt.scatter(xhat[0] + data[:, 0], xhat[1] + data[:, 1], color=[1 - colors[tidx], 0, colors[tidx]],
                        alpha=0.1)
            plt.quiver(xhat[0] + data[:, 0], xhat[1] + data[:, 1], e_dx, e_dy,
                       color=[1 - colors[tidx], 0, colors[tidx]], alpha=0.1)

            # plt.scatter(est_traj[tidx, env_id, 0], est_traj[tidx, env_id, 1], c='k', s=70, alpha=0.5)
            # plt.quiver(est_traj[tidx, env_id, 0], est_traj[tidx, env_id, 1], e_dx, e_dy, color='k', alpha=0.5)

        e_dx = np.cos(est_traj[:, env_id, -1])
        e_dy = np.sin(est_traj[:, env_id, -1])
        plt.scatter(est_traj[:, env_id, 0], est_traj[:, env_id, 1], c='k', s=70, alpha=0.5)
        plt.quiver(est_traj[:, env_id, 0], est_traj[:, env_id, 1], e_dx, e_dy, color='k', alpha=0.5)
        plt.scatter(est_traj[0, env_id, 0], est_traj[0, env_id, 1], c='w', facecolor='w', edgecolors='k', s=70)

    # plot the origin (i.e., the goal of the controller)
    plt.scatter(0, 0, c='m', edgecolors='k', marker=",", s=100)

    plt.title('State Estimate Covariance (P)')
    path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(path + '/imgs/' + 'torchfilter_kf_P_mat.png')
    if show_fig:
        plt.show()