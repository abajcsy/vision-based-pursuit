import numpy as np
import matplotlib.pyplot as plt
import pickle
import os 
import torch

from datetime import datetime
import matplotlib


def plot_theoretic_capture_time(rel_states, theoretic_capture_time, show_fig=False, filename="", folder=""):
    """
    Plots the *theoretic* pursuit-evasion capture time of each initial condition as a heatmap.

    Args:
        rel_states (np.array): array of the relative states of each agent over time, shape (num_envs, num_timesteps, num_states)
        theoretic_capture_time (np.array): shortest number of timesteps to capture for each environment
        show_fig (bool): if True, shows figure; if False, saves figure
        filename (string): name of image file
    """
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    num_envs = rel_states.shape[0]
    colors = np.linspace(0.1, 1, rel_states.shape[1])

    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
              'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'viridis',
              'plasma', 'inferno', 'magma', 'cividis']

    plt.rcParams['image.cmap'] = 'RdYlGn_r'
    cmap_rdylgr = matplotlib.cm.get_cmap('RdYlGn_r')
    normalized_len = theoretic_capture_time / 100.
    crange = np.linspace(0,1,len(normalized_len))
    alpha = 1.5
    plt.quiver(rel_states[:, 0, 0], rel_states[:, 0, 1], 
                    alpha * np.cos(rel_states[:, 0, -1]), 
                    alpha * np.sin(rel_states[:, 0, -1]),
                    color=cmap_rdylgr(normalized_len),
                    scale=6, 
                    scale_units='inches',
                    width=0.007
                    )
    plt.scatter(rel_states[:, 0, 0], 
                rel_states[:, 0, 1], 
                70,  
                # c=crange,
                color=cmap_rdylgr(normalized_len), 
                edgecolors='k', 
                linewidths=1)

    print("normalized len: ", normalized_len)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    cbar = plt.colorbar()

    plt.xlabel("rel x-pos")
    plt.ylabel("rel y-pos")
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])

    if show_fig:
        plt.show()
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + "/imgs/" + folder + "/theoretic_heatmap_" + filename + ".png")

def plot_capture_time_heatmap(rel_states, ep_lens, show_fig=False, filename="", folder=""):
    """
    Plots the capture time of each initial condition as a heatmap.

    Args:
        rel_states (np.array): array of the relative states of each agent over time, shape (num_envs, num_timesteps, num_states)
        ep_lens (np.array): episode length of each environment
        show_fig (bool): if True, shows figure; if False, saves figure
        filename (string): name of image file
    """
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    num_envs = rel_states.shape[0]
    colors = np.linspace(0.1, 1, rel_states.shape[1])


    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
              'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'viridis',
              'plasma', 'inferno', 'magma', 'cividis']

    plt.rcParams['image.cmap'] = 'RdYlGn_r'
    cmap_rdylgr = matplotlib.cm.get_cmap('RdYlGn_r')
    normalized_len = ep_lens / 100.
    crange = np.linspace(0,1,len(normalized_len))
    alpha = 1.5
    plt.quiver(rel_states[:, 0, 0], rel_states[:, 0, 1], 
                    alpha * np.cos(rel_states[:, 0, -1]), 
                    alpha * np.sin(rel_states[:, 0, -1]),
                    color=cmap_rdylgr(normalized_len),
                    scale=6, 
                    scale_units='inches',
                    width=0.007
                    )
    plt.scatter(rel_states[:, 0, 0], 
                rel_states[:, 0, 1], 
                70,  
                # c=crange,
                color=cmap_rdylgr(normalized_len), 
                edgecolors='k', 
                linewidths=1)

    print("normalized len: ", normalized_len)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    cbar = plt.colorbar()

    plt.xlabel("rel x-pos")
    plt.ylabel("rel y-pos")
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])

    if show_fig:
        plt.show()
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + "/imgs/" + folder + "/heatmap_" + filename + ".png")

def plot_capture_time_distribution(ep_lens, show_fig=False, filename=""):
    """
    Plots a histogram of the capture time. 

    Args:
        ep_lens (np.array): episode length of each environment
        show_fig (bool): if True, shows figure; if False, saves figure
        filename (string): name of image file
    """
    norm_ep_len = ep_lens/100.
    counts, bins = np.histogram(norm_ep_len)
    plt.stairs(counts, bins, fill=True)
    plt.xlabel("Time-to-capture (TTC)")
    plt.ylabel("TTC Frequency")
    plt.xlim([0, 1])
    plt.ylim([0, max(1500, np.amax(counts))])

    if show_fig:
        plt.show()
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + "/imgs/distribution_" + filename + ".png")


def plot_joint_traj(robot_traj, agent_traj, show_fig=False, filename=""):
    """
    Plots a SINGLE joint trajectory.

    Args:
        robot_traj (np.array):
        agent_traj (np.array):
        show_fig (bool): if True, shows figure; if False, saves figure
        filename (string): name of image file
    """
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot(1, 1, 1)

    colors = np.linspace(0.1, 1, robot_traj.shape[0])

    robot_cmap = 'Oranges'
    agent_cmap = 'Blues'

    cmap_r = matplotlib.cm.get_cmap(robot_cmap)
    cmap_a = matplotlib.cm.get_cmap(agent_cmap)

    num_nonzero_robot = len(robot_traj[robot_traj != 0])
    num_nonzero_agent = len(agent_traj[agent_traj != 0])
    nonzero_robot_traj = np.reshape(robot_traj[robot_traj != 0], (-1,3))
    nonzero_agent_traj = np.reshape(agent_traj[agent_traj != 0], (-1,3))
    colors_r = np.linspace(0.1, 1, nonzero_robot_traj.shape[0])
    colors_a = np.linspace(0.1, 1, nonzero_agent_traj.shape[0])

    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
              'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'viridis',
              'plasma', 'inferno', 'magma', 'cividis']

    print("nonzero_robot_traj: ", nonzero_robot_traj)
    print("nonzero_agent_traj: ", nonzero_agent_traj)

    for tidx in range(nonzero_robot_traj.shape[0]):
        plt.plot(nonzero_robot_traj[tidx:tidx+2, 0],
                 nonzero_robot_traj[tidx:tidx+2, 1], color=cmap_r(colors_r[tidx:tidx+1]), linewidth=4, markeredgecolor=cmap_r(colors_r[-1]))
        plt.plot(nonzero_agent_traj[tidx:tidx+2, 0],
                 nonzero_agent_traj[tidx:tidx+2, 1], color=cmap_a(colors_a[tidx:tidx+1]), linewidth=4, markeredgecolor=cmap_a(colors_a[-1]))

    # for tidx in range(nonzero_robot_traj.shape[0]):
    #     plt.plot(nonzero_robot_traj[tidx:tidx+2, 0],
    #              nonzero_robot_traj[tidx:tidx+2, 1], marker='o', color=cmap_r(colors_r), linewidth=6, markeredgecolor=cmap_r(colors_r))
    #     plt.plot(nonzero_agent_traj[tidx:tidx+2, 0],
    #              nonzero_agent_traj[tidx:tidx+2, 1], marker='o', color=cmap_a(colors_a), linewidth=6, markeredgecolor=cmap_a(colors_a))

    plt.scatter(nonzero_robot_traj[-6, 0], nonzero_robot_traj[-6, 1], c=colors_r[6], cmap=robot_cmap, alpha=1.0,
                edgecolors=cmap_r(colors_r[-1]))
    plt.scatter(nonzero_agent_traj[-6, 0], nonzero_agent_traj[-6, 1], c=colors_a[6], cmap=agent_cmap, alpha=1.0,
                edgecolors=cmap_a(colors_a[-1]))

    plt.quiver(nonzero_agent_traj[0, 0], nonzero_agent_traj[0, 1], 1, 0,
                    color='k',
                    scale=3,
                    scale_units='inches',
                    width=0.007)

    plt.quiver(nonzero_robot_traj[0, 0], nonzero_robot_traj[0, 1], 1, 0,
                    color='k',
                    scale=3,
                    scale_units='inches',
                    width=0.007)

    # plt.scatter(nonzero_robot_traj[:, 0], nonzero_robot_traj[:, 1], c=colors_r, cmap=robot_cmap, alpha=1.0, edgecolors=cmap_r(colors_r[-1]))
    # plt.scatter(nonzero_agent_traj[:, 0], nonzero_agent_traj[:, 1], c=colors_a, cmap=agent_cmap, alpha=1.0, edgecolors=cmap_a(colors_a[-1]))

    # plt.text(nonzero_robot_traj[-1, 0]+1, nonzero_robot_traj[-1, 1]-1, 't='+str(nonzero_robot_traj.shape[0]*0.2))

    plt.xlim([-2, 13])
    plt.ylim([-25, 2])

    # plt.xlim([-2, 6])
    # plt.ylim([-10, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)

    if show_fig:
        plt.show()
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + "/imgs/joint_traj_" + filename + ".png")

def plot_traj_data(data, robot_data=None, show_fig=False, filename=""):
    """
    Plots raw trajectories.

    Args:
        data (np.array): array containing trajectory data, shape (num_envs, num_timesteps, num_states)
        robot_data (np.array): array containing robot state data, shape (num_envs, num_timesteps, num_robot_states)
        show_fig (bool): if True, shows figure; if False, saves figure
        filename (string): name of image file
    """
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    num_envs = data.shape[0]

    colors = np.linspace(0.1, 1, data.shape[1])

    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
              'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'viridis',
              'plasma', 'inferno', 'magma', 'cividis']

    for eidx in range(num_envs):
        # plot measurements
        if num_envs > len(cmaps):
            plt.scatter(data[eidx, :, 0], data[eidx, :, 1], c=colors, cmap=cmaps[0])
        else:
            plt.scatter(data[eidx, :, 0], data[eidx, :, 1], c=colors, cmap=cmaps[eidx])

    for eidx in range(num_envs):
        # plot the starting point
        if num_envs > len(cmaps):
            cmap = matplotlib.cm.get_cmap(cmaps[0])
        else:
            cmap = matplotlib.cm.get_cmap(cmaps[eidx])
        plt.scatter(data[eidx, -1, 0], data[eidx, -1, 1], color=cmap(0.8), s=100, marker='s', edgecolors='k')
        plt.scatter(data[eidx, 0, 0], data[eidx, 0, 1], color=cmap(0.1), s=100, marker='s', edgecolors='k')

        if robot_data is not None:
            # plot the initial condition of the robot
            plt.scatter(robot_data[eidx, -1, 0], robot_data[eidx, -1, 1], color=cmap(0.8), marker='>', edgecolors='k', s=100)

    # plot the origin
    # plt.scatter(0, 0, c='k', edgecolors='k', s=100)
    # plt.plot([0, 1], [0, 0], c='r')
    # plt.plot([0, 0], [0, 1], c='g')

    plt.xlabel("x-pos")
    plt.ylabel("y-pos")
    plt.xlim([-50, 50])
    plt.ylim([-50, 50])
    
    if show_fig:
        plt.show()
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(path + "/imgs/trajs_" + filename + ".png")
