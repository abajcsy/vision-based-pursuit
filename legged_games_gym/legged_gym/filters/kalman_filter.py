import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.stats import multivariate_normal
import pdb

from datetime import datetime
import os

class KalmanFilter(object):
    def __init__(self,
                 dt,
                 num_states,
                 num_actions,
                 num_envs,
                 state_type="pos_ang",
                 device='cpu',
                 dtype=torch.float64):
        self.dt = dt
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        # if state includes the relative angle, e.g.:
        #   state = (x_rel, y_rel, z_rel, dtheta) where dtheta ~= (atan(y_rel, x_rel) - theta_robot)
        #   control = (v_x, v_y, omega)
        # then need to make sure to wrap to pi
        #
        # options:
        #   pos, pos_ang
        self.state_type = state_type

        # dynamics matricies x' = Ax + Bu + w
        self.A = torch.eye(self.num_states, dtype=self.dtype, device=self.device)
        self.B = -self.dt * torch.eye(self.num_states, self.num_actions, dtype=self.dtype, device=self.device)

        self.H = torch.eye(self.num_states, dtype=self.dtype, device=self.device)          # observation matrix y = Hx + v
        self.Q = 0.01 * torch.eye(self.num_states, dtype=self.dtype, device=self.device)    # process covariance (w ~ N(0,Q))
        self.R = 0.2 * torch.eye(self.num_states, dtype=self.dtype, device=self.device)    # measurement covariance (v ~ N(0,R))
        self.P = torch.eye(self.num_states, dtype=self.dtype, device=self.device)          # a posteriori estimate covariance matrix
        self.I = torch.eye(self.num_states, device=self.device)

        if self.state_type == "pos_ang":
            if self.num_states == 4:
                self.B[2, -1] = 0 # HACK to remap B matrix so 3rd dim (z) is not controlled
            self.B[-1, -1] = self.dt
            self.P[-1, -1] *= 0.1

            # adjust the measurement covariance for angular component
            self.R[-1, -1] = 0.1

        # current state estimate
        self.xhat = torch.zeros(self.num_envs, self.num_states, dtype=self.dtype, device=self.device)

        # torchify everything
        self.A_tensor = self.A.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.B_tensor = self.B.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_actions)
        self.H_tensor = self.H.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.Q_tensor = self.Q.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.R_tensor = self.R.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.P_tensor = self.P.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)
        self.I_tensor = self.I.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_states, self.num_states)

        # create N(0, R) to sample measurements from
        zero_mean = torch.zeros(self.num_envs, self.num_states, dtype=self.dtype, device=self.device)
        self.normal_dist = MultivariateNormal(loc=zero_mean, covariance_matrix=self.R)

    def set_init_xhat(self, xhat):
        """Initializes estimated state.
        Args: 
            xhat (torch.Tensor): initial estimated state of shape [num_envs, num_states]
        """
        self.xhat = xhat

    def reset_xhat(self, env_ids, xhat_val=None):
        """Resets the initial estimate.
        Args:
            env_ids [list of Ints]: environment indicies to be reset
            reset_val [torch.Tensor]: of length [num_env_ids x num_states] containing values to reset to
        """
        # reset initial state estimate
        if xhat_val is None:
            self.xhat[env_ids, :] = 0.
        else:
            self.xhat[env_ids, :] = xhat_val
        # reset state covariance
        self.P_tensor[env_ids, :] = self.P

    def dynamics(self, x, command_robot):
        """
        Applies linear dynamics to evolve state: x' = Ax + Bu

        Args: 
            x (torch.Tensor): current state of shape [num_envs, num_states]
            command_robot (torch.Tensor): current action of shape [num_envs, num_actions]
        Returns:
            x' (torch.Tensor): next state of shape [num_envs, num_states]
        """
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        if command_robot.dtype != self.dtype:
            command_robot = command_robot.to(self.dtype)

        Ax = torch.bmm(self.A_tensor, x.unsqueeze(-1)).squeeze(-1)
        Bu = torch.bmm(self.B_tensor, command_robot.unsqueeze(-1)).squeeze(-1)
        xnext = Ax + Bu

        if self.state_type == "pos_ang":
            # need to wrap the final state to [-pi,pi]
            xnext[:, -1] = self._wrap_to_pi(xnext[:, -1])

        return xnext

    def predict(self,
                command_robot,
                inplace=True,
                xhat=None,
                P_tensor=None):
        """
        Prediction, i.e., time update.

        Args: 
            command_robot (torch.Tensor): current action of shape [num_envs, num_actions]
            inplace (bool): if True, then modifies internal variable xhat, otherwise returns new value
            xhat (torch.Tensor): (optional) current estimate [num_envs, num_states]
            P_tensor (torch.Tensor): (optional) current covariance [num_envs, num_states, num_states]
        Returns:
            xhat (torch.Tensor): predicted (a priori) state estimate of shape [num_envs, num_states]
        """

        if command_robot.dtype != self.dtype:
            command_robot = command_robot.to(self.dtype)

        if inplace:
            # Predict the next state by applying dynamics
            #   xhat' = Axhat + Bu
            self.xhat = self.dynamics(self.xhat, command_robot)

            # Predict the error covariance
            #   Phat = A * P * A' + Q
            AP = torch.bmm(self.A_tensor, self.P_tensor)
            A_top = torch.transpose(self.A_tensor, 1, 2)
            self.P_tensor = torch.bmm(AP, A_top) + self.Q_tensor
            return self.xhat, self.P_tensor
        else:
            # Predict the next state by applying dynamics
            #   xhat' = Axhat + Bu
            xhat_new = self.dynamics(xhat, command_robot)

            # Predict the error covariance
            #   Phat = A * P * A' + Q
            AP = torch.bmm(self.A_tensor, P_tensor)
            A_top = torch.transpose(self.A_tensor, 1, 2)
            P_tensor_new = torch.bmm(AP, A_top) + self.Q_tensor
            return xhat_new, P_tensor_new

    def correct(self, z, env_ids):
        """
        Measurement update.

        Args:
            z (torch.Tensor): measurement of shape [num_envs, num_obs]
            env_ids (torch.Tensor): of length [num_env_ids] environments to be corrected
        """

        if len(env_ids) == 0:
            return

        if z.dtype != self.dtype:
            z = z.to(self.dtype)

        # Compute Kalman gain:
        #   S = H * Phat H^T + R
        #   K = Phat * H^T S^-1
        H_top = torch.transpose(self.H_tensor, 1, 2)
        S = torch.bmm(torch.bmm(self.H_tensor, self.P_tensor), H_top) + self.R_tensor
        K = torch.bmm(self.P_tensor, torch.bmm(H_top, torch.pinverse(S)))

        # Measurement residual
        #   y = z - H * xhat
        y = (z - torch.bmm(self.H_tensor, self.xhat.unsqueeze(-1)).squeeze(-1))

        if self.state_type == "pos_ang":
            # need to wrap the angular component of state estimate to [-pi,pi]
            y[env_ids, -1] = self._wrap_to_pi(y[env_ids, -1])
        
        # Get a posteriori estimate using measurement z
        #   xhat = xhat + K * y
        self.xhat[env_ids, :] = self.xhat[env_ids, :] + torch.bmm(K[env_ids, :], y[env_ids, :].unsqueeze(-1)).squeeze(-1)

        if self.state_type == "pos_ang":
            # need to wrap the angular component of state estimate to [-pi,pi]
            self.xhat[env_ids, -1] = self._wrap_to_pi(self.xhat[env_ids, -1])

        # Get a posteriori estimate covariance
        #   P = (I - K * H) * Phat
        self.P_tensor[env_ids, :] = torch.bmm(self.I_tensor[env_ids, :] -
                                              torch.bmm(K[env_ids, :], self.H_tensor[env_ids, :]), self.P_tensor[env_ids, :])

    def sim_measurement(self, xreal):
        """
        Simulate a measurement given the ground-truth state.

        Args:
            xreal (torch.Tensor): real state of shape [num_envs, num_states]

        Returns:
            z (torch.Tensor): measurement of shape [num_env_ids, num_states]
        """
        # measurements are drawn from v ~ N(0, R)
        # zero_mean = np.zeros(self.num_states)
        # v = np.random.multivariate_normal(zero_mean, self.R)
        # v = v.reshape(self.num_states, 1)

        if xreal.dtype != self.dtype:
            xreal = xreal.to(self.dtype)

        zero_mean = np.zeros((self.num_envs, self.num_states))
        v_tensor = self.sample_batch_mvn(zero_mean, self.R_tensor.cpu().numpy(), self.num_envs)

        if self.state_type == "pos_ang":
            # if we have an angular component to state, sample it separately
            v_tensor[:, -1] = self._wrap_to_pi(v_tensor[:, -1])

        # use y = Hx + v to simulate measurement
        Hx = torch.bmm(self.H_tensor, xreal.unsqueeze(-1)).squeeze(-1)
        z = Hx + v_tensor

        if self.state_type == "pos_ang":
            z[:, -1] = self._wrap_to_pi(z[:, -1])

        return z

    def sample_batch_mvn(self, mean, cov, batch_size) -> np.ndarray:
        """
        Batch sample multivariate normal distribution.

        Arguments:

            mean (np.ndarray): expected values of shape (B, D)
            cov (np.ndarray): covariance matrices of shape (B, D, D)
            batch_size (int): additional batch shape (B)

        Returns: torch.Tensor or shape: (B, D)
                 with one samples from the multivariate normal distributions
        """
        L = np.linalg.cholesky(cov)
        X = np.random.standard_normal((batch_size, mean.shape[-1], 1))
        Y = (L @ X).reshape(batch_size, mean.shape[-1]) + mean
        Y_tensor = torch.tensor(Y, dtype=self.dtype, device=self.device, requires_grad=False)
        return Y_tensor

    def _wrap_to_pi(self, angles):
        angles %= 2 * np.pi
        angles -= 2 * np.pi * (angles > np.pi)
        return angles

    def _plot_state_traj(self, real_traj, z_traj, pred_traj, est_traj):
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

        colors_z = np.linspace(0, 1, len(z_traj[:,0,0]))
        colors_r = np.linspace(0, 1, len(real_traj[:,0,0]))
        colors_e = np.linspace(0, 1, len(est_traj[:, 0, 0]))
        if pred_traj is not None:
            colors_p = np.linspace(0, 1, len(pred_traj[:, 0, 0]))

        gr_cmap = mpl.colormaps['Greens']
        rd_cmap = mpl.colormaps['Reds']
        gy_cmap = mpl.colormaps['Greys']
        if self.state_type == "pos":
            for eidx in range(self.num_envs):
                # plot measurements
                plt.scatter(z_traj[:,eidx,0], z_traj[:,eidx,1], c=colors_z, cmap='Greens', label="measurements")
                if pred_traj is not None:
                    # plot predicted state trajectory (a priori)
                    plt.scatter(pred_traj[:,eidx,0], pred_traj[:,eidx,1], c=colors_p, cmap='Blues', label="preds")
                # plot estimated state trajectory (a posteriori)
                plt.scatter(est_traj[:,eidx,0], est_traj[:,eidx,1], c=colors_e, cmap='Reds', label="estimates")
                # plot GT state trajectory
                plt.scatter(real_traj[:,eidx,0], real_traj[:,eidx,1], c=colors_r, cmap='Greys', label="real state")
                # plot initial state explicitly
                plt.scatter(real_traj[0,eidx,0], real_traj[0,eidx,1], c=colors_r[0], edgecolors='k', cmap='Greys')
        elif self.state_type == "pos_ang":
            for eidx in range(self.num_envs):
                # plot measurements
                z_dx = np.cos(z_traj[:, eidx, -1])
                z_dy = np.sin(z_traj[:, eidx, -1])
                e_dx = np.cos(est_traj[:, eidx, -1])
                e_dy = np.sin(est_traj[:, eidx, -1])
                r_dx = np.cos(real_traj[:, eidx, -1])
                r_dy = np.sin(real_traj[:, eidx, -1])

                # plot measurements
                plt.scatter(z_traj[:, eidx, 0], z_traj[:, eidx, 1], c=colors_z, cmap='Greens', label="measurements")
                plt.quiver(z_traj[:, eidx, 0], z_traj[:, eidx, 1], z_dx, z_dy, color=gr_cmap(colors_z), label="measurements")
                # plot estimated state trajectory (a posteriori)
                plt.scatter(est_traj[:, eidx, 0], est_traj[:, eidx, 1], c=colors_e, cmap='Reds', label="estimates")
                plt.quiver(est_traj[:, eidx, 0], est_traj[:, eidx, 1], e_dx, e_dy, color=rd_cmap(colors_e),label="estimates")
                # plot GT state trajectory
                # import pdb; pdb.set_trace()
                plt.scatter(real_traj[:, eidx, 0], real_traj[:, eidx, 1], c=colors_r, cmap='Greys', label="real state")
                plt.quiver(real_traj[:, eidx, 0], real_traj[:, eidx, 1], r_dx, r_dy, color=gy_cmap(colors_r), label="real state")
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
        # plt.savefig(path + '/imgs/' + 'kalman_filter'+dt_string+'.png')
        plt.show()

    def _plot_state_cov_mat(self, P_traj, est_traj):
        """
        Plots state covariance matrix P over time. 
        Args:
            P_traj (np.array): of shape [num_tsteps, num_envs, num_states, num_states]
        """
        num_tsteps = len(P_traj[:,0,0,0])
        fig = plt.figure()
        ax = plt.subplot(1,1,1)
        colors = np.linspace(0,1,num_tsteps)

        # Initializing the random seed
        random_seed = 1000

        # Setting mean of the distribution to
        # be at (0,0)
        mean = np.zeros(self.num_states)

        for env_id in range(P_traj.shape[1]):
            for tidx in range(0, num_tsteps, 20):
                # Generating a Gaussian bivariate distribution
                # with given mean and covariance matrix
                distr = multivariate_normal(cov=P_traj[tidx,env_id,:,:], mean=mean,
                                            seed=random_seed)

                # Generating samples out of the distribution
                data = distr.rvs(size=500)

                # Plotting the generated samples
                xhat = est_traj[tidx, env_id, :]
                e_dx = np.cos(xhat[-1])
                e_dy = np.sin(xhat[-1])
                plt.scatter(xhat[0] + data[:, 0], xhat[1] + data[:, 1], color=[1-colors[tidx], 0, colors[tidx]], alpha=0.1)
                plt.quiver(xhat[0] + data[:, 0], xhat[1] + data[:, 1], e_dx, e_dy, color=[1-colors[tidx], 0, colors[tidx]], alpha=0.1)

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
        plt.show()
        path = os.path.dirname(os.path.abspath(__file__))
        # plt.savefig(path + '/imgs/' + 'kalman_filter_P_mat.png')


    def _compute_mse(self, xrel_traj, pred_traj, est_traj):
        """Compute MSE per each state dimension."""
        for eidx in range(self.num_envs):
            num_tsteps = len(xrel_traj[:,0,0])
            diff_sq_est = (xrel_traj[:, eidx, :] - est_traj[:, eidx, :])**2
            mse_est = (1 / num_tsteps) * np.sum(diff_sq_est, axis=0)
            diff_sq_pred = (xrel_traj[:, eidx, :] - pred_traj[:, eidx, :])**2
            mse_pred = (1 / num_tsteps) * np.sum(diff_sq_pred, axis=0)
            print("ENV ", eidx, " | MSE[real - est] (x):",  mse_est[0], "(y): ", mse_est[1], "(z): ", mse_est[2])
            print("ENV ", eidx, " | MSE[real - pred] (x):",  mse_pred[0], "(y): ", mse_pred[1], "(z): ", mse_pred[2])

    def _turn_and_pursue_command_robot(self, rel_state):
        command_robot = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        # rel_yaw = torch.atan2(rel_state[:, 1], rel_state[:, 0]) + rel_state[:, -1] # TODO: this is a hack!
        # rel_yaw = self._wrap_to_pi(rel_yaw)
        # import pdb; pdb.set_trace()
        print("rel state: ", rel_state[:, :-1])
        print("dtheta: ", rel_state[:, -1])
        eps = 0.15
        for env_idx in range(self.num_envs):
            if torch.abs(rel_state[env_idx, -1]) < eps:
                print("GOING STRAIGHT..")
                command_robot[env_idx, :2] = torch.clip(rel_state[env_idx, :2], min=-1, max=1)
            else:
                print("TURNING..")
                command_robot[env_idx, 2] = 1 #torch.clip(rel_state[env_idx, -1],  min=-1, max=1) # just turn
        return command_robot

def sim_robot_and_kf():
    dt = 0.02
    num_states = 3
    num_actions = 3
    num_envs = 1  # 2
    min_lin_vel = -1
    max_lin_vel = 1
    min_ang_vel = -1
    max_ang_vel = 1
    state_type = "pos_ang"
    device = 'cpu'
    dtype = torch.float
    robot_full_fov = 1.20428

    # define the real system
    max_s = 7
    t = np.arange(0, max_s, dt)
    print("Simulating for ", max_s, " seconds...")

    xa0 = torch.zeros(num_envs, num_states, dtype=dtype, device=device)
    xa0[0, 0] = 0
    xa0[0, 1] = 0
    xa0[0, 2] = 0
    # xa0[1, 0] = 1
    # xa0[1, 1] = 1
    # xa0[1, 2] = 0
    xr0 = torch.zeros(num_envs, num_states, dtype=dtype, device=device)
    xr0[0, 0] = 3
    xr0[0, 1] = 2
    xr0[0, 2] = np.pi / 2
    # xr0[1, 0] = -3
    # xr0[1, 1] = -2
    # xr0[1, 2] = np.pi/4

    # define the kalman filter and set init state
    kf = KalmanFilter(dt, num_states, num_actions, num_envs, state_type, device, dtype)

    # setup the dtheta state
    xrel0 = xa0 - xr0
    xrel0[:, -1] = torch.atan2(xrel0[:, 1], xrel0[:, 0]) - xr0[:, -1]  # fourth state is dtheta
    xrel0[:, -1] = kf._wrap_to_pi(xrel0[:, -1])

    xhat0 = kf.sim_measurement(xrel0)
    all_env_ids = torch.arange(kf.num_envs, device=device)
    # xhat0[:, 0] = -2
    # xhat0[:, 1] = -2
    # xhat0[:, 2] = 0
    kf.reset_xhat(all_env_ids, xhat0)

    # create the fake ground-truth data
    xrel = xrel0
    real_traj = np.array([xrel.numpy()])
    for tidx in range(len(t)):
        action = kf._turn_and_pursue_command_robot(xrel)
        # action = torch.clip(xrel[:, :2], min=min_lin_vel, max=max_lin_vel)
        xrel = kf.dynamics(xrel, action)
        real_traj = np.append(real_traj, [xrel.numpy()], axis=0)

    # do the prediction and simulation loop
    pred_traj = np.array([xhat0.numpy()])
    est_traj = np.array([xhat0.numpy()])
    z_traj = np.zeros(0)
    P_traj = np.array([kf.P_tensor.numpy()])
    for tidx in range(len(t)):
        print("predicting...")
        action = kf._turn_and_pursue_command_robot(kf.xhat)
        # action = torch.clip(kf.xhat[:, :2], min=min_lin_vel, max=max_lin_vel)
        xhat = kf.predict(action)
        pred_traj = np.append(pred_traj, [xhat.numpy()], axis=0)

        # get a measurement
        # if tidx % 20 == 0:
        #     print("got measurement, doing corrective update...")
        #     x = real_traj[tidx, :]
        #     z = kf.sim_measurement(torch.tensor(x, dtype=torch.float64))
        #     if tidx == 0:
        #         z_traj = np.array([z.numpy()])
        #     else:
        #         z_traj = np.append(z_traj, [z.numpy()], axis=0)
        #     kf.correct(z, all_env_ids)

        # find environments where robot is visible
        x = real_traj[tidx, :]
        dtheta = x[:, -1]
        half_fov = robot_full_fov / 2.
        leq = torch.le(torch.abs(torch.tensor(dtheta.reshape(num_envs, 1))), half_fov)
        fov_bool = torch.any(leq, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()

        # simulate observations
        z = kf.sim_measurement(torch.tensor(x, dtype=torch.float64))

        if tidx == 0:
            z_traj = np.array([z.numpy()])
        else:
            z_traj = np.append(z_traj, [z.numpy()], axis=0)

        kf.correct(z, visible_env_ids)

        est_traj = np.append(est_traj, [kf.xhat.numpy()], axis=0)
        P_traj = np.append(P_traj, [kf.P_tensor.numpy()], axis=0)

    print("Plotting state traj...")
    kf._plot_state_traj(real_traj, z_traj, pred_traj, est_traj)
    print("Plotting covariance matrix traj...")
    kf._plot_state_cov_mat(P_traj, est_traj)
    print("Computing MSE...")
    kf._compute_mse(real_traj, pred_traj, est_traj)


if __name__ == '__main__':
    sim_robot_and_kf()