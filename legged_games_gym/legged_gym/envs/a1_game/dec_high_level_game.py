import pdb

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict, get_load_path
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils import task_registry
from legged_gym.utils.joypad import Joypad
from legged_gym.filters.kalman_filter import KalmanFilter
from threading import Thread
import sys

import pickle
from datetime import datetime

from rsl_rl.runners.low_level_policy_runner import LLPolicyRunner

import legged_gym.utils.math


class DecHighLevelGame():
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        print("[DecHighLevelGame] initializing ...")
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1
            

        # setup low-level policy loader
        # ll_env_cfg, ll_train_cfg = task_registry.get_cfgs(name="low_level_game")
        ll_env_cfg, ll_train_cfg = task_registry.get_cfgs(name="a1")

        # need to make sure that the low-level and high level have the same representation
        ll_env_cfg.env.num_envs = self.cfg.env.num_envs
        ll_env_cfg.noise.add_noise = self.cfg.noise.add_noise
        ll_env_cfg.domain_rand.randomize_friction = self.cfg.domain_rand.randomize_friction
        ll_env_cfg.domain_rand.push_robots = self.cfg.domain_rand.push_robots

        # let the ll env know about the robot's FOV for visualization / debug_viz purposes
        # TODO: hack to put it in the terrain class!
        ll_env_cfg.terrain.fov = self.cfg.robot_sensing.fov

        # setup terrain properties based on the high-level cfg file
        ll_env_cfg.terrain.mesh_type = self.cfg.terrain.mesh_type
        ll_env_cfg.terrain.num_rows = self.cfg.terrain.num_rows
        ll_env_cfg.terrain.num_cols = self.cfg.terrain.num_cols
        ll_env_cfg.terrain.curriculum = self.cfg.terrain.curriculum
        ll_env_cfg.terrain.horizontal_scale = self.cfg.terrain.horizontal_scale
        ll_env_cfg.terrain.vertical_scale = self.cfg.terrain.vertical_scale
        ll_env_cfg.terrain.static_friction = self.cfg.terrain.static_friction
        ll_env_cfg.terrain.dynamic_friction = self.cfg.terrain.dynamic_friction
        ll_env_cfg.terrain.restitution = self.cfg.terrain.restitution
        ll_env_cfg.terrain.measure_heights = self.cfg.terrain.measure_heights
        ll_env_cfg.terrain.fov_measure_heights = self.cfg.terrain.fov_measure_heights
        ll_env_cfg.terrain.measured_points_x = self.cfg.terrain.measured_points_x
        ll_env_cfg.terrain.measured_points_y = self.cfg.terrain.measured_points_y
        ll_env_cfg.terrain.terrain_length = self.cfg.terrain.terrain_length
        ll_env_cfg.terrain.terrain_width = self.cfg.terrain.terrain_width
        ll_env_cfg.terrain.terrain_proportions = self.cfg.terrain.terrain_proportions
        ll_env_cfg.terrain.slope_treshold = self.cfg.terrain.slope_treshold
        ll_env_cfg.terrain.num_obstacles = self.cfg.terrain.num_obstacles
        ll_env_cfg.terrain.obstacle_height = self.cfg.terrain.obstacle_height

        # TODO: HACK! Align init states between the two configs
        ll_env_cfg.init_state.pos = self.cfg.init_state.robot_pos
        ll_env_cfg.init_state.rot = self.cfg.init_state.robot_rot
        ll_env_cfg.init_state.lin_vel = self.cfg.init_state.robot_lin_vel
        ll_env_cfg.init_state.ang_vel = self.cfg.init_state.robot_ang_vel
        ll_env_cfg.init_state.agent_pos = self.cfg.init_state.agent_pos
        ll_env_cfg.init_state.agent_rot = self.cfg.init_state.agent_rot
        ll_env_cfg.init_state.agent_lin_vel = self.cfg.init_state.agent_lin_vel
        ll_env_cfg.init_state.agent_ang_vel = self.cfg.init_state.agent_ang_vel
        ll_env_cfg.commands.heading_command = self.cfg.commands.heading_command
        ll_env_cfg.commands.ranges.lin_vel_x = self.cfg.commands.ranges.lin_vel_x
        ll_env_cfg.commands.ranges.lin_vel_y = self.cfg.commands.ranges.lin_vel_y
        ll_env_cfg.commands.ranges.ang_vel_yaw = self.cfg.commands.ranges.ang_vel_yaw
        ll_env_cfg.commands.ranges.heading = self.cfg.commands.ranges.heading
        ll_env_cfg.domain_rand.randomize_friction = self.cfg.domain_rand.randomize_friction
        ll_env_cfg.domain_rand.friction_range = self.cfg.domain_rand.friction_range
        ll_env_cfg.domain_rand.randomize_base_mass = self.cfg.domain_rand.randomize_base_mass
        ll_env_cfg.domain_rand.added_mass_range = self.cfg.domain_rand.added_mass_range
        ll_env_cfg.domain_rand.push_robots = self.cfg.domain_rand.push_robots
        ll_env_cfg.domain_rand.push_interval_s = self.cfg.domain_rand.push_interval_s
        ll_env_cfg.domain_rand.max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
        ll_env_cfg.noise.add_noise = self.cfg.noise.add_noise
        ll_env_cfg.noise.noise_level = self.cfg.noise.noise_level

        # set viewer info.
        ll_env_cfg.viewer.ref_env = self.cfg.viewer.ref_env
        ll_env_cfg.viewer.pos = self.cfg.viewer.pos
        ll_env_cfg.viewer.lookat = self.cfg.viewer.lookat


        # Setup the low-level env to know about the cube agent's parameters.
        ll_env_cfg.env.agent_ang = self.cfg.env.agent_ang
        ll_env_cfg.env.agent_rad = self.cfg.env.agent_rad

        ll_env_cfg.env.agent_facing_away = self.cfg.env.agent_facing_away
        ll_env_cfg.env.randomize_on_reset = self.cfg.env.randomize_on_reset

        # TODO: NOTE -- the low-level game is in charge of creating the simulation and terrain
        #  (i.e., only the LL game calls create_sim())

        # create the policy loader
        ll_train_cfg_dict = class_to_dict(ll_train_cfg)
        ll_policy_runner = LLPolicyRunner(ll_env_cfg, ll_train_cfg_dict, self.device)

        # make the low-level environment
        print("[DecHighLevelGame] preparing low level environment...")
        self.ll_env, _ = task_registry.make_env(name="low_level_game", args=None, env_cfg=ll_env_cfg)
        self.ll_env.debug_viz = self.cfg.env.debug_viz

        # load low-level policy
        print("[DecHighLevelGame] loading low level policy... for: ", ll_train_cfg.runner.experiment_name)
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', ll_train_cfg.runner.experiment_name)
        # load_run = 'vanilla_policy'
        load_run = 'sideways_walking_policy'
        # load_run = ll_train_cfg.runner.load_run
        path = get_load_path(log_root, load_run=load_run,
                             checkpoint=ll_train_cfg.runner.checkpoint)
        self.ll_policy = ll_policy_runner.load_policy(path)

        self.num_envs = self.cfg.env.num_envs
        self.debug_viz = self.cfg.env.debug_viz
        self.hl_dt = self.cfg.env.robot_hl_dt # high level Hz

        # parse the high-level config into appropriate dicts
        self._parse_cfg(self.cfg)

        self.gym = gymapi.acquire_gym()

        # setup the capture distance between two agents
        self.capture_dist = self.cfg.env.capture_dist
        # setup the goal reaching distance for the robot
        if self.cfg.env.interaction_type == 'nav':
            self.goal_dist = self.cfg.env.goal_dist
            self.collision_dist = self.cfg.env.collision_dist
        self.MAX_REL_POS = 100.

        # setup sensing params about robot
        self.robot_full_fov = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.robot_full_fov[:] = self.cfg.robot_sensing.fov

        # setup prediction / history horizon for the robot
        self.num_pred_steps = self.cfg.env.num_pred_steps
        self.num_hist_steps = self.cfg.env.num_hist_steps

        # if using a FOV curriculum, initialize it to the starting FOV
        self.fov_curr_idx = 0
        self.cmds_curr_idx = 0
            
        if self.cfg.commands.use_joypad:
            self._joypad = Joypad()
            thread = Thread(target=self._joypad.listen)
            thread.start()
            if self.cfg.commands.record_joystick:
                self.command_agent_buf = []
                self.initial_position = []

        # if using a PREY curriculum, initialize it with starting prey relative angle range
        self.prey_curr_idx = 0
        self.max_rad = self.cfg.env.agent_rad[1] # [m] distance away from predator
        self.min_rad = self.cfg.env.agent_rad[0]

        # if using a OBSTACLE curriculum, initialize it with starting obstacle height
        self.obstacle_curr_idx = 0

        self.prey_policy_type_idx = 0

        # this keeps track of which training iteration we are in!
        self.training_iter_num = 0

        print("[DecHighLevelGame] robot full fov is: ", self.robot_full_fov[0])

        # interaction type
        self.interaction_type = self.cfg.env.interaction_type
        if self.interaction_type == 'nav':
            self.agent_init_bias = self.cfg.env.agent_init_bias
            self.agent_policy_bias = self.cfg.env.agent_policy_bias
        self.robot_policy_type = self.cfg.env.robot_policy_type
        self.agent_policy_type = self.cfg.env.agent_policy_type

        print("[DecHighLevelGame] interaction type is: ", self.interaction_type)
        print("[DecHighLevelGame] robot policy type is: ", self.robot_policy_type)
        print("[DecHighLevelGame] agent policy type is: ", self.agent_policy_type)

        if self.interaction_type == 'nav':
            robot_xyz_pos = self.ll_env.root_states[self.ll_env.robot_indices, :3]
            # initialize a suite of robot goals
            self.max_rad_goal = self.cfg.env.goal_rad[1] # [m] distance away from predator
            self.min_rad_goal = self.cfg.env.goal_rad[0]
            self.max_ang_goal = self.cfg.env.goal_ang[1]
            self.min_ang_goal = self.cfg.env.goal_ang[0]
            rand_angle_goal = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(self.min_ang_goal, self.max_ang_goal)
            rand_radius_goal = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(self.min_rad_goal, self.max_rad_goal)
            self.robot_goal = robot_xyz_pos + torch.cat((rand_radius_goal * torch.cos(rand_angle_goal),
                                                       rand_radius_goal * torch.sin(rand_angle_goal),
                                                        torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
                                                       ), dim=-1)
            self.ll_env.robot_goal_loc = self.robot_goal.clone()

            if self.agent_init_bias:
                self.agent_init_bias_scale = 0.1
                rand_offset = self.agent_init_bias_scale * torch.rand(self.num_envs, 1, device=self.device, requires_grad=False)
                rand_sign = torch.rand(self.num_envs, 1, device=self.device, requires_grad=False)
                rand_sign[rand_sign>0.5] = 1
                rand_sign[rand_sign<0.5] = -1
                # set agent starting angle to be in a similar direction as the goal
                rand_angle_agent = wrap_to_pi(rand_angle_goal + rand_sign * rand_offset)
                rand_radius_agent = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(self.min_rad, self.max_rad)
                self.agent_offset_xyz = torch.cat((rand_radius_agent * torch.cos(rand_angle_agent),
                                                   rand_radius_agent * torch.sin(rand_angle_agent),
                                                   torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
                                                   ), dim=-1)
                self.ll_env.agent_offset_xyz = self.agent_offset_xyz
                self.ll_env.rand_angle = rand_angle_agent

                # reset the low-level environment with the new initial prey positions
                self.ll_env._reset_root_states(torch.arange(self.num_envs, device=self.device))
                self._update_agent_states()
        elif self.interaction_type == 'game':
            self.ll_env._reset_root_states(torch.arange(self.num_envs, device=self.device))

        # robot policy info
        self.num_obs_robot = self.cfg.env.num_observations_robot
        self.num_privileged_obs_robot = self.cfg.env.num_privileged_obs_robot
        self.num_robot_states = self.cfg.env.num_robot_states
        self.num_actions_robot = self.cfg.env.num_actions_robot

        if self.robot_policy_type == 'po_prediction_phase2' or self.robot_policy_type == 'prediction_phase2':
            self.num_privileged_obs_priv_robot = self.cfg.env.num_privileged_obs_priv_robot
            self.num_obs_priv_robot = self.cfg.env.num_observations_priv_robot
            self.num_priv_robot_states = self.cfg.env.num_priv_robot_states
            self.num_future_robot_states = self.num_priv_robot_states
        else:
            self.num_future_robot_states = self.num_robot_states

        # agent policy info
        self.num_agent_states = self.cfg.env.num_agent_states
        self.num_obs_agent = self.cfg.env.num_observations_agent
        self.num_privileged_obs_agent = self.cfg.env.num_privileged_obs_agent
        self.num_actions_agent = self.cfg.env.num_actions_agent
        self.agent_dyn_type = self.cfg.env.agent_dyn_type
        if self.agent_dyn_type != "dubins" and self.agent_dyn_type != "integrator":
            raise NameError("Can't simulate agent type: ", self.agent_dyn_type)
            return -1

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # *SIMPLER* PARTIAL obs space: setup the indicies of relevant observation quantities
        # self.pos_idxs_robot = list(range(0,12)) # all relative position history
        # self.visible_idxs_robot = list(range(12,16))  # all visible bools

        # PARTIAL obs space: setup the indicies of relevant observation quantities
        # self.pos_idxs_robot = list(range(0,12)) # all relative position history
        # self.ang_global_idxs_robot = list(range(12,20)) # all relative (global) angle
        # self.visible_idxs_robot = list(range(20, 24))  # all visible bools
        ## self.detect_idxs_robot = list(range(20, 24))  # all detected bools
        ## self.visible_idxs_robot = list(range(24, 28))  # all visible bools

        # FULL obs space: but with extra info
        # self.pos_idxs_robot = list(range(0,3)) # all relative position history
        # self.ang_global_idxs_robot = list(range(3,5)) # all relative (global) angle
        # self.detect_idxs_robot = list(range(5,6))  # all detected bools
        # self.visible_idxs_robot = list(range(6,7))  # all visible bools

        # KALMAN FILTER obs space
        # self.pos_idxs_robot = list(range(0,4))
        # self.cov_idxs_robot = list(range(4,20))

        self.pos_idxs_robot = list(range(0, 3))

        # allocate robot buffers
        # self.obs_buf_robot = self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        # self.obs_buf_robot[:, -2:] = 0. # reset the robot's sense booleans to zero
        # self.obs_buf_robot = torch.zeros(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        # self.obs_buf_robot[:, self.pos_idxs_robot] = self.MAX_REL_POS
        self.obs_buf_robot = torch.zeros(self.num_envs, self.num_obs_robot, device=self.device, dtype=torch.float)
        if self.num_obs_robot > 3: # TODO: kind of a hack
            self.obs_buf_robot[:, self.pos_idxs_robot] = self.MAX_REL_POS

        self.rel_state_hist = torch.zeros(self.num_envs, self.num_hist_steps, self.num_robot_states, device=self.device, dtype=torch.float)
        self.agent_state_hist = torch.zeros(self.num_envs, self.num_hist_steps, self.num_agent_states, device=self.device, dtype=torch.float)
        self.robot_commands_hist = torch.zeros(self.num_envs, self.num_hist_steps, self.num_actions_robot, device=self.device, dtype=torch.float)
        self.robot_commands = torch.zeros(self.num_envs, self.num_actions_robot, dtype=torch.float, device=self.device, requires_grad=False)

        # For visualization only
        self.num_debug_hist_steps = self.cfg.env.debug_hist_steps
        self.debug_agent_state_hist = torch.zeros(self.num_envs, self.num_debug_hist_steps, self.num_agent_states,
                                            device=self.device, dtype=torch.float) # NOTE: always xyz
        self.debug_robot_state_hist = torch.zeros(self.num_envs, self.num_debug_hist_steps, self.num_agent_states,
                                            device=self.device, dtype=torch.float) # NOTE: always xyz

        self.rew_buf_robot = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # setup the low-level observation buffer
        self.ll_obs_buf_robot = self.ll_env.obs_buf

        # allocate agent buffers
        self.obs_buf_agent = -self.MAX_REL_POS * torch.ones(self.num_envs, self.num_obs_agent, device=self.device, dtype=torch.float)
        self.obs_buf_agent[:, -2:] = 0.
        # self.obs_buf_agent = torch.zeros(self.num_envs, self.num_obs_agent, device=self.device, dtype=torch.float)
        # self.obs_buf_agent[:, self.pos_idxs] = -self.MAX_REL_POS
        self.rew_buf_agent = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        self.curr_episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.capture_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # used for 'game' interaction
        self.goal_reach_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # used for 'nav' interaction
        self.collision_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # used for 'nav' interaction

        # TODO: test!
        self.last_robot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.detected_buf_agent = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool) # stores in which environment the agent has sensed the robot
        self.detected_buf_robot = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)  # stores in which environment the robot has sensed the agent

        # used for transforms.
        self.x_unit_tensor = to_torch([1., 0., 0.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0., 0., 1.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        if self.num_privileged_obs_agent is not None:
            self.privileged_obs_buf_agent = torch.zeros(self.num_envs, self.num_privileged_obs_agent, device=self.device,
                                                  dtype=torch.float)
        else:
            self.privileged_obs_buf_agent = None
            # self.num_privileged_obs = self.num_obs

        if self.num_privileged_obs_robot is not None:
            if self.robot_policy_type == 'po_prediction_phase2' or self.robot_policy_type == 'prediction_phase2':
                self.privileged_obs_buf_robot = torch.zeros(self.num_envs, self.num_privileged_obs_priv_robot, device=self.device, 
                                                      dtype=torch.float)
            # else:
            #     self.privileged_obs_buf_robot = torch.zeros(self.num_envs, self.num_privileged_obs_robot, device=self.device,
            #                                           dtype=torch.float)
        else:
            self.privileged_obs_buf_robot = None
            # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # TODO: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # Create the Kalman Filter
        self.filter_type = self.cfg.robot_sensing.filter_type
        self.real_traj = None
        self.z_traj = None
        self.est_traj = None
        self.P_traj = None 
        # self.num_states_kf = 3 # (x, y, dtheta)
        # TODO: remove hardcoding
        self.num_states_kf = 4  # (x, y, z, dtheta)
        self.num_actions_kf_r = self.num_actions_robot 
        self.num_actions_kf_a = self.num_actions_agent
        self.dyn_sys_type = "linear"
        self.filter_dt = self.hl_dt # we get measurements slower than sim: filter_dt = decimation * sim.dt
        print("[DecHighLevelGame] Filter type: ", self.filter_type)
        if self.filter_type == "kf":
            self.kf = KalmanFilter(self.filter_dt,
                                    self.num_states_kf,
                                    self.num_actions_kf_r,
                                    self.num_envs,
                                    state_type="pos_ang",
                                    device=self.device,
                                    dtype=torch.float)
        elif self.filter_type == "makf":
            self.kf = MultiAgentKalmanFilter(self.filter_dt,
                                    self.num_states_kf,
                                    self.num_actions_kf_r,
                                    self.num_actions_kf_a,
                                    self.num_envs,
                                    state_type="pos_ang",
                                    device=self.device,
                                    dtype=torch.float)
        else:
            print("[ERROR]! Invalid filter type: ", self.filter_type)
            return -1

        # KF data saving info
        self.save_kf_data = False
        self.data_save_tstep = 0
        self.data_save_interval = 50

        # state for agent policy simulation -- DUBINS WEAVING AGENT
        self.curr_turn_freq = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.int32)
        self.curr_straight_freq = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.int32)
        self.last_turn_tstep = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.int32)
        self.last_straight_tstep = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.int32)
        self.turn_or_straight_idx = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int32)   # 0 == turn, 1 == straight
        self.turn_direction_idx = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int32)    # 0 == turn left, 1 == turn right
        if self.cfg.env.randomize_ctrl_bounds:
            self.agent_max_vx = torch.zeros(self.num_envs,
                                                  1,
                                                  device=self.device).uniform_(self.cfg.env.max_vx_range[0], self.cfg.env.max_vx_range[1])
            self.agent_max_vang = torch.zeros(self.num_envs,
                                                  1,
                                                  device=self.device).uniform_(self.cfg.env.max_vang_range[0], self.cfg.env.max_vang_range[1])
        else:
            self.agent_max_vx = self.command_ranges["agent_lin_vel_x"][1] * torch.ones(self.num_envs,
                                                                                      1,
                                                                                      device=self.device)
            self.agent_max_vang = self.command_ranges["agent_ang_vel_yaw"][1] * torch.ones(self.num_envs,
                                                                                             1,
                                                                                             device=self.device)

        # state for agent policy simulation -- BROWNIAN MOTION AGENT
        self.prev_agent_command = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
        self.agent_cmd_curriculum_idx = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int32, requires_grad=False)
        curr_vx = torch.tensor([1.5, 0.5, 0.1, 0.5, 1.5], device=self.device, requires_grad=False)
        curr_omega = torch.tensor([-1, 0.5, 1, -0.5, 1], device=self.device, requires_grad=False)
        self.agent_vx_curriculum = curr_vx.repeat(self.num_envs, 1).reshape(self.num_envs, 5)
        self.agent_omega_curriculum = curr_omega.repeat(self.num_envs, 1).reshape(self.num_envs, 5)

        # state for agent policy simulation -- COMPLEX WEAVING AGENT
        # create a tensor which stores the action the agent should take at each timestep of the episode.
        # the tensor is of shape:
        #       [num_envs x episode_length (low-level timesteps) x num_agent_actions]
        self.agent_policy_schedule = torch.zeros(self.num_envs,
                                                 int(self.max_episode_length),
                                                 self.num_actions_agent,
                                                 device=self.device,
                                                 requires_grad=False)
        self.curr_agent_tstep = 0 # NOTE: this timer is "global" it just keeps increasing forever as the rollouts happen.
        vmax = self.command_ranges["agent_lin_vel_x"][1]
        vmin = self.command_ranges["agent_lin_vel_x"][0]
        steermin = self.command_ranges["agent_ang_vel_yaw"][0]
        steermax = self.command_ranges["agent_ang_vel_yaw"][1]
        tensor_vels = torch.tensor([vmin,
                                    vmax/3.,
                                    vmax/2.,
                                    vmax,
                                    vmax,
                                    vmax,
                                    vmax],
                                   device=self.device,
                                   requires_grad=False)
        tensor_steer = torch.tensor([steermin,
                                     steermin/2.,
                                     0,
                                     steermax/2.,
                                     steermax],
                                    device=self.device,
                                    requires_grad=False)
        self.motion_primitives_tensor = torch.cartesian_prod(tensor_vels,
                                                             tensor_steer)
        self.num_motion_primitives = self.motion_primitives_tensor.shape[0]

        # state for agent policy simulation -- INTEGRATOR FIXED RANDOM VEL AGENT
        self.agent_command_scale = 1.2

        # current agent command 
        self.curr_agent_command = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)


        # state for agent policy simulation -- LEARNED TRAJECTORY-PREDICTION AGENT
        self.agent_traj_len = self.num_pred_steps
        self.agent_traj = torch.zeros(self.num_envs,
                                         int(self.agent_traj_len),
                                         self.num_actions_agent,
                                         device=self.device,
                                         requires_grad=False)

        # Reward debugging saving info
        self.save_rew_data = False
        self.rew_debug_tstep = 0
        self.rew_debug_interval = 100

        # Reward debugging
        self.fov_reward = None
        self.fov_rel_yaw = None
        self.ll_env_command_ang_vel = None
        # self.r_start = None
        # self.r_goal = None
        # self.r_curr = None
        # self.r_curr_proj = None
        # self.r_last = None
        # self.r_last_proj = None

        self.command_clipping = self.cfg.commands.command_clipping

        # maintains the true state (i.e. the "perfect" obervation) of the robot
        self.true_obs_robot = torch.zeros(self.num_envs, self.num_obs_robot, device=self.device, requires_grad=False)

        # "optimal" robot actions for BC loss in PPO
        self.bc_actions_robot = torch.zeros(self.num_envs, self.num_actions_robot, device=self.device, requires_grad=False)

        print("[DecHighLevelGame] initializing buffers...")
        self._init_buffers()
        print("[DecHighLevelGame] preparing AGENT reward functions...")
        self._prepare_reward_function_agent()
        print("[DecHighLevelGame] preparing ROBOT reward functions...")
        self._prepare_reward_function_robot()
        print("[DecHighLevelGame] done with initializing!")
        self.init_done = True

        print("[DecHighLevelGame: init] agent pos: ", self.ll_env.root_states[self.ll_env.agent_indices, :3])
        print("[DecHighLevelGame: init] robot pos: ", self.ll_env.root_states[self.ll_env.robot_indices, :3])
        # print("[DecHighLevelGame: init] dist: ", torch.norm(self.ll_env.root_states[self.ll_env.agent_indices, :3] - self.ll_env.root_states[self.ll_env.robot_indices, :3], dim=-1))

    def step(self, command_agent, command_robot):
        """Applies a high-level command to a particular agent, simulates, and calls self.post_physics_step()

        Args:
            command_agent (torch.Tensor): Tensor of shape (num_envs, num_actions_agent) for the high-level command
            command_robot (torch.Tensor): Tensor of shape (num_envs, num_actions_robot) for the high-level command

        Returns:
            obs_buf_agent, obs_buf_robot, priviledged_obs_buf_agent, priviledged_obs_buf_robot, ...
            rew_buf_agent, rew_buf_robot, reset_buf, extras
        """
        
        if self.agent_policy_type == 'learned_traj':
            stopping_envs = self.curr_episode_step < self.num_future_robot_states
            command_robot[stopping_envs] *= 0

        # # bookkeeping
        # self.robot_commands = command_robot.clone()

        # clip the commands
        if self.command_clipping:
            command_robot = self.clip_command_robot(command_robot)
            command_agent = self.clip_command_agent(command_agent)

        # bookkeeping
        self.robot_commands = command_robot.clone()

        if self.agent_policy_type == 'learned_traj':
            # save the most recent predicted command for the future
            self._update_learned_traj_command_agent(command_agent.clone().detach())
            # get the actual command for this timestep
            command_agent = self._learned_traj_command_agent(self.curr_agent_tstep)

        # print("[CLIPPED] command agent: ", command_agent)
        # print("[CLIPPED] command robot: ", command_robot)
        
        if self.device == 'cpu':
            command_robot = command_robot.to(self.device)
            self.robot_commands = self.robot_commands.to(self.device)

        # NOTE: low-level policy requires 4D control
        # update the low-level simulator command since it deals with the robot
        ll_env_command_robot = torch.cat((command_robot,
                                          torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)), # heading
                                          dim=-1)

        # record the last robot state before simulating physics (get this out)
        self.last_robot_pos[:] = self.robot_states[:, :3]

        # run the low-level policy at a potentially different frequency 
        hl_freq = self.hl_dt
        ll_freq = self.ll_env.dt
        num_ll_steps = int(hl_freq / ll_freq)
        agent_command_buf = []
        
        if self.cfg.commands.use_joypad and self.cfg.commands.record_joystick:
            # Global position in xyz and (local) theta of cube
            curr_agent_state = torch.cat((self.agent_pos.clone(), 
                                    self.agent_heading.unsqueeze(-1).clone()
                                    ), 
                                    dim=-1)
            # Global position in xyz of the robot
            init_robot_states = self.robot_states[:, :3]

        for tstep in range(num_ll_steps):

            # simulate the other agent
            if self.agent_dyn_type == "integrator":
                if self.agent_policy_type == 'straight_line':
                    # command_agent = self._straight_line_command_agent()
                    command_agent = self._random_straight_line_command_agent()
            elif self.agent_dyn_type == "dubins":
                if self.cfg.commands.use_joypad:
                    command_agent = self._get_command_joy()
                    if self.cfg.commands.record_joystick:
                        agent_command_buf.append(command_agent)
                elif self.agent_policy_type == 'simple_weaving':
                    output = self._weaving_command_agent(self.turn_or_straight_idx.clone(),
                                                         self.last_turn_tstep.clone(),
                                                         self.last_straight_tstep.clone(),
                                                         self.turn_direction_idx.clone())
                    command_agent = output[0]
                    self.turn_or_straight_idx = output[1]
                    self.last_turn_tstep = output[2]
                    self.last_straight_tstep = output[3]
                    self.turn_direction_idx = output[4]
                    self.curr_agent_command = command_agent
                    #print("turn_or_straight_idx: ", self.turn_or_straight_idx)
                    #print("last_turn_tstep: ", self.last_turn_tstep)
                    #print("turn_direction_idx: ", self.turn_direction_idx)
                elif self.agent_policy_type == 'complex_weaving':
                    command_agent = self._complex_weaving_command_agent(self.curr_agent_tstep)
                elif self.agent_policy_type == 'static':
                    # if the prey agent's policy type is static, then zero out the controls.
                    command_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)

            # set the high level command in the low-level environment 
            self.ll_env.commands = ll_env_command_robot

            # get the low-level actions as a function of low-level obs
            # self.ll_env.compute_observations() # refresh the observation buffer!
            # self.ll_env._clip_obs()
            # ll_robot_obs = self.ll_env.get_observations()
            ll_robot_actions = self.ll_policy(self.ll_obs_buf_robot.detach())

            # forward simulate the low-level actions
            self.ll_obs_buf_robot, _, _, _, _ = self.ll_env.step(ll_robot_actions.detach())

            # forward simulate the agent action too
            if self.agent_dyn_type == "integrator":
                self.step_agent_single_integrator(command_agent)
            elif self.agent_dyn_type == "dubins":
                self.step_agent_dubins_car(command_agent)
        
        # print("--- simulation loop: %s seconds ---" % (time() - start_loop_time))

        # take care of terminations, compute observations, rewards, and dones
        self.post_physics_step(command_robot, command_agent) # TODO: this is the *last* command agent!
        # Save Stuff for joystick
        if self.cfg.commands.use_joypad and self.cfg.commands.record_joystick:
            self.extras['joystick_action'] = agent_command_buf[-1]
            self.extras['init_agent_state'] = curr_agent_state
            self.extras['init_robot_pos'] = init_robot_states

        if self.debug_viz:
            #   For debug visualization: update agent state history
            #   TODO: for now only using (x,y,z) components
            self.debug_agent_state_hist = torch.cat((self.agent_pos[:, :3].unsqueeze(1),
                                               self.debug_agent_state_hist[:, 0:self.num_debug_hist_steps - 1, :]),
                                              dim=1)
            self.debug_robot_state_hist = torch.cat((self.robot_states[:, :3].unsqueeze(1),
                                               self.debug_robot_state_hist[:, 0:self.num_debug_hist_steps - 1, :]),
                                              dim=1)
            self.ll_env.agent_state_hist = self.debug_agent_state_hist
            self.ll_env.robot_state_hist = self.debug_robot_state_hist

        # # update robot's visual sensing curriculum do this for all envs
        #if self.cfg.commands.cmds_curriculum:
        #    self._update_robot_cmds_curriculum()
        #     self._update_robot_sensing_curriculum(torch.arange(self.num_envs, device=self.device))

        return self.obs_buf_agent, self.obs_buf_robot, self.privileged_obs_buf_agent, self.privileged_obs_buf_robot, self.rew_buf_agent, self.rew_buf_robot, self.reset_buf, self.extras

    def _p_ctrl_robot(self):
        """Get supervisory control from P-controller."""
        p_cmd_robot = torch.zeros(self.num_envs, self.num_actions_robot, device=self.device, requires_grad=False)
        rel_pos_global = self.agent_pos[:, :3] - self.robot_states[:, :3]
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0])

        p_cmd_robot[:, 0] = torch.clip(rel_pos_local[:, 0], min=0, max=2) #torch.clip(rel_pos_local[:, 0], min=-1, max=1)
        p_cmd_robot[:, 1] = 0. #torch.clip(rel_pos_local[:, 1], min=-1, max=1)
        p_cmd_robot[:, -1] = torch.clip(rel_yaw_local, min=-3.14, max=3.14) #torch.clip(rel_yaw_local, min=-1, max=1)

        return p_cmd_robot

    def _straight_line_command_augmented_robot(self, command_robot_1d):
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        command_robot = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        command_robot[:, 0] = torch.clip(rel_agent_pos_xyz[:, 0], min=self.command_ranges["lin_vel_x"][0],
                                         max=self.command_ranges["lin_vel_x"][1])
        command_robot[:, 1] = torch.clip(rel_agent_pos_xyz[:, 1], min=self.command_ranges["lin_vel_y"][0],
                                         max=self.command_ranges["lin_vel_y"][1])
        command_robot[:, 2] = 0
        command_robot[:, :2] += command_robot_1d

        return command_robot

    def _straight_line_command_robot(self, command_robot):
        rel_pos_global = self.agent_pos[:, :3] - self.robot_states[:, :3]
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        command_robot[:, :2] = torch.clip(rel_pos_local[:, :2], min=-1, max=1)
        # command_robot[:, 0] = torch.clip(rel_pos_local[:, 0], min=self.command_ranges["lin_vel_x"][0],
        #                                  max=self.command_ranges["lin_vel_x"][1])
        # command_robot[:, 1] = torch.clip(rel_pos_local[:, 1], min=self.command_ranges["lin_vel_y"][0],
        #                                  max=self.command_ranges["lin_vel_y"][1])
        command_robot[:, 2] = 0

        return command_robot

    def _get_command_joy(self):
        command_agent = torch.zeros(self.num_envs, self.num_actions_agent,
                                    device=self.device, requires_grad=False)
        joypad_available = (time() - self._joypad.time_last_joy) < self._joypad.joypad_timeout
        if (joypad_available):
            print("Joy available")
            lin_speed_x = self._joypad.forward_value_normalized *self.command_ranges["agent_lin_vel_x"][1]
            lin_speed_ang = self._joypad.angular_value_normalized *self.command_ranges["agent_ang_vel_yaw"][1]
            #print("lin speed", lin_speed_x)
            #print("ang speed", lin_speed_ang)
        else:
            lin_speed_x = 0
            lin_speed_ang = 0
        command_agent[:,0] = lin_speed_x
        command_agent[:,1] = lin_speed_ang
        print(command_agent)
        return command_agent

    def _turn_and_pursue_command_robot(self, command_robot):
        rel_pos_global = self.agent_pos[:, :3] - self.robot_states[:, :3]
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0])

        for env_idx in range(self.num_envs):
            if torch.abs(rel_yaw_local[env_idx]) < 0.15: # pursue
                # command_robot[env_idx, 0] = 1
                # command_robot[env_idx, 1] = 0
                # command_robot[env_idx, 2] = 0
                command_robot[env_idx, :2] = 1 * torch.clip(rel_pos_local[env_idx, :2], min=-1, max=1)
                command_robot[env_idx, 2] = 0
            else: # turn
                command_robot[env_idx, 0] = 0.
                command_robot[env_idx, 1] = 0. # small bias, just for fun
                command_robot[env_idx, 2] = 1
        return command_robot

    def _straight_line_command_agent(self):
        """Agent goes straight away from the robot at max (vx, vy)"""
        # TODO: This assumes agent is prey
        rel_robot_pos_xyz = self.robot_states[:, :3] - self.agent_pos[:, :3]
        command_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
        command_agent[:, 0] = -1 * torch.clip(rel_robot_pos_xyz[:, 0], min=self.command_ranges["agent_lin_vel_x"][0],
                                         max=self.command_ranges["agent_lin_vel_x"][1])
        command_agent[:, 1] = -1 * torch.clip(rel_robot_pos_xyz[:, 1], min=self.command_ranges["agent_lin_vel_y"][0],
                                         max=self.command_ranges["agent_lin_vel_y"][1])

        return command_agent

    def _random_straight_line_command_agent(self):
        """At the beginning of each episode, choose a random (vx, vy), and then agent executes this consistently."""
        command_agent = self.curr_agent_command.clone()
        return command_agent

    def _learned_traj_command_agent(self, start_tstep, end_tstep=None):
        """Learned agent trajectory output by agent's action MLP"""
        
        # make sure you are within the time range:
        if end_tstep is not None:
            lo = 0 #int(start_tstep % self.agent_traj_len)
            hi = end_tstep - start_tstep #int(end_tstep % self.agent_traj_len)
            assert hi > 0
            command_agent = self.agent_traj[:, lo:hi, :]
        else:
            lo = int(start_tstep % self.agent_traj_len)
            command_agent = self.agent_traj[:, lo, :]
        return command_agent

    def _complex_weaving_command_agent(self, start_tstep, end_tstep=None):
        """Complex time-parameterized agent weaving behavior."""
        # make sure you are within the time range:
        if end_tstep is not None:
            lo = int(start_tstep % self.max_episode_length)
            hi = int(end_tstep % self.max_episode_length)
            if lo > hi: # if you wrapped around, then need extra bookkeeping
                command_agent_end = self.agent_policy_schedule[:, lo:, :]
                command_agent_start = self.agent_policy_schedule[:, :hi, :]
                if hi != 0:
                    command_agent = torch.cat((command_agent_end, command_agent_start), dim=1)
                else: # catch corner case
                    command_agent = command_agent_end
            else:
                command_agent = self.agent_policy_schedule[:, lo:hi, :]
        else:
            lo = int(start_tstep % self.max_episode_length)
            command_agent = self.agent_policy_schedule[:, lo, :]
        return command_agent

    def _brownian_motion_command_agent(self):
        """The agent changes the (v_lin, v_ang) with 5% probability to next in curriculum. Otherwise same cmd."""
        # TODO: This assumes agent is prey
        command_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)

        switch_indicator = torch.rand(1,1,device=self.device,requires_grad=False)

        switching_prob = 0.05 # 5% chance of changing from the prior control
        if switch_indicator[0][0] < switching_prob:
            m1 = torch.any(self.agent_cmd_curriculum_idx == 0, dim=1).nonzero(as_tuple=False).flatten()
            m2 = torch.any(self.agent_cmd_curriculum_idx == 1, dim=1).nonzero(as_tuple=False).flatten()
            m3 = torch.any(self.agent_cmd_curriculum_idx == 2, dim=1).nonzero(as_tuple=False).flatten()
            m4 = torch.any(self.agent_cmd_curriculum_idx == 3, dim=1).nonzero(as_tuple=False).flatten()
            m5 = torch.any(self.agent_cmd_curriculum_idx == 4, dim=1).nonzero(as_tuple=False).flatten()

            command_agent[m1, 0] = self.agent_vx_curriculum[m1,0]
            command_agent[m1, 1] = self.agent_omega_curriculum[m1,0]

            command_agent[m2, 0] = self.agent_vx_curriculum[m2,1]
            command_agent[m2, 1] = self.agent_omega_curriculum[m2,1]

            command_agent[m3, 0] = self.agent_vx_curriculum[m3,2]
            command_agent[m3, 1] = self.agent_omega_curriculum[m3,2]

            command_agent[m4, 0] = self.agent_vx_curriculum[m4,3]
            command_agent[m4, 1] = self.agent_omega_curriculum[m4,3]

            command_agent[m5, 0] = self.agent_vx_curriculum[m5,4]
            command_agent[m5, 1] = self.agent_omega_curriculum[m5,4]

            self.agent_cmd_curriculum_idx += 1
            self.agent_cmd_curriculum_idx %= 5
            self.prev_agent_command = command_agent
        else:
            command_agent = self.prev_agent_command

        return command_agent

    def _weaving_command_agent(self,
                                turn_or_straight_idx,
                                last_turn_tstep,
                                last_straight_tstep,
                                turn_direction_idx):
        """Agent executes hard-coded snake-like pattern of (v_lin, v_ang)."""
        # TODO: This assumes agent is prey
        command_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)

        turn_mode = torch.any(turn_or_straight_idx == 0, dim=1)

        turn_mode_ids = turn_mode.nonzero(as_tuple=False).flatten()
        straight_mode_ids = (~turn_mode).nonzero(as_tuple=False).flatten()

        turn_dir_0_ids = torch.any(turn_direction_idx == 0, dim=1).nonzero(as_tuple=False).flatten()
        turn_dir_1_ids = torch.any(turn_direction_idx == 1, dim=1).nonzero(as_tuple=False).flatten()

        turn_right_ids = self._intersect_vecs(turn_mode_ids, turn_dir_0_ids)
        turn_left_ids = self._intersect_vecs(turn_mode_ids, turn_dir_1_ids)

        # if in turn mode, setup the turn commands
        command_agent[turn_mode_ids, 0] = self.agent_max_vx[turn_mode_ids, 0] #self.command_ranges["agent_lin_vel_x"][1]
        command_agent[turn_right_ids, 1] = -self.agent_max_vang[turn_right_ids, 0] #self.command_ranges["agent_ang_vel_yaw"][0]
        command_agent[turn_left_ids, 1] = self.agent_max_vang[turn_left_ids, 0] #self.command_ranges["agent_ang_vel_yaw"][1]

        # if in straight mode, setup the straight command
        command_agent[straight_mode_ids, 0] = self.agent_max_vx[straight_mode_ids, 0] #self.command_ranges["agent_lin_vel_x"][1]
        command_agent[straight_mode_ids, 1] = 0

        # if time to switch from turn --> straight mode
        switch_from_turn_ids = torch.any(last_turn_tstep == self.curr_turn_freq, dim=1).nonzero(as_tuple=False).flatten() # switch to straight
        keep_turning_ids = torch.any(last_turn_tstep != self.curr_turn_freq, dim=1).nonzero(as_tuple=False).flatten()

        # if time to switch from straight --> turn mode
        switch_from_straight_ids = torch.any(last_straight_tstep == self.curr_straight_freq, dim=1).nonzero(as_tuple=False).flatten() # keep going in turn mode
        keep_straight_ids = torch.any(last_straight_tstep != self.curr_straight_freq, dim=1).nonzero(as_tuple=False).flatten() # keep going in straight mode

        # if TURNING and SWITCHING FROM TURNING
        turn_mode_and_switch_ids = self._intersect_vecs(turn_mode_ids, switch_from_turn_ids)
        
        # if TURNING and KEEP TURNING
        turn_mode_and_NOT_switch_ids = self._intersect_vecs(turn_mode_ids, keep_turning_ids)
        
        # if STRAIGHT and SWITCHING FROM STRAIGHT
        straight_mode_and_switch_ids = self._intersect_vecs(straight_mode_ids, switch_from_straight_ids)

        # if STRAIGHT and KEEP GOING STRAIGHT
        straight_mode_and_NOT_switch_ids = self._intersect_vecs(straight_mode_ids, keep_straight_ids)

        # if TURNING and SWITCHING FROM TURNING
        last_turn_tstep[turn_mode_and_switch_ids, 0] = 1 # reset counter
        turn_or_straight_idx[turn_mode_and_switch_ids, 0] += 1
        turn_or_straight_idx[turn_mode_and_switch_ids, 0] %= 2

        if self.cfg.env.randomize_turn_dir:
            # randomly sample if you are turning left or right
            if len(turn_mode_and_switch_ids) > 0:
                pswitch = torch.rand(self.num_envs, 1, device=self.device, requires_grad=False)
                pswitch[pswitch >= 0.3] = 1 # with 70% probability, you change the turn direction
                pswitch[pswitch < 0.3] = 0 # with 30% probability, you keep the turn direction
                turn_direction_idx[turn_mode_and_switch_ids, 0] += pswitch[turn_mode_and_switch_ids, 0].int()
        else:
            turn_direction_idx[turn_mode_and_switch_ids, 0] += 1
            turn_direction_idx[turn_mode_and_switch_ids, 0] %= 2

        # if STRAIGHT and SWITCHING FROM STRAIGHT
        last_straight_tstep[straight_mode_and_switch_ids, 0] = 1 # reset counter
        turn_or_straight_idx[straight_mode_and_switch_ids, 0] += 1
        turn_or_straight_idx[straight_mode_and_switch_ids, 0] %= 2

        # if not ready to switch out of turn mode
        last_turn_tstep[turn_mode_and_NOT_switch_ids, 0] += 1
        # if not ready to switch out of straight mode
        last_straight_tstep[straight_mode_and_NOT_switch_ids, 0] += 1

        return command_agent, turn_or_straight_idx, last_turn_tstep, last_straight_tstep, turn_direction_idx

    def _intersect_vecs(self, v1, v2):
        """Returns the intersection of the vectors v1 and v2."""
        combined1 = torch.cat((v1, v2))
        uniques1, counts1 = combined1.unique(return_counts=True)
        difference1 = uniques1[counts1 == 1]
        intersection1 = uniques1[counts1 > 1]

        return intersection1

    def _predict_random_straight_line_command_agent(self,
                                                    pred_hor):
        """Predicts the straight line command of the agent and returns the ctrls"""

        # agent commands in the agent's coordinate frame of shape: [num_envs x N x (vx, vy)]
        future_agent_cmds_agent_frame = torch.zeros(self.num_envs,
                                    pred_hor,
                                    self.num_actions_agent,
                                    device=self.device,
                                    requires_grad=False)
        # agent will just keep applying their current command over entire time horizon
        future_agent_cmds_agent_frame[:] = self.curr_agent_command.unsqueeze(dim=1)

        # TODO: Note: assumes integrator dynamics! state = (x,y,z)
        curr_agent_state = self.agent_pos[:, :3].clone()
        # future agent state in tensor of shape: [num_envs x (N+1) x (x,y,z)]
        future_agent_states = torch.zeros(self.num_envs,
                                                   pred_hor + 1,
                                                   self.num_agent_states,
                                                   device=self.device,
                                                   requires_grad=False)
        future_agent_states[:, 0, :] = curr_agent_state

        # future relative state w.r.t robot base frame assuming uR == 0 over prediction horizon.
        #   4-dim relative state = (x, y, z, theta)
        # future relative state in tensor of shape: [num_envs x (N+1) x (x,y,z,theta)_rel]
        future_rel_states_robot_frame = torch.zeros(self.num_envs,
                                                pred_hor + 1,
                                                self.num_future_robot_states,
                                                device=self.device,
                                                requires_grad=False)

        # get current relative state in robot's base frame
        rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)
        rel_state = torch.cat((rel_pos_local, rel_yaw_local), dim=-1)
        future_rel_states_robot_frame[:, 0, :] = rel_state

        # B_a = self.hl_dt * torch.eye(self.num_robot_states, self.num_actions_agent, device=self.device, requires_grad=False)
        # B_a[2, -1] = 0 # TODO: this is a hack; agent doesn't ctrl z-vel and isn't modelled to influence rel angle
        # B_a[-1, -1] = 0
        # B_a_tensor = B_a.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_robot_states, self.num_actions_agent)

        # for each future timestep
        for tstep in range(pred_hor):

            # simulate the other agent at the same frequency as the high-level policy gets executed
            curr_agent_state = self.agent_single_integrator_dynamics(curr_agent_state, future_agent_cmds_agent_frame[:, tstep, :])
            future_agent_states[:, tstep + 1, :] = curr_agent_state

            # using the future state of the other agent (assuming the robot is standing still) get the relative position
            future_rel_pos_global = (curr_agent_state - self.robot_states[:, :3]).clone()
            future_rel_pos_local = self.global_to_robot_frame(future_rel_pos_global)
            future_rel_yaw_local = torch.atan2(future_rel_pos_local[:, 1], future_rel_pos_local[:, 0]).unsqueeze(-1)
            future_rel_state = torch.cat((future_rel_pos_local, future_rel_yaw_local), dim=-1)

            # record the future relative state
            future_rel_states_robot_frame[:, tstep + 1, :] = future_rel_state

            # # Convert the agent's actions fo the robot coordinate frame.
            # #   Ra_r = Rw_r * Ra_w = (Rr_w)^-1 * Ra_w
            # quat_r_w = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
            # quat_a_w = self.ll_env.root_states[self.ll_env.agent_indices, 3:7]
            # quat_w_r = quat_conjugate(quat_r_w)
            # quat_a_r = quat_mul(quat_w_r, quat_a_w)
            # # pad the linear and angular velocity commands with zeros for stationary dimensions
            # command_agent_lin_vel = torch.cat((future_agent_cmds_agent_frame[:, tstep, :],
            #                                    torch.zeros(self.num_envs, 1, device=self.device,
            #                                                requires_grad=False)), dim=-1)
            # # transform from agent to the robot's coordinate frame
            # command_agent_lin_vel_r = quat_rotate_inverse(quat_a_r, command_agent_lin_vel)
            # # extract only vx and angular vel around the z-axis
            # command_agent = torch.cat((command_agent_lin_vel_r[:, 0].unsqueeze(-1),
            #                            command_agent_lin_vel_r[:, 1].unsqueeze(-1)),
            #                           dim=-1)
            # # TODO: CHECK THIS!
            # # evolve the relative dynamics
            # Baua = torch.bmm(B_a_tensor, command_agent.unsqueeze(-1)).squeeze(-1)
            # future_rel_state2 = future_rel_states_robot_frame[:, tstep, :] + Baua
            # future_rel_states_robot_frame[:, tstep+1, :] = future_rel_state2

            #print("future_rel_state (v1): ", future_rel_state)
            # print("future_rel_state (v2): ", future_rel_state2)

        return future_rel_states_robot_frame, future_agent_states

    def _predict_simple_weaving_command_agent(self,
                                        pred_hor):
        """Predicts the weaving command of the agent and returns the ctrls"""

        # agent commands in the agent's coordinate frame
        future_agent_cmds_agent_frame = torch.zeros(self.num_envs,
                                        pred_hor, 
                                        self.num_actions_agent, 
                                        device=self.device, 
                                        requires_grad=False)

        # TODO: Note: assumes dubins car dynamics! state = (x,y,z,theta)
        curr_agent_state = torch.cat((self.agent_pos.clone(), 
                                self.agent_heading.unsqueeze(-1).clone()
                                ), 
                                dim=-1)
        # future agent state in tensor of shape: [num_envs x (N+1) x (x,y,z)]
        future_agent_states = torch.zeros(self.num_envs,
                                    pred_hor + 1, 
                                    self.num_agent_states, 
                                    device=self.device, 
                                    requires_grad=False)
        future_agent_states[:, 0, :] = curr_agent_state[:, :3] # only keep track of xyz

        # future relative state w.r.t robot base frame assuming uR == 0 over prediction horizon.
        #   4-dim relative state = (x, y, z, theta)
        # future relative state in tensor of shape: [num_envs x (N+1) x (x,y,z,theta)_rel]
        future_rel_states_robot_frame = torch.zeros(self.num_envs,
                                                pred_hor + 1,
                                                self.num_future_robot_states,
                                                device=self.device,
                                                requires_grad=False)
        curr_rel_pos_global = (curr_agent_state[:, :3] - self.robot_states[:, :3]).clone()
        curr_rel_pos_local = self.global_to_robot_frame(curr_rel_pos_global)
        curr_rel_yaw_local = torch.atan2(curr_rel_pos_local[:, 1], curr_rel_pos_local[:, 0]).unsqueeze(-1)
        curr_rel_state = torch.cat((curr_rel_pos_local, curr_rel_yaw_local), dim=-1)
        future_rel_states_robot_frame[:, 0, :] = curr_rel_state

        turn_or_straight_idx = self.turn_or_straight_idx.clone()
        last_turn_tstep = self.last_turn_tstep.clone()
        last_straight_tstep = self.last_straight_tstep.clone()
        turn_direction_idx = self.turn_direction_idx.clone()

        for tstep in range(pred_hor):
            for i in range(int(self.hl_dt / self.ll_env.dt)):
                # NOTE: this weaving command is based on the LOW LEVEL environment dt.
                # it has to be called repeatedly to simulate the agent at the HIGH LEVEL environment's dt
                output = self._weaving_command_agent(turn_or_straight_idx,
                                         last_turn_tstep,
                                         last_straight_tstep,
                                         turn_direction_idx)
                command_agent = output[0]
                turn_or_straight_idx = output[1]
                last_turn_tstep = output[2]
                last_straight_tstep = output[3]
                turn_direction_idx = output[4]

            future_agent_cmds_agent_frame[:, tstep, :] = command_agent.clone()

            curr_agent_state = self.agent_dubins_car_dynamics(curr_agent_state, future_agent_cmds_agent_frame[:, tstep, :])
            future_agent_states[:, tstep + 1, :] = curr_agent_state[:, :3]

            # using the future state of the other agent (assuming the robot is standing still) get the relative position
            future_rel_pos_global = (curr_agent_state[:, :3] - self.robot_states[:, :3]).clone()
            future_rel_pos_local = self.global_to_robot_frame(future_rel_pos_global)
            future_rel_yaw_local = torch.atan2(future_rel_pos_local[:, 1], future_rel_pos_local[:, 0]).unsqueeze(-1)
            future_rel_state = torch.cat((future_rel_pos_local, future_rel_yaw_local), dim=-1)

            # # Convert the agent's actions fo the robot coordinate frame.
            # #   Ra_r = Rw_r * Ra_w = (Rr_w)^-1 * Ra_w
            # quat_r_w = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
            # quat_a_w = self.ll_env.root_states[self.ll_env.agent_indices, 3:7]
            # quat_w_r = quat_conjugate(quat_r_w)
            # quat_a_r = quat_mul(quat_w_r, quat_a_w)
            # # pad the linear and angular velocity commands with zeros for stationary dimensions
            # command_agent_lin_vel = torch.cat((command_agent[:, 0].unsqueeze(-1),
            #                                     torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False),
            #                                     torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)),
            #                                     dim=-1)
            # command_agent_ang_vel = torch.cat((torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False),
            #                                     torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False),
            #                                     command_agent[:, 1].unsqueeze(-1)),
            #                                     dim=-1)
            # # transform from agent to the robot's coordinate frame
            # command_agent_vlin = quat_rotate_inverse(quat_a_r, command_agent_lin_vel)
            # command_agent_vang = quat_rotate_inverse(quat_a_r, command_agent_ang_vel)
            # # extract only vx and angular vel around the z-axis
            # command_agent = torch.cat((command_agent_vlin[:, 0].unsqueeze(-1),
            #                            command_agent_vang[:, -1].unsqueeze(-1)),
            #                           dim=-1)

            future_rel_states_robot_frame[:, tstep + 1, :] = future_rel_state

        return future_rel_states_robot_frame, future_agent_states

    def _predict_complex_weaving_command_agent(self, pred_hor):
        """Predicts the COMPLEX weaving command of the agent and returns the ctrls"""

        # agent commands in the agent's coordinate frame
        future_agent_cmds_agent_frame = self._complex_weaving_command_agent(self.curr_agent_tstep,
                                                                            self.curr_agent_tstep+pred_hor)

        assert future_agent_cmds_agent_frame.shape[1] == pred_hor
        # TODO: Note: assumes dubins car dynamics! state = (x,y,z,theta)
        curr_agent_state = torch.cat((self.agent_pos.clone(),
                                      self.agent_heading.unsqueeze(-1).clone()
                                      ),
                                     dim=-1)
        # future agent state in tensor of shape: [num_envs x (N+1) x (x,y,z)]
        future_agent_states = torch.zeros(self.num_envs,
                                          pred_hor + 1,
                                          self.num_agent_states,
                                          device=self.device,
                                          requires_grad=False)
        future_agent_states[:, 0, :] = curr_agent_state[:, :3]  # only keep track of xyz

        # future relative state w.r.t robot base frame assuming uR == 0 over prediction horizon.
        #   4-dim relative state = (x, y, z, theta)
        # future relative state in tensor of shape: [num_envs x (N+1) x (x,y,z,theta)_rel]
        future_rel_states_robot_frame = torch.zeros(self.num_envs,
                                                    pred_hor + 1,
                                                    self.num_future_robot_states,
                                                    device=self.device,
                                                    requires_grad=False)
        curr_rel_pos_global = (curr_agent_state[:, :3] - self.robot_states[:, :3]).clone()
        curr_rel_pos_local = self.global_to_robot_frame(curr_rel_pos_global)
        curr_rel_yaw_local = torch.atan2(curr_rel_pos_local[:, 1], curr_rel_pos_local[:, 0]).unsqueeze(-1)
        curr_rel_state = torch.cat((curr_rel_pos_local, curr_rel_yaw_local), dim=-1)
        future_rel_states_robot_frame[:, 0, :] = curr_rel_state

        for tstep in range(pred_hor):
            curr_agent_state = self.agent_dubins_car_dynamics(curr_agent_state,
                                                              future_agent_cmds_agent_frame[:, tstep, :])
            future_agent_states[:, tstep + 1, :] = curr_agent_state[:, :3]

            # using the future state of the other agent (assuming the robot is standing still) get the relative position
            future_rel_pos_global = (curr_agent_state[:, :3] - self.robot_states[:, :3]).clone()
            future_rel_pos_local = self.global_to_robot_frame(future_rel_pos_global)
            future_rel_yaw_local = torch.atan2(future_rel_pos_local[:, 1], future_rel_pos_local[:, 0]).unsqueeze(-1)
            future_rel_state = torch.cat((future_rel_pos_local, future_rel_yaw_local), dim=-1)
            future_rel_states_robot_frame[:, tstep + 1, :] = future_rel_state

        return future_rel_states_robot_frame, future_agent_states

    def _predict_learned_traj_command_agent(self, pred_hor):
        """Predicts the LEARNED command-trajectory of the agent and returns the ctrls"""

        # agent commands in the agent's coordinate frame
        future_agent_cmds_agent_frame = self._learned_traj_command_agent(self.curr_agent_tstep,
                                                                         self.curr_agent_tstep+pred_hor)

        assert future_agent_cmds_agent_frame.shape[1] == pred_hor

        # TODO: Note: assumes dubins car dynamics! state = (x,y,z,theta)
        curr_agent_state = torch.cat((self.agent_pos.clone(),
                                      self.agent_heading.unsqueeze(-1).clone()
                                      ),
                                     dim=-1)
        # future agent state in tensor of shape: [num_envs x (N+1) x (x,y,z)]
        future_agent_states = torch.zeros(self.num_envs,
                                          pred_hor + 1,
                                          self.num_agent_states,
                                          device=self.device,
                                          requires_grad=False)
        future_agent_states[:, 0, :] = curr_agent_state[:, :3]  # only keep track of xyz

        # future relative state w.r.t robot base frame assuming uR == 0 over prediction horizon.
        #   4-dim relative state = (x, y, z, theta)
        # future relative state in tensor of shape: [num_envs x (N+1) x (x,y,z,theta)_rel]
        future_rel_states_robot_frame = torch.zeros(self.num_envs,
                                                    pred_hor + 1,
                                                    self.num_future_robot_states,
                                                    device=self.device,
                                                    requires_grad=False)
        curr_rel_pos_global = (curr_agent_state[:, :3] - self.robot_states[:, :3]).clone()
        curr_rel_pos_local = self.global_to_robot_frame(curr_rel_pos_global)
        curr_rel_yaw_local = torch.atan2(curr_rel_pos_local[:, 1], curr_rel_pos_local[:, 0]).unsqueeze(-1)
        curr_rel_state = torch.cat((curr_rel_pos_local, curr_rel_yaw_local), dim=-1)
        future_rel_states_robot_frame[:, 0, :] = curr_rel_state


        for tstep in range(pred_hor):
            if self.agent_dyn_type == 'dubins':
                curr_agent_state = self.agent_dubins_car_dynamics(curr_agent_state,
                                                                  future_agent_cmds_agent_frame[:, tstep, :])
            elif self.agent_dyn_type == 'integrator':
                curr_agent_state = self.agent_single_integrator_dynamics(curr_agent_state,
                                                                  future_agent_cmds_agent_frame[:, tstep, :])
            future_agent_states[:, tstep + 1, :] = curr_agent_state[:, :3]

            # using the future state of the other agent (assuming the robot is standing still) get the relative position
            future_rel_pos_global = (curr_agent_state[:, :3] - self.robot_states[:, :3]).clone()
            future_rel_pos_local = self.global_to_robot_frame(future_rel_pos_global)
            future_rel_yaw_local = torch.atan2(future_rel_pos_local[:, 1], future_rel_pos_local[:, 0]).unsqueeze(-1)
            future_rel_state = torch.cat((future_rel_pos_local, future_rel_yaw_local), dim=-1)
            future_rel_states_robot_frame[:, tstep + 1, :] = future_rel_state

        return future_rel_states_robot_frame, future_agent_states

    def clip_command_robot(self, command_robot):
        """Clips the robot's commands"""
        # clip the robot's commands
        command_robot[:, 0] = torch.clip(command_robot[:, 0], min=self.command_ranges["lin_vel_x"][0], max=self.command_ranges["lin_vel_x"][1])
        command_robot[:, 1] = torch.clip(command_robot[:, 1], min=self.command_ranges["lin_vel_y"][0], max=self.command_ranges["lin_vel_y"][1])
        command_robot[:, 2] = torch.clip(command_robot[:, 2], min=self.command_ranges["ang_vel_yaw"][0], max=self.command_ranges["ang_vel_yaw"][1])

        # set small commands to zero; this was originally used by lowlevel code to enable robot to learn to stand still
        # command_robot[:, :2] *= (torch.norm(command_robot[:, :2], dim=1) > 0.2).unsqueeze(1)

        return command_robot

    def clip_command_agent(self, command_agent):
        """Clips the agent's commands"""
        if self.agent_dyn_type == 'dubins':
            command_agent[:, 0] = torch.clip(command_agent[:, 0], min=self.command_ranges["agent_lin_vel_x"][0], max=self.command_ranges["agent_lin_vel_x"][1])
            command_agent[:, 1] = torch.clip(command_agent[:, 1], min=self.command_ranges["agent_ang_vel_yaw"][0], max=self.command_ranges["agent_ang_vel_yaw"][1])
        elif self.agent_dyn_type == 'integrator':
            command_agent[:, 0] = torch.clip(command_agent[:, 0], min=self.command_ranges["agent_lin_vel_x"][0], max=self.command_ranges["agent_lin_vel_x"][1])
            command_agent[:, 1] = torch.clip(command_agent[:, 1], min=self.command_ranges["agent_lin_vel_y"][0], max=self.command_ranges["agent_lin_vel_y"][1])

        return command_agent

    def step_agent_omnidirectional(self, command_agent):
        """
        Steps agent modeled as an omnidirectional agent
            x' = x + dt * u1 * cos(theta)
            y' = y + dt * u2 * sin(theta)
            z' = z
            theta' = theta + dt * u3
        where the control is 3-dimensional:
            u = [u1 u2 u3] = [vx, vy, omega]
        """
        lin_vel_x = command_agent[:, 0]
        lin_vel_y = command_agent[:, 1]
        ang_vel = command_agent[:, 2]

        # TODO: agent gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.agent_pos[:, 0] += self.ll_env.cfg.sim.dt * lin_vel_x #* torch.cos(self.agent_heading)
            self.agent_pos[:, 1] += self.ll_env.cfg.sim.dt * lin_vel_y #* torch.sin(self.agent_heading)
            self.agent_heading += self.ll_env.cfg.sim.dt * ang_vel
            self.agent_heading = wrap_to_pi(self.agent_heading) # make sure heading is between -pi and pi

        # convert from heading angle to quaternion
        heading_quat = quat_from_angle_axis(self.agent_heading, self.z_unit_tensor)

        # update the simulator state for the agent!
        self.ll_env.root_states[self.ll_env.agent_indices, :3] = self.agent_pos
        self.ll_env.root_states[self.ll_env.agent_indices, 3:7] = heading_quat

        self.ll_env.root_states[self.ll_env.agent_indices, 7] = lin_vel_x  # lin vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 8] = lin_vel_y  # lin vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 9] = 0.  # lin vel z
        self.ll_env.root_states[self.ll_env.agent_indices, 10] = 0. # ang vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 11] = 0. # ang_vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 12] = ang_vel # ang_vel z

        # self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))
        agent_env_ids_int32 = self.ll_env.agent_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.ll_env.sim,
                                                     gymtorch.unwrap_tensor(self.ll_env.root_states),
                                                     gymtorch.unwrap_tensor(agent_env_ids_int32),
                                                     len(agent_env_ids_int32))

    def step_agent_single_integrator(self, command_agent):
        """
        Steps agent modeled as a single integrator:
            x' = x + dt * u1
            y' = y + dt * u2
            z' = z          (assume z-dim is constant)
        """

        # agent that has full observability of the robot state
        command_lin_vel_x = command_agent[:, 0]
        command_lin_vel_y = command_agent[:, 1]

        # TODO: agent gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.agent_pos[:, 0] += self.ll_env.cfg.sim.dt * command_lin_vel_x
            self.agent_pos[:, 1] += self.ll_env.cfg.sim.dt * command_lin_vel_y

        heading_quat = quat_from_angle_axis(self.agent_heading, self.z_unit_tensor)
        # print("[DecHighLevelGame | step_agent_single_integrator] heading_quat (z-unit): ", heading_quat)
        # print("[DecHighLevelGame] curr agent quaternion: ", self.ll_env.root_states[self.ll_env.agent_indices, 3:7])

        # update the simulator state for the agent!
        self.ll_env.root_states[self.ll_env.agent_indices, :3] = self.agent_pos
        self.ll_env.root_states[self.ll_env.agent_indices, 3:7] = heading_quat

        self.ll_env.root_states[self.ll_env.agent_indices, 7] = command_lin_vel_x # lin vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 8] = command_lin_vel_y # lin vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 9] = 0.  # lin vel z
        self.ll_env.root_states[self.ll_env.agent_indices, 10:13] = 0. # ang vel xyz

        # self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))
        agent_env_ids_int32 = self.ll_env.agent_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.ll_env.sim,
                                                     gymtorch.unwrap_tensor(self.ll_env.root_states),
                                                     gymtorch.unwrap_tensor(agent_env_ids_int32),
                                                     len(agent_env_ids_int32))

    def agent_single_integrator_dynamics(self, agent_state, command_agent):
        """
        Simulates the agent physics according to the HIGH LEVEL timestep:
            x' = x + hl_dt * u1
            y' = y + hl_dt * u2
            z' = z
        """
        next_state = torch.zeros(agent_state.shape[0], agent_state.shape[1], device=self.device, requires_grad=False)
        next_state[:, 0] = agent_state[:, 0] + self.hl_dt * command_agent[:, 0]
        next_state[:, 1] = agent_state[:, 1] + self.hl_dt * command_agent[:, 1]
        next_state[:, 2] = agent_state[:, 2] 

        return next_state

    def agent_dubins_car_dynamics(self, agent_state, command_agent):
        """
        Simulates the agent physics according to the HIGH LEVEL timestep:
            x' = x + hl_dt * u1 * cos(yaw)
            y' = y + hl_dt * u1 * cos(yaw)
            z' = z
            yaw' = yaw + hl_dt * u2
        """
        command_lin_vel = command_agent[:, 0]
        command_ang_vel = command_agent[:, 1]
        next_state = torch.zeros_like(agent_state)

        next_state[:, 0] = agent_state[:, 0] + (command_lin_vel/command_ang_vel) * (torch.sin(agent_state[:, 3] +
                                                                                              command_ang_vel * self.hl_dt) - torch.sin(agent_state[:, 3]))
        next_state[:, 1] = agent_state[:, 1] - (command_lin_vel / command_ang_vel) * (torch.cos(agent_state[:, 3] +
                                                                                                command_ang_vel * self.hl_dt) - torch.cos(
            agent_state[:, 3]))
        next_state[:, 2] = agent_state[:, 2]
        next_state[:, 3] = agent_state[:, 3] + self.hl_dt * command_ang_vel
        next_state[:, 3] = wrap_to_pi(next_state[:, 3])

        return next_state

    def step_agent_dubins_car(self, command_agent):
        """
        Steps agent modeled as a single integrator:
            x' = x + dt * u1 * cos(yaw)
            y' = y + dt * u1 * cos(yaw)
            yaw' = yaw + dt * u2
        """

        # agent that has full observability of the robot state
        command_lin_vel = command_agent[:, 0]
        command_ang_vel = command_agent[:, 1]

        # TODO: agent gets simulated at the same Hz as the low-level controller!
        for _ in range(self.ll_env.cfg.control.decimation):
            self.agent_pos[:, 0] += self.ll_env.cfg.sim.dt * command_lin_vel * torch.cos(self.agent_heading)
            self.agent_pos[:, 1] += self.ll_env.cfg.sim.dt * command_lin_vel * torch.sin(self.agent_heading)
            self.agent_heading += self.ll_env.cfg.sim.dt * command_ang_vel
            self.agent_heading = wrap_to_pi(self.agent_heading)

        heading_quat = quat_from_angle_axis(self.agent_heading, self.z_unit_tensor)

        # update the simulator state for the agent!
        self.ll_env.root_states[self.ll_env.agent_indices, :3] = self.agent_pos
        self.ll_env.root_states[self.ll_env.agent_indices, 3:7] = heading_quat

        self.ll_env.root_states[self.ll_env.agent_indices, 7] = command_lin_vel # lin vel x
        self.ll_env.root_states[self.ll_env.agent_indices, 8] = 0. # lin vel y
        self.ll_env.root_states[self.ll_env.agent_indices, 9] = 0.  # lin vel z
        self.ll_env.root_states[self.ll_env.agent_indices, 10:13] = 0. # ang vel xyz
        # self.ll_env.root_states[self.ll_env.agent_indices, 12] = command_ang_vel

        # self.ll_env.gym.set_actor_root_state_tensor(self.ll_env.sim, gymtorch.unwrap_tensor(self.ll_env.root_states))
        agent_env_ids_int32 = self.ll_env.agent_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.ll_env.sim,
                                                     gymtorch.unwrap_tensor(self.ll_env.root_states),
                                                     gymtorch.unwrap_tensor(agent_env_ids_int32),
                                                     len(agent_env_ids_int32))

    def post_physics_step(self, command_robot, command_agent):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
        """
        self.gym.refresh_actor_root_state_tensor(self.ll_env.sim)
        self.gym.refresh_net_contact_force_tensor(self.ll_env.sim)

        # updates the local copies of agent state info
        self._update_agent_states()

        self.episode_length_buf += 1
        self.curr_episode_step += 1 # this is used to model the progress of "time" in the episode
        self.curr_agent_tstep += 1  # this is used to model the progress of "time" for the agent's policy

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward_robot() # robot (robot) combines the high-level and low-level rewards
        self.compute_reward_agent()
        # print("agent_dist: ", torch.norm(self.robot_states[:, :3] - self.agent_pos, dim=-1))
        # print("rew_buf_agent: ", self.rew_buf_agent)

        # reset environments
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.reset_idx(env_ids)

        # refresh the low-level policy's observation buf
        self.ll_env.compute_observations()
        self.ll_env._update_last_quantities()
        self.ll_env._clip_obs()
        self.ll_obs_buf_robot = self.ll_env.get_observations()

        # compute the high-level observation for each agent
        self.compute_observations_agent()               # compute high-level observations for agent
        self.compute_observations_robot(command_robot, command_agent)  # compute high-level observations for robot
        if self.num_privileged_obs_robot is not None:
            self.compute_privileged_observations_robot(command_robot, command_agent)

        # print("[in post_physics_step] obs_buf_robot: ", self.obs_buf_robot)

    def check_termination(self):
        """ Check if environments need to be reset under various conditions.
        """
        # reset agent-robot if they are within the capture distance

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs

        if self.interaction_type == 'game':
            self.capture_buf = torch.norm(self.robot_states[:, :2] - self.agent_pos[:, :2], dim=-1) < self.capture_dist
            self.reset_buf = self.capture_buf.clone()
        elif self.interaction_type == 'nav':
            self.goal_reach_buf = torch.norm(self.robot_states[:, :2] - self.robot_goal[:, :2], dim=-1) < self.goal_dist
            self.collision_buf = torch.norm(self.robot_states[:, :2] - self.agent_pos[:, :2], dim=-1) < self.capture_dist
            goal_or_coll_buf = self.goal_reach_buf | self.collision_buf
            self.reset_buf = goal_or_coll_buf

        self.reset_buf |= self.time_out_buf

    def _update_learned_traj_command_agent(self, learned_traj_command):
        """Updates the last command of the agent command trajectory.

        self.agent_traj is of shape [num_envs, num_pred_steps, num_actions_agent]
        and is laid out:
            self.agent_traj[0,0,:] = uA^t
            self.agent_traj[0,1,:] = uA^t+1
            self.agent_traj[0,2,:] = uA^t+2
            ...
        """
        assert learned_traj_command.shape[0] == self.num_envs
        assert learned_traj_command.shape[1] == self.num_actions_agent
        
        if self.device == 'cpu':
            learned_traj_command = learned_traj_command.to(self.device)
        
        self.agent_traj = torch.cat((self.agent_traj[:, 1:, :], learned_traj_command.unsqueeze(1)), dim=1)

    def _initialize_complex_weaving_command_agent(self, env_ids):
        """Initializes the complex weaving command agent policy."""

        # print("[complex_weaving] Initializing the complex_weaving_command_agent...")

        # get a random duration at which all motion primitives will be applied this episode
        #min_dur = 4 # 0.8 seconds
        #max_dur = 10 # 2 seconds
        min_dur = 5 # 1 seconds
        max_dur = 15 # 3 seconds
        duration = np.random.randint(min_dur, max_dur, size=1)

        # how many chunks to split episode time into
        num_chunks = int(self.max_episode_length // duration)
        remainder = int(self.max_episode_length % duration)

        # get a random sequence of motion primitive indexes
        rand_mp_seq = torch.randint(0,
                                    self.num_motion_primitives,
                                    (self.num_envs, num_chunks),
                                    device=self.device, requires_grad=False)

        counter = 0
        # import pdb;
        # pdb.set_trace()
        for chunk_id in range(num_chunks):
            mp_ids = rand_mp_seq[env_ids, chunk_id]     # shape: size of env_ids, with an index into the motion primitive per env
            curr_motion_primitives = self.motion_primitives_tensor[mp_ids] # mp_per_env_tensor[env_ids, mp_ids]

            vx = curr_motion_primitives[:, 0].repeat(duration[0]).view(duration[0], len(env_ids))
            vsteer = curr_motion_primitives[:, 1].repeat(duration[0]).view(duration[0], len(env_ids))

            self.agent_policy_schedule[env_ids, counter:counter+duration[0], 0] = torch.transpose(vx, 0, 1)
            self.agent_policy_schedule[env_ids, counter:counter+duration[0], 1] = torch.transpose(vsteer, 0, 1)

            # setup the angular velocity
            counter += duration[0]

        if self.interaction_type == 'nav':
            if self.agent_policy_bias:
                # if we are biasing a subset of the policies to be "pursuit", set it up.
                percent_unbiased_envs = 1.0
                unbiasd_envs = torch.rand(self.num_envs, device=self.device, requires_grad=False)
                unbiasd_envs[unbiasd_envs > percent_unbiased_envs] = 1
                unbiasd_envs[unbiasd_envs <= percent_unbiased_envs] = 0
                unbiasd_envs = unbiasd_envs.bool()

                # import pdb; pdb.set_trace()
                rel_pos_agent = self.robot_states[~unbiasd_envs, :3] - self.agent_pos[~unbiasd_envs, :3]
                rel_pos_agent_local = self.global_to_agent_frame(rel_pos_agent)
                self.agent_policy_schedule[~unbiasd_envs, :, 0] = torch.clip(rel_pos_agent_local[:, 0],
                                                                             min=self.command_ranges["agent_lin_vel_x"][0],
                                                                             max=self.command_ranges["agent_lin_vel_x"][1])
                self.agent_policy_schedule[~unbiasd_envs, :, 1] = torch.clip(rel_pos_agent_local[:, 1],
                                                                             min=self.command_ranges["agent_ang_vel_yaw"][0],
                                                                             max=self.command_ranges["agent_ang_vel_yaw"][1]) # TODO: THIS IS NOT CORRECT

        # populate the remainder of the episode with the last command
        mp_ids = rand_mp_seq[env_ids, num_chunks-1]  # shape: size of env_ids, with an index into the motion primitive per env
        curr_motion_primitives = self.motion_primitives_tensor[mp_ids]

        vx = curr_motion_primitives[:, 0].repeat(remainder).view(remainder, len(env_ids))
        vsteer = curr_motion_primitives[:, 1].repeat(remainder).view(remainder, len(env_ids))

        self.agent_policy_schedule[env_ids, counter:counter + remainder, 0] = torch.transpose(vx, 0, 1)
        self.agent_policy_schedule[env_ids, counter:counter + remainder, 1] = torch.transpose(vsteer, 0, 1)

        # print("==> Done!")

    def _gym_to_game_state(self, gym_xyz, gym_q):
        """Converts from gym representation of state to representation of state 
            that is compatible with the game-theory based state representation. 

        Args:
            gym_xyz (tensor): gym-based representation of state, shape (num_envs, 3)
            gym_q (tensor): gym-based quaternion, shape (num_envs, 4)
        Returns:
            game_xyth (tensor): game-compatible (x, y, yaw) state, shape (num_envs, 3)
        """
        _, _, yaw = get_euler_xyz(gym_q)
        return torch.cat((gym_xyz[:, 0:2], yaw.unsqueeze(-1)), dim=-1)

    def _update_game_state(self, env_ids, reset=False):
        self.game_x_agent[env_ids, :] = self._gym_to_game_state(self.agent_pos[env_ids, :].clone(), self.agent_states[env_ids, 3:7].clone()) 
        self.game_x_robot[env_ids, :] = self._gym_to_game_state(self.robot_states[env_ids, 0:3].clone(), self.base_quat[env_ids, :].clone()) 

        # process game state
        xrel_agent_np = self.game_helper.get_state_relative_evader_batched(self.game_x_agent[env_ids, :].detach().cpu().numpy(), 
                                                                            self.game_x_robot[env_ids, :].detach().cpu().numpy())
        xrel_robot_np = self.game_helper.get_state_relative_pursuer_batched(self.game_x_robot[env_ids, :].detach().cpu().numpy(), 
                                                                            self.game_x_agent[env_ids, :].detach().cpu().numpy())
        # reset the time index
        if reset:
            # ts_f1 = time()
            game_tidx_np, value_np = self.game_helper.find_policy_start_tidx_batched(xrel_robot_np, "pursuer")
            # te_f1 = time()
            self.game_tidx[env_ids, :] = torch.from_numpy(game_tidx_np).to(self.device)
            # te_f2 = time()
            # print("find_policy_start_tidx_batched: %f (s)", te_f1 - ts_f1)
            # print("    converting to torch: %f (s) ", te_f2 - te_f1)
            infeasible_cond = value_np[:, 0] > 1e-4
            infeasible_cond = infeasible_cond.nonzero()[0]
            if len(infeasible_cond) > 0:
                print("    ==> infeasible capture initial condition: x_agent = ", self.game_x_agent[infeasible_cond])
            
        self.game_xrel_agent[env_ids, :] = torch.from_numpy(xrel_agent_np).to(self.device)
        self.game_xrel_robot[env_ids, :] = torch.from_numpy(xrel_robot_np).to(self.device)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # resets the agent and robot states in the low-level simulator environment
        self.ll_env._reset_dofs(env_ids)
        self.ll_env._reset_root_states(env_ids)

        # reset low-level buffers
        self.ll_env.last_actions[env_ids] = 0.
        self.ll_env.last_dof_vel[env_ids] = 0.
        self.ll_env.feet_air_time[env_ids] = 0.
        self.ll_env.episode_length_buf[env_ids] = 0
        self.ll_env.reset_buf[env_ids] = 1

        # update the state variables
        self._update_agent_states()
        self.agent_heading[env_ids] = self.ll_env.init_agent_heading[env_ids].clone()

        # reset the optimal robot actions
        self.bc_actions_robot[env_ids, :] = 0.

        # reset agent observation buffers
        self.obs_buf_agent[env_ids, :] = -self.MAX_REL_POS
        self.obs_buf_agent[env_ids, -2:] = 0.
        # print("reset obs buf agent: ", self.obs_buf_agent[env_ids, :])

        # reset robot observation buffers
        reset_robot_obs = torch.zeros(len(env_ids), self.num_obs_robot, device=self.device, requires_grad=False)
        if self.num_obs_robot > 3:  # TODO: kind of a hack
            reset_robot_obs[:, self.pos_idxs_robot] = self.MAX_REL_POS # only the initial relative position is reset different
        self.obs_buf_robot[env_ids, :] = reset_robot_obs

        # reset the privileged observation buffers
        if self.num_privileged_obs_robot is not None:
            self.privileged_obs_buf_robot[env_ids, :] = 0
        if self.num_privileged_obs_agent is not None:
            self.privileged_obs_buf_agent[env_ids, :] = 0

        # reset the kalman filter
        if self.num_states_kf == 4:
            rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
            rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        elif self.num_states_kf == 3:
            rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
            rel_pos_local = self.global_to_robot_frame(rel_pos_global)[:, :2]
        else:
            print("[DecHighLevelGame: reset_idx] ERROR: self.num_states_kf", self.num_states_kf, " is not supported.")
            return

        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)
        rel_state = torch.cat((rel_pos_local, rel_yaw_local), dim=-1)

        # ---- choose right filter calls ---- #
        if self.filter_type == "kf":
            xhat0 = self.kf.sim_measurement(rel_state)
            self.kf.reset_xhat(env_ids=env_ids, xhat_val=xhat0[env_ids, :])
        # elif self.filter_type == "ukf":
        #     xhat0 = self.kf.sim_observations(states=rel_state)
        #     self.kf.reset_mean_cov(env_ids=env_ids, mean=xhat0[env_ids, :])
        #     xhat0_og = self.kf_og.sim_measurement(rel_state)
        #     self.kf_og.reset_xhat(env_ids=env_ids, xhat_val=xhat0[env_ids, :])
            # import pdb; pdb.set_trace()

        # reset the high-level buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.curr_episode_step[env_ids] = 0
        self.capture_buf[env_ids] = 0
        self.detected_buf_agent[env_ids, :] = 0
        self.detected_buf_robot[env_ids, :] = 0

        # simulated agent time-based state
        self.agent_cmd_curriculum_idx[env_ids, 0] = 0
        self.prev_agent_command[env_ids, :] = 0

        self.rel_state_hist[env_ids, :, :] = 0.
        self.robot_commands_hist[env_ids, :, :] = 0.
        self.robot_commands[env_ids, :] = 0.
        self.agent_state_hist[env_ids, :, :] = 0.

        # for debugging, initialize with current state
        agent_xyz = self.ll_env.root_states[self.ll_env.agent_indices[env_ids], :3].clone().unsqueeze(1) # xyz
        robot_xyz = self.ll_env.root_states[self.ll_env.robot_indices[env_ids], :3].clone().unsqueeze(1) # xyz
        agent_xyz = torch.repeat_interleave(agent_xyz, self.num_debug_hist_steps, dim=1)
        robot_xyz = torch.repeat_interleave(robot_xyz, self.num_debug_hist_steps, dim=1)
        self.debug_agent_state_hist[env_ids, :, 0] = agent_xyz[:, :, 0] # agent
        self.debug_agent_state_hist[env_ids, :, 1] = agent_xyz[:, :, 1]
        self.debug_agent_state_hist[env_ids, :, 2] = agent_xyz[:, :, 2]
        self.debug_robot_state_hist[env_ids, :, 0] = robot_xyz[:, :, 0] # robot
        self.debug_robot_state_hist[env_ids, :, 1] = robot_xyz[:, :, 1]
        self.debug_robot_state_hist[env_ids, :, 2] = robot_xyz[:, :, 2]

        # reset the robot goal if in the "navigation" interaction
        if self.interaction_type == 'nav':
            rand_angle_goal = torch.zeros(len(env_ids), 1, device=self.device, requires_grad=False).uniform_(self.min_ang_goal, self.max_ang_goal)
            rand_radius_goal = torch.zeros(len(env_ids), 1, device=self.device, requires_grad=False).uniform_(self.min_rad_goal, self.max_rad_goal)
            self.robot_goal[env_ids, :] = self.robot_states[env_ids, :3] + torch.cat((rand_radius_goal * torch.cos(rand_angle_goal),
                                                                                       rand_radius_goal * torch.sin(rand_angle_goal),
                                                                                      torch.zeros(len(env_ids), 1,
                                                                                                  device=self.device,
                                                                                                  requires_grad=False)
                                                                                       ), dim=-1)
            self.ll_env.robot_goal_loc = self.robot_goal.clone()
            self.goal_reach_buf[env_ids] = 0
            self.collision_buf[env_ids] = 0

            if self.agent_init_bias:
                # if we are biasing where the agent gets initialized...
                rand_offset = self.agent_init_bias_scale * torch.rand(len(env_ids), 1, device=self.device, requires_grad=False)
                rand_sign = torch.rand(len(env_ids), 1, device=self.device, requires_grad=False)
                rand_sign[rand_sign > 0.5] = 1
                rand_sign[rand_sign < 0.5] = -1

                # set agent starting angle to be in a similar direction as the goal
                rand_angle_agent = wrap_to_pi(rand_angle_goal + rand_sign * rand_offset)
                rand_radius_agent = torch.zeros(len(env_ids), 1, device=self.device, requires_grad=False).uniform_(self.min_rad, self.max_rad)
                self.agent_offset_xyz[env_ids] = torch.cat((rand_radius_agent * torch.cos(rand_angle_agent),
                                                   rand_radius_agent * torch.sin(rand_angle_agent),
                                                   torch.zeros(len(env_ids), 1, device=self.device, requires_grad=False)
                                                   ), dim=-1)
                self.ll_env.agent_offset_xyz = self.agent_offset_xyz
                self.ll_env.rand_angle = rand_angle_agent
                self.ll_env._reset_root_states(env_ids) # refresh low-level environment state
                self._update_agent_states() # refresh internal state variables

        if self.agent_dyn_type == 'integrator':

            # =========== INTEGRATOR -- Moving Agent Policy Setup ========= #
            # simulated agent information
            rand_vel_cmds = self.agent_command_scale * torch.rand(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
            rand_sign = torch.rand(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
            rand_sign[rand_sign > 0.5] = 1
            rand_sign[rand_sign < 0.5] = -1
            self.curr_agent_command[env_ids, :] = rand_sign[env_ids, :] * rand_vel_cmds[env_ids, :]

            # check the angle between the agent's velocity and the relative agent-robot position is making the agent go *towards* the robot
            rel_xy_global = self.agent_pos[env_ids, :2] - self.robot_states[env_ids, :2]
            agent_base_quat = self.ll_env.root_states[self.ll_env.agent_indices[env_ids], 3:7]
            padded_agent_cmd_local = torch.cat((self.curr_agent_command[env_ids, :], torch.zeros(len(env_ids), 1, device=self.device, requires_grad=False)), dim=-1)
            agent_cmd_global = quat_rotate_inverse(agent_base_quat, padded_agent_cmd_local)[:, :2]

            dot = torch.bmm(rel_xy_global.view(len(env_ids), 1, self.num_actions_agent), agent_cmd_global.view(len(env_ids), self.num_actions_agent, 1)).squeeze(-1)
            inner = torch.div(dot, (torch.norm(rel_xy_global, dim=-1) * torch.norm(agent_cmd_global, dim=-1)).view(len(env_ids), 1))
            ang = torch.acos(inner)

            # find the environments where the simulated agent is going *towards* the robot too much, and flip the sign of the commands
            geq = torch.ge(torch.abs(ang), 2.8) # ~160 degrees
            otw_bool = torch.any(geq, dim=1)
            moving_towards_robot_ids = otw_bool.nonzero(as_tuple=False).flatten()
            self.curr_agent_command[env_ids[moving_towards_robot_ids], :] *= -1.0

        elif self.agent_dyn_type == 'dubins':
            if self.agent_policy_type == 'complex_weaving':
                self._initialize_complex_weaving_command_agent(env_ids)
            elif self.agent_policy_type == 'learned_traj':
                self.agent_traj[env_ids, :, :] = 0
                # print("time-to-capture:")
                # print("   (sec): ", (self.max_episode_length - self.game_tidx)*self.hl_dt)
                # print("   (step): ", (self.max_episode_length - self.game_tidx))
            elif self.agent_policy_type == 'simple_weaving':
                # reset simulated agent behavior variables
                self.curr_agent_command[env_ids, :] = 0.
                self.last_turn_tstep[env_ids, 0] = 1
                self.last_straight_tstep[env_ids, 0] = 1
                self.turn_or_straight_idx[env_ids, 0] = 0   # 0 == turn, 1 == straight
                self.turn_direction_idx[env_ids, :] = torch.randint(0, 2,
                                                                    (len(env_ids), 1),
                                                                    dtype=torch.int32,
                                                                    device=self.device,
                                                                    requires_grad=False)    # 0 == turn left, 1 == turn right
                if not self.cfg.env.randomize_init_turn_dir:
                    # if not randomizing initial turn direction, then always turn left first
                    self.turn_direction_idx[env_ids, :] *= 0

                if self.cfg.env.randomize_ctrl_bounds:
                    self.agent_max_vx[env_ids, :] = torch.zeros(len(env_ids),
                                                    1,
                                                    device=self.device).uniform_(self.cfg.env.max_vx_range[0],
                                                                                self.cfg.env.max_vx_range[1])
                    self.agent_max_vang[env_ids, :] = torch.zeros(len(env_ids),
                                                      1,
                                                      device=self.device).uniform_(self.cfg.env.max_vang_range[0],
                                                                                  self.cfg.env.max_vang_range[1])
                else:
                    self.agent_max_vx[env_ids, :] = self.command_ranges["agent_lin_vel_x"][1] * torch.ones(len(env_ids),
                                                                                               1,
                                                                                               device=self.device)
                    self.agent_max_vang[env_ids, :] = self.command_ranges["agent_ang_vel_yaw"][1] * torch.ones(len(env_ids),
                                                                                                   1,
                                                                                                   device=self.device)

                    # choose a random turning frequency and straight frequency
                self.curr_turn_freq[env_ids, :] = torch.randint(self.cfg.env.agent_turn_freq[0],
                                                                self.cfg.env.agent_turn_freq[1]+1,
                                                                (len(env_ids), 1),
                                                                dtype=torch.int32,
                                                                device=self.device,
                                                                requires_grad=False)
                self.curr_straight_freq[env_ids, :] = torch.randint(self.cfg.env.agent_straight_freq[0],
                                                                    self.cfg.env.agent_straight_freq[0]+1,
                                                                    (len(env_ids), 1),
                                                                    dtype=torch.int32,
                                                                    device=self.device,
                                                                    requires_grad=False)

        else:
            print("[reset_idx()] ERROR: unsupported agent dynamics type.")
            return -1

         # reset last robot position to the reset pos
        self.last_robot_pos[env_ids, :] = self.ll_env.env_origins[env_ids, :3]

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums_agent.keys():
            self.extras["episode"]['rew_agent_' + key] = torch.mean(self.episode_sums_agent[key][env_ids]) / self.max_episode_length_s
            self.episode_sums_agent[key][env_ids] = 0.
        for key in self.episode_sums_robot.keys():
            self.extras["episode"]['rew_robot_' + key] = torch.mean(self.episode_sums_robot[key][env_ids]) / self.max_episode_length_s
            self.episode_sums_robot[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # send BC actions to the algorithm
        if self.cfg.env.send_BC_actions:
            self.extras["bc_actions"] = self.bc_actions_robot

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        actions_agent = torch.zeros(self.num_envs, self.num_actions_agent, device=self.device, requires_grad=False)
        actions_robot = torch.zeros(self.num_envs, self.num_actions_robot, device=self.device, requires_grad=False)
        if self.ll_env.cfg.terrain.measure_heights:
            self.ll_env.measured_heights = self.ll_env._get_heights() # TODO: need to do this for the observation buffer to match in size!
        obs_agent, obs_robot, privileged_obs_agent, privileged_obs_robot, _, _, _, _ = self.step(actions_agent, actions_robot)
        return obs_agent, obs_robot, privileged_obs_agent, privileged_obs_robot

    def get_rel_yaw_global_robot(self):
        """Returns relative angle between the robot's local yaw angle and
        the angle between the robot's base and the agent's base:
            i.e., angle_btw_bases - robot_yaw
        """
        # from robot's POV, get its sensing
        rel_pos = self.agent_pos[:, :3] - self.robot_states[:, :3]

        # get relative yaw between the agent's heading and the robot's heading (global)
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        rel_yaw_global = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]) - robot_yaw
        rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        rel_yaw_global = wrap_to_pi(rel_yaw_global)
        return rel_yaw_global

    def compute_reward_robot(self):
        """ Compute rewards for the robot
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward

            Args: ll_rews (torch.Tensor) of size num_envs containing low-level reward per environment
        """
        self.rew_buf_robot[:] = 0.
        for i in range(len(self.reward_functions_robot)):
            name = self.reward_names_robot[i]
            rew = self.reward_functions_robot[i]() * self.reward_scales_robot[name]
            self.rew_buf_robot += rew
            self.episode_sums_robot[name] += rew
            # print("[", name, "] rew :", rew)
        # sum together the low-level reward and the high-level reward
        # ll_rew_weight = 2.0
        # self.rew_buf_robot += ll_rew_weight * ll_rews
        if self.cfg.rewards_robot.only_positive_rewards:
            self.rew_buf_robot[:] = torch.clip(self.rew_buf_robot[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales_robot:
            # rew = self._reward_termination() * self.reward_scales_robot["termination"]
            if self.interaction_type == 'game':
                # capture bonus
                rew_term = self.capture_buf * self.reward_scales_robot["termination"] # TODO: THIS IS A HACK!!!
            elif self.interaction_type == 'nav':
                # goal reach bonus
                rew_term = self.goal_reach_buf * self.reward_scales_robot["termination"]
                # penalty for terminating episode because of collision
                rew_term += self.collision_buf * -1 * self.reward_scales_robot["termination"]
            else:
                print("[ERROR] invalid interaction type: ", self.interaction_type)
                return
            rew_other = 0. #torch.logical_xor(self.reset_buf, self.capture_buf) * -1 * self.reward_scales_robot["termination"]
            rew = rew_term + rew_other
            # rew = self.capture_buf * self.reward_scales_robot["termination"] # TODO: THIS IS A HACK!!!
            self.rew_buf_robot += rew
            self.episode_sums_robot["termination"] += rew

        # rew_name = "path_progress"
        # print("episode_sum_robot[", rew_name, "]: ", self.episode_sums_robot[rew_name])

    def compute_reward_agent(self):
        """ Compute rewards for the AGENT
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf_agent[:] = 0.
        for i in range(len(self.reward_functions_agent)):
            name = self.reward_names_agent[i]
            rew = self.reward_functions_agent[i]() * self.reward_scales_agent[name]
            self.rew_buf_agent += rew
            self.episode_sums_agent[name] += rew
        if self.cfg.rewards_agent.only_positive_rewards:
            self.rew_buf_agent[:] = torch.clip(self.rew_buf_agent[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales_agent:
            # rew = self._reward_termination() * self.reward_scales_agent["termination"]
            rew = self.capture_buf * self.reward_scales_agent["termination"]
            self.rew_buf_agent += rew
            self.episode_sums_agent["termination"] += rew
        # print("high level rewards + low-level rewards: ", self.rew_buf)

    def compute_privileged_observations_robot(self, command_robot, command_agent):
        if self.interaction_type == 'game':
            add_goal = False
        elif self.interaction_type == 'nav':
            add_goal = True
        else:
            print("[ERROR: compute_privileged_observations_robot()] unsupported interaction type: ", self.interaction_type)

        # FUTURE OBS: (x^t, x^{t+1:t+N}) or (x^t, goal, x^{t+1:t+N})
        self.compute_observations_RMA_predictions_robot(add_goal=add_goal,
                                                        add_noise=False,
                                                        privileged=True)  # FUTURE OBS: (x^t, x^{t+1:t+N})

    def compute_observations_robot(self, command_robot, command_agent):
        """ Computes observations of the robot
        """
        # === PHASE 1 === #
        if self.interaction_type == 'game':
            add_goal = False
        elif self.interaction_type == 'nav':
            add_goal = True
        else:
            print("[ERROR: compute_observations_robot()] unsupported interaction type: ", self.interaction_type)

        if self.robot_policy_type == 'reaction':
            # CURRENT STATE OBS: (x_rel)
            self.compute_observations_pos_angle_robot()
        elif self.robot_policy_type == 'estimation' or self.robot_policy_type == 'prediction_phase2':
            # HISTORY OBS: (x^t, x^{t-1:t-N}, uR^{t-1:t-N}) or (x^t, goal, x^{t-1:t-N}, uR^{t-1:t-N})
            self.compute_observations_RMA_history_robot(add_goal=add_goal)
            # HISTORY of Partial OBS: (x^t, x^{t-1:t-N}, uR^{t-1:t-N}) or (x^t, goal, x^{t-1:t-N}, uR^{t-1:t-N})
            # uncomment for baseline
            #self.compute_observations_RMA_history_KF_robot(command_robot,
            #                                               command_agent,
            #                                               limited_fov=True,
            #                                               sense_obstacles=False,
            #                                               observe_variance=False) 
        elif self.robot_policy_type == 'prediction_phase1':
            # FUTURE OBS: (x^t, x^{t+1:t+N}) or (x^t, goal, x^{t+1:t+N})
            self.compute_observations_RMA_predictions_robot(add_goal=add_goal,
                                                            add_noise=False,
                                                            privileged=False)
        elif self.robot_policy_type == 'po_prediction_phase2':
            if self.num_hist_steps > 0:
                self.compute_observations_RMA_history_KF_robot(command_robot,
                                                               command_agent,
                                                               limited_fov=True,
                                                               sense_obstacles=False,
                                                               observe_variance=True) 
            else:
                self.compute_observations_KF_robot(command_robot,
                                                   command_agent,
                                                   limited_fov=True,
                                                   sense_obstacles=False,
                                                   observe_variance=True) 
        else:
            print("[ERROR compute_observations_robot()] unsupported robot policy type: ", self.robot_policy_type)

     
        # print("[DecHighLevelGame] self.obs_buf_robot: ", self.obs_buf_robot)

    def compute_observations_pos_robot(self):
        """ Computes observations of the agent with full FOV in the GLOBAL coordinate frame
        """
        # from robot's POV, get the relative position to the agent
        rel_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        self.obs_buf_robot = rel_pos_xyz

    def compute_observations_pos_angle_robot(self):
        """ Computes relative position and angle w.r.t robot's base frame.
        """
        # from robot's POV, get the relative position to the agent
        rel_pos_global = self.agent_pos[:, :3] - self.robot_states[:, :3]
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)

        pos_scale = 0.1
        self.obs_buf_robot = torch.cat((rel_pos_local * pos_scale,
                                        rel_yaw_local), dim=-1)

        # if self.debug_viz:
            # if self.agent_dyn_type == 'dubins':
            #     if self.agent_policy_type == 'simple_weaving':
            #         # get future states of the agent (for visualization)
            #         future_rel_states_robot_frame, future_agent_states = \
            #             self._predict_simple_weaving_command_agent(pred_hor=self.num_pred_steps)
            #     elif self.agent_policy_type == 'complex_weaving':
            #         future_rel_states_robot_frame, future_agent_states = \
            #             self._predict_complex_weaving_command_agent(pred_hor=self.num_pred_steps)
            #     self.ll_env.agent_state_future = future_agent_states  # true future
            #
            # # update the visualization variables of agent history, ground-truth future, and predicted future
            # self.ll_env.agent_state_hist = self.agent_state_hist # true history
            #
            # #   For debug visualization: update agent state history
            # #   TODO: for now only using (x,y,z) components
            # self.agent_state_hist = torch.cat((self.agent_pos[:, :3].unsqueeze(1),
            #                                    self.agent_state_hist[:, 0:self.num_hist_steps - 1, :]),
            #                                   dim=1)

    def compute_observations_pos_angle_time_robot(self):
        """ Computes relative position and angle w.r.t robot's base frame.
        """
        # from robot's POV, get the relative position to the agent
        rel_pos_global = self.agent_pos[:, :3] - self.robot_states[:, :3]
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)

        pos_scale = 0.1
        ep_step_scale = 1 / self.max_episode_length
        self.obs_buf_robot = torch.cat((rel_pos_local * pos_scale,
                                        rel_yaw_local,
                                        self.curr_episode_step.unsqueeze(-1) * ep_step_scale), dim=-1)

    def compute_observations_full_obs_robot(self):
        """ Computes observations of the robot with partial observability.
            obs_buf is laid out as:

           [s1, sin(h1_g), cos(h1_g), sin(h1_l), cos(h1_l), d1, v1]

           where s1 = (px1, py1, pz1) is the relative position (of size 3)
                 h1_g = is the *global* relative yaw angle between robot yaw and COM of agent (of size 1)
                 h1_l = is the *local* relative yaw angle between robot yaw and agent yaw (of size 1)
                 d1 = is "detected" boolean which tracks if the agent was detected once (of size 1)
                 v1 = is "visible" boolean which tracks if the agent was visible in FOV (of size 1)
        """
        sense_rel_pos, sense_rel_yaw_global, _, detected_bool, visible_bool = self.robot_sense_agent(limited_fov=False)

        self.obs_buf_robot = torch.cat((sense_rel_pos * 0.1,
                                        sense_rel_yaw_global,
                                        detected_bool.long(),
                                        visible_bool.long(),
                                        ), dim=-1)
        
    def compute_observations_KF_robot(self, command_robot, command_agent,
                                            limited_fov=False, sense_obstacles=False, observe_variance=True):
        """ Computes observations of the robot using a Kalman Filter state estimate.
            obs_buf is laid out as:

            pi_R(x^t, uR^t-1, elapsed_t)     where the x is relative state in robot frame plus its covariance from a KF estimate
                                                                   uR^t-1 are past robot actions
                                                                   elapsed_t (optional) is the elapsed time
        """
        actions_kf_r = command_robot[:, :self.num_actions_kf_r]

        # ---- predict a priori state estimate  ---- #
        if self.filter_type == "kf":
            self.kf.predict(actions_kf_r)
            rel_state_a_priori = self.kf.xhat

        if self.num_states_kf == 4:
            rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
            rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        #elif self.num_states_kf == 3:
        #    rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
        #    rel_pos_local = self.global_to_robot_frame(rel_pos_global)[:, :2]
        else:
            print("[DecHighLevelGame: reset_idx] ERROR: self.num_states_kf", self.num_states_kf, " is not supported.")
            return

        # from robot's POV, get its sensing
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)
        rel_state = torch.cat((rel_pos_local, rel_yaw_local), dim=-1)

        # simulate getting a noisy measurement
        if limited_fov is True:
            half_fov = self.robot_full_fov / 2.

            # find environments where agent is visible
            leq = torch.le(torch.abs(rel_yaw_local), half_fov)
            fov_bool = torch.any(leq, dim=1)
            # fov_bool = torch.any(torch.abs(rel_yaw_local) <= half_fov, dim=1)
            visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        else:
            # if full FOV, then always get measurement and do update
            visible_env_ids = torch.arange(self.num_envs, device=self.device)
        
        # ---- perform Kalman update to only environments that can get measurements ---- #
        if self.filter_type == "kf":
            z = self.kf.sim_measurement(rel_state)
            self.kf.correct(z, env_ids=visible_env_ids)
            rel_state_a_posteriori = self.kf.xhat.clone()
            covariance_a_posteriori = self.kf.P_tensor.clone()

        # TODO: Hack for logging!
        if limited_fov is True:
            hidden_env_ids = (~fov_bool).nonzero(as_tuple=False).flatten()
            z[hidden_env_ids, :] = 0

        # pack observation buf (getting diagonal elements)
        P_flattened = torch.diagonal(covariance_a_posteriori, offset=0, dim1=1, dim2=2) 
        pos_scale = 0.1
        scaled_rel_state_a_posteriori = rel_state_a_posteriori.clone() 
        scaled_rel_state_a_posteriori[:, :-1] *= pos_scale # don't scale the angular dimension since it is already small
        
        # TODO: HACK!! Remove z from the observation by putting it to zero
        scaled_rel_state_a_posteriori[:, 2] = 0
        P_flattened[:, 2] = 0

        # Important part
        if observe_variance:
            current_state = torch.cat((scaled_rel_state_a_posteriori,
                                       P_flattened
                                       ), dim=-1)
        else:
            current_state = scaled_rel_state_a_posteriori
        
        state_dim = current_state.shape[1]

        self.true_obs_robot = torch.cat((rel_state,
                                        torch.zeros(self.num_envs, P_flattened.shape[1], device=self.device)
                                        ), dim=-1)
        
        self.obs_buf_robot = torch.cat((current_state,
                                        self.robot_commands),
                                        dim=-1)
        
        
    def compute_observations_RMA_history_KF_robot(self, command_robot, command_agent,
                                                  limited_fov=False, sense_obstacles=False, observe_variance=True):
        """ Computes an history of observations of the robot using a Kalman Filter state estimate.
            obs_buf is laid out as:

            pi_R(x^t, x^t-1:t-N, uR^t-1:t-N, elapsed_t)     where the x is relative state in robot frame plus its covariance from a KF estimate
                                                                   x^t-1:t-N is past relative state assuming robot is static
                                                                   uR^t-1:t-N are past robot actions
                                                                   elapsed_t (optional) is the elapsed time
        """
        actions_kf_r = command_robot[:, :self.num_actions_kf_r]

        # ---- predict a priori state estimate  ---- #
        if self.filter_type == "kf":
            self.kf.predict(actions_kf_r)
            rel_state_a_priori = self.kf.xhat

        if self.num_states_kf == 4:
            rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
            rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        #elif self.num_states_kf == 3:
        #    rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
        #    rel_pos_local = self.global_to_robot_frame(rel_pos_global)[:, :2]
        else:
            print("[DecHighLevelGame: reset_idx] ERROR: self.num_states_kf", self.num_states_kf, " is not supported.")
            return

        # from robot's POV, get its sensing
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)
        rel_state = torch.cat((rel_pos_local, rel_yaw_local), dim=-1)

        # simulate getting a noisy measurement
        if limited_fov is True:
            half_fov = self.robot_full_fov / 2.

            # find environments where agent is visible
            leq = torch.le(torch.abs(rel_yaw_local), half_fov)
            fov_bool = torch.any(leq, dim=1)
            # fov_bool = torch.any(torch.abs(rel_yaw_local) <= half_fov, dim=1)
            visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        else:
            # if full FOV, then always get measurement and do update
            visible_env_ids = torch.arange(self.num_envs, device=self.device)
        
        # ---- perform Kalman update to only environments that can get measurements ---- #
        if self.filter_type == "kf":
            z = self.kf.sim_measurement(rel_state)
            self.kf.correct(z, env_ids=visible_env_ids)
            rel_state_a_posteriori = self.kf.xhat.clone()
            covariance_a_posteriori = self.kf.P_tensor.clone()

        # TODO: Hack for logging!
        if limited_fov is True:
            hidden_env_ids = (~fov_bool).nonzero(as_tuple=False).flatten()
            z[hidden_env_ids, :] = 0

        # pack observation buf (getting diagonal elements)
        P_flattened = torch.diagonal(covariance_a_posteriori, offset=0, dim1=1, dim2=2) 
        pos_scale = 0.1
        scaled_rel_state_a_posteriori = rel_state_a_posteriori.clone() 
        scaled_rel_state_a_posteriori[:, :-1] *= pos_scale # don't scale the angular dimension since it is already small
        
        # TODO: HACK!! Remove z from the observation by putting it to zero
        scaled_rel_state_a_posteriori[:, 2] = 0
        P_flattened[:, 2] = 0

        # Important part
        if observe_variance:
            current_state = torch.cat((scaled_rel_state_a_posteriori,
                                       P_flattened
                                       ), dim=-1)
        else:
            current_state = scaled_rel_state_a_posteriori
        
        state_dim = current_state.shape[1]

        self.true_obs_robot = torch.cat((rel_state,
                                        torch.zeros(self.num_envs, P_flattened.shape[1], device=self.device)
                                        ), dim=-1)
        
        self.obs_buf_robot = torch.cat((current_state,
                                        self.rel_state_hist.reshape(self.num_envs, self.num_hist_steps*state_dim),
                                        self.robot_commands_hist.reshape(self.num_envs, self.num_hist_steps*self.num_actions_robot)),
                                        dim=-1)
        
        # if self.debug_viz:
        #     # get future states of the agent (for visualization)
        #     # future_rel_states_robot_frame, future_agent_states = \
        #     #     self._predict_weaving_command_agent(pred_hor=self.num_pred_steps)
        #     future_rel_states_robot_frame, future_agent_states = \
        #         self._predict_complex_weaving_command_agent(pred_hor=self.num_pred_steps)
        #     # update the visualization variables of agent history, ground-truth future, and predicted future
        #     self.ll_env.agent_state_hist = self.agent_state_hist # true history
        #     #self.ll_env.agent_state_preds = next_agent_pos_global.unsqueeze(1) # predicted 1-step future
        #     self.ll_env.agent_state_future = future_agent_states # true future

        # update the history vectors:
        #   rel state hist is of shape [num_envs x num_hist_steps x num_robot_states]
        self.rel_state_hist = torch.cat((current_state.unsqueeze(1),
                                        self.rel_state_hist[:, 0:self.num_hist_steps-1, :]), 
                                        dim=1)
        #   robot cmd hist is of shape [num_envs x num_hist_steps x num_robot_actions]
        self.robot_commands_hist = torch.cat((self.robot_commands.unsqueeze(1),
                                        self.robot_commands_hist[:, 0:self.num_hist_steps-1, :]), dim=1)
        #   For debug visualization: update agent state history 
        #   TODO: for now only using (x,y,z) components
        self.agent_state_hist = torch.cat((self.agent_pos[:, :3].unsqueeze(1),
                                        self.agent_state_hist[:, 0:self.num_hist_steps-1, :]),
                                        dim=1)

        # ------------------------------------------------------------------- #
        # =========================== Book Keeping ========================== #
        # ------------------------------------------------------------------- #
        if self.save_kf_data:
            # record real data
            if self.real_traj is None:
                self.real_traj = np.array([rel_state.cpu().numpy()])
            else:
                self.real_traj = np.append(self.real_traj, [rel_state.cpu().numpy()], axis=0)

            # record state estimate
            if self.est_traj is None:
                self.est_traj = np.array([rel_state_a_posteriori.cpu().numpy()])
            else:
                self.est_traj = np.append(self.est_traj, [rel_state_a_posteriori.cpu().numpy()], axis=0)

            # record state covariance
            if self.P_traj is None:
                self.P_traj = np.array([covariance_a_posteriori.cpu().numpy()])
            else:
                self.P_traj = np.append(self.P_traj, [covariance_a_posteriori.cpu().numpy()], axis=0)

            # record measurements
            if self.z_traj is None:
                self.z_traj = np.array([z.cpu().numpy()])
            else:
                self.z_traj = np.append(self.z_traj, [z.cpu().numpy()], axis=0)

            if self.data_save_tstep % self.data_save_interval == 0: 
                print("Saving state estimation trajectory at tstep ", self.data_save_tstep, "...")
                now = datetime.now()
                dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
                filename = "kf_data_" + dt_string + "_" + str(self.data_save_tstep) + ".pickle"
                data_dict = {"real_traj": self.real_traj,
                                "est_traj": self.est_traj, 
                                "z_traj": self.z_traj, 
                                "P_traj": self.P_traj, 
                                "kf": self.kf}
                with open(filename, 'wb') as handle:
                    pickle.dump(data_dict, handle)

            # advance timestep
            self.data_save_tstep += 1
        # ------------------------------------------------------------------- #

    def compute_observations_RMA_predictions_robot(self, add_goal=False, add_noise=False, privileged=False):
        """This function can handle observations like:
                pi_R(x^t, x^t+1:t+N)     where x is true relative state in robot base frame
                                               x^t+1:t+N is future rel state assuming uR == 0

            if add_goal is True, then the policy includes the goal state
                pi_R(x^t, goal, x^t+1:t+N, elapsed_t)
        """

        # get the future relative states (in robot coord frame) and agent states
        if self.agent_dyn_type == 'integrator':
            if self.agent_policy_type == 'learned_traj':
                future_rel_states_robot_frame, future_agent_states = \
                    self._predict_learned_traj_command_agent(pred_hor=self.num_pred_steps)
            else:
                future_rel_states_robot_frame, future_agent_states = \
                self._predict_random_straight_line_command_agent(pred_hor=self.num_pred_steps)
        elif self.agent_dyn_type == 'dubins':
            if self.agent_policy_type == 'simple_weaving':
                future_rel_states_robot_frame, future_agent_states = \
                    self._predict_simple_weaving_command_agent(pred_hor=self.num_pred_steps)
            elif self.agent_policy_type == 'complex_weaving':
                future_rel_states_robot_frame, future_agent_states = \
                    self._predict_complex_weaving_command_agent(pred_hor=self.num_pred_steps)
            elif self.agent_policy_type == 'learned_traj':
                future_rel_states_robot_frame, future_agent_states = \
                    self._predict_learned_traj_command_agent(pred_hor=self.num_pred_steps)

        if add_noise:
            # add noise to the predicted states
            mean_np = np.zeros((self.num_envs, self.num_robot_states))
            cov_tensor = 0.1 * torch.eye(self.num_robot_states, dtype=torch.float32, device=self.device)
            cov_batch_tensor = cov_tensor.repeat(self.num_envs, 1).reshape(self.num_envs, self.num_robot_states, self.num_robot_states)
            for tstep in range(self.num_pred_steps):
                v_tensor = self.sample_batch_mvn(mean_np, cov_batch_tensor.cpu().numpy(), self.num_envs)
                future_rel_states_robot_frame[:, tstep, :] += v_tensor

        # scale just the positional entries (xyz)
        pos_scale = 0.1
        future_rel_states_robot_frame_scaled = future_rel_states_robot_frame.clone()
        future_rel_states_robot_frame_scaled[:, :, :-1] *= pos_scale

        if add_goal:
            # compute rel (x, y, z theta) from robot to the goal.
            rel_pos_to_goal_global = (self.robot_goal[:, :3] - self.robot_states[:, :3]).clone()
            rel_pos_to_goal_local = self.global_to_robot_frame(rel_pos_to_goal_global)
            rel_yaw_to_goal_local = torch.atan2(rel_pos_to_goal_local[:, 1], rel_pos_to_goal_local[:, 0]).unsqueeze(-1)
            rel_state_to_goal = torch.cat((rel_pos_to_goal_local[:, :2], rel_yaw_to_goal_local), dim=-1)  # construct rel_x, rel_y, rel_theta
            rel_state_to_goal[:, :-1] *= pos_scale  # scale the position component

            # extract the current state from the predictions for packing obs buf
            curr_state_scaled = future_rel_states_robot_frame_scaled[:, 0, :]
            pred_states_scaled = future_rel_states_robot_frame_scaled[:, 1:, :]
            pred_states_scaled_flat = torch.flatten(pred_states_scaled, start_dim=1)
            final_obs = torch.cat((curr_state_scaled, rel_state_to_goal, pred_states_scaled_flat), dim=-1)
        else:
            # NOTE: pred_rel_state_robot_frame includes current state and future states:
            #       it's of shape: [num_envs x (N+1) x (x,y,z,theta)_rel]
            future_rel_state_robot_frame_flat = torch.flatten(future_rel_states_robot_frame_scaled, start_dim=1)
            final_obs = future_rel_state_robot_frame_flat

        # obs buf is laid out:
        #   (x^t, x^t+1:t+N) or (goal, x^t, x^t+1:t+N) if add_goal=True
        if privileged:
            self.privileged_obs_buf_robot = final_obs
        else:
            self.obs_buf_robot = final_obs

            # if self.debug_viz:
            #     # update the visualization variables of agent history, ground-truth future, and predicted future
            #     self.ll_env.agent_state_hist = self.agent_state_hist # true history
            #     self.ll_env.agent_state_preds = future_agent_states # predicted future agent states (is the true future)
            #     self.ll_env.agent_state_future = future_agent_states  # true future (is the predicted future)
            #     self.ll_env.rel_state_preds = future_rel_states_robot_frame # predicted future relative states
            #     self.ll_env.robot_state_at_pred_start = self.robot_states[:, :3].clone()
            #     self.ll_env.rel_state_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7].clone()

            #   For debug visualization: update agent state history
            #   TODO: for now only using (x,y,z) components
            self.agent_state_hist = torch.cat((self.agent_pos[:, :3].unsqueeze(1),
                                                self.agent_state_hist[:, 0:self.num_hist_steps-1, :]),
                                                dim=1)

    def compute_observations_RMA_predictions_and_goal_robot(self, privileged=False):
        """This function can handle observations like:
            pi_R(x^t, x^t+1:t+N, elapsed_t)     where x is true relative state in robot base frame
                                                     x^t+1:t+N is future rel state assuming uR == 0
                                                     elapsed_t (optional) is elapsed time in episode
        """

        # get the future relative states (in robot coord frame) and agent states
        if self.agent_dyn_type == 'integrator':
            future_rel_states_robot_frame, future_agent_states = \
                self._predict_random_straight_line_command_agent(pred_hor=self.num_pred_steps)
        elif self.agent_dyn_type == 'dubins':
            if self.agent_policy_type == 'simple_weaving':
                future_rel_states_robot_frame, future_agent_states = \
                    self._predict_simple_weaving_command_agent(pred_hor=self.num_pred_steps)
            elif self.agent_policy_type == 'complex_weaving':
                future_rel_states_robot_frame, future_agent_states = \
                    self._predict_complex_weaving_command_agent(pred_hor=self.num_pred_steps)

        # scale just the positional entries (xyz) of the future states and the (xy) goal
        pos_scale = 0.1
        future_rel_states_robot_frame_scaled = future_rel_states_robot_frame.clone()
        future_rel_states_robot_frame_scaled[:, :, :-1] *= pos_scale

        # import pdb; pdb.set_trace()
        # NOTE: pred_rel_state_robot_frame includes current state and future states:
        #       it's of shape: [num_envs x (N+1) x (x,y,z,theta)_rel]
        future_rel_state_robot_frame_flat = torch.flatten(future_rel_states_robot_frame_scaled, start_dim=1)

        rel_pos_to_goal_global = (self.robot_goal[:, :3] - self.robot_states[:, :3]).clone()
        rel_pos_to_goal_local = self.global_to_robot_frame(rel_pos_to_goal_global)
        rel_yaw_to_goal_local = torch.atan2(rel_pos_to_goal_local[:, 1], rel_pos_to_goal_local[:, 0]).unsqueeze(-1)
        rel_state_to_goal = torch.cat((rel_pos_to_goal_local[:, :2], rel_yaw_to_goal_local), dim=-1) # construct relx, rely, reltheta
        rel_state_to_goal[:, :-1] *= pos_scale # scale the position component
        final_obs = torch.cat((rel_state_to_goal, future_rel_state_robot_frame_flat), dim=-1)

        # obs buf is laid out:
        #   (x^t, x^t+1:t+N)
        if privileged:
            self.privileged_obs_buf_robot = final_obs
        else:
            self.obs_buf_robot = final_obs

            if self.debug_viz:
                # update the visualization variables of agent history, ground-truth future, and predicted future
                self.ll_env.agent_state_hist = self.agent_state_hist # true history
                self.ll_env.agent_state_preds = future_agent_states # predicted future agent states (is the true future)
                self.ll_env.agent_state_future = future_agent_states  # true future (is the predicted future)
                self.ll_env.rel_state_preds = future_rel_states_robot_frame # predicted future relative states
                self.ll_env.robot_state_at_pred_start = self.robot_states[:, :3].clone()
                self.ll_env.rel_state_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7].clone()

            #   For debug visualization: update agent state history
            #   TODO: for now only using (x,y,z) components
            self.agent_state_hist = torch.cat((self.agent_pos[:, :3].unsqueeze(1),
                                                self.agent_state_hist[:, 0:self.num_hist_steps-1, :]),
                                                dim=1)

    def compute_observations_RMA_history_robot(self, add_goal=False):
        """This function can handle observations like:
            pi_R(x^t, x^t-1:t-N, uR^t-1:t-N)     where the x is true relative state in robot frame
                                                   x^t-1:t-N is future relative state assuming robot is static
                                                   uR^t-1:t-N are past robot actions
        """
        rel_pos_global = (self.agent_pos[:, :3] - self.robot_states[:, :3]).clone()
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)

        # from robot's POV, get its sensing
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)
        rel_state = torch.cat((rel_pos_local, rel_yaw_local), dim=-1)

        state_dim = rel_state.shape[1]
        pos_scale = 0.1 # scale the relative position components to match the scale of other observations

        # scale the positional (xyz) components
        rel_state[:, :-1] *= pos_scale

        if add_goal:
            # compute rel (x, y, theta) from robot to the goal.
            rel_pos_to_goal_global = (self.robot_goal[:, :3] - self.robot_states[:, :3]).clone()
            rel_pos_to_goal_local = self.global_to_robot_frame(rel_pos_to_goal_global)
            rel_yaw_to_goal_local = torch.atan2(rel_pos_to_goal_local[:, 1], rel_pos_to_goal_local[:, 0]).unsqueeze(-1)
            rel_state_to_goal = torch.cat((rel_pos_to_goal_local[:, :2], rel_yaw_to_goal_local),
                                          dim=-1)  # construct rel_x, rel_y, rel_theta
            rel_state_to_goal[:, :-1] *= pos_scale  # scale the position component
            # obs buf is laid out:
            #   (x^t, rel_goal, x^t-1:x^t-N, u^t-1:u^t-N) in robot body frame
            self.obs_buf_robot = torch.cat((rel_state,
                                            rel_state_to_goal,
                                            self.rel_state_hist.reshape(self.num_envs, self.num_hist_steps * state_dim),
                                            self.robot_commands_hist.reshape(self.num_envs, self.num_hist_steps * self.num_actions_robot)),
                                           dim=-1)
        else:
            # obs buf is laid out:
            #   (x^t, x^t-1:x^t-N, u^t-1:u^t-N) in robot body frame
            self.obs_buf_robot = torch.cat((rel_state,
                                            self.rel_state_hist.reshape(self.num_envs, self.num_hist_steps*state_dim),
                                            self.robot_commands_hist.reshape(self.num_envs, self.num_hist_steps*self.num_actions_robot)),
                                            dim=-1)


        # if self.debug_viz:
        #     # get future states of the agent (for visualization)
        #     if self.agent_policy_type == 'simple_weaving':
        #         future_rel_states_robot_frame, future_agent_states = \
        #             self._predict_simple_weaving_command_agent(pred_hor=self.num_pred_steps)
        #     elif self.agent_policy_type == 'complex_weaving':
        #         future_rel_states_robot_frame, future_agent_states = \
        #             self._predict_complex_weaving_command_agent(pred_hor=self.num_pred_steps)
        #     elif self.agent_policy_type == 'learned_traj':
        #         future_rel_states_robot_frame, future_agent_states = \
        #             self._predict_learned_traj_command_agent(pred_hor=self.num_pred_steps)
        #     # update the visualization variables of agent history, ground-truth future, and predicted future
        #     self.ll_env.agent_state_hist = self.agent_state_hist # true history
        #     self.ll_env.agent_state_future = future_agent_states # true future

        # update the history vectors:
        #   rel state hist is of shape [num_envs x num_hist_steps x num_robot_states]
        self.rel_state_hist = torch.cat((rel_state.unsqueeze(1),
                                        self.rel_state_hist[:, 0:self.num_hist_steps-1, :]), 
                                        dim=1)
        #   robot cmd hist is of shape [num_envs x num_hist_steps x num_robot_actions]
        self.robot_commands_hist = torch.cat((self.robot_commands.unsqueeze(1),
                                        self.robot_commands_hist[:, 0:self.num_hist_steps-1, :]), dim=1)
        #   For debug visualization: update agent state history 
        #   TODO: for now only using (x,y,z) components
        self.agent_state_hist = torch.cat((self.agent_pos[:, :3].unsqueeze(1),
                                        self.agent_state_hist[:, 0:self.num_hist_steps-1, :]),
                                        dim=1)

    def compute_observations_state_hist_robot(self, limited_fov=False):
        """ Computes observations of the robot with a state history (instead of estimator)
                    obs_buf is laid out as:

                   [s^t, s^t-1, s^t-2, s^t-3,
                    v^t, v^t-1, v^t-2, v^t-3]

                   where s^i = (px^i, py^i, pz^i) is the relative position (of size 3) at timestep i
                         v^i = is "visible" boolean which tracks if the agent was visible in FOV (of size 1)
         """
        # from agent's POV, get its sensing
        sense_rel_pos, _, visible_bool = self.robot_sense_agent_simpler(limited_fov=limited_fov)

        old_sense_rel_pos = self.obs_buf_robot[:,
                            self.pos_idxs_robot[:-3]].clone()  # remove the oldest relative position observation
        old_visible_bool = self.obs_buf_robot[:,
                           self.visible_idxs_robot[:-1]].clone()  # remove the corresponding oldest visible bool

        # a new observation has the form: (rel_opponent_pos, rel_opponent_heading, detected_bool, visible_bool)
        self.obs_buf_robot = torch.cat((sense_rel_pos,
                                        old_sense_rel_pos,
                                        visible_bool.long(),
                                        old_visible_bool
                                        ), dim=-1)

    def compute_observations_agent(self):
        """ Computes observations of the agent
        """
        self.compute_observations_pos_angle_agent()


    def compute_observations_pos_angle_agent(self):
        """ Computes observations of the agent with full FOV
        """
        # from agent's POV, get the relative position to the robot
        rel_pos_global = self.robot_states[:, :3] - self.agent_pos[:, :3]
        rel_pos_local = self.global_to_agent_frame(rel_pos_global)
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1) 

        pos_scale = 0.1

        self.obs_buf_agent = torch.cat((rel_pos_global * pos_scale,
                                        rel_yaw_local), dim=-1)

    def get_observations_agent(self):
        # self.compute_observations_agent()
        return self.obs_buf_agent

    def get_observations_robot(self):
        # self.compute_observations_robot()
        return self.obs_buf_robot

    def get_privileged_observations_agent(self):
        return self.privileged_obs_buf_agent

    def get_privileged_observations_robot(self):
        return self.privileged_obs_buf_robot

    def robot_sense_agent_simpler(self, limited_fov):
        """
        Args:
            limited_fov (bool): true if limited fov, false if otherwise

        Returns: sensing information the POV of the robot. Returns 3 values:

            sense_rel_pos (torch.Tensor): [num_envs * 3] if is visible, contains the relative agent xyz-position;
                                                if not visible, copies the last relative agent xyz-position
            sense_rel_yaw_global (torch.Tensor): [num_envs * 1] if visible, contains global_yaw
                                                if not visible, copies the last global yaw
            visible_bool (torch.Tensor): [num_envs * 1] boolean if the agent has is currently visible

        TODO: I treat the FOV the same for horiz, vert, etc. For realsense camera:
                (HorizFOV)  = 64 degrees (~1.20428 rad)
                (DiagFOV)   = 72 degrees
                (VertFOV)   = 41 degrees
        """
        half_fov = self.robot_full_fov /2.

        # rel_agent_pos_xy = self.agent_pos[:, :2] - self.robot_states[:, :2]
        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        # robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        # _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        # robot_yaw = wrap_to_pi(robot_yaw)

        # relative yaw between the agent's heading and the robot's heading (global)
        # rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        # rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        # rel_yaw_global = wrap_to_pi(rel_yaw_global)

        rel_yaw_global = self.get_rel_yaw_global_robot()

        # pack the final sensing measurements
        sense_rel_pos_xyz = rel_agent_pos_xyz.clone()
        # sense_rel_pos_xy = rel_agent_pos_xy.clone()
        sense_rel_yaw_global = torch.cat((torch.cos(rel_yaw_global), torch.sin(rel_yaw_global)), dim=-1)

        # find environments where robot is visible
        fov_bool = torch.any(torch.abs(rel_yaw_global) <= half_fov, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        hidden_env_ids = (fov_bool == 0).nonzero(as_tuple=False).flatten()

        # mark envs where we robot was visible
        visible_bool = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
        visible_bool[visible_env_ids, :] = 1

        if limited_fov:
            # if we didn't sense the robot now, copy over the agent's previous sensed position as our new "measurement"
            sense_rel_pos_xyz[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.pos_idxs_robot[:3]].clone()
            # sense_rel_pos_xy[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.pos_idxs_robot[:2]].clone()
            # sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_global_idxs_robot[:2]].clone()

        # print("===================================")
        # print("agent_pos: ", self.agent_pos[:, :3])
        # print("robot_pos: ", self.robot_states[:, :3])
        # print("sense_rel_pos_xy: ", sense_rel_pos_xy)
        # print("rel_yaw_global: ", rel_yaw_global)
        # print("cos(rel_yaw_global) and sin(rel_yaw_global): ", sense_rel_yaw_global)
        # # # print("detected_buf_robot: ", self.detected_buf_robot)
        # print("visible_bool: ", visible_bool)
        # print("===================================")

        return sense_rel_pos_xyz, None, visible_bool

    def robot_sense_agent(self, limited_fov):
        """
        Args:
            limited_fov (bool): true if limited field of view, false if otherwise

        Returns: sensing information the POV of the robot. Returns five values:

            sense_rel_pos (torch.Tensor): [num_envs * 3] if is visible, contains the relative agent position;
                                                if not visible, copies the last relative agent position
            sense_rel_yaw_global (torch.Tensor): [num_envs * 2] if visible, contains cos(global_yaw) and sin(global_yaw);
                                                if not visible, copies the last cos / sin of last global yaw
            sense_rel_yaw_local (torch.Tensor): [num_envs * 2] if visible, contains cos(local_yaw) and sin(local_yaw);
                                                if not visible, copies the last cos / sin of last local yaw
            detected_buf_agent (torch.Tensor): [num_envs * 1] boolean if the agent has been detected before
            visible_bool (torch.Tensor): [num_envs * 1] boolean if the agent has is currently visible

        TODO: I treat the FOV the same for horiz, vert, etc. For realsense camera:
                (HorizFOV)  = 64 degrees (~1.20428 rad)
                (DiagFOV)   = 72 degrees
                (VertFOV)   = 41 degrees
        """
        half_fov = self.robot_full_fov /2.

        rel_agent_pos_xyz = self.agent_pos[:, :3] - self.robot_states[:, :3]
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        _, _, robot_yaw = get_euler_xyz(robot_base_quat)
        robot_yaw = wrap_to_pi(robot_yaw)

        # # relative yaw between the two agents (global)
        # rel_yaw_global = torch.atan2(rel_agent_pos_xyz[:, 1], rel_agent_pos_xyz[:, 0]) - robot_yaw
        # rel_yaw_global = rel_yaw_global.unsqueeze(-1)
        # rel_yaw_global = wrap_to_pi(rel_yaw_global)
        rel_yaw_global = self.get_rel_yaw_global_robot()

        # relative yaw between the agent's heading and the robot's heading (local)
        # robot_forward = quat_apply_yaw(self.ll_env.base_quat, self.ll_env.forward_vec)
        # robot_heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])
        rel_yaw_local = wrap_to_pi(robot_yaw - self.agent_heading) # make sure between -pi and pi
        rel_yaw_local = rel_yaw_local.unsqueeze(-1)

        # pack the final sensing measurements
        sense_rel_pos = rel_agent_pos_xyz.clone()
        sense_rel_yaw_global = torch.cat((torch.cos(rel_yaw_global), torch.sin(rel_yaw_global)), dim=-1)
        sense_rel_yaw_local = torch.cat((torch.cos(rel_yaw_local), torch.sin(rel_yaw_local)), dim=-1)

        # find environments where robot is visible
        fov_bool = torch.any(torch.abs(rel_yaw_global) <= half_fov, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        hidden_env_ids = (fov_bool == 0).nonzero(as_tuple=False).flatten()

        # mark envs where robot has been detected at least once
        self.detected_buf_robot[visible_env_ids, :] = 1

        # mark envs where we robot was visible
        visible_bool = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
        visible_bool[visible_env_ids, :] = 1

        if limited_fov:
            # if we didn't sense the robot now, copy over the agent's previous sensed position as our new "measurement"
            sense_rel_pos[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.pos_idxs_robot[:3]].clone()
            sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_global_idxs_robot[:2]].clone()
            # sense_rel_yaw_local[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_local_idxs[:2]].clone()

            # dist_outside_fov = torch.norm(wrap_to_pi(rel_yaw_global[hidden_env_ids] - half_fov), dim=-1)
            # zero_mean_pos = torch.zeros(len(dist_outside_fov), device=self.device, requires_grad=False)
            # zero_mean_ang = torch.zeros(len(dist_outside_fov), device=self.device, requires_grad=False)
            # noise_pos = torch.normal(mean=zero_mean_pos, std=dist_outside_fov).unsqueeze(-1)
            # noise_pos *= 0.1
            # noise_ang = torch.normal(mean=zero_mean_ang, std=dist_outside_fov).unsqueeze(-1)
            # noise_ang *= 0.01 # scale down the noise in the angular space

            # print("real real pos: ", sense_rel_pos[hidden_env_ids, :])
            # sense_rel_pos[hidden_env_ids, :] = torch.add(sense_rel_pos[hidden_env_ids, :], noise_pos)
            # print("*noisy* real pos: ", sense_rel_pos[hidden_env_ids, :])
            # sense_rel_yaw_global[hidden_env_ids, :] = torch.add(sense_rel_yaw_global[hidden_env_ids, :], noise_ang)  # TODO: should add this to angle directly, not sin/cos
            # sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_robot[hidden_env_ids][:, self.ang_global_idxs_robot[:2]].clone()

        # print("===================================")
        # print("agent_pos: ", self.agent_pos[:, :3])
        # print("robot_pos: ", self.robot_states[:, :3])
        # print("sense_rel_pos: ", sense_rel_pos)
        # print("rel_yaw_global: ", rel_yaw_global)
        # print("cos(rel_yaw_global) and sin(rel_yaw_global): ", sense_rel_yaw_global)
        # # # print("detected_buf_robot: ", self.detected_buf_robot)
        # print("visible_bool: ", visible_bool)
        # print("===================================")

        return sense_rel_pos, sense_rel_yaw_global, sense_rel_yaw_local, self.detected_buf_robot, visible_bool

    def agent_sense_robot(self):
        """
        Returns: sensing information the POV of the agent. Returns five values:

            sense_rel_pos (torch.Tensor): [num_envs * 3] if is visible, contains the relative robot position;
                                                if not visible, copies the last relative robot position
            sense_rel_yaw_global (torch.Tensor): [num_envs * 2] if visible, contains cos(global_yaw) and sin(global_yaw);
                                                if not visible, copies the last cos / sin of last global yaw
            sense_rel_yaw_local (torch.Tensor): [num_envs * 2] if visible, contains cos(local_yaw) and sin(local_yaw);
                                                if not visible, copies the last cos / sin of last local yaw
            detected_buf_agent (torch.Tensor): [num_envs * 1] boolean if the robot has been detected before
            visible_bool (torch.Tensor): [num_envs * 1] boolean if the robot has is currently visible
        """
        half_fov = 1.20428 / 2. # full FOV is ~64 degrees; same FOV as robot

        # relative position of robot w.r.t. agent
        rel_robot_pos = self.robot_states[:, :3] - self.agent_pos

        # angle between the agent's heading and the robot's COM (global)
        angle_btwn_agents_global = torch.atan2(rel_robot_pos[:, 1], rel_robot_pos[:, 0]) - self.agent_heading
        angle_btwn_agents_global = wrap_to_pi(angle_btwn_agents_global.unsqueeze(-1)) # make sure between -pi and pi

        # relative heading between the agent's heading and the robot's heading (local)
        robot_base_quat = self.ll_env.root_states[self.ll_env.robot_indices, 3:7]
        robot_forward = quat_apply_yaw(robot_base_quat, self.ll_env.forward_vec)
        robot_heading = torch.atan2(robot_forward[:, 1], robot_forward[:, 0])
        angle_btwn_heading_local = wrap_to_pi(robot_heading - self.agent_heading) # make sure between -pi and pi
        angle_btwn_heading_local = angle_btwn_heading_local.unsqueeze(-1)

        # pack the final sensing measurements
        sense_rel_pos = rel_robot_pos.clone()
        sense_rel_yaw_global = torch.cat((torch.cos(angle_btwn_agents_global),
                                              torch.sin(angle_btwn_agents_global)), dim=-1)
        sense_rel_yaw_local = torch.cat((torch.cos(angle_btwn_heading_local),
                                              torch.sin(angle_btwn_heading_local)), dim=-1)

        # find environments where robot is visible
        fov_bool = torch.any(torch.abs(angle_btwn_agents_global) <= half_fov, dim=1)
        visible_env_ids = fov_bool.nonzero(as_tuple=False).flatten()
        hidden_env_ids = (fov_bool == 0).nonzero(as_tuple=False).flatten()

        # mark envs where robot has been detected at least once
        self.detected_buf_agent[visible_env_ids, :] = 1

        # mark envs where we robot was visible
        visible_bool = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
        visible_bool[visible_env_ids, :] = 1

        # if we didn't sense the robot now, copy over the agent's previous sensed position as our new "measurement"
        sense_rel_pos[hidden_env_ids, :] = self.obs_buf_agent[hidden_env_ids][:, self.pos_idxs[9:]].clone()
        sense_rel_yaw_global[hidden_env_ids, :] = self.obs_buf_agent[hidden_env_ids][:, self.ang_global_idxs[6:]].clone()
        sense_rel_yaw_local[hidden_env_ids, :] = self.obs_buf_agent[hidden_env_ids][:, self.ang_local_idxs[6:]].clone()

        return sense_rel_pos, sense_rel_yaw_global, sense_rel_yaw_local, self.detected_buf_agent, visible_bool

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    # ------------- Helpers -------------- #

    def update_training_iter_num(self, iter):
        """Updates the internal variable keeping track of training iteration"""
        self.training_iter_num = iter

    def global_to_robot_frame(self, global_vec):
        """Transforms (xyz) global vector to robot's local coordinate frame."""
        robot_base_quat_global = self.ll_env.root_states[self.ll_env.robot_indices, 3:7].clone()

        # robot_base_quat_global = self.ll_env.root_states[self.ll_env.robot_indices, 3:7].clone()
        # local_vec = quat_rotate(robot_base_quat_global, rel_pos_global) # quat_rotate -- coordinate system is rotated with respect to the point (?)
        # local_yaw = torch.atan2(local_vec[:, 1], local_vec[:, 0]).unsqueeze(-1)
        # rel_state_B = torch.cat((local_vec, local_yaw), dim=-1)
        #
        # rel_pos_global_w = torch.cat((rel_pos_global, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)
        # local_vec2 = quat_mul(quat_conjugate(robot_base_quat_global), quat_mul(robot_base_quat_global, rel_pos_global_w))
        # local_yaw2 = torch.atan2(local_vec2[:, 1], local_vec2[:, 0]).unsqueeze(-1)
        # rel_state_C = torch.cat((local_vec2[:, :3], local_yaw2), dim=-1)

        # print("(quat_rotate_inverse) rel state robot frame: ", rel_state_A)
        # print("(quat_rotate) rel state robot frame: ", rel_state_B)
        # print("(eqn) rel state robot frame: ", rel_state_C)

        local_vec = quat_rotate_inverse(robot_base_quat_global, global_vec)
        return local_vec

    def global_to_agent_frame(self, global_vec):
        """Transforms (xyz) global vector to agent's local coordinate frame."""
        agent_base_quat_global = self.ll_env.root_states[self.ll_env.agent_indices, 3:7].clone()
        local_vec = quat_rotate_inverse(agent_base_quat_global, global_vec)
        return local_vec

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
        Y_tensor = torch.tensor(Y, dtype=torch.float32, device=self.device, requires_grad=False)
        return Y_tensor

    def expand_with_zeros_at_dim(self, tensor_in, dim):
        """Expands the tensor_in by putting zeros at the old "dim" location.
        """
        tensor_out = torch.cat((tensor_in[:, :dim],
                                torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False),
                                tensor_in[:, dim:]
                                ), dim=-1)
        return tensor_out

    # ------------- Callbacks --------------#

    def _update_agent_states(self):
        """Goes to the low-level environment and grabs the most recent simulator states for the agent and robot."""
        self.robot_states = self.ll_env.root_states[self.ll_env.robot_indices, :]
        self.base_quat = self.robot_states[:, 3:7]
        self.robot_heading = get_euler_xyz(self.base_quat)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 10:13])
        self.agent_pos = self.ll_env.root_states[self.ll_env.agent_indices, :3]
        self.agent_states = self.ll_env.root_states[self.ll_env.agent_indices, :]

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.robot_states = self.ll_env.root_states[self.ll_env.robot_indices, :]
        self.base_quat = self.robot_states[:, 3:7]
        self.robot_heading = get_euler_xyz(self.base_quat)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_states[:, 10:13])

        self.agent_pos = self.ll_env.root_states[self.ll_env.agent_indices, :3]
        self.agent_states = self.ll_env.root_states[self.ll_env.agent_indices, :]
        self.agent_heading = self.ll_env.init_agent_heading.clone()

    def _update_robot_sensing_curriculum(self, env_ids):
        """ Implements the curriculum for the robot's FOV.

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        if self.training_iter_num == self.curriculum_target_iters[self.fov_curr_idx]:
            self.fov_curr_idx += 1
            self.fov_curr_idx = min(self.fov_curr_idx, len(self.cfg.robot_sensing.fov_levels)-1)
            self.robot_full_fov[env_ids] = self.cfg.robot_sensing.fov_levels[self.fov_curr_idx]
            print("[DecHighLevelGame] in update_robot_sensing_curriculum():")
            print("                   training iter num: ", self.training_iter_num)
            print("                   robot FOV is: ", self.robot_full_fov[0])
            
    def _update_robot_cmds_curriculum(self):
        """ Implements the curriculum for the robot's control authority.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        
        
        if self.training_iter_num == self.cfg.commands.curriculum_iters[self.cmds_curr_idx]:
            self.cmds_curr_idx += 1
            self.cmds_curr_idx = min(self.cmds_curr_idx, len(self.cfg.commands.curriculum_iters)-1)
            self.command_ranges["lin_vel_x"][1] *=1.2
            self.command_ranges["lin_vel_x"][1] = np.minimum(self.command_ranges["lin_vel_x"][1],self.cfg.commands.ranges.max_lin_speed)
            print("[DecHighLevelGame] in update_robot_cmds_curriculum():")
            print("                   training iter num: ", self.training_iter_num)
            print("                   robot max lin speed is: ", self.command_ranges["lin_vel_x"][1])

    def _update_prey_curriculum(self, env_ids):
        """ Implements the curriculum for the prey initial position

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if self.training_iter_num == self.curriculum_target_iters[self.prey_curr_idx]:
            self.prey_curr_idx += 1 
            self.prey_curr_idx = min(self.prey_curr_idx, len(self.cfg.robot_sensing.prey_angs)-1)
            max_ang = self.cfg.robot_sensing.prey_angs[self.prey_curr_idx]
            min_ang = -max_ang
            self.rand_angle = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(min_ang, max_ang)
            rand_radius = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False).uniform_(self.min_rad, self.max_rad)
            self.agent_offset_xyz = torch.cat((rand_radius * torch.cos(self.rand_angle),
                                               rand_radius * torch.sin(self.rand_angle),
                                               torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
                                               ), dim=-1)
            self.ll_env.agent_offset_xyz = self.agent_offset_xyz
            self.ll_env.rand_angle = self.rand_angle

            print("[DecHighLevelGame] in _update_prey_curriculum():")
            print("                   training iter num: ", self.training_iter_num)
            print("                   prey ang is: ", max_ang)

    def _update_prey_policy_curriculum(self, env_ids):
        """ Implements the curriculum for the prey policy.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if self.training_iter_num == self.curriculum_target_iters[self.prey_policy_type_idx]:
            self.prey_policy_type_idx += 1
            self.prey_policy_type_idx = min(self.prey_policy_type_idx, len(self.cfg.robot_sensing.prey_policy) - 1)
            self.prey_policy_type = self.cfg.robot_sensing.prey_policy[self.prey_policy_type_idx]

            print("[DecHighLevelGame] in _update_prey_policy_curriculum():")
            print("                   training iter num: ", self.training_iter_num)
            print("                   prey policy type is: ", self.prey_policy_type)

    # def _update_obstacle_curriculum(self, env_ids):
    #     """ Implements the curriculum for the obstacle heights
    #
    #     Args:
    #         env_ids (List[int]): ids of environments being reset
    #     """
    #     if self.training_iter_num == self.curriculum_target_iters[self.prey_curr_idx]:

    def _prepare_reward_function_agent(self):
        """ Prepares a list of reward functions for the agent, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales_agent.keys()):
            scale = self.reward_scales_agent[key]
            if scale == 0:
                self.reward_scales_agent.pop(key)
            else:
                self.reward_scales_agent[key] *= self.hl_dt #self.ll_env.dt
        # prepare list of functions
        self.reward_functions_agent = []
        self.reward_names_agent = []
        for name, scale in self.reward_scales_agent.items():
            if name == "termination":
                continue
            self.reward_names_agent.append(name)
            name = '_reward_' + name
            print("[_prepare_reward_function_agent]: reward = ", name)
            print("[_prepare_reward_function_agent]:     scale: ", scale)
            self.reward_functions_agent.append(getattr(self, name))

        # reward episode sums
        self.episode_sums_agent = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales_agent.keys()}

    def _prepare_reward_function_robot(self):
        """ Prepares a list of reward functions for the robot, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales_robot.keys()):
            scale = self.reward_scales_robot[key]
            if scale == 0:
                self.reward_scales_robot.pop(key)
            else:
                self.reward_scales_robot[key] *= self.hl_dt #self.ll_env.dt
        # prepare list of functions
        self.reward_functions_robot = []
        self.reward_names_robot = []
        for name, scale in self.reward_scales_robot.items():
            if name == "termination":
                continue
            self.reward_names_robot.append(name)
            name = '_reward_' + name
            print("[_prepare_reward_function_robot]: reward = ", name)
            print("[_prepare_reward_function_robot]:     scale: ", scale)
            self.reward_functions_robot.append(getattr(self, name))

        # reward episode sums
        self.episode_sums_robot = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales_robot.keys()}

    def _parse_cfg(self, cfg):
        # self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales_robot = class_to_dict(self.cfg.rewards_robot.scales)
        self.reward_scales_agent = class_to_dict(self.cfg.rewards_agent.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.hl_dt) #self.ll_env.dt)
        # self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.ll_env.dt)

    # ------------ reward functions----------------
    def _reward_evasion(self):
        """Reward for evading"""
        rew = torch.square(torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1))
        return rew

    def _reward_pursuit(self):
        """Reward for pursuing"""
        rew = torch.square(torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1))
        return rew

    def _reward_robot_dist_to_goal(self):
        """Reward for robot getting closer to its goal.
            Smaller the closer to the goal.
            Bigger the further from the goal. 
        """
        rew = torch.square(torch.norm(self.robot_goal[:, :2] - self.robot_states[:, :2], p=2, dim=-1))
        return rew 

    def _reward_robot_collision_avoid(self):
        """Reward for robot avoiding collision with the other agent.
            Higher the closer to the other agent. Lower the farther. 
        """
        # dist = torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1)
        # max_d = torch.max(dist)
        # rew = torch.exp(-dist)
        rew = torch.square(torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1))
        far_enough_idxs = rew > self.collision_dist
        rew[far_enough_idxs] *= 0
        return rew 

    def _reward_time_elapsed(self):
        """Reward for episode length"""
        rew = self.curr_episode_step
        return rew

    def _reward_exp_pursuit(self):
        """Exponentially shaped pursuit"""
        dist = torch.norm(self.agent_pos[:, :2] - self.robot_states[:, :2], p=2, dim=-1)
        max_d = torch.max(dist)
        rew = torch.exp(-dist)
        return rew

    def _reward_robot_foveation(self):
        """Reward for the robot facing the agent"""
        rel_pos_global = self.agent_pos[:, :3] - self.robot_states[:, :3]
        rel_pos_local = self.global_to_robot_frame(rel_pos_global)
        rel_yaw_local = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0])

        # Exponential-type reward
        # rew = torch.exp(-torch.abs(rel_yaw_local))

        # "Relu"-type reward
        offset = np.pi / 3
        max_rew_val = 0.9

        slope = 1.0 #0.45
        diff_left = slope * rel_yaw_local + offset
        diff_right = slope * rel_yaw_local - offset

        # relu_left = torch.clamp(diff_left, min=0)   # max(0, diff_left)
        # relu_right = -torch.clamp(diff_right, max=0) # -min(0, diff_right)

        relu_left = diff_left  # max(0, diff_left)
        relu_right = -diff_right # -min(0, diff_right)

        val = torch.zeros_like(relu_left)
        val[rel_yaw_local > 0] = relu_right[rel_yaw_local > 0]
        val[rel_yaw_local <= 0] = relu_left[rel_yaw_local <= 0]
        rew = torch.clamp(val, max=max_rew_val) # min(min(a,b), max_rew_val)

        # ------------------------------------------------------------------- #
        # =========================== Book Keeping ========================== #
        # ------------------------------------------------------------------- #
        if self.save_rew_data:
            if self.fov_reward is None:
                self.fov_reward = rew.unsqueeze(0)
                self.fov_rel_yaw = rel_yaw_local.unsqueeze(0)
                self.ll_env_command_ang_vel = self.ll_env.commands[:, 2].unsqueeze(0)
            else:
                self.fov_reward = torch.cat((self.fov_reward, rew.unsqueeze(0)), dim=0)
                self.fov_rel_yaw = torch.cat((self.fov_rel_yaw, rel_yaw_local.unsqueeze(0)), dim=0)
                self.ll_env_command_ang_vel = torch.cat((self.ll_env_command_ang_vel, self.ll_env.commands[:, 2].unsqueeze(0)), dim=0)

            if self.rew_debug_tstep % self.rew_debug_interval == 0:
                print("[Foveation] Saving reward debugging information at ", self.rew_debug_tstep, "...")
                now = datetime.now()
                dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
                filename = "foveation_rew_debug_" + dt_string + "_" + str(self.rew_debug_tstep) + ".pickle"
                data_dict = {"fov_reward": self.fov_reward.cpu().numpy(),
                             "fov_rel_yaw": self.fov_rel_yaw.cpu().numpy(),
                             "ll_env_command_ang_vel": self.ll_env_command_ang_vel.cpu().numpy()}
                with open(filename, 'wb') as handle:
                    pickle.dump(data_dict, handle)

            self.rew_debug_tstep += 1

        # print("foveation reward:", rew)
        return rew

    def _reward_robot_ang_vel(self):
        """Reward for the robot's angular velocity"""
        rew = torch.norm(self.ll_env.commands[:, 2], p=2, dim=-1)
        return rew

    def _reward_command_norm(self):
        """Reward for the robot's command"""
        rew = torch.square(torch.norm(self.ll_env.commands, p=2, dim=-1))
        return rew

    def _reward_path_progress(self):
        """Reward for progress along the path that connects the initial robot state to the agent state.
        r(t) = proj(s(t)) - proj(s(t-1))
        """
        curr_robot_pos = self.robot_states[:, :3]
        robot_start_pos = self.ll_env.env_origins
        robot_goal_pos = self.agent_pos[:, :3] # TODO: NOTE THIS ASSUMES THE AGENT IS STATIC

        curr_progress_to_agent = self.proj_state_to_path(curr_robot_pos, robot_start_pos, robot_goal_pos)
        last_progress_to_agent = self.proj_state_to_path(self.last_robot_pos, robot_start_pos, robot_goal_pos)

        return curr_progress_to_agent - last_progress_to_agent

    def proj_state_to_path(self, curr_pos, start, goal):
        """Projects point curr_pos onto the line formed by start and goal points"""
        gs = goal - start
        cs = curr_pos - start
        gs_norm = torch.norm(gs, dim=-1)
        # gs_norm = torch.sum(gs * gs, dim=-1)
        progress = torch.sum(cs * gs, dim=-1) / gs_norm
        return progress
