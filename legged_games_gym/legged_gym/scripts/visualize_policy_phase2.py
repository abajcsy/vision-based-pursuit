# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import pdb

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_dec_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import pandas as pd 

from collections import deque
import statistics
import random
import pickle

from isaacgym import gymapi


def play_pursuit_evasion_game(args):
    # set the seed
    torch.manual_seed(0)
    random.seed(0)
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # load policies of agent and robot
    train_cfg.runner.load_run_robot = 'po_p2_complexWeave_lstm'
    train_cfg.runner.load_run_agent = ''
    
    evol_checkpoint = 0
    checkpoint_number = 1600
    train_cfg.runner.resume_robot = True # pursuer
    train_cfg.runner.resume_agent = False # evader
    
    # override some parameters for testing
    max_num_envs = 5
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max_num_envs)
    env_cfg.commands.use_joypad = False
    env_cfg.commands.record_joystick = False
    
    if env_cfg.commands.record_joystick:
        human_name = input("Please enter your name: ")
    
    # # prepare environment
    print("[play_pursuit evasion_game] making environment...")
    env, _ = task_registry.make_dec_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[play_pursuit_evasion_game] getting observations for both agents..")
    obs_agent = env.get_observations_agent()
    privileged_obs_robot = env.get_privileged_observations_robot()
    obs_robot = env.get_observations_robot()

    # setup for visualization time
    train_cfg.policy.use_estimator = True
    train_cfg.policy.use_privilege_enc = False
    train_cfg.policy.eval_time = True

    train_cfg.runner.learn_checkpoint_robot = checkpoint_number 
    train_cfg.runner.evol_checkpoint_robot = evol_checkpoint  

    dagger_runner, train_cfg = task_registry.make_dagger_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    action_policy_robot = dagger_runner.get_estimator_inference_policy(device=env.device)
    if train_cfg.runner.resume_agent:
        policy_agent = dagger_runner.get_inference_policy(agent_id=0, device=env.device)

    # camera info.
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # if True, computes stats for one episode: episode length and reward
    logging = False
    # LOGGING
    ep_infos = []
    rewbuffer_robot = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum_robot = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    
    
    if env_cfg.env.robot_policy_type == 'prediction_phase2':
        policy = dagger_runner.alg
    elif env_cfg.env.robot_policy_type == 'po_prediction_phase2':
        policy = dagger_runner.alg_student
    else:
        raise IOError("This Script Can only be used for phase 2 policies")
    
    # Get the estimator information
    h = torch.zeros((policy.actor_critic.estimator.num_layers, env.num_envs,
                     policy.actor_critic.estimator.hidden_size), device=env.device, requires_grad=True)
    c = torch.zeros((policy.actor_critic.estimator.num_layers, env.num_envs,
                     policy.actor_critic.estimator.hidden_size), device=env.device, requires_grad=True)
    hidden_state = (h, c)
    mask = torch.ones((env.num_envs,), device=env.device)
    
    # record the last robot state before simulating physics (get this out)
    if env_cfg.commands.record_joystick:
        joystick_actions_buf = []
        robot_states_buf = []
        agent_states_buf = []
        rollout_idx = 0
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                                'dec_high_level_game', train_cfg.runner.load_run,
                                human_name)
        os.mkdir(log_root)

    coeff = 10 # number of runs
    if logging:
        coeff = 1

    for i in range(coeff * int(env.max_episode_length)):
        if logging:
            print("Iter ", i, "  /  ", int(env.max_episode_length))
        
        hidden_state = (torch.einsum("ijk,j->ijk", hidden_state[0], mask),
                        torch.einsum("ijk,j->ijk", hidden_state[1], mask))    

        # get the estimator latent
        with torch.no_grad():
            estimator_obs = obs_robot.clone().unsqueeze(0)
            zhat, hidden_state = policy.actor_critic.estimate_latent_lstm(estimator_obs, hidden_state)
            actions_robot = action_policy_robot(obs_robot.detach(), zhat[0].detach())
            if train_cfg.runner.resume_agent:
                actions_agent = policy_agent(obs_agent.detach())
            else:
                actions_agent = torch.zeros(env.num_envs, env.num_actions_agent, device=env.device, requires_grad=False)

        # spoof agent actions, since they are overridden anyway.
        obs_agent, obs_robot , _, privileged_obs_robot, rews_agent, rews_robot, dones, infos = env.step(actions_agent.detach(), actions_robot.detach())

        # Time to save
        if env_cfg.commands.record_joystick:
            joystick_actions = infos['joystick_action']
            joystick_actions_buf.append(joystick_actions)
            agent_states = infos['init_agent_state']
            agent_states_buf.append(agent_states)
            robot_states = infos['init_robot_pos']
            robot_states_buf.append(robot_states)
            if dones[0]:
                episode_joy_actions = torch.vstack(joystick_actions_buf).cpu().numpy()
                episode_agent_states = torch.vstack(agent_states_buf).cpu().numpy()
                episode_robot_states = torch.vstack(robot_states_buf).cpu().numpy()
                save_dict = {'joy_actions' : episode_joy_actions,
                             'robot_pos' : episode_robot_states,
                             'episode_joy_state': episode_agent_states}
                save_path = os.path.join(log_root, "rollout_{}.pickle".format(rollout_idx))
                with open(save_path, 'wb') as handle:
                    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                rollout_idx += 1
                # Save
                joystick_actions_buf = []
                robot_states_buf = []
                agent_states_buf = []
        
        # mask out finished envs
        mask = ~dones

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # Book keeping
        if logging:
            if 'episode' in infos:
                ep_infos.append(infos['episode'])
                cur_reward_sum_robot += rews_robot
            cur_episode_length += 1
            new_ids = (dones > 0).nonzero(as_tuple=False)
            rewbuffer_robot.extend(cur_reward_sum_robot[new_ids][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
            cur_reward_sum_robot[new_ids] = 0
            cur_episode_length[new_ids] = 0

    if logging:
        log_str = f"""{'Mean reward (robot):':>{35}} {statistics.mean(rewbuffer_robot):.2f}\n"""
        log_str += f"""{'Mean episode length:':>{35}} {statistics.mean(lenbuffer):.2f}\n"""
        print(log_str)

if __name__ == '__main__':
    args = get_dec_args()
    play_pursuit_evasion_game(args)
