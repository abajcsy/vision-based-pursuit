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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_dec_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from collections import deque
import statistics
import random

from isaacgym import gymapi


def play_dec_game(args):
    # set the seed
    torch.manual_seed(0)
    random.seed(0)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    max_num_envs = 1 #800
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, max_num_envs)
    env_cfg.env.debug_viz = False
    env_cfg.commands.use_joypad = False

    # # prepare environment
    print("[play_dec_game] making environment...")
    env, _ = task_registry.make_dec_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[play_dec_game] getting observations for both agents..")
    obs_agent = env.get_observations_agent()
    obs_robot = env.get_observations_robot()

    logging = False

    # setup what modules the policy is using
    train_cfg.policy.eval_time = True

    # load policies of agent and robot
    evol_checkpoint = 0
    checkpoint_number = 1600
    train_cfg.runner.resume_robot = True
    train_cfg.runner.load_run_robot = 'phase_1_policy_v3' 
    train_cfg.runner.learn_checkpoint_robot = checkpoint_number  # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT
    train_cfg.runner.evol_checkpoint_robot = evol_checkpoint  # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT

    if env_cfg.env.agent_policy_type == 'learned_traj':
        train_cfg.runner.resume_agent = True 
        train_cfg.runner.learn_checkpoint_agent = 5600
        train_cfg.runner.evol_checkpoint_agent = evol_checkpoint
        train_cfg.runner.load_run_agent = 'po_p2_marl_lstm'
    else:
        train_cfg.runner.resume_agent = False

    dec_ppo_runner, train_cfg = task_registry.make_dec_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

    # bind to the correct policy function call
    policy_agent = dec_ppo_runner.get_inference_policy(agent_id=0, device=env.device)
    policy_robot = dec_ppo_runner.get_inference_policy(agent_id=1, device=env.device)

    # camera info.
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # LOGGING
    ep_infos = []
    rewbuffer_robot = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum_robot = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    coeff = 10
    if logging:
        coeff = 1

    # simulate an episode
    for i in range(coeff * int(env.max_episode_length)):
        if logging:
            print("Iter ", i, "  /  ", int(env.max_episode_length))

        # print("[play_dec_game] current obs_robot: ", obs_robot.detach())
        actions_robot = policy_robot(obs_robot.detach())
        actions_agent = policy_agent(obs_agent.detach())
        
        obs_agent, obs_robot , _, _, rews_agent, rews_robot, dones, infos = env.step(actions_agent.detach(), actions_robot.detach())

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
    play_dec_game(args)