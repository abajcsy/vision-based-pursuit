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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_dec_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymapi

import pickle
from datetime import datetime

from collections import deque
import statistics
import random

def collect_data(args):
    # set the seed
    torch.manual_seed(0)
    random.seed(0)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    max_num_envs = 500
    env_cfg.env.num_envs = max_num_envs # min(env_cfg.env.num_envs, max_num_envs)
    env_cfg.env.debug_viz = False #True
    # env_cfg.env.capture_dist = 0 #THIS IS NEVER TRUE SO THEY WILL NEVER BE CAPTURED

    # prepare environment
    print("[collect_data] making environment...")
    env, _ = task_registry.make_dec_env(name=args.task, args=args, env_cfg=env_cfg)
    print("[collect_data] getting observations for both agents..")
    obs_agent = env.get_observations_agent()
    obs_robot = env.get_observations_robot()

    # load policies of agent and robot
    evol_checkpoint = 0
    learn_checkpoint = 5600

    # TEACHER POLICY DESIGN ABLATIONS
    # train_cfg.runner.load_run_robot = 'reactive_policy'             # REACTION w/ heuristic opponent (simple weaving)?; 1600
    # train_cfg.runner.load_run_robot = 'estimation_policy'             # ESTIMATION w/ heuristic opponent (simple weaving)?; 1600
    # train_cfg.runner.load_run_robot = 'p2_simpleWeave_randEverything_lstm'             # PREDICTION w/ heuristic opponent (simple weaving); 1600

    # TEACHER: FULLY OBSERVABLE EXPERIMENTS
    # train_cfg.runner.load_run_robot = 'p2_complexWeave_lstm'              # HEURISTIC opponent (complex weaving); 1600
    # train_cfg.runner.load_run_robot = 'p2_marl_lstm'                     # MARL opponent; 1600

    # STUDENT:PARTIALLY OBSERVABLE EXPERIMENTS
    # train_cfg.runner.load_run_robot = 'po_p2_complexWeave_lstm'                     # heuristic (Dubins++) teacher policy; 1600
    train_cfg.runner.load_run_robot = 'po_p2_marl_lstm'                           # MARL opponent teacher policy; 5600
    # train_cfg.runner.load_run_robot = 'May29_20-32-51_po_game_theory_teacher'     # game theory teacher policy; 2600

    # only load the agent policy if it was learned via MARL
    if env_cfg.env.agent_policy_type == "learned_traj":
        train_cfg.runner.resume_agent = True 
        train_cfg.runner.learn_checkpoint_agent = 5600 
        train_cfg.runner.evol_checkpoint_agent = evol_checkpoint
        train_cfg.runner.load_run_agent = 'po_p2_marl_lstm' #train_cfg.runner.load_run_robot # NOTE: assumes that the agent policy lives in same place as robot policy 
    else: 
         train_cfg.runner.resume_agent = False

    # load the robot policy unless using pre-computed game theory policy
    if env_cfg.env.robot_policy_type == "game_theory":
        train_cfg.runner.resume_robot = False
    else:
        train_cfg.runner.resume_robot = True
        train_cfg.runner.learn_checkpoint_robot = learn_checkpoint # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT
        train_cfg.runner.evol_checkpoint_robot = evol_checkpoint  # TODO: WITHOUT THIS IT GRABS WRONG CHECKPOINT
   
    # if True, then executes policy
    # if False, then robot is forced to stand in place
    run_robot_policy = True
    save_data = True #False

    if env_cfg.env.robot_policy_type == 'prediction_phase2':# or env_cfg.env.robot_policy_type == 'prediction_phase1':
        # setup what modules the policy is using
        train_cfg.policy.use_estimator = True
        train_cfg.policy.use_privilege_enc = True
        train_cfg.policy.eval_time = True
        dagger_runner, train_cfg = task_registry.make_dagger_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy_robot = dagger_runner.get_estimator_inference_policy(device=env.device)
        runner_alg = dagger_runner.alg
        if train_cfg.runner.resume_agent:
            policy_agent = dagger_runner.get_inference_policy(agent_id=0, device=env.device)
    elif env_cfg.env.robot_policy_type == 'po_prediction_phase2':
        # setup what modules the policy is using
        train_cfg.policy.use_estimator = True
        train_cfg.policy.use_privilege_enc = False
        train_cfg.policy.eval_time = True
        dagger_runner, train_cfg = task_registry.make_dagger_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy_robot =  dagger_runner.get_estimator_inference_policy(device=env.device)
        runner_alg = dagger_runner.alg_student
        if train_cfg.runner.resume_agent:
            policy_agent = dagger_runner.get_inference_policy(agent_id=0, device=env.device)
    else:
        dec_ppo_runner, train_cfg = task_registry.make_dec_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy_robot = dec_ppo_runner.get_inference_policy(agent_id=1, device=env.device)
        runner_alg = dec_ppo_runner.alg_robot
        if train_cfg.runner.resume_agent:
            policy_agent = dec_ppo_runner.get_inference_policy(agent_id=0, device=env.device)


    if env_cfg.env.robot_policy_type == 'prediction_phase2' or env_cfg.env.robot_policy_type == 'po_prediction_phase2':
        # Get the estimator information
        h = torch.zeros((runner_alg.actor_critic.estimator.num_layers, env.num_envs,
                         runner_alg.actor_critic.estimator.hidden_size), device=env.device, requires_grad=True)
        c = torch.zeros((runner_alg.actor_critic.estimator.num_layers, env.num_envs,
                         runner_alg.actor_critic.estimator.hidden_size), device=env.device, requires_grad=True)
        hidden_state = (h, c)


    # mask for keeping track of environments that are not done
    mask = torch.ones((env.num_envs,), device=env.device)

    # camera info.
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # data saving info.
    agent_state_data = None
    robot_state_data = None
    rel_pos_data = None
    rel_state_robot_frame_data = None
    theoretic_capture_time = None # just for game theory agent-robot
    step_scale = 1 #100
    num_sim_steps = step_scale * int(env_cfg.env.episode_length_s / env_cfg.env.robot_hl_dt)

    # LOGGING
    ep_infos = []
    rewbuffer_robot = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum_robot = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    capture_buf = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # do a single episode rollout for each environment
    for i in range(num_sim_steps):
        print("Collecting data at tstep: ", i, " / ", num_sim_steps, "...")

        # save the current agent state.
        agent_state = env.ll_env.root_states[env.ll_env.agent_indices, :3]
        robot_state = env.ll_env.root_states[env.ll_env.robot_indices, :3]
        rel_pos = agent_state - robot_state
        rel_pos_local = env.global_to_robot_frame(rel_pos)
        rel_heading = torch.atan2(rel_pos_local[:, 1], rel_pos_local[:, 0]).unsqueeze(-1)
        rel_state = torch.cat((rel_pos, rel_heading), dim=-1)

        # ===== LOGGING ===== #
        if rel_state_robot_frame_data is None:
            agent_state_data = np.array(agent_state.unsqueeze(1).cpu().numpy())
            rel_pos_data = np.array(rel_pos.unsqueeze(1).cpu().numpy())
            rel_state_robot_frame_data = np.array(rel_state.unsqueeze(1).cpu().numpy())
            robot_state_data = np.array(robot_state.unsqueeze(1).cpu().numpy())

            if env_cfg.env.agent_policy_type == 'game_theory' and env_cfg.env.robot_policy_type == 'game_theory':
                print("     Saving theoretic capture time...")
                theoretic_capture_time = env.max_episode_length - env.game_tidx # theoretic number of timesteps
        else:
            # TODO: if the environment is done, then mask it out
            # import pdb; pdb.set_trace()
            captured_ids = (capture_buf > 0).nonzero(as_tuple=False).cpu()
            agent_state[captured_ids, :] *= 0
            rel_pos[captured_ids, :] *= 0
            rel_state[captured_ids, :] *= 0
            robot_state[captured_ids, :] *= 0

            agent_state_data = np.append(agent_state_data, agent_state.unsqueeze(1).cpu().numpy(), axis=1)
            rel_pos_data = np.append(rel_pos_data, rel_pos.unsqueeze(1).cpu().numpy(), axis=1)
            rel_state_robot_frame_data = np.append(rel_state_robot_frame_data, rel_state.unsqueeze(1).cpu().numpy(), axis=1)
            robot_state_data = np.append(robot_state_data, robot_state.unsqueeze(1).cpu().numpy(), axis=1)
        # =================== #

        # if the robot is using an LSTM predictor, then set it up; otherwise, query the appropriate policy.
        if env_cfg.env.robot_policy_type == 'prediction_phase2' or env_cfg.env.robot_policy_type == 'po_prediction_phase2':
            hidden_state = (torch.einsum("ijk,j->ijk", hidden_state[0], mask),
                            torch.einsum("ijk,j->ijk", hidden_state[1], mask))

            # get the estimator latent
            estimator_obs = obs_robot.clone().unsqueeze(0)
            zhat, hidden_state = runner_alg.actor_critic.estimate_latent_lstm(estimator_obs, hidden_state)
            actions_robot = policy_robot(obs_robot.detach(), zhat[0].detach())
        elif env_cfg.env.robot_policy_type == 'game_theory':
            actions_robot = torch.zeros(env.num_envs, env.num_actions_robot, device=env.device, requires_grad=False)
        else:
            actions_robot = policy_robot(obs_robot.detach())

        # if the agent is using a learned policy, query it; otherwise it doesn't matter.
        if env.agent_policy_type == 'learned_traj':
            actions_agent = policy_agent(obs_agent.detach())
            actions_agent = actions_agent.detach()
        else:
            # Note: in all scenarios but the "learned traj" (i.e., MARL agent), the agent actions get spoofed in the step() function
            actions_agent = torch.zeros(env.num_envs, env.num_actions_agent, device=env.device, requires_grad=False)
            

        # zero out robot controls if running sanity check
        if not run_robot_policy:
            actions_robot *= 0

        # step the environment
        obs_agent, obs_robot , _, _, rews_agent, rews_robot, dones, infos = env.step(actions_agent, actions_robot.detach())

        # keep track of which environments had captures so far
        capture_buf |= dones

        # mask out finished envs
        mask = ~dones

        # book keeping about episode length
        if 'episode' in infos:
            ep_infos.append(infos['episode'])
            cur_reward_sum_robot += rews_robot

        # check which environments are done during THIS STEP
        dones_ids = (dones > 0).nonzero(as_tuple=False)
        # which environments are NOT done 
        not_dones_ids = (dones == 0).nonzero(as_tuple=False)

        # which environments have NOT YET had captures
        not_captured_ids = (capture_buf == 0).nonzero(as_tuple=False)
        # increment the episode length counter for NOT captured environments.
        cur_episode_length[not_captured_ids] += 1
        print("cur_episode_length: ", cur_episode_length)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

    if save_data:
        print("DONE! Saving data...")
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H-%M-%S")
        f = "Agent_" + str(env_cfg.env.agent_policy_type) +  "_Robot_" + str(env_cfg.env.robot_policy_type) + "_Policy" + str(train_cfg.runner.load_run_robot) + ".pickle"
        filename = LEGGED_GYM_ROOT_DIR + "/legged_gym/sim_experiments/data/" + f
        data_dict = {"agent_state_data": agent_state_data,
                     "robot_state_data": robot_state_data,
                     "rel_pos_data": rel_pos_data,
                     "rel_state_robot_frame_data": rel_state_robot_frame_data,
                     "dt": env_cfg.env.robot_hl_dt,
                     "traj_length_s": env_cfg.env.episode_length_s,
                     "cur_episode_length": cur_episode_length.detach().cpu().numpy(),
                     "capture_buf": capture_buf.detach().cpu().numpy()}
        if theoretic_capture_time is not None:
            data_dict["theoretic_capture_time"] = theoretic_capture_time.detach().cpu().numpy()

        with open(filename, 'wb') as handle:
            pickle.dump(data_dict, handle)
    else:
        print("DONE! ")

if __name__ == '__main__':
    args = get_dec_args()
    collect_data(args)
