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

# TODO: maybe remove this file, the normal PPO should work

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticGamesRMA
from rsl_rl.env import VecEnv
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

import numpy as np

class OnPolicyDagger:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # Robot: Setup the student and teacher policy observation spaces
        if self.env.num_privileged_obs_robot is not None:
            # need privileged observations to do Dagger!
            num_critic_obs_expert = self.env.num_privileged_obs_priv_robot
            num_critic_obs_student = self.env.num_privileged_obs_robot
        else:
            raise(IOError)
            # num_critic_obs_expert = self.env.num_obs_robot
            # num_critic_obs_student = self.env.num_privileged_obs_robot

        # Agent: Setup the observation spaces
        if self.env.num_privileged_obs_agent is not None:
            num_critic_obs_agent = self.env.num_privileged_obs_agent
        else:
            num_critic_obs_agent = self.env.num_obs_agent

        ac_policy_name = self.cfg["policy_class_name"]
        ac_class = eval(ac_policy_name) 

        self.num_expert_states = self.env.num_priv_robot_states
        self.num_student_states = self.env.num_robot_states
        
        # Hack: When creating the *expert*, then the number of states 
        # should be different in the policy config used to create the ActorCritic class. 
        # Setup the correct number of robot states based on the expert or student. 
        self.policy_cfg['num_robot_states'] = self.num_expert_states

        # create the robot expert policy (teaching the student)
        print("self.env.num_obs_priv_robot: ", self.env.num_obs_priv_robot)
        expert = ac_class(self.env.num_obs_priv_robot,
                          num_critic_obs_expert,
                          self.env.num_actions_robot,
                          device=self.device,
                          # num_robot_states=self.num_expert_states,
                          **self.policy_cfg).to(self.device)
        
        # Hack: When creating the *student*, then the number of states 
        # should be different in the policy config used to create the ActorCritic class. 
        self.policy_cfg['num_robot_states'] = self.num_student_states
        
        # create the robot student policy (learning from the expert)
        student = ac_class( self.env.num_obs_robot,
                           num_critic_obs_student,
                           self.env.num_actions_robot,
                           device=self.device,
                           # num_robot_states=self.env.num_student_states,
                           **self.policy_cfg).to(self.device)
        
        # create the non-robot agent policy
        actor_critic_agent = ActorCritic( self.env.num_obs_agent,
                                          num_critic_obs_agent,
                                          self.env.num_actions_agent,
                                          **self.policy_cfg).to(self.device)
        
        
        # PPO alg information. 
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg_expert: PPO = alg_class(expert, device=self.device, **self.alg_cfg)
        self.alg_student: PPO = alg_class(student, device=self.device, **self.alg_cfg)
        self.alg_agent: PPO = alg_class(actor_critic_agent, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_learn_interval"]

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # we are updating the student policy's estimator and policy parameters. 
        learn_rate = 5e-3 #1e-3
        self.optimizer = optim.Adam(params=(*self.alg_student.actor_critic.estimator.parameters(),
                                    *self.alg_student.actor_critic.actor.parameters()),
                                    lr=learn_rate)

        # optional: learning rate scheduler. 
        lr_decay_rate = 0.2
        num_learn_iters = self.current_learning_iteration + self.cfg["max_iterations"]
        # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[int(0.4*num_learn_iters),
                                                                 int(0.8*num_learn_iters)],
                                        gamma=lr_decay_rate)
        # define the loss function
        self.Loss = nn.MSELoss()

        print("[OnPolicyDagger] finished creating the actor-critic and PPO alg... resetting robots..")
        _, _, _, _ = self.env.reset()

    def learn(self,
              num_learning_iterations,
              init_at_random_ep_len=False,
              early_stop=-1):

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # STUDENT: get observations of the student robot policy
        obs = self.env.get_observations_robot()

        # TEACHER: get observations of the teacher robot policy 
        privileged_obs = self.env.get_privileged_observations_robot()

        # Agent: get observations for the agent
        obs_agent = self.env.get_observations_agent()

        # bookkeeping. 
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        self.alg_student.actor_critic.train()  # switch student to train mode (for dropout for example)
        self.alg_agent.actor_critic.eval()  # NOTE: training the agent is not supported

        # logging. 
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # control bounds for the robot. Used to make outputs of the policy more dynamically feasible. 
        max_actions = torch.ones((self.env.num_envs,self.env.num_actions_robot), dtype=torch.float, device=self.device)
        max_actions[:,0] = 3
        max_actions[:,1] = 0
        max_actions[:,2] = 3.14
        min_actions = torch.ones((self.env.num_envs,self.env.num_actions_robot), dtype=torch.float, device=self.device)
        min_actions[:,0] = 0
        min_actions[:,1] = 0
        min_actions[:,2] = -3.14


        # ==== LSTM Predictor state keeping ==== #
        h = np.zeros((self.alg_student.actor_critic.estimator.num_layers, self.env.num_envs, self.alg_student.actor_critic.estimator.hidden_size))
        c = np.zeros((self.alg_student.actor_critic.estimator.num_layers, self.env.num_envs,self.alg_student.actor_critic.estimator.hidden_size))
        mask = np.ones((self.env.num_envs,))
        # ====================================== #
        
        # Learning loop. 
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(0, tot_iter):
            start = time.time()
            self.env.update_training_iter_num(it)

            # create LSTM hidden states
            h_curr = torch.tensor(h, device=self.device, requires_grad=True, dtype=torch.float32)
            c_curr = torch.tensor(c, device=self.device, requires_grad=True, dtype=torch.float32)
            hidden_state = (h_curr, c_curr)
            mask_curr = torch.tensor(mask, device=self.device, dtype=torch.float32)

            loss = 0
            latent_loss = 0
            imitation_loss = 0
            for _ in range(self.num_steps_per_env):
                hidden_state = (torch.einsum("ijk,j->ijk", hidden_state[0], mask_curr),
                                torch.einsum("ijk,j->ijk", hidden_state[1], mask_curr))
                
                # STUDENT: get the student's estimate of the latent intent and its actions.
                #   estimator:      E(hist) -> zhat
                #   policy:         pi(x, zhat) -> action 
                estimator_obs = obs.clone().unsqueeze(0)
                zhat, hidden_state = self.alg_student.actor_critic.estimate_latent_lstm(estimator_obs, hidden_state)
                actions_robot = self.alg_student.actor_critic.estimate_actor_inference(obs[:,:self.num_student_states], zhat[0].detach(), mask=False)
                
                with torch.no_grad():
                    # TEACHER: get the teacher's estimate of the latent intent and its actions.
                    #   (ideal) estimator:      E*(future) -> zexpert
                    #   (ideal) policy:         pi*(x, zhat) -> gt_action 
                    zexpert = self.alg_expert.actor_critic.privileged_latent(privileged_obs.to(self.device))
                    gt_actions = self.alg_expert.actor_critic.estimate_actor_inference(privileged_obs[:,:self.num_expert_states].to(self.device), zexpert.detach(), mask=False)
                    
                    # clip the gt_actions to make them sensible
                    gt_actions = torch.minimum(gt_actions,max_actions)
                    gt_actions = torch.maximum(gt_actions,min_actions)
                    actions_agent = self.alg_agent.actor_critic.act_inference(obs_agent)

                # store the MSE loss on the latent
                latent_loss += self.Loss(zhat[0], zexpert)
                imitation_loss += self.Loss(actions_robot,gt_actions)
                loss += latent_loss + imitation_loss

                # take step in environment
                env_actions = actions_robot.detach()
                env_actions = torch.minimum(env_actions,max_actions)
                env_actions = torch.maximum(env_actions,min_actions)
                ret_val = self.env.step(actions_agent, env_actions)
                obs = ret_val[1]  # extract robot-centric return values
                obs_agent = ret_val[0] # observation for the agent
                privileged_obs = ret_val[3]
                rewards = ret_val[5]
                dones = ret_val[6]
                infos = ret_val[7]

                # process the info
                obs, privileged_obs, critic_obs, rewards, dones = obs.to(self.device), privileged_obs.to(self.device), \
                    critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                obs_agent = obs_agent.to(self.device)
                        
                # mask out finished envs
                mask_curr = ~dones

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop

            loss = loss / self.num_steps_per_env
            latent_loss = latent_loss / self.num_steps_per_env
            imitation_loss = imitation_loss / self.num_steps_per_env
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # save the hidden states, but detach from compute graph
            h = h_curr.detach().cpu().numpy()
            c = c_curr.detach().cpu().numpy()
            mask = mask_curr.detach().cpu().numpy()
            
            self.scheduler.step()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'robot_model_e{}_l{}.pt'.format(0, it)))
            ep_infos.clear()

            if early_stop > 0 and it > early_stop:
                if statistics.mean(rewbuffer) <= 1e-5:
                    self.current_learning_iteration += num_learning_iterations
                    self.save(os.path.join(self.log_dir, 'robot_model_e{}_l{}.pt'.format(0, self.current_learning_iteration)))
                    return
        self.current_learning_iteration += num_learning_iterations

        self.save(os.path.join(self.log_dir, 'robot_model_e{}_l{}.pt'.format(0, self.current_learning_iteration)))
        return

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg_student.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/Latent_Loss', locs['latent_loss'], locs['it'])
        self.writer.add_scalar('Loss/Imitation_loss', locs['imitation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.scheduler.get_last_lr()[0], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "



        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Latent loss:':>{pad}} {locs['latent_loss']:.4f}\n"""
                          f"""{'Action loss:':>{pad}} {locs['imitation_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'BC loss:':>{pad}} {locs['loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            "estimator": self.alg_student.actor_critic.estimator.state_dict(),
            "actor": self.alg_student.actor_critic.actor.state_dict(),
            'model_state_dict': self.alg_student.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg_student.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, agent_id, path, load_optimizer=True, reset_std=False, student_policy=False):
        loaded_dict = torch.load(path, map_location=self.device)
        if agent_id == 0: # AGENT
            policy = self.alg_agent
        elif agent_id == 1: # ROBOT
            if student_policy:
                policy = self.alg_student
            else:
                policy = self.alg_expert
            example_input = torch.rand(1, policy.actor_critic.actor_obs_dim)
            print("[OnPolicyDagger] actor_obs_dim", policy.actor_critic.actor_obs_dim)
            # actor_graph = torch.jit.trace(self.alg.actor_critic.actor.to('cpu'), example_input)
            # torch.jit.save(actor_graph, LEGGED_GYM_ROOT_DIR + "/logs/phase_2_policy/actor.pt")

            example_input = torch.rand(1, policy.actor_critic.num_privilege_obs_estimator)
            print("[OnPolicyDagger] num_privilege_obs_estimator", policy.actor_critic.num_privilege_obs_estimator)
            # h = torch.zeros((self.alg.actor_critic.estimator.num_layers, 1, self.alg.actor_critic.estimator.hidden_size))
            # c = torch.zeros((self.alg.actor_critic.estimator.num_layers, 1, self.alg.actor_critic.estimator.hidden_size))

            # actor_graph = torch.jit.trace(self.alg.actor_critic.estimator.to('cpu'), (example_input[None, ...], h, c))
            # torch.jit.save(actor_graph, LEGGED_GYM_ROOT_DIR + "/logs/phase_2_policy/estimator.pt")

            if reset_std:
                self.alg_student.actor_critic.std = nn.Parameter(
                    self.alg_student.actor_critic.init_noise_std * torch.ones(self.alg_student.actor_critic.num_actions,
                                                                      device=self.device))

            if load_optimizer:
                self.alg_student.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            
        policy.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)

        # self.current_learning_iteration = loaded_dict['learn_it']
        return loaded_dict['infos']

    def load_jit(self, path):
        loaded_policy = torch.jit.load(path)
        return loaded_policy

    def get_inference_policy(self, agent_id, device=None, sample=False):
        if agent_id == 0: # AGENT
            alg = self.alg_agent
        elif agent_id == 1: # ROBOT
            alg = self.alg_student
        if device is not None:
            alg.actor_critic.to(device)
        alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if sample:
            return alg.actor_critic.act
        else:
            return alg.actor_critic.act_inference

    def get_estimator_inference_policy(self, device=None, sample=False):
        self.alg_student.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg_student.actor_critic.to(device)
        if sample:
            return self.alg_student.actor_critic.estimate_actor
        else:
            return self.alg_student.actor_critic.estimate_actor_inference
