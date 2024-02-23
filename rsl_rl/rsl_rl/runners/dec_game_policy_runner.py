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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticGamesRMA
from rsl_rl.env import VecEnv

class DecGamePolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs_robot is not None:
            num_critic_obs_robot = self.env.num_privileged_obs_robot
        else:
            num_critic_obs_robot = self.env.num_obs_robot

        if self.env.num_privileged_obs_agent is not None:
            num_critic_obs_agent = self.env.num_privileged_obs_agent
        else:
            num_critic_obs_agent = self.env.num_obs_agent
        
        actor_critic_name = self.cfg["policy_class_name"]
        actor_critic_class = eval(actor_critic_name) # ActorCritic

        self.num_actions_policy_agent = self.env.num_actions_agent

        if actor_critic_name == "ActorCriticGamesRMA":
            actor_critic_robot: ActorCritic = actor_critic_class( self.env.num_obs_robot,
                                                            num_critic_obs_robot,
                                                            self.env.num_actions_robot,
                                                            self.device,
                                                            **self.policy_cfg).to(self.device)
            # TODO: note -- the cube agent is always a vanilla actor critic agent
            actor_critic_agent: ActorCritic = ActorCritic( self.env.num_obs_agent,
                                                            num_critic_obs_agent,
                                                            self.num_actions_policy_agent,
                                                            **self.policy_cfg).to(self.device)
        elif actor_critic_name == "ActorCriticGames":
            actor_critic_robot: ActorCritic = actor_critic_class( self.env.num_obs_robot,
                                                            self.env.num_obs_encoded_robot,
                                                            num_critic_obs_robot,
                                                            self.env.embedding_sz_robot, 
                                                            self.env.num_actions_robot,
                                                            **self.policy_cfg).to(self.device)
            actor_critic_agent: ActorCritic = actor_critic_class( self.env.num_obs_agent,
                                                            self.env.num_obs_encoded_agent,
                                                            num_critic_obs_agent,
                                                            self.env.embedding_sz_agent,
                                                            self.num_actions_policy_agent,
                                                            **self.policy_cfg).to(self.device)
        elif actor_critic_name == "ActorCritic":
            actor_critic_robot: ActorCritic = actor_critic_class( self.env.num_obs_robot,
                                                            num_critic_obs_robot,
                                                            self.env.num_actions_robot,
                                                            self.device,
                                                            **self.policy_cfg).to(self.device)
            actor_critic_agent: ActorCritic = actor_critic_class( self.env.num_obs_agent,
                                                            num_critic_obs_agent,
                                                            self.num_actions_policy_agent,
                                                            self.device,
                                                            **self.policy_cfg).to(self.device)
        else:
            print("==> ERROR: actor_critic ", actor_critic_name, "is not supported!")
            return

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg_robot: PPO = alg_class(actor_critic_robot, device=self.device, **self.alg_cfg)
        self.alg_agent: PPO = alg_class(actor_critic_agent, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_learn_interval = self.cfg["save_learn_interval"]
        self.save_evol_interval = self.cfg["save_evol_interval"]

        # init storage and model
        self.alg_robot.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs_robot], [self.env.num_privileged_obs_robot], [self.env.num_actions_robot])
        self.alg_agent.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs_agent], [self.env.num_privileged_obs_agent], [self.env.num_actions_agent])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.curr_evolution_iteration = 0

        self.agent_id_to_train = self.cfg['agent_id_to_train'] # which agent to start training first
        self.agent_dict = {"agent" : 0, "robot" : 1}
        self.agent_names = list(self.agent_dict.keys())

        print("[DecGamePolicyRunner] finished creating the actor-critics and the PPO alg... resetting robots..")
        _, _, _, _ = self.env.reset()
    
    def learn(self,
              max_num_evolutions,
              num_learning_iterations,
              init_at_random_ep_len=False):
        """
        Args:
            max_num_evolutions (int): number of times that two agents alternate policy updates
            num_learning_iterations (int): number of policy updates per agent
        """
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        ego_agent_id = self.agent_id_to_train 
        ego_agent_name = self.agent_names[ego_agent_id]
       
        for evol_it in range(0, max_num_evolutions):

            print("[DecGamePolicyRunner] the ", ego_agent_name.upper(), " is learning...")
            # update the learning agent's policy
            self.update_policy(ego_agent_id, num_learning_iterations, evol_it, max_num_evolutions)

            # save the current pair of policies
            if evol_it % self.save_evol_interval == 0:
                self.save(os.path.join(self.log_dir, 'agent_model_e{}_l{}.pt'.format(evol_it, self.current_learning_iteration)), 
                    self.alg_agent, evol_it, self.current_learning_iteration)
                self.save(os.path.join(self.log_dir, 'robot_model_e{}_l{}.pt'.format(evol_it, self.current_learning_iteration)), 
                    self.alg_robot, evol_it, self.current_learning_iteration)

            # alternate who is learning
            ego_agent_id = (ego_agent_id + 1) % 2
            ego_agent_name = self.agent_names[ego_agent_id]

            self.curr_evolution_iteration += 1

    def update_policy(self, ego_agent_id, num_learning_iterations, evol_it, max_num_evolutions):
        """
        We use the terminology:
            EGO to refer to the agent that is actively learning during interaction
            EXO to refer to the agent whose policy is fixed for this interaction
        Args:
            ego_agent_id (int): ID of the agent that is learning (0 = AGENT, 1 = ROBOT)
        """

        ep_infos = []
        rewbuffer_agent = deque(maxlen=100)
        rewbuffer_robot = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum_agent = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_sum_robot = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if ego_agent_id == 0: # AGENT is learning
            obs_ego = self.env.get_observations_agent()
            obs_exo = self.env.get_observations_robot()
            privileged_obs_ego = self.env.get_privileged_observations_agent()
            privileged_obs_exo = self.env.get_privileged_observations_robot()
            self.env.ll_obs_buf_robot = self.env.ll_env.get_observations()  # set the low-level observations
            alg_ego = self.alg_agent
            alg_exo = self.alg_robot
        elif ego_agent_id == 1: # ROBOT is learning
            obs_ego = self.env.get_observations_robot()
            obs_exo = self.env.get_observations_agent()
            privileged_obs_ego = self.env.get_privileged_observations_robot()
            privileged_obs_exo = self.env.get_privileged_observations_agent()
            self.env.ll_obs_buf_robot = self.env.ll_env.get_observations()  # set the low-level observations
            alg_ego = self.alg_robot
            alg_exo = self.alg_agent
        else:
            print("[DecGamePolicyRunner] ERROR! Wrong agent id.")
            return 

        # get the learning agent's initial observations
        critic_obs_ego = privileged_obs_ego if privileged_obs_ego is not None else obs_ego
        obs_ego, critic_obs_ego = obs_ego.to(self.device), critic_obs_ego.to(self.device)  

        # get the other agent's initial observations
        critic_obs_exo = privileged_obs_exo if privileged_obs_exo is not None else obs_exo
        obs_exo, critic_obs_exo = obs_exo.to(self.device), critic_obs_exo.to(self.device)   

        alg_ego.actor_critic.train()    # switch the learning agent to train mode (for dropout for example)
        alg_exo.actor_critic.eval()     # switch the fixed agent to eval mode

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for learn_it in range(self.current_learning_iteration, tot_iter):
            self.env.update_training_iter_num(learn_it)
            
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # get both agent's actions
                    actions_ego = alg_ego.act(obs_ego, critic_obs_ego)
                    actions_exo = alg_exo.actor_critic.act_inference(obs_exo)

                    # step the environment forward
                    if ego_agent_id == 0: # AGENT is learning and is treated as "ego"
                        # NOTE: 
                        # the function signature of step() always requires
                        # AGENT [non-robot] actions to be first, and then ROBOT [a1] actions
                        # 
                        #       step(actions_agent, actions_robot)
                        # 
                        # the return value follows the same convention. for each
                        # return type (obs, privi_obs, rewards,..) the AGENT is
                        # returned first, and the ROBOT is returned second.
                        ret_val = self.env.step(actions_ego, actions_exo)
                        obs_ego = ret_val[0] # agent
                        obs_exo = ret_val[1] # robot
                        privileged_obs_ego = ret_val[2]
                        privileged_obs_exo = ret_val[3]
                        rewards_ego = ret_val[4]
                        rewards_exo = ret_val[5]
                        dones = ret_val[6]
                        infos = ret_val[7]
                    else: # ROBOT is learning and as is treated as "ego"
                        ret_val = self.env.step(actions_exo, actions_ego)
                        obs_exo = ret_val[0] # agent
                        obs_ego = ret_val[1] # robot
                        privileged_obs_exo = ret_val[2]
                        privileged_obs_ego = ret_val[3]
                        rewards_exo = ret_val[4]
                        rewards_ego = ret_val[5]
                        dones = ret_val[6]
                        infos = ret_val[7]

                    # process the info
                    critic_obs_ego = privileged_obs_ego if privileged_obs_ego is not None else obs_ego
                    obs_ego, critic_obs_ego, rewards_ego, dones = obs_ego.to(self.device), critic_obs_ego.to(self.device), rewards_ego.to(self.device), dones.to(self.device)
                    obs_exo, critic_obs_exo, rewards_exo = obs_exo.to(self.device), critic_obs_exo.to(self.device), rewards_exo.to(self.device)
                    alg_ego.process_env_step(rewards_ego, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if ego_agent_id == 0:  # AGENT is learning
                            cur_reward_sum_agent += rewards_ego
                            cur_reward_sum_robot += rewards_exo
                        else: # ROBOT is learning
                            cur_reward_sum_robot += rewards_ego
                            cur_reward_sum_agent += rewards_exo
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer_agent.extend(cur_reward_sum_agent[new_ids][:, 0].cpu().numpy().tolist())
                        rewbuffer_robot.extend(cur_reward_sum_robot[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum_agent[new_ids] = 0
                        cur_reward_sum_robot[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                alg_ego.compute_returns(critic_obs_ego)

            mean_value_loss, mean_surrogate_loss, mean_bc_loss = alg_ego.update()

            # enforce a minimum std
            min_std = 0.2
            alg_ego.actor_critic.enforce_minimum_std(
                (torch.ones(alg_ego.actor_critic.num_actions) * min_std).to(self.device))
            alg_exo.actor_critic.enforce_minimum_std(
                (torch.ones(alg_exo.actor_critic.num_actions) * min_std).to(self.device))

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals(), alg_ego)
            if learn_it % self.save_learn_interval == 0:
                self.save(os.path.join(self.log_dir, 'agent_model_e{}_l{}.pt'.format(evol_it, learn_it)), 
                    self.alg_agent, evol_it, learn_it)
                self.save(os.path.join(self.log_dir, 'robot_model_e{}_l{}.pt'.format(evol_it, learn_it)), 
                    self.alg_robot, evol_it, learn_it)
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        # self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, alg, width=80, pad=35):
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
                self.writer.add_scalar('Episode/' + key, value, locs['learn_it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        ego_agent_name = self.agent_names[locs['ego_agent_id']]
        policy_info_str = '[' + ego_agent_name + '] Policy/mean_noise_std'
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['learn_it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['learn_it'])
        self.writer.add_scalar('Loss/bc_loss', locs['mean_bc_loss'], locs['learn_it'])
        self.writer.add_scalar('Loss/learning_rate', alg.learning_rate, locs['learn_it'])
        self.writer.add_scalar(policy_info_str, mean_std.item(), locs['learn_it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['learn_it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['learn_it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['learn_it'])
        if len(locs['rewbuffer_robot']) > 0:
            self.writer.add_scalar('Train/mean_reward_robot', statistics.mean(locs['rewbuffer_robot']), locs['learn_it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['learn_it'])
            self.writer.add_scalar('Train/mean_reward_robot/time', statistics.mean(locs['rewbuffer_robot']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        if len(locs['rewbuffer_agent']) > 0:
            self.writer.add_scalar('Train/mean_reward_agent', statistics.mean(locs['rewbuffer_agent']), locs['learn_it'])
            self.writer.add_scalar('Train/mean_reward_agent/time', statistics.mean(locs['rewbuffer_agent']), self.tot_time)

        str_header = f"\033[1m Currently training: {ego_agent_name} \033[0m "
        str_evo = f" \033[1m Evolution number {self.curr_evolution_iteration}/{locs['max_num_evolutions']} \033[0m "
        str = f" \033[1m Learning iteration {locs['learn_it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer_robot']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_header.center(width, ' ')}\n"""
                          f"""{str_evo.center(width, ' ')}\n"""
                          f"""{'=' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (robot):':>{pad}} {statistics.mean(locs['rewbuffer_robot']):.2f}\n"""
                          f"""{'Mean reward (agent):':>{pad}} {statistics.mean(locs['rewbuffer_agent']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_evo.center(width, ' ')}\n\n"""
                          f"""{'-' * width}\n\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'BC loss:':>{pad}} {locs['mean_bc_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['learn_it'] + 1) * (
                               locs['num_learning_iterations'] - locs['learn_it']):.1f}s\n""")
        print(log_string)


    def save(self, path, alg, evol_iteration, learn_iteration, infos=None):
        torch.save({
            'model_state_dict': alg.actor_critic.state_dict(),
            'optimizer_state_dict': alg.optimizer.state_dict(),
            'evol_iter': evol_iteration,
            'learn_iter': learn_iteration,
            'infos': infos,
            }, path)

    def load(self, agent_id, path, load_optimizer=True):
        print("IN LOAD path: ", path)
        loaded_dict = torch.load(path, map_location=self.device)
        if agent_id == 0: # AGENT
            alg = self.alg_agent
        elif agent_id == 1: # ROBOT
            alg = self.alg_robot
        else:
            print("[DecGamePolicyRunner] ERROR in load(). Invalid agent_id: ", agent_id)
            return None

        alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])

        if 'learn_iter' in loaded_dict:
            self.current_learning_iteration = loaded_dict['learn_iter']
        else:
            self.current_learning_iteration = 0
        if 'evol_iter' in loaded_dict:
            self.curr_evolution_iteration = loaded_dict['evol_iter']
        else:
            self.curr_evolution_iteration = 0
        return loaded_dict['infos']

    def get_inference_policy(self, agent_id, device=None):
        if agent_id == 0: # AGENT
            alg = self.alg_agent
        elif agent_id == 1: # ROBOT
            alg = self.alg_robot
        else:
            print("[DecGamePolicyRunner] ERROR in get_inference_policy(). Invalid agent_id: ", agent_id)
            return None

        alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            alg.actor_critic.to(device)
        return alg.actor_critic.act_inference

