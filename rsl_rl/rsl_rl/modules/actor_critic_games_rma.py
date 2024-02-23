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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import pdb

class RNNEstimator(torch.nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 activation,
                 type='lstm',
                 num_layers=1,
                 hidden_size=256):
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.activation = activation

        #layers = [nn.Linear(hidden_size, output_size),
        #          self.activation]
        layers = [nn.Linear(hidden_size, 128),
                  self.activation,
                   nn.Linear(128, output_size)]
        self.last_mlp = nn.Sequential(*layers)

    def forward(self, observations, h, c):
        hidden_state = (h, c)
        out, hidden_state = self.rnn(observations, hidden_state)
        latent = self.last_mlp(out)
        return latent, hidden_state[0], hidden_state[1]

class MLPEstimator(torch.nn.Module):
    def __init__(self, input_size, num_latent, estimator_hidden_dims, activation):
        super().__init__()
        estimator_layers = []
        estimator_layers.append(nn.Linear(input_size, estimator_hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(estimator_hidden_dims)):
            if l == len(estimator_hidden_dims) - 1:
                estimator_layers.append(nn.Linear(estimator_hidden_dims[l], num_latent))
            else:
                estimator_layers.append(nn.Linear(estimator_hidden_dims[l], estimator_hidden_dims[l + 1]))
                estimator_layers.append(activation)
        self.estimator_mlp = nn.Sequential(*estimator_layers)

    def forward(self, observations):
        return self.estimator_mlp(observations)

class ActorCriticGamesRMA(nn.Module):
    is_recurrent = False

    # This class should have 4 neural networks:
    #   1. actor policy network:            pi(x^t, z)
    #   2. critic value function network:   V(x, x^future) for learning the priviledged policy + teacher estimator
    #   3. teacher latent estimator:        z* = T(x^future)                ==> MLP
    #   4. learned latent estimator:           zhat = E(x^history, uR^history)    ==> LSTM (?)
    #
    # The way they are active during the different phases are:
    #   Phase 1) train pi(x^t, z) where z* = T(x^future) and the critic is V(x, x^future)
    #   Phase 2) freeze pi(x^t, z)
    #   Phase 3) train zhat = E(x^history, uR^history) to match z* = T(x^future)
    #       where you get the rollouts by running the frozen policy with the
    #       *estimated* latent: pi(x^t, zhat). Then do MSE(z*, zhat) and update E to match.

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 device,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 use_privilege_enc=False,
                 privilege_enc_hidden_dims=[128, 128],
                 num_privilege_enc_obs=0,
                 num_estimator_obs=0,
                 num_latent=8, # i.e., embedding sz
                 use_estimator=False,
                 history_len=8,
                 num_robot_states=4,
                 **kwargs):
        if kwargs:
            print("ActorCriticGamesRMA.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCriticGamesRMA, self).__init__()

        activation = get_activation(activation)

        self.init_noise_std = init_noise_std
        self.num_actions = num_actions
        self.num_robot_states = num_robot_states
        self.device = device
        self.curr_state_dim = num_robot_states # current state is always the "first" four entries of the observation

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        if use_privilege_enc or use_estimator:
            # note: (mlp_input_dim_a - num_privilege_obs) equals number of observations passed in "raw"
            #              the actor MLP always gets (x^t, z)
            #              privileged encoder always separates the present from the future and encodes (x^future) into z
            mlp_input_dim_a = num_robot_states + num_latent
            self.privilege_obs_start_idx = num_robot_states

        self.actor_obs_dim = mlp_input_dim_a
        self.num_privilege_obs_estimator = num_estimator_obs
        self.num_privilege_enc_obs = num_privilege_enc_obs

        assert mlp_input_dim_a > 0
        assert mlp_input_dim_c > 0

        # Policy: pi(x, z)
        actor_layers = []
        actor_layers.append(nn.Linear(self.actor_obs_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # RMA teacher estimator: z* = T(x^future)
        self.use_privilege_enc = use_privilege_enc
        self.use_estimator = use_estimator

        # Construct the mask to separate current state from the rest of the observations
        curr_state_mask = torch.zeros(num_actor_obs)
        curr_state_mask[0:self.curr_state_dim] = 1  # masks out the future and only gives you the current state (and possibly goal)
        self.curr_state_mask = curr_state_mask.to(bool)

        if self.use_privilege_enc or self.use_estimator:
            # Construct the mask used by the RMA module that gets to see the future.
            privilege_mask = torch.zeros(self.num_privilege_enc_obs + self.num_robot_states)
            privilege_mask[self.privilege_obs_start_idx:] = 1  # masks out the future and only gives you the current state
            self.privilege_mask = privilege_mask.to(bool)
        # elif not self.use_privilege_enc and not self.use_estimator:
        #     raise NotImplementedError()

        if self.use_privilege_enc:
            enc_layers = []
            enc_layers.append(nn.Linear(self.num_privilege_enc_obs, privilege_enc_hidden_dims[0]))
            enc_layers.append(activation)
            for l in range(len(privilege_enc_hidden_dims)):
                if l == len(privilege_enc_hidden_dims) - 1:
                    enc_layers.append(nn.Linear(privilege_enc_hidden_dims[l], num_latent))
                else:
                    enc_layers.append(nn.Linear(privilege_enc_hidden_dims[l], privilege_enc_hidden_dims[l + 1]))
                    enc_layers.append(activation)
            self.privilege_enc = nn.Sequential(*enc_layers)
            self.privilege_enc.to(self.device)

        # Learned estimator: zhat = E(x^hist, uR^hist)
        if self.use_estimator:

            self.estimator = RNNEstimator(num_estimator_obs,
                                           num_latent,
                                           activation,
                                           type="lstm",
                                           num_layers=1,
                                           hidden_size=256)
            
            # self.estimator = MLPEstimator(num_privilege_obs_estimator,
            #                               num_latent,
            #                               RMA_hidden_dims,
            #                               activation)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def estimate_latent_lstm(self, observations, hidden_states=None):
        latent, h, c = self.estimator.forward(observations,
                                              hidden_states[0],
                                              hidden_states[1])
        return latent, (h, c)

    def estimate_latent(self, observations):
        latent = self.estimator.forward(observations)
        return latent

    def estimate_actor(self, observations, latent):
        new_obs = torch.cat([latent, observations[..., self.curr_state_mask]], dim=1).to(observations.device)
        mean = self.actor(new_obs)
        self.distribution = Normal(mean, mean * 0. + self.std)
        return self.distribution.sample()

    def estimate_actor_inference(self, observations, latent, mask=True):
        if mask:
            new_obs = torch.cat([latent, observations[..., self.curr_state_mask]], dim=1).to(observations.device)
        else:
            new_obs = torch.cat([latent, observations], dim=1).to(observations.device)
        mean = self.actor(new_obs)
        return mean
    
    def privileged_latent(self, observations):
        privilege_obs = observations[:, self.privilege_mask]
        # return self.privilege_enc(privilege_obs)
        return self.privilege_enc(privilege_obs)

    def privileged_actor(self, observations):
        latent = self.privileged_latent(observations)
        new_obs = torch.cat([latent, observations[:, ~self.privilege_mask]], dim=1).to(observations.device)
        return self.actor(new_obs)

    def update_distribution(self, observations):
        if self.use_privilege_enc:
            mean = self.privileged_actor(observations)
        else:
            mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)
        # self.distribution = Normal(mean,
        #                            mean * 0. + torch.min(self.std, self.init_noise_std * torch.ones_like(self.std)))

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self.use_privilege_enc:
            actions_mean = self.privileged_actor(observations)
        else:
            actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
