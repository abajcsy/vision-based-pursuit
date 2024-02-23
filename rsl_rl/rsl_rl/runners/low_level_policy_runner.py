
import torch
from rsl_rl.modules import ActorCritic

class LLPolicyRunner:
    """Class for loading the pre-trained low-level walking policy."""
    def __init__(self,
                env_cfg,
                train_cfg_dict,
                device):
        cfg = train_cfg_dict["runner"]
        policy_cfg = train_cfg_dict["policy"]
        actor_critic_class = eval(cfg["policy_class_name"])  # ActorCritic
        if env_cfg.env.num_privileged_obs is not None:
            num_critic_obs = env_cfg.env.num_privileged_obs
        else:
            num_critic_obs = env_cfg.env.num_observations
        num_obs = env_cfg.env.num_observations
        num_actions = env_cfg.env.num_actions
        self.actor_critic: ActorCritic = actor_critic_class(num_obs,
                                                       num_critic_obs,
                                                       num_actions,
                                                       **policy_cfg).to(device)
    def load_policy(self, path, device=None):
        #print("TODO (@ANTONIO): REMOVE CPU FROM LOAD (file rsl_rl/runners/low_lewel_policy_runner.py ")
        loaded_dict = torch.load(path, map_location=device)
        self.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference
