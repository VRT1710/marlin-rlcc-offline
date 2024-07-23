import d3rlpy
from d3rlpy.algos import SACConfig, SAC
from d3rlpy.datasets import MDPDataset
import h5py
import torch
import stable_baselines3 as sb3
from stable_baselines3 import SAC as SB3_SAC
from stable_baselines3.common.utils import set_random_seed
import gym

from utils.wrappers import HistoryWrapper
import utils.import_envs

def convert_sb3_to_d3rlpy(sb3_model_path, env):
    sb3_model = SB3_SAC.load(sb3_model_path, env=env)
    
    actor = sb3_model.actor
    critic1 = sb3_model.critic
    critic2 = sb3_model.critic_target
    
    actor_params = {name: param.detach().cpu().numpy() for name, param in actor.named_parameters()}
    critic1_params = {name: param.detach().cpu().numpy() for name, param in critic1.named_parameters()}
    critic2_params = {name: param.detach().cpu().numpy() for name, param in critic2.named_parameters()}
    
    
    
    return actor_params, critic1_params, critic2_params

def set_d3rlpy_params(network, params):
    with torch.no_grad():
        for name, param in network.named_parameters():
            if name in params:
                param.copy_(torch.tensor(params[name]))

def load_sb3_model_to_d3rlpy(sac, sb3_model_path, env):
    actor_params, critic1_params, critic2_params = convert_sb3_to_d3rlpy(sb3_model_path, env)
    
    set_d3rlpy_params(sac.impl.actor.impl, actor_params)
    set_d3rlpy_params(sac.impl.q_function.impl[0], critic1_params)
    set_d3rlpy_params(sac.impl.q_function.impl[1], critic2_params)

env = gym.make('Marlin-v1')
env = HistoryWrapper(env, 10)

set_random_seed(1)

# loading dataset from the .h5 file
with h5py.File('/home/devuser/app/training/dataset_marlin_100k_trained_for_200k.h5', 'r') as f:
    observations = f['observations'][:]
    actions = f['actions'][:]
    rewards = f['rewards'][:]
    next_observations = f['next_observations'][:]
    terminals = f['terminals'][:]
    timeouts = f['timeouts'][:]

# Create an MDPDataset
dataset = MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals
)

# configuring SAC
sac_config = SACConfig(
    batch_size=256,
    gamma=0.99,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-4,
    tau=0.005,
    n_critics=2,
    initial_temperature=1.0
)

# initializing SAC
sac = sac_config.create(device='cpu')

# Build SAC with dataset
sac.build_with_dataset(dataset)

# Load SB3 trained SAC model
sb3_model_path = "/home/devuser/app/training/Marlin-v1.zip"
load_sb3_model_to_d3rlpy(sac, sb3_model_path, env)

# training
sac.fit(dataset,
        n_steps=1000,
        n_steps_per_epoch=200,
        experiment_name='sac_offline_rl_custom_dataset')

# save the trained model
sac.save_model('/home/devuser/app/training/model_marlin_continued_S=200k+1k_B=100k.pt')
