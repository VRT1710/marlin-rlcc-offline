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

from d3rlpy.models.encoders import VectorEncoderFactory

def convert_sb3_to_d3rlpy(sb3_model_path, env):
    sb3_model = SB3_SAC.load(sb3_model_path, env=env)
    
    actor = sb3_model.actor
    critic1 = sb3_model.critic
    critic2 = sb3_model.critic_target
    
    actor_params = {name: param.detach().cpu().numpy() for name, param in actor.named_parameters()}
    critic1_params = {name: param.detach().cpu().numpy() for name, param in critic1.named_parameters()}
    critic2_params = {name: param.detach().cpu().numpy() for name, param in critic2.named_parameters()}
    
    return actor_params, critic1_params, critic2_params

def map_keys(state_dict, mapping):
    mapped_state_dict = {}
    for sb3_key, d3rlpy_key in mapping.items():
        if sb3_key in state_dict:
            mapped_state_dict[d3rlpy_key] = state_dict[sb3_key]
    return mapped_state_dict

def convert_to_tensors(state_dict):
    tensor_state_dict = {}
    for key, value in state_dict.items():
        tensor_state_dict[key] = torch.tensor(value)
    return tensor_state_dict

def load_sb3_model_to_d3rlpy(sac, sb3_model_path, env):
    actor_params, critic1_params, critic2_params = convert_sb3_to_d3rlpy(sb3_model_path, env)
    
    # Mapping for actor parameters
    actor_mapping = {
        "latent_pi.0.weight": "_encoder._layers.0.weight",
        "latent_pi.0.bias": "_encoder._layers.0.bias",
        "latent_pi.2.weight": "_encoder._layers.2.weight",
        "latent_pi.2.bias": "_encoder._layers.2.bias",
        "mu.weight": "_mu.weight",
        "mu.bias": "_mu.bias",
        "log_std.weight": "_logstd.weight",
        "log_std.bias": "_logstd.bias"
    }
    
    # Mapping for critic parameters
    critic_mapping = {
        "qf1.0.weight": "_encoder._layers.0.weight",
        "qf1.0.bias": "_encoder._layers.0.bias",
        "qf1.2.weight": "_encoder._layers.2.weight",
        "qf1.2.bias": "_encoder._layers.2.bias",
        "qf1.4.weight": "_output.weight",
        "qf1.4.bias": "_output.bias",
        "qf2.0.weight": "_encoder._layers.0.weight",
        "qf2.0.bias": "_encoder._layers.0.bias",
        "qf2.2.weight": "_encoder._layers.2.weight",
        "qf2.2.bias": "_encoder._layers.2.bias",
        "qf2.4.weight": "_output.weight",
        "qf2.4.bias": "_output.bias"
    }
    
    # Map the keys
    mapped_actor_params = map_keys(actor_params, actor_mapping)
    mapped_critic1_params = map_keys(critic1_params, critic_mapping)
    mapped_critic2_params = map_keys(critic2_params, critic_mapping)
    
    # Convert numpy arrays to PyTorch tensors
    tensor_actor_params = convert_to_tensors(mapped_actor_params)
    tensor_critic1_params = convert_to_tensors(mapped_critic1_params)
    tensor_critic2_params = convert_to_tensors(mapped_critic2_params)
    
    # Set the state dictionaries
    sac.impl._actor.load_state_dict(tensor_actor_params)
    sac.impl._q_func1.load_state_dict(tensor_critic1_params)
    sac.impl._q_func2.load_state_dict(tensor_critic2_params)

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

# configuring SAC with custom architecture
sac_config = SACConfig(
    batch_size=256,
    gamma=0.99,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-4,
    tau=0.005,
    n_critics=2,
    initial_temperature=1.0,
    actor_encoder_factory=VectorEncoderFactory([400, 300]),  # Match the SB3 architecture
    critic_encoder_factory=VectorEncoderFactory([400, 300])  # Match the SB3 architecture
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
        n_steps=200000,
        n_steps_per_epoch=200,
        experiment_name='sac_offline_rl_custom_dataset')

# save the trained model
sac.save_model('/home/devuser/app/training/model_marlin_continued_S=200k+200k_B=100k.pt')
