import d3rlpy
from d3rlpy.algos import SACConfig, SAC
from d3rlpy.datasets import MDPDataset
import h5py
import torch
import stable_baselines3 as sb3
import gym

# Import and register the custom environment
import env

# Function to convert SB3 policy to d3rlpy compatible format
def convert_sb3_to_d3rlpy(sb3_model_path, env):
    # Load SB3 model
    sb3_model = sb3.SAC.load(sb3_model_path, env=env)
    
    # Extract policy components
    actor = sb3_model.actor
    critic1 = sb3_model.critic
    critic2 = sb3_model.critic_target
    
    # Convert to torch tensors
    actor_params = {name: param.detach().cpu().numpy() for name, param in actor.named_parameters()}
    critic1_params = {name: param.detach().cpu().numpy() for name, param in critic1.named_parameters()}
    critic2_params = {name: param.detach().cpu().numpy() for name, param in critic2.named_parameters()}
    
    return actor_params, critic1_params, critic2_params

# Custom function to load SB3 model and set to d3rlpy SAC
def load_sb3_model_to_d3rlpy(sac, sb3_model_path, env):
    actor_params, critic1_params, critic2_params = convert_sb3_to_d3rlpy(sb3_model_path, env)
    sac.impl.set_params({'actor': actor_params, 'critic1': critic1_params, 'critic2': critic2_params})

# loading dataset from the .h5 file
with h5py.File('dataset_marlin_100k_trained_for_200k.h5', 'r') as f:
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
sac = SAC(config=sac_config, device='cpu')

# Build SAC with dataset
sac.build_with_dataset(dataset)

# Create the custom Gym environment
env = gym.make('Marlin-v1')

# Load SB3 trained SAC model
sb3_model_path = "best_model.zip"  # Replace with the path to your SB3 model
load_sb3_model_to_d3rlpy(sac, sb3_model_path, env)

# training
sac.fit(dataset,
        n_steps=200000,
        n_steps_per_epoch=200,
        experiment_name='sac_offline_rl_custom_dataset')

# save the trained model
sac.save_model('model_marlin_continued_S=200+200k_B=100k.pt')
