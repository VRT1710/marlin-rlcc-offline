import d3rlpy
from d3rlpy.algos import SACConfig, SAC
from d3rlpy.datasets import MDPDataset
import h5py

# loading dataset from the .h5 file
with h5py.File('dataset_marlin_100k_trained_for_100k.h5', 'r') as f:
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
    batch_size=512,
    gamma=0.97,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-3,
    tau=0.01,
    n_critics=2,
    initial_temperature=1.0
)

# initializing SAC
sac = SAC(config=sac_config, device='cpu')

sac.build_with_dataset(dataset)
sac.load_model("online_marlin_S=200k_B=100k.pt")

# training
sac.fit(dataset,
        n_steps=200000,
        n_steps_per_epoch=200,
        experiment_name='sac_offline_rl_custom_dataset')

# save the trained model
sac.save_model('model_marlin_S=200k_B=100k_BS=100.pt')
