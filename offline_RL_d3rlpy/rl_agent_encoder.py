import dataclasses
import torch
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.algos import SACConfig, SAC
from stable_baselines3 import SAC as sb3_SAC
from d3rlpy.datasets import MDPDataset
import h5py

def load_policy_state(model, state_dict):
    model.impl.policy.load_state_dict(state_dict)

def rename_keys(state_dict):
    renamed_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('actor.latent_pi'):
            if '.0.weight' in key:
                new_key = new_key.replace('actor.latent_pi.0.weight', '_encoder.fc1.weight')
            elif '.0.bias' in key:
                new_key = new_key.replace('actor.latent_pi.0.bias', '_encoder.fc1.bias')
            elif '.2.weight' in key:
                new_key = new_key.replace('actor.latent_pi.2.weight', '_encoder.fc2.weight')
            elif '.2.bias' in key:
                new_key = new_key.replace('actor.latent_pi.2.bias', '_encoder.fc2.bias')
            elif '.4.weight' in key:
                new_key = new_key.replace('actor.latent_pi.4.weight', '_encoder.fc3.weight')
            elif '.4.bias' in key:
                new_key = new_key.replace('actor.latent_pi.4.bias', '_encoder.fc3.bias')
        elif key.startswith('actor.mu'):
            new_key = new_key.replace('actor.mu', '_mu')
        elif key.startswith('actor.log_std'):
            new_key = new_key.replace('actor.log_std', '_logstd')
        
        if any(substring in new_key for substring in ["_encoder", "_mu", "_logstd"]):
            renamed_state_dict[new_key] = value

    # add the missing keys
    if "_encoder.fc3.weight" not in renamed_state_dict:
        renamed_state_dict["_encoder.fc3.weight"] = torch.randn(300, 300)
    if "_encoder.fc3.bias" not in renamed_state_dict:
        renamed_state_dict["_encoder.fc3.bias"] = torch.randn(300)

    return renamed_state_dict

# Custom Encoder
class CustomEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super(CustomEncoder, self).__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0], 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, feature_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = self.fc3(h)
        return h

class CustomEncoderWithAction(nn.Module):
    def __init__(self, observation_shape, action_size, feature_size):
        super(CustomEncoderWithAction, self).__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0] + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, feature_size)

    def forward(self, x, action):
        h = torch.cat([x, action], dim=1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        h = self.fc3(h)
        return h

# Custom Encoder Factory
@dataclasses.dataclass
class CustomEncoderFactory(EncoderFactory):
    feature_size: int

    def create(self, observation_shape):
        return CustomEncoder(observation_shape, self.feature_size)

    def create_with_action(self, observation_shape, action_size):
        return CustomEncoderWithAction(observation_shape, action_size, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "custom"

feature_size = 300

# Load the dataset from the .h5 file
with h5py.File('/home/nomads/Documents/elia/d3rlpy/test_policy/dataset_marlin_100k_trained_for_200k.h5', 'r') as f:
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

# Configure SAC with the custom encoder
sac_config = SACConfig(
    batch_size=512,
    gamma=0.97,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=3e-3,
    tau=0.01,
    n_critics=2,
    initial_temperature=1.0,
    actor_encoder_factory=CustomEncoderFactory(feature_size=feature_size),
    critic_encoder_factory=CustomEncoderFactory(feature_size=feature_size)
)

# Initialize SAC
sac = SAC(config=sac_config, device='cpu')

# Build the model with the dataset
sac.build_with_dataset(dataset)

# Print the d3rlpy model keys
#print("D3RLPY Model Keys:")
#print(sac.impl.policy.state_dict().keys())

# Load the trained model
model = sb3_SAC.load("/home/nomads/Documents/elia/d3rlpy/test_policy/Marlin-v1")


#model.save("/home/nomads/Documents/elia/d3rlpy/test_policy/online_policy_S=200k_B=100k_SAVED.zip")
#model = sb3_SAC.load("/home/nomads/Documents/elia/d3rlpy/test_policy/online_policy_S=200k_B=100k_SAVED.zip")

# Save the policy's state dict
#torch.save(model.policy.state_dict(), "/home/nomads/Documents/elia/d3rlpy/test_policy/online_policy_S=200k_B=100k_state.pth")
#state_dict = torch.load("/home/nomads/Documents/elia/d3rlpy/test_policy/online_policy_S=200k_B=100k_state.pth")

state_dict = model.policy.state_dict()

# Print the SB3 model state dictionary keys
#print("SB3 Model State Dictionary Keys:")
#print(state_dict.keys())

# Rename the state dictionary keys
renamed_state_dict = rename_keys(state_dict)

# Print the renamed keys
#print("Renamed State Dictionary Keys:")
#print(renamed_state_dict.keys())

# Loading the state dictionary into the d3rlpy model
load_policy_state(sac, renamed_state_dict)

# training
sac.fit(dataset,
        n_steps=1000,
        n_steps_per_epoch=200,
        experiment_name='sac_offline_rl_custom_dataset')

# Save model
sac.save_model('model_marlin_continued.pt')
