import time
from pettingzoo.atari import tennis_v3
import supersuit as ss
from manual_policy import ManualPolicy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, env.action_space(env.agents[0]).n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.clone()
        x = x.unsqueeze(0)
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x = x.unsqueeze(0)
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


env = tennis_v3.env(render_mode="human")
env = ss.max_observation_v0(env, 2)
env = ss.frame_skip_v0(env, 4)
env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v1(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 4)
env = ss.agent_indicator_v0(env, type_only=False)

env.reset()

manual_policy = ManualPolicy(env, agent_id=1)

agent_model = Agent(env)

state_dict = torch.load("model.state", map_location=torch.device("cpu"))
agent_model.load_state_dict(state_dict)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if agent == manual_policy.agent:
        action = manual_policy(observation, agent)
    else:
        # this is where you would insert your policy
        # action = env.action_space(agent).sample()
        action, _, _, _ = agent_model.get_action_and_value(torch.Tensor(observation))
        action = action.item()

    env.step(action)

    if termination or truncation:
        env.reset()
    time.sleep(1000/60/1000)
