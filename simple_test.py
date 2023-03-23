from env_wrapper import EnvWrapper
from ppo_agent.meta.config import Config
from ppo_agent.agent import CadreAgent
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def visualize_rgb(obs):
    obs = np.array(obs, dtype=np.uint8)
    obs1 = np.hstack(obs[0:4])
    obs2 = np.hstack(obs[4:8])
    obs = np.vstack([obs1, obs2])
    img = Image.fromarray(obs)
    plt.axis('off')
    plt.imshow(img)
    plt.draw()
    plt.pause(0.1)
    plt.clf()

all_config = Config.fromfile("config_files/agent_config.py")

env_cfg = all_config.env_cfg
rank = 0
env_cfg.rank = rank
env_cfg.port = env_cfg.port[rank]
env_cfg.routes = env_cfg.routes[rank]
env_cfg.scenarios = env_cfg.scenarios[rank]
env_cfg.town = env_cfg.town[rank]
env_cfg.seq_length = all_config.rollout_cfg.seq_length
env = EnvWrapper(env_cfg)

agent_cfg = all_config.agent_cfg
agent_cfg.rank = rank
agent = CadreAgent(**agent_cfg)

# render = False
render = True
for episode in range(5):
    done = False
    obs = env.reset()
    sum_reward = 0
    while not done:
        _, action, *_ = agent.act(obs)
        control = agent.convert_action(action)

        # action = env.action_space.sample()
        control[2] = 0
        obs, reward, done, info = env.step(control)
        sum_reward += reward
        if render:
            visualize_rgb(obs['rgb'])
        if done:
            print('===========> in episode {}, sum reward for steer is {:.2f}, sum reward for throttle is {:.2f}\n'.format(episode, sum_reward[0].item(), sum_reward[1].item()))
            break



