import copy

import carla
from ppo_agent.meta.config import Config
from ppo_agent.agent import CadreAgent
from env_wrapper import EnvWrapper
import torch

import os


if __name__ == '__main__':
    # 1. Load your config
    all_config = Config.fromfile("config_files/eval_agent_config.py")
    agent_cfg = all_config.agent_cfg
    device = agent_cfg.model_cfg.device_num

    rank = 0
    agent_cfg.rank = rank

    rollout_cfg = all_config.rollout_cfg

    eval_cfg = all_config.eval_cfg
    pretrained_path = eval_cfg.pretrained_path
    load_episode = eval_cfg.load_episode
    agent_num = len(load_episode)
    eval_episode = eval_cfg.eval_episode

    env_cfg = all_config.env_cfg
    env_cfg.rank = rank
    env_cfg.port = env_cfg.port[rank]
    env_cfg.routes = env_cfg.routes[rank]
    env_cfg.scenarios = env_cfg.scenarios[rank]
    env_cfg.town = env_cfg.town[rank]
    env_cfg.seq_length = rollout_cfg.seq_length
    env_cfg.pretrained_path = pretrained_path
    if device == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(device))

    # 2. Create eval env
    env = EnvWrapper(env_cfg)

    # 3. Create agent
    agent_group = []
    for i in range(agent_num):
        agent = CadreAgent(**agent_cfg)
        snapshot_path = os.path.join(pretrained_path, 'models', 'ppo_model_{}.pt'.format(load_episode[i]))
        agent.load_snapshot(snapshot_path, device)
        agent_group.append(agent)

    for episode in range(eval_episode):
        obs = env.reset()
        done = False
        while not done:
            action_list = []
            for i in range(agent_num):
                agent = agent_group[i]
                _, action, *_ = agent.act(obs)
                action_list.append(action)
            control = agent.avg_action(action_list)
            obs, reward, done, info = env.step(control)
    print('Evaluation done. Results are saved under {}.'.format(env.average_completion_ratio_path))






