from ppo_agent.agent import CadreAgent
from ppo_agent.storage import RolloutStorage
# from leaderboard.leaderboard.env_wrapper import EnvWrapper
from env_wrapper import EnvWrapper
from ppo_agent.models import get_vae_output
import torch
import time
from tqdm import tqdm

def train(rank, train_cfg, agent_cfg, env_cfg, rollout_cfg, traffic_light=None, counter=None,
          shared_model_list=None, shared_grad_buffers=None, son_process_counter=None):
    env_cfg.rank = rank
    env_cfg.port = env_cfg.port[rank]
    env_cfg.routes = env_cfg.routes[rank]
    env_cfg.scenarios = env_cfg.scenarios[rank]
    env_cfg.town = env_cfg.town[rank]
    env_cfg.seq_length = rollout_cfg.seq_length
    env = EnvWrapper(env_cfg)

    max_episode = train_cfg.max_episode
    use_adv_norm = train_cfg.use_adv_norm
    ppo_epoch = train_cfg.ppo_epoch

    num_steps = rollout_cfg.num_steps
    hidden_size, _ = get_vae_output(agent_cfg.model_cfg)
    agent_cfg.rank = rank
    agent = CadreAgent(**agent_cfg)

    device = agent_cfg.model_cfg.device_num
    if device == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(device))

    rollout_cfg.hidden_size = hidden_size
    steer_rollout = RolloutStorage(**rollout_cfg)
    steer_rollout.to(device)

    throttle_rollout = RolloutStorage(**rollout_cfg)
    throttle_rollout.to(device)

    obs = env.reset()
    done = False

    for episode in tqdm(range(max_episode)):

        for steps in range(num_steps):
            command = obs['command']
            obs_feature, action, action_log_probs, value_preds, hidden_state = agent.act(obs)
            control = agent.convert_action(action)
            obs, reward, done, info = env.step(control)
            action_done = info['action_done']
            steer_masks = torch.tensor([[0.0] if action_done[0] else [1.0]])
            throttle_masks = torch.tensor([[0.0] if action_done[1] else [1.0]])

            steer_action, throttle_action = action
            steer_action_log_probs, throttle_action_log_probs = action_log_probs
            steer_value, throttle_value = value_preds
            steer_reward, throttle_reward = reward

            steer_rollout.insert(obs_feature, steer_action, steer_action_log_probs, steer_value, steer_reward,
                                 steer_masks, hidden_state, command)
            throttle_rollout.insert(obs_feature, throttle_action, throttle_action_log_probs, throttle_value,
                                    throttle_reward, throttle_masks, hidden_state, command)
            if done:
                obs = env.reset()

        steer_batch = steer_rollout.get_last()
        throttle_batch = throttle_rollout.get_last()
        next_steer_value, next_throttle_value = agent.get_value(done, steer_batch, throttle_batch)

        steer_rollout.compute_returns(next_steer_value.detach())
        steer_advantages = steer_rollout.returns[:-1] - steer_rollout.value_preds[:-1]
        throttle_rollout.compute_returns(next_throttle_value.detach())
        throttle_advantages = throttle_rollout.returns[:-1] - throttle_rollout.value_preds[:-1]
        if use_adv_norm:
            steer_advantages = (steer_advantages - steer_advantages.mean()) / (steer_advantages.std() + 1e-8)
            throttle_advantages = (throttle_advantages - throttle_advantages.mean()) / (
                    throttle_advantages.std() + 1e-8)

        for _ in range(ppo_epoch):
            steer_data_generator = steer_rollout.feed_forward_generator(steer_advantages)
            throttle_data_generator = throttle_rollout.feed_forward_generator(throttle_advantages)
            for steer_samples, throttle_samples in zip(steer_data_generator, throttle_data_generator):
                agent.update_policy(steer_samples, throttle_samples)

                signal_init = traffic_light.get()
                shared_grad_buffers.add_gradient(agent.model_dict)
                counter.increment()
                st_time = time.time()
                while traffic_light.get() == signal_init:
                    last_time = time.time() - st_time
                    if last_time % (60 * 60) // 60 == 5 and last_time % (60 * 60 * 60) == 0:
                        print('timeout in ', rank, counter.get(), ' in episode ', episode)
                    continue

    son_process_counter.increment()
    print('process {} finished.'.format(rank))
    return
