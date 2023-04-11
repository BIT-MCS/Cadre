from ppo_agent.models import create_model, get_vae_output
import torch
import numpy as np
import time
from utils.logger import logger


class CadreAgent(object):
    def __init__(self, rank, model_cfg, frame, STEER_CONTROL, THROTTLE_CONTROL, ent_coeff,
                 value_coeff, clip_coeff, clip):
        self.rank = rank
        self.vae_model, self.model_dict = create_model(model_cfg, load_vae=True)
        self.use_lstm = model_cfg.use_lstm
        device = model_cfg.device_num
        self.command_num = model_cfg.command_num
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(device))
        self.device = device

        vae_device = model_cfg.vae_device
        if vae_device == -1:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda:" + str(vae_device))
        self.vae_device = vae_device
        self.STEER_CONTROL = STEER_CONTROL
        self.THROTTLE_CONTROL = THROTTLE_CONTROL
        self.ent_coeff = ent_coeff
        self.value_coeff = value_coeff
        self.clip_coeff = clip_coeff
        self.clip = clip
        self.lstm_input, self.vae_params = get_vae_output(model_cfg)
        self.use_vae = True
        self.frame = frame
        self.pre_latent_feature = None
        self.hidden_state = (
            torch.zeros(1, self.lstm_input).to(self.device),
            torch.zeros(1, self.lstm_input).to(self.device))
        self.pre_control = None


    def pre_process(self, tick_data):
        img = None
        if self.use_vae:
            rgb = np.array(tick_data['rgb'] / 255., dtype=np.float32)
            # img = torch.from_numpy(rgb)
            img = rgb.transpose(0, 3, 1, 2)

            if self.vae_params.in_route:
                for index in range(tick_data['route_fig'].shape[0]):
                    max_data = np.max(tick_data['route_fig'][index]) * 1.0
                    if max_data > 0:
                        tick_data['route_fig'][index] = 1.0 * tick_data['route_fig'][index] / max_data
                route_fig = np.array(tick_data['route_fig'], dtype=np.float32)
                route_fig = route_fig.swapaxes(1, 2)
                route_fig = np.expand_dims(route_fig, 1)
            else:
                route_fig = None
            rgb_topdown = route_fig
        else:
            route_fig = tick_data['route_fig']
            if np.max(tick_data['route_fig']) > 0:
                route_fig = tick_data['route_fig'] / np.max(tick_data['route_fig'])
            route_fig = np.array(route_fig, dtype=np.float32)
            route_fig = route_fig.swapaxes(0, 1)
            route_fig = np.expand_dims(route_fig, 0)

            rgb_fig = np.array(tick_data['rgb'] / 255., dtype=np.float32)
            rgb_fig = rgb_fig.swapaxes(0, 2)
            rgb_fig = rgb_fig.swapaxes(1, 2)
            rgb_topdown = np.concatenate([route_fig, rgb_fig], axis=0)

        output = np.concatenate([img, rgb_topdown], axis=1)
        return output

    def convert_action(self, discrete_action):
        steer = self.STEER_CONTROL[discrete_action[0].item()]
        throttle, brake = self.THROTTLE_CONTROL[discrete_action[1].item()]
        action = [steer, throttle, brake]
        self.pre_control = [steer, throttle, brake]
        return action

    def get_latent_feature(self, tick_data):
        image_output = self.pre_process(tick_data)
        # image_output = [rgb, route] 8 * 4 * 144 * 256
        image_output = torch.from_numpy(image_output).to(self.vae_device)
        # latent_feature: 8 * 512
        latent_feature = self.vae_model.get_latent_feature(image_output, "concate")
        latent_feature = latent_feature.clone().detach().to(self.device)

        # tick_data['measurements'] 8*3
        measurements = tick_data['measurements']

        # measurements 8*3 -> 8*18
        measurements = torch.from_numpy(measurements).to(self.device)
        measurements = measurements.repeat(1, 6)
        latent_feature = torch.cat([latent_feature, measurements], dim=-1).float()
        return latent_feature

    def act(self, tick_data):
        command = tick_data['command']

        ppo_feature = self.get_latent_feature(tick_data)
        if self.use_lstm:

            steer_lstm_model = self.model_dict['steer_lstm_' + str(command)]
            throttle_lstm_model = self.model_dict['throttle_lstm_' + str(command)]

            steer_ppo_feature, hidden_state = steer_lstm_model(ppo_feature, self.hidden_state)
            throttle_ppo_feature, hidden_state = throttle_lstm_model(ppo_feature, self.hidden_state)

        else:
            steer_ppo_feature = ppo_feature
            throttle_ppo_feature = ppo_feature

        steer_ppo_model = self.model_dict['steer_ppo_' + str(command)]
        throttle_ppo_model = self.model_dict['throttle_ppo_' + str(command)]
        steer_value, steer_action, _ = steer_ppo_model.act(steer_ppo_feature)
        throttle_value, throttle_action, _ = throttle_ppo_model.act(throttle_ppo_feature)

        steer_action_log_probs = steer_ppo_model.get_log_probs(steer_action)
        throttle_action_log_probs = throttle_ppo_model.get_log_probs(throttle_action)
        return ppo_feature, \
               [steer_action[0], throttle_action[0]], \
               [steer_action_log_probs, throttle_action_log_probs], \
               [steer_value, throttle_value], \
               self.hidden_state

    def get_value(self, done, steer_batch, throttle_batch):
        with torch.no_grad():
            if done:
                next_value_steer = torch.zeros(1)
                next_value_throttle = torch.zeros(1)
            else:
                steer_obs_batch, steer_command = steer_batch
                throttle_obs_batch, throttle_command = throttle_batch
                if self.use_lstm:
                    steer_lstm_model = self.model_dict['steer_lstm_' + str(steer_command)]
                    # throttle_lstm_model = self.model_dict['throttle_lstm_3']
                    throttle_lstm_model = self.model_dict['throttle_lstm_' + str(throttle_command)]
                    steer_obs_batch, hidden_state = steer_lstm_model(steer_obs_batch,
                                                                     self.hidden_state)
                    throttle_obs_batch, hidden_state = throttle_lstm_model(throttle_obs_batch,
                                                                           self.hidden_state)

                next_value_steer = self.model_dict['steer_ppo_' + str(steer_command)].get_value(steer_obs_batch)
                next_value_throttle = self.model_dict['throttle_ppo_' + str(throttle_command)].get_value(
                    throttle_obs_batch)

            return next_value_steer, next_value_throttle

    def update_policy(self, steer_samples, throttle_samples):
        obs_batch, action_batch, old_values, return_batch, masks_batch, old_action_log_probs, advantages_batch, \
        hidden_state_batch, command_batch = steer_samples
        cur_values = cur_action_log_probs = dist_entropy = 0
        for _command in range(self.command_num):
            steer_obs_batch = obs_batch.clone()
            if self.use_lstm:
                lstm_model = self.model_dict['steer_lstm_' + str(_command)]
                steer_obs_batch, _ = lstm_model(steer_obs_batch, hidden_state_batch)
            steer_ppo_model = self.model_dict['steer_ppo_' + str(_command)]
            _command_batch = command_batch == _command
            _cur_values, _cur_action_log_probs, _dist_entropy = steer_ppo_model.evaluate_actions(
                steer_obs_batch, action_batch)
            cur_values = cur_values + _cur_values * _command_batch
            cur_action_log_probs = cur_action_log_probs + _cur_action_log_probs * _command_batch

            dist_entropy = dist_entropy + _dist_entropy * _command_batch

        ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages_batch
        action_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = old_values + (cur_values - old_values).clamp(-self.clip, self.clip)
        value_losses = (cur_values - return_batch).pow(2)
        value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        entropy_loss = dist_entropy.mean()

        obs_batch, action_batch, old_values, return_batch, masks_batch, old_action_log_probs, advantages_batch, \
        hidden_state_batch, command_batch = throttle_samples

        cur_values = cur_action_log_probs = dist_entropy = 0
        for _command in range(self.command_num):
            throttle_obs_batch = obs_batch.clone()
            # throttle_obs_batch = [obs_batch[0].clone()]
            if self.use_lstm:
                lstm_model = self.model_dict['throttle_lstm_' + str(_command)]
                throttle_obs_batch, _ = lstm_model(throttle_obs_batch, hidden_state_batch)

            throttle_ppo_model = self.model_dict['throttle_ppo_' + str(_command)]
            _command_batch = command_batch == _command
            _cur_values, _cur_action_log_probs, _dist_entropy = throttle_ppo_model.evaluate_actions(
                throttle_obs_batch, action_batch)

            cur_values = cur_values + _cur_values * _command_batch
            cur_action_log_probs = cur_action_log_probs + _cur_action_log_probs * _command_batch
            dist_entropy = dist_entropy + _dist_entropy * _command_batch

        ratio = torch.exp(cur_action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages_batch

        action_loss += -torch.min(surr1, surr2).mean()
        value_pred_clipped = old_values + (cur_values - old_values).clamp(-self.clip, self.clip)
        value_losses = (cur_values - return_batch).pow(2)
        value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
        value_loss += 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        entropy_loss += dist_entropy.mean()

        value_loss = value_loss * self.value_coeff
        action_loss = action_loss * self.clip_coeff
        ent_loss = entropy_loss * self.ent_coeff
        total_loss = value_loss + action_loss - ent_loss

        for model_name in self.model_dict:
            model = self.model_dict[model_name]
            model.zero_grad()

        total_loss.backward()

        return value_loss.item(), action_loss.item(), ent_loss.item()

    def update_model(self, shared_model_list):
        for model_name in self.model_dict:
            model = self.model_dict[model_name]
            shared_model = shared_model_list[model_name]
            model.load_state_dict(shared_model.state_dict())

    def save_snapshot(self, model_path):
        model_dict = {}
        for _command in range(self.command_num):
            model_name = 'throttle_ppo_' + str(_command)
            model_dict[model_name] = self.model_dict[model_name]

            model_name = 'steer_ppo_' + str(_command)
            model_dict[model_name] = self.model_dict[model_name]

            model_name = 'steer_lstm_' + str(_command)
            model_dict[model_name] = self.model_dict[model_name]

            model_name = 'steer_ppo_' + str(_command)
            model_dict[model_name] = self.model_dict[model_name]

        torch.save(model_dict, model_path)

    def load_snapshot(self, model_path):
        try:
            model_dict = torch.load(model_path, map_location=self.device)
            for name in model_dict:
                self.model_dict[name].load_state_dict(model_dict[name])
        except Exception as e:
            print('load snapshot error due to ', e)
            exit(-1)
