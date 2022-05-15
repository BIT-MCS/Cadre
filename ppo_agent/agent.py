from ppo_agent.models import create_model, get_vae_output
import torch
import numpy as np
import time


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
        self.hidden_state = None
        self.pre_control = None
        self.reset()

    # def update(self):
    def reset(self):
        self.pre_latent_feature = np.zeros((self.frame, self.lstm_input), dtype=np.float32)
        if self.use_lstm:
            self.hidden_state = (
                torch.zeros(1, self.lstm_input).to(self.device), torch.zeros(1, self.lstm_input).to(self.device))

        else:
            self.hidden_state = None
        self.pre_control = [0, 0, 0]

    def pre_process(self, tick_data):
        img = None
        if self.use_vae:
            rgb = np.array(tick_data['rgb'] / 255., dtype=np.float32)
            img = torch.from_numpy(rgb)
            # img = torch.from_numpy(tick_data['rgb']).float() / 255
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)

            if self.vae_params.in_route:
                max_data = np.max(tick_data['route_fig'])
                if max_data == 0:
                    route_fig = tick_data['route_fig']
                else:
                    route_fig = tick_data['route_fig'] / max_data

                route_fig = np.array(route_fig, dtype=np.float32)
                route_fig = route_fig.swapaxes(0, 1)
                route_fig = np.expand_dims(route_fig, 0)
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
        return img, rgb_topdown

    def convert_action(self, discrete_action):
        steer = self.STEER_CONTROL[discrete_action[0].item()]
        throttle, brake = self.THROTTLE_CONTROL[discrete_action[1].item()]
        action = [steer, throttle, brake]
        self.pre_control = [steer, throttle, brake]
        return action

    def act(self, tick_data):
        command = tick_data['command']
        image, topdown = self.pre_process(tick_data)
        speed = tick_data['speed']

        # [1, 3, 144, 256]
        input_list = image
        input_list = input_list.to(self.vae_device)
        if self.vae_params.in_route:
            route_fig = torch.from_numpy(topdown).float()
            route_fig = route_fig.view(1, 1, 144, 256).to(self.vae_device)
            input_list = torch.cat((input_list, route_fig), dim=1)

        if self.vae_params.in_speed:
            cur_speed = speed * torch.ones(1, 1, 144, 256).to(self.vae_device)
            input_list = torch.cat((input_list, cur_speed), dim=1)

        latent_feature = self.vae_model.get_latent_feature(input_list, "concate")

        latent_feature = latent_feature.clone().detach().to(self.device)

        measurements = tick_data['measurements']

        measurements = torch.tensor(
            [measurements[0], measurements[1], measurements[2], self.pre_control[0], self.pre_control[1],
             self.pre_control[2]], dtype=torch.float32).to(self.device)

        measurements = measurements.repeat(3).unsqueeze(0)
        latent_feature = torch.cat((latent_feature, measurements), dim=-1)

        self.pre_latent_feature[:-1] = self.pre_latent_feature[1:]
        self.pre_latent_feature[-1] = latent_feature.clone().cpu().numpy()
        ppo_feature = torch.from_numpy(self.pre_latent_feature).to(self.device)

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
        return latent_feature, \
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
                    if self.frame == 1:
                        self.hidden_state = hidden_state
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
        hidden_state_batch,command_batch = throttle_samples

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
        # ------------------ for curiosity driven--------------------------
        total_loss = value_loss + action_loss - ent_loss

        for model_name in self.model_dict:
            model = self.model_dict[model_name]
            model.zero_grad()
        total_loss.backward()

        # if self.thread_num == 1:
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()
        # else:
        #     # --------------- add gradient and update parameters--------------
        #     signal_init = self.traffic_light.get()
        #     self.shared_grad_buffers.add_gradient(self.model_dict)
        #     self.counter.increment()
        #     st_time = time.time()
        #     while self.traffic_light.get() == signal_init:
        #         last_time = time.time() - st_time
        #         if last_time % (60 * 60) // 60 == 5 and last_time % (60 * 60 * 60) == 0:
        #             print('timeout in ', self.rank, self.counter.get(), ' in episode ', self.episode, '_',
        #                   cnt)
        #             # while True:
        #             #     continue
        #             # # print('timeout in ', self.rank, self.counter[self.rank].get())
        #             # timeout = True
        #             # break
        #         continue
        #     cnt += 1
        #
        #     for model_name in self.model_dict:
        #         model = self.model_dict[model_name]
        #         shared_model = self.shared_model_list[model_name]
        #         model.load_state_dict(shared_model.state_dict())

        return value_loss.item(), action_loss.item(), ent_loss.item()
