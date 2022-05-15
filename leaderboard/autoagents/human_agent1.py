#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import time
import json
from threading import Thread
import cv2
import numpy as np
import torch
from agents.navigation.local_planner import RoadOption

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import carla
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# from team_code.planner import RoutePlanner, Plotter, DynamicRoutePlanner
from team_code.planner import RoutePlanner, Plotter, RoutePlanner
import math
from carla_perception.Data.datasets import get_img_transform, get_lidar_transform
from torch.distributions import Bernoulli

# from carla_perception.Networks.vanilla_vae import VanillaVAE
from carla_perception.Networks.danet import DANet
from carla_perception.Networks.da_beta_vae import DABetaVae
from carla_perception.Networks.beta_vae import BetaVAE
from carla_perception.Data.datasets import get_img_transform, get_lidar_transform
from carla_perception.Config.auto_vanilla_vae import vanilla_vae_config
# vae_params = vanilla_vae_config()
from carla_perception.Config.auto_danet import danet_config
from leaderboard.utils.converter import Converter

vae_params = danet_config()
# todo: debug
# vae_params = da_beta_vae_config()
from config_files.config_ppo_model import Params

ppo_params = Params()

import os

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
print('DEBUG', DEBUG)

from config_files.config_hdmap_agent import CONF

config_agent = CONF()

from carla import WeatherParameters
from team_code.base_agent import PRESET_WEATHERS


def get_entry_point():
    return 'HumanAgent1'


def draw_bbx(tick_data, _bbx):
    # print(bbx)

    _rgb = Image.fromarray(tick_data['rgb_light'])
    _draw_rgb = ImageDraw.Draw(_rgb)

    left_up, right_down = _bbx
    print(_bbx)
    _draw_rgb.rectangle((right_down[1] + 5, right_down[0] + 5, left_up[1] - 5, left_up[0] - 5), outline=(0, 0, 255),
                        fill=None, width=3)
    cv2.imshow('traffic-light-rgb', cv2.cvtColor(np.array(_rgb), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


def draw_light(tick_data, cam_point):
    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    x, y = cam_point
    # x = (x + 1) / 2 * 256
    # y = (y + 1) / 2 * 144
    print('point:', x, y)
    _draw_rgb.ellipse((x - 10, y - 10, x + 10, y + 10), (0, 0, 255))
    cv2.imshow('traffic-light-rgb', cv2.cvtColor(np.array(_rgb), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)


def debug_display(tick_data):
    # rgb_topdown = Image.fromarray(tick_data['topdown'])
    # cv2.imshow('seg-topdown', cv2.cvtColor(np.array(rgb_topdown), cv2.COLOR_BGR2RGB))
    #
    # rgb_center = Image.fromarray(tick_data['rgb'])
    # cv2.imshow('rgb', cv2.cvtColor(np.array(rgb_center), cv2.COLOR_BGR2RGB))

    _route = tick_data['route_fig']
    cv2.imshow('human_agent_route_fig', np.array(_route).astype('uint8'))

    # _road = tick_data['road_fig']
    # cv2.imshow('road_fig', np.array(_road))
    #
    # _vehicle = tick_data['vehicle_fig']
    # cv2.imshow('vehicle_fig', np.array(_vehicle))


def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]
    ])


def decode_segmap_cv(label_mask, dataset, sailent_class_index, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
        the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
        in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        print()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        if ll in sailent_class_index:
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        else:
            r[label_mask == ll] = 0
            g[label_mask == ll] = 0
            b[label_mask == ll] = 0

    r[label_mask == 255] = 0
    g[label_mask == 255] = 0
    b[label_mask == 255] = 0

    rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
    # replace blue with red as opencv uses bgr
    rgb[:, :, 0] = b  # /255.0
    rgb[:, :, 1] = g  # /255.0
    rgb[:, :, 2] = r  # /255.0
    #
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def debug_save(tick_data, step):
    _rgb = Image.fromarray(tick_data['rgb_light'])
    _rgb.save(os.path.join('datasets', ('rgb-%s.png' % str(step))))

def save_vae(rgb_seg_pred_list, step):
    display_path = '/home/quan/distribution/result/vary_beta_vae_1'
    _combine_image = []
    for rgb_seg_pred in rgb_seg_pred_list:
        seg_predictions = torch.max(rgb_seg_pred, 1)[1]
        seg_pred = seg_predictions.detach().cpu().numpy()

        sailent_class_index = [0, 1, 2, 3, 4, 5, 6, 7]

        cur_img_index = 0

        cur_seg_pred = np.expand_dims(seg_pred[cur_img_index], axis=0)
        pred_color = decode_segmap_cv(cur_seg_pred, 'cityscapes', sailent_class_index)
        pred_color = pred_color[..., ::-1]
        _combine_image.append(pred_color)

    _combine_img = np.hstack(_combine_image)
    _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'RGB')
    # todo:debug
    _combine_img.save(os.path.join(display_path, ('rgb-%04d.png' % step)))

    # cv2.imshow('rgb, rgb_pred, rgb_tar', cv2.cvtColor(np.array(_combine_img), cv2.COLOR_BGR2RGB))


def debug_vae(img_target, rgb_seg_pred, img_seg_list, \
              left_img_input_list, left_img_pred, left_img_seg_resort, \
              right_img_input_list, right_img_pred, right_img_seg_resort, \
              route_list, route_pred, step):
    # rgb = Image.fromarray(rgb)
    # cv2.imshow('rgb', cv2.cvtColor(np.array(rgb), cv2.COLOR_BGR2RGB))

    # lidar = Image.fromarray(lidar)
    # cv2.imshow('lidar', cv2.cvtColor(np.array(lidar), cv2.COLOR_BGR2RGB))
    #
    # topdown = Image.fromarray(topdown)
    # cv2.imshow('topdown', cv2.cvtColor(np.array(topdown), cv2.COLOR_BGR2RGB))

    b, c, h, w = img_seg_list.size()
    img_seg_list = img_seg_list.detach().cpu().numpy()
    left_img_seg_resort = left_img_seg_resort.detach().cpu().numpy()
    right_img_seg_resort = right_img_seg_resort.detach().cpu().numpy()

    seg_predictions = torch.max(rgb_seg_pred, 1)[1]
    seg_pred = seg_predictions.detach().cpu().numpy()

    if left_img_pred is not None:
        left_seg_predictions = torch.max(left_img_pred, 1)[1]
        left_seg_pred = left_seg_predictions.detach().cpu().numpy()

    if right_img_pred is not None:
        right_seg_predictions = torch.max(right_img_pred, 1)[1]
        right_seg_pred = right_seg_predictions.detach().cpu().numpy()

    sailent_class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    cur_img_index = 0

    ################
    cur_img_target = img_target[cur_img_index]
    cur_img_target = np.clip(cur_img_target, 0.0, 1.0)
    cur_img_target = np.transpose(cur_img_target, (1, 2, 0))
    cur_img_target = cur_img_target * 255.0

    cur_seg_pred = np.expand_dims(seg_pred[cur_img_index], axis=0)
    pred_color = decode_segmap_cv(cur_seg_pred, 'cityscapes', sailent_class_index)
    pred_color = pred_color[..., ::-1]

    cur_seg_tar = img_seg_list[cur_img_index]
    target_color = decode_segmap_cv(cur_seg_tar, 'cityscapes', sailent_class_index)
    target_color = target_color[..., ::-1]

    _combine_img = np.hstack([cur_img_target, pred_color, target_color])
    _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'RGB')
    # todo:debug
    cv2.imshow('rgb, rgb_pred, rgb_tar', cv2.cvtColor(np.array(_combine_img), cv2.COLOR_BGR2RGB))

    left_img_input_list = left_img_input_list[cur_img_index]
    left_img_input_list = np.clip(left_img_input_list, 0.0, 1.0)
    left_img_input_list = np.transpose(left_img_input_list, (1, 2, 0))
    left_img_input_list = left_img_input_list * 255.0

    right_img_input_list = right_img_input_list[cur_img_index]
    right_img_input_list = np.clip(right_img_input_list, 0.0, 1.0)
    right_img_input_list = np.transpose(right_img_input_list, (1, 2, 0))
    right_img_input_list = right_img_input_list * 255.0

    _rgb_combine_img = np.hstack([left_img_input_list, cur_img_target, right_img_input_list])
    _rgb_combine_img = Image.fromarray(_rgb_combine_img.astype('uint8'), 'RGB')
    cv2.imshow('left rgb, rgb, right rgb', cv2.cvtColor(np.array(_rgb_combine_img), cv2.COLOR_BGR2RGB))

    cur_img_target = cur_img_target.clone().detach().cpu().numpy()
    _save_cur_image_target = Image.fromarray(cur_img_target.astype('uint8'), 'RGB')
    # _save_cur_image_target.save('result/image_' + str(step) + '.jpg')

    _save_target_color = Image.fromarray(target_color.astype('uint8'), 'RGB')
    # _save_target_color.save('result/seg_image_' + str(step) + '.jpg')

    # _combine_img.show('rgb, rgb_pred, rgb_tar')
    # _combine_img.save('/data2/wk/carla/1_2020_CARLA_challenge/test1.jpg')

    #######################
    if left_img_pred is not None:
        cur_left_img_target = left_img_input_list[cur_img_index]
        cur_left_img_target = np.clip(cur_left_img_target, 0.0, 1.0)
        cur_left_img_target = np.transpose(cur_left_img_target, (1, 2, 0))
        cur_left_img_target = cur_left_img_target * 255.0

        cur_left_seg_pred = np.expand_dims(left_seg_pred[cur_img_index], axis=0)
        left_pred_color = decode_segmap_cv(cur_left_seg_pred, 'cityscapes', sailent_class_index)
        left_pred_color = left_pred_color[..., ::-1]

        cur_left_seg_tar = left_img_seg_resort[cur_img_index]
        left_target_color = decode_segmap_cv(cur_left_seg_tar, 'cityscapes', sailent_class_index)
        left_target_color = left_target_color[..., ::-1]

        _combine_img = np.hstack([cur_left_img_target, left_pred_color, left_target_color])
        _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'RGB')

    #######################
    if right_img_pred is not None:
        cur_right_img_target = right_img_input_list[cur_img_index]
        cur_right_img_target = np.clip(cur_right_img_target, 0.0, 1.0)
        cur_right_img_target = np.transpose(cur_right_img_target, (1, 2, 0))
        cur_right_img_target = cur_right_img_target * 255.0

        cur_right_seg_pred = np.expand_dims(right_seg_pred[cur_img_index], axis=0)
        right_pred_color = decode_segmap_cv(cur_right_seg_pred, 'cityscapes', sailent_class_index)
        right_pred_color = right_pred_color[..., ::-1]

        cur_right_seg_tar = right_img_seg_resort[cur_img_index]
        right_target_color = decode_segmap_cv(cur_right_seg_tar, 'cityscapes', sailent_class_index)
        right_target_color = right_target_color[..., ::-1]

        _combine_img = np.hstack([cur_right_img_target, right_pred_color, right_target_color])
        _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'RGB')

    ################
    if route_pred is not None:
        b, c, h, w = route_list.size()
        route_list = route_list.detach().cpu().numpy()
        route_pred = route_pred.detach().cpu().numpy()

        cur_route_target = route_list[cur_img_index]
        cur_route_pred = route_pred[cur_img_index]

        _combine_img = np.hstack([cur_route_pred * 255, cur_route_target * 255])
        _combine_img = np.squeeze(_combine_img, axis=0)
        _combine_img = _combine_img.swapaxes(0, 1)
        _combine_img = Image.fromarray(_combine_img.astype('uint8'), 'L')
        # cv2.imshow('route_pred, route_target', cv2.cvtColor(np.array(_combine_img), cv2.COLOR_BGR2RGB))

        # todo: don't show route_pred and route_target
        cv2.imshow('route_pred, route_target', np.array(_combine_img))

        # _combine_img.show('route_pred, route_target')
        # _combine_img.save('/data2/wk/carla/1_2020_CARLA_challenge/test2.jpg')


def seg_to_rgb(image):
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3))
    rgb_image[image == 1] = (0, 0, 255)
    rgb_image[image == 2] = (0, 128, 128)
    rgb_image[image == 3] = (255, 0, 0)
    rgb_image[image == 4] = (255, 255, 0)
    rgb_image[image == 5] = (0, 255, 0)
    rgb_image = np.array(rgb_image, dtype=np.uint8)
    rgb_image = Image.fromarray(rgb_image, 'RGB')

    return rgb_image


import queue


class BFS(object):
    def __init__(self, _threshold=20):
        self.dx = [0, 0, 1, -1, -1, -1, 1, 1]
        self.dy = [1, -1, 0, 0, -1, 1, -1, 1]
        self._threshold = _threshold

    def reset(self, rgb_seg_tar):
        rgb_seg_tar = rgb_seg_tar
        self.img_seg = np.asarray(rgb_seg_tar)
        self.img_vis = np.zeros((self.img_seg.shape[0], self.img_seg.shape[1]))

    def find_bbx(self):
        q = queue.Queue()
        bbx = []
        max_area = self._threshold
        for h in range(self.img_seg.shape[0]):
            for w in range(self.img_seg.shape[1]):
                if not self.img_vis[h][w]:
                    self.img_vis[h][w] = 1
                    if self.img_seg[h][w] == 18:
                        q.put((h, w))
                        left_up = [h, w]
                        right_down = [h, w]
                        while not q.empty():
                            p = q.get()
                            for i in range(8):
                                next_h = p[0] + self.dx[i]
                                next_w = p[1] + self.dy[i]
                                if next_h < self.img_seg.shape[0] and next_w < self.img_seg.shape[
                                    1] and next_h >= 0 and next_w >= 0:
                                    if not self.img_vis[next_h][next_w]:
                                        self.img_vis[next_h][next_w] = 1
                                        if self.img_seg[next_h][next_w] == 18:
                                            left_up[0] = min(left_up[0], next_h)
                                            left_up[1] = min(left_up[1], next_w)
                                            right_down[0] = max(right_down[0], next_h)
                                            right_down[1] = max(right_down[1], next_w)
                                            q.put((next_h, next_w))
                        if (right_down[0] - left_up[0]) * (right_down[1] - left_up[1]) > max_area:
                            bbx = (left_up, right_down)
                            max_area = (right_down[0] - left_up[0]) * (right_down[1] - left_up[1])
        return bbx


class HumanInterface(object):
    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self):
        self._width = 400  # 800
        self._height = 300  # 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Human Agent")

    def run_interface(self, input_data):
        """
        Run the GUI
        """

        # process sensor data
        image_center = input_data['rgb_light'][1][:, :, -2::-1]
        self._clock.tick_busy_loop(20)
        # display image
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()


def save_display(rgb, route_fig, step, result_root_path):
    display_path = os.path.join(result_root_path, 'display_path')
    if not os.path.exists(display_path):
        # os.mkdir(display_path)
        os.makedirs(display_path)
    # print(route_fig)
    # print(route_fig.shape)
    route_fig = np.array(255 * route_fig).astype('uint8')
    route_fig = route_fig.squeeze(0)
    _route = Image.fromarray(route_fig)
    _route.save(os.path.join(display_path, ('route-%04d.png' % step)))

    rgb = np.array(255 * rgb[0]).astype('uint8')
    rgb = rgb.swapaxes(0, 2)
    rgb = rgb.swapaxes(0, 1)
    _rgb = Image.fromarray(rgb)
    _rgb.save(os.path.join(display_path, ('rgb-%04d.png' % step)))


from xml.dom import minidom


def generateXml(route_list):
    impl = minidom.getDOMImplementation()

    # 创建一个xml dom
    # 三个参数分别对应为 ：namespaceURI, qualifiedName, doctype
    doc = impl.createDocument(None, None, None)

    # 创建根元素
    rootElement = doc.createElement('routes')
    pythonId = 0

    # 为根元素添加10个子元素
    for route in route_list:
        routeElement = doc.createElement('route')
        routeElement.setAttribute('id', str(pythonId))
        routeElement.setAttribute('map', "Town02")
        pythonId += 1
        for waypoint in route:
            # 创建子元素
            childElement = doc.createElement('waypoint')
            # 为子元素添加id属性
            childElement.setAttribute('pitch', str(waypoint['pitch']))
            childElement.setAttribute('roll', str(waypoint['roll']))
            childElement.setAttribute('x', str(waypoint['x']))
            childElement.setAttribute('y', str(waypoint['y']))
            childElement.setAttribute('yaw', str(waypoint['yaw']))
            childElement.setAttribute('z', str(waypoint['z']))
            # 将子元素追加到根元素中
            routeElement.appendChild(childElement)

        # print(childElement.firstChild.data)
        rootElement.appendChild(routeElement)

    # 将拼接好的根元素追加到dom对象
    doc.appendChild(rootElement)

    # 打开test.xml文件 准备写入
    f = open('Nocrash_test_route.xml', 'a')
    # 写入文件
    doc.writexml(f, addindent='  ', newl='\n')
    # 关闭
    f.close()
    print('write done')


class HumanAgent1(AutonomousAgent):
    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False

    def setup(self, path_to_conf_file, a, b, c, d, e):
        """
        Setup the agent parameters
        """
        self._bfs = BFS()
        self.track = Track.SENSORS
        self.intersection_num = 0
        self.success_intersection_num = 0
        self.agent_engaged = False
        self._hic = HumanInterface()
        self._controller = KeyboardControl(path_to_conf_file)
        self.last_event_timestamp = 0
        self.use_vae = config_agent.use_vae
        self.detect_traffic_light = config_agent.detect_traffic_light
        self.use_light_vae = config_agent.use_light_vae
        self.vae_params = vae_params
        self._prev_timestamp = 0
        self.initialized = False
        self.step = 0
        self.last_corner_gps = [0, 0]
        self.rank = 0
        self.max_degree = ppo_params.max_degree
        self.max_speed = ppo_params.max_speed
        self.min_speed = ppo_params.min_speed
        self.target_speed = ppo_params.target_speed
        self.training = path_to_conf_file.training
        self.distance_mode = ppo_params.distance_mode
        self.traffic_light_state = [carla.TrafficLightState.Red, carla.TrafficLightState.Yellow,
                                    carla.TrafficLightState.Green]
        self.traffic_light_time = [50, 50, 150]
        self.traffic_light_counter = 0
        self.pre_traffic_light_time = 0
        self.episode = -1
        if self.use_vae:
            self.vae_params = vae_params
            z_dims = self.vae_params.networks['autoencoder']['z_dims']
            # load vae model
            vae_model_path = self.vae_params.networks['autoencoder']['pretrained_path']
            pretrained_model = torch.load(vae_model_path, map_location='cpu')
            # network = BetaVAE(self.vae_params.networks['autoencoder'])
            # self.vae_model = VanillaVAE(self.vae_params.networks['autoencoder'])
            # network = VanillaVAE(self.vae_params.networks['autoencoder'])
            network = DANet(self.vae_params.networks['autoencoder'])
            # todo: debug
            # network = DABetaVae(self.vae_params.networks['autoencoder'])
            key = 'autoencoder'
            if key in pretrained_model.keys():
                if pretrained_model[key].keys() == network.state_dict().keys():
                    print('==> network parameters in pre-trained file'
                          ' %s can strictly match' % (vae_model_path))
                    network.load_state_dict(pretrained_model[key])
                else:
                    print('VAE model load fail in ', vae_model_path)
            self.vae_model = network
            self.vae_device = 'cpu'
            self.vae_model = self.vae_model.to(self.vae_device)
            # self.vae_model.encoder = self.vae_model.encoder.to(self.vae_device)
            # self.vae_model.image_mu = self.vae_model.image_mu.to(self.vae_device)
            # self.vae_model.image_logvar = self.vae_model.image_logvar.to(self.vae_device)
            self.vae_model.eval()

        if self.use_light_vae:
            from carla_perception_light.Config.auto_danet import danet_config as danet_config_light
            light_vae_params = danet_config_light()
            self.light_vae_params = light_vae_params
            light_vae_model_path = light_vae_params.networks['autoencoder']['pretrained_path']
            pretrained_model = torch.load(light_vae_model_path, map_location=self.vae_device)
            from carla_perception_light.Networks.danet import DANet as LightDANet
            network = LightDANet(light_vae_params.networks['autoencoder'])
            key = 'autoencoder'
            if key in pretrained_model.keys():
                if pretrained_model[key].keys() == network.state_dict().keys():
                    if self.rank == 0:
                        print('==> network parameters in pre-trained file'
                              ' %s can strictly match' % (light_vae_model_path))
                    network.load_state_dict(pretrained_model[key])
                else:
                    if self.rank == 0:
                        pretrained_model_dict = list(pretrained_model[key].keys())
                        network_keys = list(network.state_dict().keys())
                        for _key in pretrained_model_dict:
                            if _key not in network_keys:
                                print(_key, ' does not exist in model config!')
            self.light_vae_model = network
            self.light_vae_model.to(self.vae_device)
            if self.training:
                # del self.vae_model.reverse_image
                if self.light_vae_params.pred_left_camera_seg:
                    del self.light_vae_model.reverse_left_image
                if self.light_vae_params.pred_right_camera_seg:
                    del self.light_vae_model.reverse_right_image
                # if self.vae_params.pred_route:
                #     del self.vae_model.reverse_route
                if self.light_vae_params.pred_light_dist:
                    del self.light_vae_model.reverse_lightDist
                if self.light_vae_params.pred_lidar:
                    del self.light_vae_model.reverse_lidar
                if self.light_vae_params.pred_topdown_rgb:
                    del self.light_vae_model.reverse_topdown_rgb
                if self.light_vae_params.pred_topdown_seg:
                    del self.light_vae_model.reverse_topdown_seg
                self.light_vae_model.eval()

        self.converter = Converter()

    def _init(self):
        # todo: check
        self.pre_far_command = RoadOption.LANEFOLLOW
        self.near_lights = np.zeros(20)
        self.intersection_stay_time = 0

        self.has_red_light = False
        self.pre_light_id = -100

        self.debug = Plotter(256, "debug")
        self.begin = True
        self.in_turn = False
        self.turn_first_node = None
        self.turn_last_node = None
        self.first_direction = 0
        self.last_direction = 0
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self.pre_pos = None
        self.pre_theta = 0
        self.pre_latent_feature = None
        # points = self._map.get_spawn_points()
        # total_route_list = []
        # for _point in points:
        #     location = _point.location
        #     rotation = _point.rotation
        #     waypoint = {'x': location.x, 'y': location.y, 'z': location.z, 'yaw': rotation.yaw, 'roll': rotation.roll,
        #                 'pitch': rotation.pitch}
        #     total_route_list.append(waypoint)
        # no_crash_file = open('nocrash/nocrash_Town02.txt')
        # no_crash_route_list = []
        # for _line in no_crash_file:
        #     st, ed = _line.strip().split(' ')
        #     st = int(st)
        #     ed = int(ed)
        #     no_crash_route_list.append([total_route_list[st], total_route_list[ed]])
        #
        # generateXml(no_crash_route_list)
        # print("successfully generate!")
        self.route_mode = ppo_params.route_mode
        self.debug = Plotter(256, "debug")
        self.last_corner_gps = [0, 0]
        self.in_turn = False
        self.turn_first_node = None
        self.turn_last_node = None
        self.first_direction = 0
        self.last_direction = 0
        self.initialized = True

        # self._light_planner = DynamicRoutePlanner(16, 200.0, 257)
        # self._light_planner = DynamicRoutePlanner(16, 200.0, 257)
        # self._light_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)
        # self._waypoint_planner = RoutePlanner(1.0, 50)
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)

        self.pre_waypoint = None
        self.sum_distance = 0
        self.route_list = []

        self.approach_light_time = 0
        self.pre_light_id = -1
        self.light_id = -1

        logits = torch.empty(1, self.vae_params.networks['autoencoder']['z_dims']).uniform_(0, 1)
        if self.use_vae:
            self.causal_graph = torch.bernoulli(logits).to(self.vae_device)
        # print(self.causal_graph)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        # gps = (gps - self._command_planner.mean) * self._command_planner.scale
        gps = (gps - self._waypoint_planner.mean) * self._waypoint_planner.scale

        return gps

    def _de_position(self, node):
        node1 = node / self._waypoint_planner.scale + self._waypoint_planner.mean
        return node1

    def pre_process(self, tick_data, vehicle_speed):
        lidar_img = None
        img = None
        rgb_topdown = None
        if self.vae_params.in_lidar:
            x_obs_range = 64
            y_obs_range = 36
            lidar_bin = 0.25
            lidar_height = 1.6
            x_bins = np.arange(-x_obs_range / 2, x_obs_range / 2 + lidar_bin, lidar_bin)
            y_bins = np.arange(-y_obs_range / 2, y_obs_range / 2 + lidar_bin, lidar_bin)
            # z_bins = [-lidar_height - 3, 1.0, 15, 700]
            point_cloud = tick_data['point_cloud']
            point_cloud = point_cloud[:, 0:2]
            lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins))
            lidar_label = np.array(lidar > 0, dtype=np.uint8)
            size = lidar_label.shape
            # lidar_transform = get_lidar_transform(self.vae_params.lidar_transform_optionsF, size)
            lidar_transform = get_lidar_transform(self.vae_params.lidar_transform_optionsF)
            if lidar_transform is not None:
                lidar_img = seg_to_rgb(lidar_label)
                lidar_img = lidar_transform(lidar_img)
                lidar_img = lidar_img.unsqueeze(0)

        img = Image.fromarray(tick_data['rgb'])
        size = img.size
        # img_transform = get_img_transform(self.vae_params.img_transform_optionsF, size)
        phase = 'fixed'
        img_transform = get_img_transform(phase, self.vae_params.img_transform_optionsF)
        if img_transform is not None:
            img = img_transform(img)
            img = img.unsqueeze(0)

        rgb_light = np.array(tick_data['rgb_light'] / 255., dtype=np.float32)
        img_light = torch.from_numpy(rgb_light)
        img_light = img_light.permute(2, 0, 1)
        img_light = img_light.unsqueeze(0)

        route_fig = tick_data['route_fig'] / np.max(tick_data['route_fig'])

        if self.vae_params.in_speed:
            print('in speed!')
            route_fig = np.array(route_fig, dtype=np.float32)
        else:
            # route_fig = np.array(route_fig * ((vehicle_speed + 1) / max_speed), dtype=np.float32)
            route_fig = np.array(route_fig, dtype=np.float32)

        route_fig = route_fig.swapaxes(0, 1)
        route_fig = np.expand_dims(route_fig, 0)
        rgb_topdown = route_fig

        return img, lidar_img, rgb_topdown, img_light

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """

        # sensors = [
        #     {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
        #      'width': 800, 'height': 600, 'fov': 100, 'id': 'Center'},
        #     {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
        # ]
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 256, 'height': 144, 'fov': 90,
                'id': 'rgb'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 2.0, 'y': 0.0, 'z': 1.4,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 90,
                'id': 'rgb_light'
            },
            # {
            #     'type': 'sensor.camera.rgb',
            #     'x': 1.3, 'y': 0.0, 'z': 1.3,
            #     'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #     'width': 400, 'height': 300, 'fov': 90,
            #     'id': 'rgb_light'
            # },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 1.3, 'y': 0.0, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 90,
                'id': 'rgb_seg_large'
            },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 1.3, 'y': 0.0, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 256, 'height': 144, 'fov': 90,
                'id': 'rgb_seg'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.2, 'y': -0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                'width': 256, 'height': 144, 'fov': 90,
                'id': 'rgb_left'
            },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 1.2, 'y': -0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                'width': 256, 'height': 144, 'fov': 90,
                'id': 'rgb_left_seg'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.2, 'y': 0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                'width': 256, 'height': 144, 'fov': 90,
                'id': 'rgb_right'
            },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 1.2, 'y': 0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                'width': 256, 'height': 144, 'fov': 90,
                'id': 'rgb_right_seg'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 0.0, 'y': 0.0, 'z': 2.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 90.0,
                'id': 'LIDAR'
            },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 0, 'y': 0.0, 'z': 35.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 256, 'height': 256, 'fov': 90,
                'id': 'map'
            },
            {
                'type': 'sensor.other.obstacle',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'id': 'obstacle'
            },
        ]

        return sensors

    def compute_rewards_17(self, steer, direction, speed, dis, theta, new_event_list, brake, obstacle, throttle,
                           in_corner, max_block_time=400):
        event_reward = 0
        target_reached = False
        done = 0
        if self.begin is False:
            for event in new_event_list:
                # print(self.step, event.get_type())
                if event.get_type() == TrafficEventType.COLLISION_STATIC:
                    if self.rank == 0:
                        print('collision static')
                    # todo: add penalty
                    event_reward -= 2
                    done = 1
                elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN or event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                    intensity = 400
                    if event.get_dict():
                        intensity = event.get_dict()['intensity']
                    event_reward -= 2
                    # if intensity == 400:
                    if self.rank == 0:
                        print("collision vehicle with intensity:", intensity)
                    done = 1
                elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                    done = 1
                    event_reward -= 1
                elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                    done = 1
                    event_reward -= 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                    event_reward += 5
                    target_reached = True
                    done = 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                    if not target_reached:
                        if event.get_dict():
                            score_route = event.get_dict()['route_completed']
                        else:
                            score_route = 0
                        print('score_route:', score_route)
                        event_reward += 5 * score_route
                    done = 1
        else:
            self.begin = False
        # if self.completion_ratio > self.target_completion:
        #     event_reward += 2
        #     self.target_completion += 5

        # ================= theta_reward: 0-1 =====================
        degree = abs(180 * theta / np.pi)

        theta_reward = max(0, 1 - degree / self.max_degree)
        # if degree >= self.max_degree:
        #     event_reward -= 2
        #     if self.training:
        #         done = True
        #     if self.rank == 0:
        #         print('degree is too large!')
        # ================ speed_reward: 0-1 or -1 ~ 2 =======================
        if speed > self.max_speed:
            event_reward -= 2
            # if self.training:
            done = True
            if self.rank == 0:
                print('exceed speed')
        detect_obstacle = obstacle > -1 and obstacle < 9
        if detect_obstacle:  # detect obstacles
            # print('distance:', obstacle)
            self.last_event_timestamp = self.step
            if obstacle > 5:
                target_speed = max(0, (obstacle - 5))
                speed_reward = 1 - max((speed - target_speed), 0) / (self.max_speed - target_speed)
            else:
                if speed > 0.01 or throttle > 0.01:
                    speed_reward = 0.1
                    event_reward -= 1
                else:
                    speed_reward = 1

        else:
            if speed < self.min_speed:
                speed_reward = speed / self.min_speed
            elif speed > self.target_speed:
                speed_reward = max(0, 1 - (speed - self.target_speed) / (self.max_speed - self.target_speed))
            else:
                speed_reward = 1

        # # ================ deviation reward: 0 ~ -1 ============================
        if self.in_turn:
            D_max = 3
        else:
            D_max = 2.5
        if dis > D_max:
            if self.rank == 0:
                print('route deviation:', dis, D_max)
            # if self.training:
            done = True
            event_reward -= 2

        # deviation_reward = max(-1, -1 * dis / D_max)
        deviation_reward = max(0.0, 1.0 - dis / D_max)

        if speed < 0.1 and (self.step - self.last_event_timestamp) > max_block_time:
            # if not detect_obstacle or detect_obstacle and (self.step - self.last_event_timestamp) > 2 * max_block_time:
            if self.rank == 0:
                print('vehicle block')
            self.last_event_timestamp = self.step
            if self.training:
                done = 1
                event_reward -= 2

        if len(new_event_list) > 0 or speed > 1:
            self.last_event_timestamp = self.step

        self.pre_commmand = direction
        # reward = (2 * speed_reward + event_reward + deviation_reward) / 2
        reward = 2 * speed_reward * deviation_reward * theta_reward + event_reward

        # if self.training is False:
        # print(self.step, round(reward, 2), " s_r: ", round(speed_reward, 2), '; e_r: ', round(event_reward, 2),
        #       '; d_r: ', round(deviation_reward, 2), 't_r:', round(theta_reward, 2), 'deviation: ', round(dis, 2),
        #       '; theta: ',
        #       round(degree, 2))
        return torch.tensor([reward]), done

    def compute_rewards_19(self, steer, direction, speed, dis, theta, new_event_list, brake, obstacle, throttle,
                           in_corner, max_block_time=400):
        event_reward = 0
        throttle_event_reward = 0
        steer_event_reward = 0
        target_reached = False
        done = 0
        throttle_done = 0
        steer_done = 0
        error_message = ""
        approach_light_distance = -1
        if self.begin is False:
            for event in new_event_list:
                if event.get_type() == TrafficEventType.COLLISION_STATIC:
                    if self.rank == 0:
                        print('collision static')
                        error_message = "collision static"
                    # todo: add penalty
                    steer_event_reward -= 1
                    # event_reward -= 2
                    steer_done = 1
                    done = 1
                elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN or event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                    intensity = 400
                    if event.get_dict():
                        intensity = event.get_dict()['intensity']
                    intensity = max(intensity, 0)
                    # intensity = min(intensity, 400)
                    # event_reward -= 2
                    throttle_event_reward -= 1
                    throttle_done = 1

                    # if intensity >= 400:
                    if self.rank == 0:
                        print("collision vehicle!")
                    done = 1
                    error_message = "collision vehicle"
                elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                    error_message = "vehicle blocked"
                    done = 1
                    throttle_done = 1
                    # event_reward -= 1
                    throttle_event_reward -= 1
                elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                    error_message = "route deviation"
                    done = 1
                    # event_reward -= 1
                    steer_event_reward -= 1
                    steer_done = 1
                # elif event.get_type() == TrafficEventType.APPROACH_LIGHT:
                #     if event.get_dict():
                #         state = event.get_dict()['state']
                #         self.approach_light_time += 1
                #         if state == carla.TrafficLightState.Red:
                #             approach_light_distance = event.get_dict()['distance']
                elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                    if self.rank == 0:
                        print("traffic light infraction!")
                    throttle_event_reward -= 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                    error_message = "success"
                    event_reward += 5
                    target_reached = True
                    done = 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                    if not target_reached:
                        if event.get_dict():
                            score_route = event.get_dict()['route_completed']
                        else:
                            score_route = 0
                        print('score_route:', score_route)
                        event_reward += 5 * score_route
                    done = 1
        else:
            self.begin = False
        # if self.completion_ratio > self.target_completion:
        #     event_reward += 2
        #     self.target_completion += 5

        # ================= theta_reward: 0-1 =====================
        degree = abs(180 * theta / np.pi)

        # theta_reward = max(0, 1 - degree / self.max_degree)
        theta_reward = - degree / self.max_degree
        # if degree >= self.max_degree:
        #     event_reward -= 2
        #     if self.training:
        #         done = True
        #     if self.rank == 0:
        #         print('degree is too large!')
        # ================ speed_reward: 0-1 or -1 ~ 2 =======================
        if speed > self.max_speed:
            # event_reward -= 2
            throttle_event_reward -= 1
            if self.training:
                done = True
                throttle_done = 1
                if self.rank == 0:
                    print('exceed speed')
        detect_obstacle = obstacle > -1 and obstacle < 12
        if not detect_obstacle:
            detect_obstacle = approach_light_distance > -1 and approach_light_distance < 12
            obstacle = approach_light_distance
        else:
            if approach_light_distance > -1 and approach_light_distance < 12:
                obstacle = min(approach_light_distance, obstacle)
        if detect_obstacle:  # detect obstacles
            self.last_event_timestamp = self.step
            # if self.training is False:
            #     print('distance:', obstacle)
            target_speed = max(0, (obstacle - 5))
            speed_reward = 1 - max((speed - target_speed), 0) / (self.max_speed - target_speed)
            if obstacle < 5:
                # todo: change
                if speed > 0.1:
                    # if self.training is False:
                    # if speed > 0.1:
                    #     print('speed > 0', speed)
                    # elif throttle > 0:
                    #     print('throttle>0', throttle)
                    speed_reward = -1
                else:
                    speed_reward = 1

        else:
            if speed < self.min_speed:
                speed_reward = speed / self.min_speed
            elif speed > self.target_speed:
                speed_reward = max(0, 1 - (speed - self.target_speed) / (self.max_speed - self.target_speed))
            else:
                speed_reward = 1

        # # ================ deviation reward: 0 ~ -1 ============================
        if self.in_turn:
            D_max = 3
        else:
            D_max = 2.5
        if self.training is False:
            D_max = 10
        if dis > D_max:
            if self.rank == 0:
                print('route deviation:', dis, D_max)
            # if self.training:
            done = True
            # event_reward -= 2
            steer_event_reward -= 1
            steer_done = 1
            error_message = "route deviation"

        # deviation_reward = max(-1, -1 * dis / D_max)
        deviation_reward = max(0.0, 1.0 - dis / D_max)

        if speed < 0.1 and (self.step - self.last_event_timestamp) > max_block_time:
            if self.rank == 0:
                print('vehicle block')
            self.last_event_timestamp = self.step
            # if self.training:
            done = 1
            # event_reward -= 2
            throttle_event_reward -= 2
            throttle_done = 1
            error_message = "vehicle blocked"

        if len(new_event_list) > 0 and not (
                approach_light_distance > -1 and state == carla.TrafficLightState.Green) or speed > 0.1:
            self.last_event_timestamp = self.step

        self.pre_commmand = direction

        # reward = (2 * speed_reward + event_reward + deviation_reward) / 2
        throttle_reward = speed_reward + throttle_event_reward
        steer_reward = (deviation_reward + theta_reward) / 2 + steer_event_reward

        # # if self.training is False:
        # print(self.step % 202, direction, " steer_r: ", round(steer_reward, 2), '; throttled_r: ', round(throttle_reward, 2),
        #       '; distance: ', obstacle, self.in_turn)
        return torch.tensor([steer_reward, throttle_reward]), done, error_message, [steer_done, throttle_done]

    def compute_rewards_18(self, steer, direction, speed, dis, theta, new_event_list, brake, obstacle, throttle,
                           in_corner, max_block_time=400):
        event_reward = 0
        throttle_event_reward = 0
        steer_event_reward = 0
        target_reached = False
        done = 0
        throttle_done = 0
        steer_done = 0
        error_message = ""
        if self.begin is False:
            for event in new_event_list:
                if event.get_type() == TrafficEventType.COLLISION_STATIC:
                    if self.rank == 0:
                        print('collision static')
                        error_message = "collision static"
                    # todo: add penalty
                    steer_event_reward -= 1
                    # event_reward -= 2
                    steer_done = 1
                    if self.training:
                        done = 1
                elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN or event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                    # if event.get_dict():
                    #     intensity = event.get_dict()['intensity']
                    #
                    throttle_event_reward -= 1
                    throttle_done = 1

                    # if intensity >= 400:
                    if event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                        if self.rank == 0:
                            print("collision pedestrians!")
                        error_message = "collision pedestrians!"
                    else:
                        if self.rank == 0:
                            print("collision vehicle")
                        error_message = "collision vehicles!"
                    if self.training:
                        done = 1
                elif event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                    if self.rank == 0:
                        print("traffic light infraction!")
                    throttle_event_reward -= 1
                    throttle_done = 1
                    # done = 1
                # elif event.get_type() == TrafficEventType.APPROACH_LIGHT:
                #     if event.get_dict():
                #         state = event.get_dict()['state']
                #         self.approach_light_time += 1
                #         if state == carla.TrafficLightState.Red:
                #             approach_light_distance = event.get_dict()['distance']
                #             print('approach_light_distance', approach_light_distance)
                elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                    error_message = "vehicle blocked"
                    # if self.training:
                    #     done = 1
                    throttle_done = 1
                    # event_reward -= 1
                    throttle_event_reward -= 1
                elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                    error_message = "route deviation"
                    # if self.training:
                    #     done = 1
                    # event_reward -= 1
                    steer_event_reward -= 1
                    steer_done = 1
                    print('route deviation')
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                    print('route completed!!!')
                    error_message = "success"
                    event_reward += 5
                    target_reached = True
                    done = 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                    print('route completion!!!')
                    if not target_reached:
                        if event.get_dict():
                            score_route = event.get_dict()['route_completed']
                        else:
                            score_route = 0
                        print('score_route:', score_route)
                        event_reward += 5 * score_route
                    done = 1
                    error_message = "success"
                elif event.get_type() == TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION:
                    print('outside route!')
                    error_message = "outside route!"
                    if self.training:
                        done = 1
        else:
            self.begin = False
        # if self.completion_ratio > self.target_completion:
        #     event_reward += 2
        #     self.target_completion += 5

        # ================= theta_reward: 0-1 =====================
        degree = abs(180 * theta / np.pi)
        if self.in_turn:
            degree = max(0, degree - 30)

        theta_reward = max(0, 1 - degree / self.max_degree)
        # theta_reward = - degree / self.max_degree
        # if degree >= self.max_degree:
        #     event_reward -= 2
        #     if self.training:
        #         done = True
        #     if self.rank == 0:
        #         print('degree is too large!')
        # ================ speed_reward: 0-1 or -1 ~ 2 =======================
        if speed > self.max_speed:
            # event_reward -= 2
            throttle_event_reward -= 1
            if self.training:
                # done = True
                throttle_done = 1
                if self.rank == 0:
                    print('exceed speed')
        detect_obstacle = obstacle > -1 and obstacle < 12
        if detect_obstacle:  # detect obstacles
            self.last_event_timestamp = self.step
            # if self.training is False:
            #     print('distance:', obstacle)
            target_speed = max(0, (obstacle - 5))
            speed_reward = 1 - max((speed - target_speed), 0) / (self.max_speed - target_speed)
            if obstacle < 5:
                # todo: change
                if speed > 0.1:
                    # if self.training is False:
                    # if speed > 0.1:
                    #     print('speed > 0', speed)
                    # elif throttle > 0:
                    #     print('throttle>0', throttle)
                    speed_reward = -1
                else:
                    speed_reward = 1

        else:
            if speed < self.min_speed:
                speed_reward = speed / self.min_speed
            elif speed > self.target_speed:
                speed_reward = max(0, 1 - (speed - self.target_speed) / (self.max_speed - self.target_speed))
            else:
                speed_reward = 1

        # # ================ deviation reward: 0 ~ -1 ============================
        if self.in_turn or self.near_command != RoadOption.LANEFOLLOW:
            # todo: debug
            D_max = 5
        else:
            D_max = 2.5
        if self.training is False:
            D_max = 10
        # if dis > D_max:
        #     if self.rank == 0:
        #         print('route deviation:', dis, D_max)
        #     # if self.training:
        #     # todo: debug
        #     done = True
        #     # event_reward -= 2
        #     steer_event_reward -= 2
        #     steer_done = 1
        #     error_message = "route deviation in command " + str(self.near_command)

        # deviation_reward = max(-1, -1 * dis / D_max)
        deviation_reward = max(0.0, 1.0 - dis / D_max)

        if speed < 0.5 and (self.step - self.last_event_timestamp) > max_block_time:
            # if not detect_obstacle or detect_obstacle and (self.step - self.last_event_timestamp) > 2 * max_block_time:
            if self.rank == 0:
                print('vehicle block')
            self.last_event_timestamp = self.step
            if self.training:
                done = 1
            # event_reward -= 2
            throttle_event_reward -= 2
            throttle_done = 1
            error_message = "vehicle blocked"

        if len(new_event_list) > 0 or speed > 0.5:
            self.last_event_timestamp = self.step

        # reward = (2 * speed_reward + event_reward + deviation_reward) / 2
        throttle_reward = speed_reward + throttle_event_reward
        steer_reward = (deviation_reward + theta_reward) / 2 + steer_event_reward

        # # if self.training is False:
        # print(self.step % 202, direction, " steer_r: [", round(steer, 2), round(steer_reward, 2), ']; throttled_r: [',
        #       round(throttle, 2), round(throttle_reward, 2), ']; distance: ', obstacle)
        return torch.tensor([steer_reward, throttle_reward]), done, error_message, [steer_done, throttle_done]

    def compute_distance(self, node1, node2):
        return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

    def draw_route(self, tick_data, route_list, pixels_per_meter=3.66, size_x=144, size_y=256,
                   color=(255, 0, 0)):
        if self.vae_params.in_route:
            route_fig = np.zeros((256, 144), dtype=np.uint8)
            route_fig = Image.fromarray(route_fig)
            route_draw = ImageDraw.Draw(route_fig)

        color = 255

        compass = tick_data['compass']
        compass = 0.0 if np.isnan(compass) else compass
        compass = compass + np.pi / 2
        R = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)],
        ])
        pos = self._get_position(tick_data)
        far_node = None

        turn_pre_node = route_list[0]
        direction = 0
        pep_dis = np.inf
        ori_pep_dis = np.inf
        gps_x = size_x / 2
        gps_y = size_y / 2

        zeros_node_x, zeros_node_y = pixels_per_meter * (R.T.dot(route_list[0] - pos))
        zeros_node_x += size_x / 2
        zeros_node_y += size_y / 2
        turn_first_waypoint = turn_last_waypoint = None

        # forward_dis = self.compute_distance(pos, route_list[0])
        for i in range(1, len(route_list)):
            cur_node = route_list[i]
            pre_node = route_list[i - 1]
            _pre_x, _pre_y = pixels_per_meter * (R.T.dot(pre_node - pos))
            pre_x = _pre_x + size_x / 2
            pre_y = _pre_y + size_y / 2

            _cur_x, _cur_y = pixels_per_meter * (R.T.dot(cur_node - pos))
            cur_x = _cur_x + size_x / 2
            cur_y = _cur_y + size_y / 2
            if self.vae_params.in_route:
                route_draw.line((_pre_x + size_x / 2, _pre_y + size_y / 2, _cur_x + size_x / 2, _cur_y + size_y / 2),
                                color, width=15)
            if i == 1:
                turn_first_waypoint = [pre_x, pre_y]
            if i == len(route_list) - 1:
                turn_last_waypoint = [cur_x, cur_y]
            if math.sqrt((zeros_node_y - cur_y) ** 2 + (zeros_node_x - cur_x) ** 2) > 1e-3 and pep_dis == np.inf:
                ori_pep_dis = abs(
                    ((cur_y - zeros_node_y) * (gps_x - zeros_node_x) - (cur_x - zeros_node_x) * (
                            gps_y - zeros_node_y)) / math.sqrt(
                        (cur_y - zeros_node_y) ** 2 + (cur_x - zeros_node_x) ** 2))
                pep_dis = abs(
                    ((cur_node[1] - route_list[0][1]) * (pos[0] - route_list[0][0]) - (
                            cur_node[0] - route_list[0][0]) * (
                             pos[1] - route_list[0][1])) / math.sqrt(
                        (cur_node[1] - route_list[0][1]) ** 2 + (cur_node[0] - route_list[0][0]) ** 2))

            if abs(route_list[i][0] - route_list[0][0]) + abs(
                    route_list[i][1] - route_list[0][1]) > 1e-3 and far_node is None:
                far_node = route_list[i]

            turn_cur_node = route_list[i]
            if self.in_turn is False:
                if abs(turn_cur_node[0] - turn_pre_node[0]) < 1 or abs(turn_cur_node[1] - turn_pre_node[1]) < 1:
                    continue
                else:
                    # print(abs(turn_cur_node[0] - turn_pre_node[0]), abs(turn_cur_node[1] - turn_pre_node[1]))
                    if abs(turn_cur_node[0] - turn_pre_node[0]) < abs(turn_cur_node[1] - turn_pre_node[1]):
                        direction = 0
                    else:
                        direction = 1
                    if self.turn_first_node is None:
                        self.first_direction = direction
                        self.turn_first_node = turn_cur_node
                    else:
                        self.last_direction = direction
                        self.turn_last_node = turn_cur_node
                turn_pre_node = turn_cur_node
        if pep_dis == np.inf or np.isnan(pep_dis):
            pep_dis = 0
        if self.vae_params.in_route:
            route_fig = np.array(route_fig)
        else:
            route_fig = np.zeros((256, 144), dtype=np.uint8)
        tick_data['route_fig'] = route_fig

        theta, distance = self.get_theta(far_node, route_list[0], pos, tick_data, len(route_list), pixels_per_meter)
        if len(route_list) == 2:
            distance = pep_dis
        #     print('theta: ', theta)
        #     theta = abs(theta)
        #     if theta > np.pi / 2:
        #         theta -= np.pi / 2

        if self.turn_first_node is not None and self.turn_last_node is not None:
            if self.first_direction == 0:
                turn_middle_node = [self.turn_last_node[0], self.turn_first_node[1]]
            else:
                turn_middle_node = [self.turn_first_node[0], self.turn_last_node[1]]
            turn_dis = self.compute_distance(turn_middle_node, pos)
            max_dis = max(self.compute_distance(turn_middle_node, self.turn_first_node),
                          self.compute_distance(turn_middle_node, self.turn_last_node))

            if turn_dis < max_dis + 5:
                # print('in turn', max_dis, turn_dis)
                self.in_turn = True
            elif self.in_turn == True:
                # print('straight, ', max_dis, turn_dis)
                self.in_turn = False
                self.turn_first_node = None
                self.turn_last_node = None
                self.first_direction = 0
                self.last_direction = 0

        return tick_data, distance, theta, self.in_turn

    def normalize(self, point, length):
        length = abs(length)
        len = math.sqrt(point.x ** 2 + point.y ** 2)
        point.x = point.x * length / len
        point.y = point.y * length / len
        return point

    def get_theta(self, cur_node, pre_node, pos, tick_data, route_len, pixels_per_meter=3.66):
        if cur_node is None:
            return 0, 0
        size_x = 144
        size_y = 256
        compass = tick_data['compass']
        compass = 0.0 if np.isnan(compass) else compass
        compass = compass + np.pi / 2
        R = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)],
        ])
        x1, y1 = pixels_per_meter * (R.T.dot(pre_node - pos))
        x1 += size_x / 2
        y1 += size_y / 2
        x2, y2 = pixels_per_meter * (R.T.dot(cur_node - pos))
        x2 += size_x / 2
        y2 += size_y / 2

        x0, y0 = size_x / 2, size_y / 2
        # version-1
        # location = self._vehicle.get_transform().location
        # tail_close_pt = self.rotate_point(carla.Vector3D(-3, 0.0, location.z),
        #                                   self._vehicle.get_transform().rotation.yaw)

        # version-2
        # location = carla.Vector3D(tick_data['measurements'][0], tick_data['measurements'][1],
        #                           tick_data['measurements'][2])
        # tail_close_pt = self.rotate_point(carla.Vector3D(-3, 0.0, tick_data['measurements'][2]),
        #                                   tick_data['measurements'][3])
        # tail_close_pt = location + carla.Location(tail_close_pt)
        # tail_close_pt = self._map.transform_to_geolocation(tail_close_pt)
        # tail_close_pt = [tail_close_pt.latitude, tail_close_pt.longitude]

        # version-3
        location = carla.Vector3D(tick_data['full_gps'][0], tick_data['full_gps'][1], tick_data['full_gps'][2])
        tail_close_pt = self.rotate_point(carla.Vector3D(0.0001, 0.0, tick_data['full_gps'][2]),
                                          tick_data['measurements'][3] - 90)

        tail_close_pt = location + tail_close_pt
        tail_close_pt = [tail_close_pt.x, tail_close_pt.y]

        tail_close_pt = (tail_close_pt - self._waypoint_planner.mean) * self._waypoint_planner.scale

        head_close_pt = self.rotate_point(carla.Vector3D(-0.000025, 0.0, tick_data['full_gps'][2]),
                                          tick_data['measurements'][3] - 90)
        head_close_pt = self.normalize(head_close_pt, 0.000025)
        head_close_pt = location + head_close_pt
        head_close_pt = [head_close_pt.x, head_close_pt.y]
        head_close_pt = (head_close_pt - self._waypoint_planner.mean) * self._waypoint_planner.scale
        distance = self.compute_distance(pre_node, head_close_pt)
        if DEBUG:
            self.debug.clear()

            # self.debug.line(pos, pre_node, cur_node)
            self.debug.line(pos, pos, tail_close_pt, color=(255, 0, 0))
            self.debug.line(pos, pos, head_close_pt, color=(0, 255, 0))
            self.debug.dot(pos, pre_node, color=(0, 0, 255), r=5)
            self.debug.show()
        # print(round(dis1, 2), round(dis2, 2), round(dis3, 2))

        pre_pos = tail_close_pt
        x3, y3 = pixels_per_meter * (R.T.dot(pre_pos - pos))
        # if self.pre_pos is None:
        #     self.pre_pos = pos
        # x3, y3 = pixels_per_meter * (R.T.dot(self.pre_pos - pos))
        x3 += size_x / 2
        y3 += size_y / 2
        a = [x2 - x1, y2 - y1]  # route vector
        b = [x0 - x3, y0 - y3]  # vehicle vector
        location = [location.x, location.y]
        gps_location = (location - self._waypoint_planner.mean) * self._waypoint_planner.scale
        vector1 = gps_location - tail_close_pt
        # if vector1[1] < 0:
        #     vector1 = -vector1
        vector2 = cur_node - gps_location
        x4, y4 = pixels_per_meter * (R.T.dot(vector2))
        if math.sqrt(a[0] * a[0] + a[1] * a[1]) < 1e-3 or math.sqrt(b[0] * b[0] + b[1] * b[1]) < 1e-3:
            theta = self.pre_theta
        else:
            # theta = math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0])

            theta = (vector1[0] * vector2[0] + vector1[1] * vector2[1]) / (
                    math.sqrt(vector1[0] ** 2 + vector1[1] ** 2) * math.sqrt(vector2[0] ** 2 + vector2[1] ** 2))
            theta = max(theta, -1)
            theta = min(theta, 1)
            theta = np.arccos(theta)
            if route_len == 2 and y4 > 0:
                theta = np.pi - theta

            # ori_theta = (a[0] * b[0] + a[1] * b[1]) / (
            #         math.sqrt(a[0] * a[0] + a[1] * a[1]) * math.sqrt(b[0] * b[0] + b[1] * b[1]))
            # ori_theta = max(ori_theta, -1)
            # ori_theta = min(ori_theta, 1)
            # ori_theta = np.arccos(ori_theta)

            # print('theta is ', round(180 * theta / np.pi, 2), round(180 * ori_theta / np.pi, 2), self.near_command,
            #       round(x4, 3), round(y4, 3))

        # if DEBUG:
        #     self.debug.clear()
        #
        #     self.debug.line(pos, pre_node, cur_node)
        #     self.debug.line(pos, pos, tail_close_pt, color=(255, 0, 0))
        #
        #     self.debug.show()
        self.pre_theta = theta
        self.pre_pos = pos
        if np.isnan(theta):
            return 0, distance
        return theta, distance

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        # y_ = math.sin(math.radians(angle)) * point.x - math.cos(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def tick(self, input_data):
        point_cloud = []
        # for location in input_data['LIDAR'][1]:
        #     point_cloud.append([location[0], location[1], -location[2]])
        # point_cloud = np.array(point_cloud)
        gps = input_data['gps'][1][:2]
        full_gps = input_data['gps'][1][:3]

        self._actors = self._world.get_actors()
        # self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'), size=256,
        #                                          pixels_per_meter=3.66)

        # topdown = input_data['map'][1][:, :, 2]
        # topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights, size=256, pixels_per_meter=3.66,
        #                               radius=5)

        target_diff = 0
        # if self.pre_topdown is None:
        #     target_diff = 100.0
        # else:
        #     target_diff = abs(self.pre_topdown - topdown)
        #     target_diff = np.array(target_diff[:, :] > 0).sum()
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        measurements = [input_data['imu'][1][0], input_data['imu'][1][1], input_data['imu'][1][2],
                        input_data['imu'][1][3]]

        # self.pre_topdown = topdown
        # rgb = Image.open('/home/qua n/distribution/result/01-20/22-28-18/0/display_path/rgb-0001.png')
        # rgb = np.array(rgb)

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_light = cv2.cvtColor(input_data['rgb_light'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        rgb_seg = input_data['rgb_seg'][1][:, :, 2]
        rgb_left_seg = input_data['rgb_left_seg'][1][:, :, 2]
        rgb_right_seg = input_data['rgb_right_seg'][1][:, :, 2]
        rgb_seg_large = input_data['rgb_seg_large'][1][:, :, 2]

        obstacle_data = input_data['obstacle']
        obstacle_distance = -1
        self._vehicle = CarlaDataProvider.get_hero_actor()

        if obstacle_data[0] > -1:
            obstacle_distance = obstacle_data[1][0]
            actor = obstacle_data[1][1]
            ego_vehicle_point = self._map.get_waypoint(CarlaDataProvider.get_location(self._vehicle),
                                                       project_to_road=False)
            ego_vehicle_road_point = self._map.get_waypoint(CarlaDataProvider.get_location(self._vehicle),
                                                            lane_type=carla.LaneType.Driving, project_to_road=True)
            if ego_vehicle_point:
                ego_vehicle_lane_id = ego_vehicle_point.lane_id
                ego_vehicle_road_id = ego_vehicle_road_point.road_id
            else:
                ego_vehicle_lane_id = -100
                ego_vehicle_road_id = -100

            other_vehicle_point = self._map.get_waypoint(CarlaDataProvider.get_location(actor), project_to_road=False)
            other_vehicle_road_point = self._map.get_waypoint(CarlaDataProvider.get_location(actor),
                                                              lane_type=carla.LaneType.Driving, project_to_road=True)
            if other_vehicle_point:
                other_vehicle_lane_id = other_vehicle_point.lane_id
                other_vehicle_road_id = other_vehicle_road_point.road_id
            else:
                other_vehicle_lane_id = -101
                other_vehicle_road_id = -101

            if ego_vehicle_lane_id != other_vehicle_lane_id and ego_vehicle_road_id == other_vehicle_road_id:
                # print('ego vehicle id is ', ego_vehicle_lane_id, ', but other id is ', other_vehicle_lane_id)

                # print('land id is not same ', ego_vehicle_lane_id, other_vehicle_lane_id)
                obstacle_distance = -1
            else:
                # print('obstacle distance:', round(obstacle_distance, 2))
                transforms = CarlaDataProvider.get_transform(actor)
                actor_speed = CarlaDataProvider.get_velocity(actor)
                vehicle_theta = abs(transforms.rotation.yaw - input_data['imu'][1][3])
                # todo: change vehicle_theta
                if vehicle_theta > 180:
                    vehicle_theta = 360 - vehicle_theta
                if vehicle_theta > 90 and actor_speed < 0.01 and 'vehicle' in actor.type_id:
                    obstacle_distance = -1
        # if obstacle_distance > -1:
        #     print('detect obstacle:', obstacle_distance)
        result = {
            'rgb': rgb,
            'rgb_light': rgb_light,
            'rgb_seg_large': rgb_seg_large,
            'rgb_seg': rgb_seg,
            'rgb_left': rgb_left,
            'rgb_left_seg': rgb_left_seg,
            'rgb_right': rgb_right,
            'rgb_right_seg': rgb_right_seg,
            'gps': gps,
            'full_gps': full_gps,
            'speed': speed,
            'compass': compass,
            'target_diff': target_diff,
            'topdown_seg': None,
            'point_cloud': point_cloud,
            'obstacle': obstacle_distance,
            'measurements': measurements
        }

        # result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)
        compass = 0.0 if np.isnan(compass) else compass
        compass = compass + np.pi / 2
        R = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)],
        ])

        gps = self._get_position(result)

        # regardless of command
        far_node, _, _ = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        # result['target'] = topdown
        return result

    def filter_event(self, new_event_list):
        if self.detect_traffic_light is True:
            return new_event_list
        else:
            for i, event in enumerate(new_event_list):
                if event.get_type() == TrafficEventType.APPROACH_LIGHT:
                    del new_event_list[i]
                    break
            for i, event in enumerate(new_event_list):
                if event.get_type() == TrafficEventType.TRAFFIC_LIGHT_INFRACTION:
                    del new_event_list[i]
                    break
            return new_event_list

    def get_traffic_light(self, new_event_list):
        light_state = -1
        red_light_distance = -1
        for event in new_event_list:
            if event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                self.target_reached = True
                done = 1
                print("Target reached!!!")
            # elif event.get_type() == TrafficEventType.APPROACH_LIGHT:
            #     if event.get_dict():
            #         red_light_distance = 1.0 * event.get_dict()['distance']
            #         light_state = event.get_dict()['state']
            elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN or event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                intensity = 400
                if event.get_dict():
                    intensity = event.get_dict()['intensity']
                intensity = max(intensity, 0)
                intensity = min(intensity, 400)
                print("collision vehicle!")
                if intensity == 400:
                    # print("collision vehicle!")
                    done = 1
        return light_state, red_light_distance

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        if not self.initialized:
            self._init()

        x = self._vehicle.get_location().x
        y = self._vehicle.get_location().y
        z = self._vehicle.get_location().z
        yaw = self._vehicle.get_transform().rotation.yaw
        pitch = self._vehicle.get_transform().rotation.pitch
        roll = self._vehicle.get_transform().rotation.roll
        waypoint = {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'roll': roll, 'pitch': pitch}

        if self.pre_waypoint is None:
            self.pre_waypoint = waypoint
        else:
            _pre_waypoint = [self.pre_waypoint['x'], self.pre_waypoint['y']]
            _cur_waypoint = [waypoint['x'], waypoint['y']]
            waypoint_dis = self.compute_distance(_pre_waypoint, _cur_waypoint)
            self.sum_distance += waypoint_dis
            if self.sum_distance > 50:
                self.sum_distance = 0

        # route = {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'roll': roll, 'pitch': pitch}

        # todo: debug

        # if config_agent.benchmark == 'nocrash' and self.step % 100 == 0:
        #     weather_index = np.random.choice(config_agent.weather)
        #     weather = PRESET_WEATHERS[weather_index]
        #     self._world.set_weather(weather)
        #
        # if config_agent.benchmark == 'challenge' and self.step % 100 == 0:
        #     weather_index = np.random.choice(config_agent.weather)
        #     weather = PRESET_WEATHERS[weather_index]
        #     # weather_index = np.random.choice(config_agent.weather)
        #     # weather = PRESET_WEATHERS[weather_index]
        #     self._world.set_weather(weather)

        self.agent_engaged = True
        self._hic.run_interface(input_data)
        obstacle_data = input_data['obstacle']
        obstacle_distance = -1
        if obstacle_data[0] > -1:
            obstacle_distance = obstacle_data[1][0]
            # print('obstacle:', obstacle_distance)
        self.step += 1
        tick_data = self.tick(input_data)

        new_event_list = self.get_event()
        new_event_list = self.filter_event(new_event_list)

        traffic_light_state, light_distance = self.get_traffic_light(new_event_list)

        # _actor = CarlaDataProvider.get_next_traffic_light_by_location(self._vehicle.get_location())
        # # change light
        # if _actor:
        #     if _actor.id != self.light_id:
        #         self.has_red_light = False
        #     self.light_id = _actor.id
        #     if light_distance > 0:
        #         # # ========================= process light camera ==========================
        #         # delta = _actor.get_transform().location - self._vehicle.get_transform().location
        #         # transform = self._vehicle.get_transform()
        #         # theta = np.radians(90 + transform.rotation.yaw)
        #         # R = np.array([
        #         #     [np.cos(theta), -np.sin(theta)],
        #         #     [np.sin(theta), np.cos(theta)],
        #         # ])
        #         # map_point = R.T.dot([delta.x, delta.y])
        #         # pixels_per_meter = 5.5
        #         # map_point *= pixels_per_meter
        #         #
        #         # map_point[0] = 256 // 2 + map_point[0]
        #         # map_point[1] = 256 + map_point[1]
        #         # map_point = torch.FloatTensor([map_point[0], map_point[1]])
        #         # #
        #         # # light_box = _actor.trigger_volume
        #         # # print(light_box)
        #         # cam_point = self.converter.map_to_cam(map_point)
        #         # cam_point[1] -= max(0, pixels_per_meter * 5)
        #         # draw_light(tick_data, cam_point)
        #
        #         if traffic_light_state == carla.TrafficLightState.Red:
        #             self.has_red_light = True
        #         elif traffic_light_state == carla.TrafficLightState.Green:
        #             if self.has_red_light == True:
        #                 _actor.set_state(carla.TrafficLightState.Green)
        #
        #     #     self.approach_light_time = 0
        #     #     self.light_id = _actor.id
        #     #     print('approach light: ', _actor.id)
        #     # if self.approach_light_time > 0:
        #     #     if self.approach_light_time > 200:
        #     #         _actor.set_state(carla.TrafficLightState.Green)
        #     #     else:
        #     #         _actor.set_state(carla.TrafficLightState.Red)
        #     # else:
        #     #     traffic_light_time = self.traffic_light_time[self.traffic_light_counter]
        #     #     _actor.set_state(self.traffic_light_state[self.traffic_light_counter])
        #     #     if self.step - self.pre_traffic_light_time == traffic_light_time:
        #     #         self.traffic_light_counter = (self.traffic_light_counter + 1) % 3
        #     #         self.pre_traffic_light_time = self.step
        #     # # # print(self.step, 'green: ', _actor.get_green_time(), ' red: ', _actor.get_red_time(), ' yellow: ',
        #     # # #       _actor.get_yellow_time(), _actor.get_state())
        #     #

        gps = self._get_position(tick_data)
        near_node, near_command, route_list = self._waypoint_planner.run_step(gps)
        self.near_command = near_command
        # print('near_command:', near_command)
        # print(self._vehicle.get_transform().location)

        speed = tick_data['speed']
        far_node, far_command, _ = self._command_planner.run_step(gps)
        # far_node, far_command, _ = self._light_planner.run_step(gps)
        # far_command = near_command
        # print(far_command, near_command)
        # print(RoadOption.LEFT, int(RoadOption.LEFT.value), RoadOption.RIGHT, int(RoadOption.RIGHT.value),
        #       RoadOption.STRAIGHT, int(RoadOption.STRAIGHT.value), RoadOption.LANEFOLLOW, int(RoadOption.LANEFOLLOW.value))
        tick_data, dis, degree, in_corner = self.draw_route(tick_data, route_list, 10)

        # if DEBUG:
        #     debug_display(tick_data)
        # control = self._controller.parse_events(timestamp - self._prev_timestamp)
        control = self._controller.parse_events(self._hic._clock)

        # rewards, done, _, _ = self.compute_rewards_19(control.steer, near_command, speed, dis, degree, new_event_list,
        #                                               control.brake, tick_data['obstacle'], control.throttle, in_corner,
        #                                               max_block_time=400)

        rewards, done, error_message, _ = self.compute_rewards_18(control.steer, near_command, speed, dis, degree,
                                                                  new_event_list,
                                                                  control.brake, tick_data['obstacle'],
                                                                  control.throttle, in_corner,
                                                                  max_block_time=400)
        # rewards, done = self.compute_rewards_17(control.steer, near_command, speed, dis, degree, new_event_list,
        #                                         control.brake, tick_data['obstacle'], control.throttle, in_corner,
        #                                         max_block_time=400)
        self._prev_timestamp = timestamp

        image, lidar, topdown, img_light = self.pre_process(tick_data, speed)


        if self.use_light_vae:
            traffic_light = self.light_vae_model.get_light_state(img_light)
            traffic_light = traffic_light.item()
            if self.pre_far_command == RoadOption.LANEFOLLOW and (
                    far_command == RoadOption.LEFT or far_command == RoadOption.RIGHT or far_command == RoadOption.STRAIGHT):
                self.intersection_stay_time = 1
                # self.intersection_stay_time = 0
                self.near_lights = np.zeros(20)

            if self.intersection_stay_time > 300:
                # print('stay enough')
                self.intersection_stay_time = 0
                self.near_lights = np.zeros(20)
            elif self.intersection_stay_time > 0:
                self.near_lights[0:-1] = self.near_lights[1:]
                self.near_lights[-1] = traffic_light
                # if np.sum(self.near_lights) < 10:
                #     # print('pred red light')
                #     # throttle = 0.0
                #     # brake = 1.0
                #     # control.throttle = 0.0
                #     # control.brake = 1.0
                # else:
                #     # print('pred green light')
                #     # self.intersection_stay_time = 0
                #     # self.near_lights = np.zeros(20)
        # print('near_command:', near_command, 'far_command', far_command)
        self.pre_far_command = far_command

        input_list = image

        ##### load img from left camer
        left_img = Image.fromarray(tick_data['rgb_left'])
        left_size = left_img.size
        # img_transform = get_img_transform(self.vae_params.img_transform_optionsF, size)
        phase = 'fixed'
        left_img_transform = get_img_transform(phase, self.vae_params.img_transform_optionsF)
        if left_img_transform is not None:
            left_img = left_img_transform(left_img)
        left_img_input_list = left_img.unsqueeze(0)

        ##### load img from right camer
        right_img = Image.fromarray(tick_data['rgb_right'])
        right_size = right_img.size
        # img_transform = get_img_transform(self.vae_params.img_transform_optionsF, size)
        phase = 'fixed'
        right_img_transform = get_img_transform(phase, self.vae_params.img_transform_optionsF)
        if right_img_transform is not None:
            right_img = right_img_transform(right_img)
        right_img_input_list = right_img.unsqueeze(0)

        # input_list = torch.cat((image, lidar), dim=1)
        # input_list = input_list.to(self.vae_device)
        if self.use_vae:
            input_list = input_list.to(self.vae_device)
            route_fig = torch.from_numpy(topdown)
            route_fig = route_fig.view(1, 1, input_list.size()[2], input_list.size()[3]).to(self.vae_device)
            # if self.step % 3 == 0:
            #     def flip(x, dim):
            #         indices = [slice(None)] * x.dim()
            #         indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
            #                                     dtype=torch.long, device=x.device)
            #         return x[tuple(indices)]
            #
            #     route_fig = flip(route_fig, 2)

            if self.vae_params.in_left_camera:
                input_list = torch.cat((input_list, left_img_input_list), dim=1)
            if self.vae_params.in_right_camera:
                input_list = torch.cat((input_list, right_img_input_list), dim=1)
            if self.vae_params.in_lidar:
                input_list = torch.cat((input_list, lidar), dim=1)
            if self.vae_params.in_route:
                input_list = torch.cat((input_list, route_fig), dim=1)
            # if self.vae_params.in_speed:
            #     cur_speed = speed * torch.ones(1, 1, 144, 256)
            #     input_list = torch.cat((input_list, cur_speed), dim=1)

            latent_feature = self.vae_model.get_latent_feature(input_list, 'add')
            if self.pre_latent_feature is not None:
                latent_feature = latent_feature.clone().detach()
                difference = torch.sum((latent_feature - self.pre_latent_feature) ** 2)
            self.pre_latent_feature = latent_feature.clone().detach()

            # img_pred, lidar_pred, topdown_pred, lightState_pred, \
            #     lightDist_pred, route_pred, mu, logvar = self.vae_model.test_forward(input_list)
            if self.step % 5 == 0 and self.use_vae:
                logits = torch.empty(1, self.vae_params.networks['autoencoder']['z_dims']).uniform_(0, 1)
                self.causal_graph = torch.bernoulli(logits).to(self.vae_device)

            # lightState_pred, \
            # lightDist_pred, \
            # img_pred, \
            # lidar_pred, \
            # topdown_pred, \
            # route_pred, \
            # left_img_pred, \
            # right_img_pred, \
            # mu, logvar = self.vae_model.test_forward(input_list)

            self.causal_graph = torch.ones(self.vae_params.networks['autoencoder']['z_dims'])
            lightState_pred, \
            lightDist_pred, \
            img_pred, \
            lidar_pred, \
            topdown_pred, \
            route_pred, \
            left_img_pred, \
            right_img_pred, \
            mu, logvar = self.vae_model.decode_with_graph(input_list, self.causal_graph)
            img_pred_list = [img_pred]

            # print(self.step)
            # # if self.step < self.vae_params.networks['autoencoder']['z_dims']:
            # if self.step == 38:
            #     for i in range(10):
            #         value = -10 + 2 * i
            #         self.causal_graph[self.step - 1] = value
            #         lightState_pred, \
            #         lightDist_pred, \
            #         img_pred, \
            #         lidar_pred, \
            #         topdown_pred, \
            #         route_pred, \
            #         left_img_pred, \
            #         right_img_pred, \
            #         mu, logvar = self.vae_model.decode_with_graph(input_list, self.causal_graph, value=value,
            #                                                       step=self.step - 1)
            #         img_pred_list.append(img_pred)
            #         rgb_seg_tar = tick_data['rgb_seg']
            #         img_seg_resort = self.resort_seg(rgb_seg_tar)
            #
            #         rgb_left_seg_tar = tick_data['rgb_left_seg']
            #         left_img_seg_resort = self.resort_seg(rgb_left_seg_tar)
            #
            #         rgb_right_seg_tar = tick_data['rgb_right_seg']
            #         right_img_seg_resort = self.resort_seg(rgb_right_seg_tar)
            #
            #     save_vae(img_pred_list, self.step)

            rgb_seg_tar = tick_data['rgb_seg']
            img_seg_resort = self.resort_seg(rgb_seg_tar)
            # self._bfs.reset(tick_data['rgb_seg_large'])
            # bbx = self._bfs.find_bbx()
            # if len(bbx) > 0:
            #     draw_bbx(tick_data, bbx)

            rgb_left_seg_tar = tick_data['rgb_left_seg']
            left_img_seg_resort = self.resort_seg(rgb_left_seg_tar)

            rgb_right_seg_tar = tick_data['rgb_right_seg']
            right_img_seg_resort = self.resort_seg(rgb_right_seg_tar)

            if DEBUG:
                rgb = tick_data['rgb']
                _rgb = Image.fromarray(rgb)
                # _rgb_seg = Image.fromarray(COLOR[CONVERTER[rgb_seg]])
                # _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
                # _draw = ImageDraw.Draw(_topdown)
                # _topdown.thumbnail((256, 256))
                _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))
                # _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
                _draw = ImageDraw.Draw(_rgb)
                # _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
                # _draw.text((5, 10), 'Steer: %.3f' % steer)
                # _draw.text((5, 30), 'Throttle: %.3f' % throttle)
                # _draw.text((5, 50), 'Brake: %s' % brake)
                _draw.text((5, 50), 'Speed: %s' % speed)
                _draw.text((5, 70), 'Light_state: %s' % str(traffic_light_state + 1))
                if self.vae_params.pred_light_state:
                    _, lightState_pred_indices = torch.max(lightState_pred.data, 1)
                    _draw.text((5, 90), 'Pred_Light_state: %s' % lightState_pred_indices[0])
                # cv2.imshow('Current State', cv2.cvtColor(np.array(_rgb), cv2.COLOR_BGR2RGB))

                debug_vae(image, img_pred, img_seg_resort, \
                          left_img_input_list, left_img_pred, left_img_seg_resort, \
                          right_img_input_list, right_img_pred, right_img_seg_resort, \
                          route_fig, route_pred, self.step)

        # if self.step % 10 ==0:
        #     # route_dir = '/home/quan/distribution'
        #     route_dir = '/home/cst/wk/carla/1_2020_CARLA_challenge/vae_test_results'
        #     save_display(image, topdown, self.step, route_dir)
        # error_message = ""
        # todo: debug
        # import time
        # debug_save(tick_data, time.time())
        return control, done, error_message

    def resort_seg(self, rgb_seg_tar):
        img_seg = np.asarray(rgb_seg_tar)
        img_seg_resort = np.zeros((img_seg.shape[0], img_seg.shape[1]))
        # img_seg_resort[img_seg == 7] = 1
        # img_seg_resort[img_seg == 10] = 2
        # img_seg_resort[img_seg == 4] = 3

        img_seg_resort[img_seg == 7] = 1
        # img_seg_resort[img_seg == 15] = 1
        img_seg_resort[img_seg == 10] = 2
        img_seg_resort[img_seg == 4] = 3
        img_seg_resort[img_seg == 1] = 4
        img_seg_resort[img_seg == 11] = 4
        img_seg_resort[img_seg == 2] = 5
        img_seg_resort[img_seg == 5] = 5
        img_seg_resort[img_seg == 12] = 5
        img_seg_resort[img_seg == 19] = 5
        img_seg_resort[img_seg == 20] = 5
        img_seg_resort[img_seg == 9] = 6
        img_seg_resort[img_seg == 21] = 6
        img_seg_resort[img_seg == 22] = 6
        img_seg_resort[img_seg == 6] = 7
        img_seg_resort[img_seg == 18] = 8

        img_seg_resort = transforms.functional.to_tensor(img_seg_resort)
        img_seg_resort = img_seg_resort.unsqueeze(0)

        return img_seg_resort

    def find_bbx(self, rgb_seg_tar):
        img_seg = np.asarray(rgb_seg_tar)
        img_vis = np.zeros((img_seg.shape[0], img_seg.shape[1]))
        # img_seg_resort[img_seg == 7] = 1
        # img_seg_resort[img_seg == 10] = 2
        # img_seg_resort[img_seg == 4] = 3
        left_up = [img_seg.shape[0], img_seg.shape[1]]
        right_down = [0, 0]
        bbx = []

        for h in range(img_seg.shape[0]):
            for w in range(img_seg.shape[1]):
                if img_seg[h][w] == 18:
                    img_vis[h][w] = 1
                    left_up = (h, w)
                    for i in range(h, img_seg.shape[0]):
                        for j in range(w, img_seg.shape[1]):
                            if i == h and j == w:
                                continue

            if np.sum(img_seg[h] == 18) > 0:
                print(h, img_seg[h])
            # for w in range(img_seg.shape[1]):
            #     if img_seg[h][w] == 18:
            #         if ((int(img_seg[max(0, h - 1)][w] == 18)) + int(img_seg[min(img_seg.shape[0] - 1, h + 1)][w] == 18) + int(
            #                 img_seg[h][max(0, w - 1)] == 18) + int(
            #                     img_seg[h][min(img_seg.shape[1] - 1, w + 1)] == 18)) >= 2:
            #
            #             left_up[0] = min(h, left_up[0])
            #             left_up[1] = min(w, left_up[1])
            #             _tmp_right_down_0 = right_down[0]
            #             right_down[0] = max(h, _tmp_right_down_0)
            #             right_down[1] = max(w, right_down[1])
        print('============================================')
        bbx = [left_up, right_down]
        return bbx

    def destroy(self):
        """
        Cleanup
        """
        self._hic._quit = True


class KeyboardControl(object):
    """
    Keyboard control for the human agent
    """

    def __init__(self, path_to_conf_file):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()

        # # Get the mode
        # if path_to_conf_file:
        #
        #     with (open(path_to_conf_file, "r")) as f:
        #         lines = f.read().split("\n")
        #         self._mode = lines[0].split(" ")[1]
        #         self._endpoint = lines[1].split(" ")[1]
        #
        #     # Get the needed vars
        #     if self._mode == "log":
        #         self._log_data = {'records': []}
        #
        #     elif self._mode == "playback":
        #         self._index = 0
        #         self._control_list = []
        #
        #         with open(self._endpoint) as fd:
        #             try:
        #                 self._records = json.load(fd)
        #                 self._json_to_control()
        #             except json.JSONDecodeError:
        #                 pass
        # else:
        #     self._mode = "normal"
        #     self._endpoint = None
        self._mode = "normal"
        self._endpoint = None

    def _json_to_control(self):

        # transform strs into VehicleControl commands
        for entry in self._records['records']:
            control = carla.VehicleControl(throttle=entry['control']['throttle'],
                                           steer=entry['control']['steer'],
                                           brake=entry['control']['brake'],
                                           hand_brake=entry['control']['hand_brake'],
                                           reverse=entry['control']['reverse'],
                                           manual_gear_shift=entry['control']['manual_gear_shift'],
                                           gear=entry['control']['gear'])
            self._control_list.append(control)

    # def parse_events(self, timestamp):
    def parse_events(self, clock):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            # self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp * 1000)
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 1.0
            self._control.brake = 0.0
            self._control.reverse = False
        elif keys[K_DOWN] or keys[K_s]:
            # self._control.brake = 0.0
            # self._control.reverse = True
            # self._control.throttle = 1.0
            self._control.brake = 1.0
            self._control.throttle = 0.0
        else:
            self._control.reverse = False
            self._control.throttle = 0.0
            self._control.brake = 0.0

        # steer_increment = 3e-4 * milliseconds
        steer_increment = 15 * 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        steer_cache = min(0.95, max(-0.95, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        # self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        new_record = {
            'control': {
                'throttle': self._control.throttle,
                'steer': self._control.steer,
                'brake': self._control.brake,
                'hand_brake': self._control.hand_brake,
                'reverse': self._control.reverse,
                'manual_gear_shift': self._control.manual_gear_shift,
                'gear': self._control.gear
            }
        }

        self._log_data['records'].append(new_record)

    def __del__(self):
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, 'w') as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)
