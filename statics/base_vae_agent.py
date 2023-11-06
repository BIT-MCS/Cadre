import os
import numpy as np
import cv2
import torch
import torchvision
import carla
# from srunner.scenariomanager.carla_data_provider import CarlaActorPool
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import pygame
from PIL import Image, ImageDraw

from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter
from torchvision.transforms import CenterCrop

from team_code.pid_controller import PIDController
from leaderboard.autoagents.autonomous_agent import Track
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights
import pickle

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


def get_entry_point():
    return 'VaeAgent'


import time

import cv2
import carla

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner

# root_path = '/home/luban/test_weather'
# if not os.path.exists(root_path):
#     os.mkdir(root_path)


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
    _combined = Image.fromarray(tick_data['lidar'])
    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))

    _topdown = Image.fromarray(tick_data['topdown'])
    cv2.imshow('topdown', cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB))

    _rgb = Image.fromarray(tick_data['rgb'])
    cv2.imshow('rgb', cv2.cvtColor(np.array(_rgb), cv2.COLOR_BGR2RGB))

    _rgb_left = Image.fromarray(tick_data['rgb_left'])
    cv2.imshow('rgb_left', cv2.cvtColor(np.array(_rgb_left), cv2.COLOR_BGR2RGB))

    _rgb_right = Image.fromarray(tick_data['rgb_right'])
    cv2.imshow('rgb_right', cv2.cvtColor(np.array(_rgb_right), cv2.COLOR_BGR2RGB))


# def debug_save(tick_data, step):
#     route = os.environ['ROUTES'].strip().split('/')[-1].split('.')[0]
#
#     _topdown = Image.fromarray(tick_data['topdown'])
#     _rgb = Image.fromarray(tick_data['rgb'])
#     _rgb_left = Image.fromarray(tick_data['rgb_left'])
#     _rgb_right = Image.fromarray(tick_data['rgb_right'])
#     route_path = os.path.join(root_path, route)
#     center_cam_path = os.path.join(route_path, 'center_cam')
#     left_cam_path = os.path.join(route_path, 'left_cam')
#     right_cam_path = os.path.join(route_path, 'right_cam')
#     topdown_seg_path = os.path.join(route_path, 'topdown_seg')
#     lidar_path = os.path.join(route_path, 'lidar')
#
#     if not os.path.exists(route_path):
#         os.mkdir(route_path)
#         os.mkdir(center_cam_path)
#         os.mkdir(left_cam_path)
#         os.mkdir(right_cam_path)
#         os.mkdir(topdown_seg_path)
#         os.mkdir(lidar_path)
#
#     _rgb.save(os.path.join(center_cam_path, ('%04d.png' % step)))
#     _rgb_left.save(os.path.join(left_cam_path, ('%04d.png' % step)))
#     _rgb_right.save(os.path.join(right_cam_path, ('%04d.png' % step)))
#     Image.fromarray(tick_data['topdown_seg']).save(os.path.join(topdown_seg_path, ('%04d.png' % step)))
#     df_lidar = open(os.path.join(lidar_path, ('%04d.pkl' % step)), 'wb')
#     pickle.dump(tick_data['point_cloud'], df_lidar)
#     df_lidar.close()


class BaseVaeAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.converter = Converter()
        self.debug_step = -1

    def _init(self):
        self.lidar_bin = 0.25
        self.x_obs_range = 64
        self.y_obs_range = 36
        self.pre_topdown = None
        # self.lidar_bin = 0.5
        self.bias = 0
        self.lidar_height = 1.6

        self._traffic_lights = list()
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)
        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()
        # self._waypoint_planner = RoutePlanner(4.0, 50)
        # todo: change_1
        self._waypoint_planner = RoutePlanner(2.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 90,
                'id': 'rgb'
            },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 1.3, 'y': 0.0, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 90,
                'id': 'rgb_seg'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.2, 'y': -0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                'width': 400, 'height': 300, 'fov': 90,
                'id': 'rgb_left'
            },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 1.2, 'y': -0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                'width': 400, 'height': 300, 'fov': 90,
                'id': 'rgb_left_seg'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.2, 'y': 0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                'width': 400, 'height': 300, 'fov': 90,
                'id': 'rgb_right'
            },
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 1.2, 'y': 0.25, 'z': 1.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                'width': 400, 'height': 300, 'fov': 90,
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
                'x': 0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'id': 'obstacle'
            }
        ]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def tick(self, input_data):

        self.step += 1
        point_cloud = []
        for location in input_data['LIDAR'][1]:
            point_cloud.append([location[0], -location[1], -location[2]])
        point_cloud = np.array(point_cloud)
        x_bins = np.arange(-self.x_obs_range / 2, self.x_obs_range / 2 + self.lidar_bin, self.lidar_bin)
        y_bins = np.arange(-self.y_obs_range / 2, self.y_obs_range / 2 + self.lidar_bin, self.lidar_bin)
        # z_bins = [-self.lidar_height - 3, 1.0, 15, 700]
        # lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
        lidar, _ = np.histogramdd(point_cloud[:, 0:2], bins=(x_bins, y_bins))
        # lidar[:, :, 0] = 255 * np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
        # lidar[:, :, 1] = 255 * np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
        # lidar[:, :, 2] = 255 * np.array(lidar[:, :, 2] > 0, dtype=np.uint8)
        lidar = 255 * np.array(lidar > 0, dtype=np.uint8)
        # lidar = np.array(lidar, dtype=np.uint8)
        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'), size=256,
                                                 pixels_per_meter=3.66)

        rgb_seg = input_data['rgb_seg'][1][:, :, 2]
        rgb_left_seg = input_data['rgb_left_seg'][1][:, :, 2]
        rgb_right_seg = input_data['rgb_right_seg'][1][:, :, 2]

        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights, size=256, pixels_per_meter=3.66,
                                      radius=5)
        left_line = 56
        right_line = 200
        topdown = topdown[:, left_line:right_line]
        rgb_topdown = np.zeros((topdown.shape[0], topdown.shape[1], 3))
        rgb_topdown[topdown == 7] = (0, 255, 255)
        rgb_topdown[topdown == 10] = (0, 255, 0)
        rgb_topdown[topdown == 4] = (0, 255, 0)
        rgb_topdown[topdown == 11] = (0, 0, 255)

        rgb_topdown[topdown == 13] = (255, 0, 0)
        rgb_topdown[topdown == 14] = (255, 0, 255)
        rgb_topdown[topdown == 15] = (255, 255, 0)
        rgb_topdown[topdown == 16] = (128, 0, 128)
        rgb_topdown[topdown == 17] = (128, 128, 0)
        rgb_topdown = np.array(rgb_topdown, dtype=np.uint8)

        target_diff = 0
        if self.pre_topdown is None:
            target_diff = 100.0
        else:
            target_diff = abs(self.pre_topdown - topdown)
            target_diff = np.array(target_diff[:, :] > 0).sum()
        if target_diff >= 20:
            self.pre_topdown = topdown

        obstacle_data = input_data['obstacle']
        obstacle_distance = -1
        if obstacle_data[0] > -1:
            obstacle_distance = obstacle_data[1][0]
        result = {
            'rgb': rgb,
            'rgb_seg': rgb_seg,
            'rgb_left': rgb_left,
            'rgb_left_seg': rgb_left_seg,
            'rgb_right': rgb_right,
            'rgb_right_seg': rgb_right_seg,
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'lidar': lidar,
            'topdown': rgb_topdown,
            'target_diff': target_diff,
            'topdown_seg': topdown,
            'point_cloud': np.array(point_cloud),
            'obstacle': obstacle_distance
        }

        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        gps = self._get_position(result)
        # regardless of command
        far_node, _, _ = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target
        return result
