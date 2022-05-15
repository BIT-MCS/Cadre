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

import os

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
print('DEBUG', DEBUG)


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
        # image_center = input_data['rgb'][1][:, :, -2::-1]
        image_center = np.array(input_data['rgb'])
        self._clock.tick_busy_loop(20)
        # display image
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()



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


class HumanAgent(object):
    """
    Human agent to control the ego vehicle via keyboard
    """

    current_control = None
    agent_engaged = False

    def __init__(self):
        """
        Setup the agent parameters
        """
        self.agent_engaged = False
        self._hic = HumanInterface()
        self._controller = KeyboardControl()

    def act(self, input_data):

        self.agent_engaged = True
        self._hic.run_interface(input_data)

        control = self._controller.parse_events(self._hic._clock)

        return control

    def destroy(self):
        """
        Cleanup
        """
        self._hic._quit = True


class KeyboardControl(object):
    """
    Keyboard control for the human agent
    """

    def __init__(self):
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
