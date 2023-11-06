import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw

from carla_project.src.common import CONVERTER, COLOR
# from team_code.map_agent import MapAgent
from team_code.base_vae_agent import BaseVaeAgent
from team_code.pid_controller import PIDController
import pickle
from srunner.scenariomanager.traffic_events import TrafficEventType

from carla import WeatherParameters
import sys
import importlib
import os

# PRESET_WEATHERS = {
#     1: WeatherParameters.ClearNoon,
#     2: WeatherParameters.CloudyNoon,
#     3: WeatherParameters.WetNoon,
#     4: WeatherParameters.WetCloudyNoon,
#     5: WeatherParameters.MidRainyNoon,
#     6: WeatherParameters.HardRainNoon,
#     7: WeatherParameters.SoftRainNoon,
#     8: WeatherParameters.ClearSunset,
#     9: WeatherParameters.CloudySunset,
#     10: WeatherParameters.WetSunset,
#     11: WeatherParameters.WetCloudySunset,
#     12: WeatherParameters.MidRainSunset,
#     13: WeatherParameters.HardRainSunset,
#     14: WeatherParameters.SoftRainSunset,
# }


## update for cvpr challenge
ClearNight = WeatherParameters.ClearNoon
ClearNight.sun_altitude_angle = -90

CloudyNight = WeatherParameters.CloudyNoon
CloudyNight.sun_altitude_angle = -90

WetNight = WeatherParameters.WetNoon
WetNight.sun_altitude_angle = -90

WetCloudyNight = WeatherParameters.WetCloudyNoon
WetCloudyNight.sun_altitude_angle = -90

SoftRainNight = WeatherParameters.SoftRainNoon
SoftRainNight.sun_altitude_angle = -90

MidRainyNight = WeatherParameters.MidRainyNoon
MidRainyNight.sun_altitude_angle = -90

HardRainNight = WeatherParameters.HardRainNoon
HardRainNight.sun_altitude_angle = -90


PRESET_WEATHERS = {
    1: WeatherParameters.ClearNoon,
    2: WeatherParameters.CloudyNoon,
    3: WeatherParameters.WetNoon,
    4: WeatherParameters.WetCloudyNoon,
    5: WeatherParameters.MidRainyNoon,
    6: WeatherParameters.HardRainNoon,
    7: WeatherParameters.SoftRainNoon,
    8: WeatherParameters.ClearSunset,
    9: WeatherParameters.CloudySunset,
    10: WeatherParameters.WetSunset,
    11: WeatherParameters.WetCloudySunset,
    12: WeatherParameters.MidRainSunset,
    13: WeatherParameters.HardRainSunset,
    14: WeatherParameters.SoftRainSunset,
    15: ClearNight,
    16: CloudyNight,
    17: WetNight,
    18: WetCloudyNight,
    19: SoftRainNight,
    20: MidRainyNight,
    21: HardRainNight,
}


def get_entry_point():
    return 'VaeAgent'


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def debug_display(tick_data):
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

    _route_fig = Image.fromarray(tick_data['route_fig'])
    cv2.imshow('route_fig', cv2.cvtColor(np.array(_route_fig), cv2.COLOR_BGR2RGB))

def debug_save(conf, routes, train_path, test_path, tick_data, repeat_id, step, light_state, light_distance, far_node=None, near_command=None, steer=None,
               throttle=None, brake=None, target_speed=None, pos=None, weather=None, xml_route_id=None):
    # route_id = int(os.environ['ROUTE_ID'])
    # route = 'route_' + ('%02d' % route_id)
    # if int(route_id) < 9:
    #     save_path = train_path
    # else:
    #     save_path = test_path

    # repeat_id += 41

    # route = os.environ['ROUTES'].strip().split('/')[-1].split('.')[0]
    route = routes.strip().split('/')[-1].split('.')[0]
    # print('route: ' + str(route))
    save_path = None
    
    tmp = route.strip().split('_')[1]
    if tmp.find('Town') >= 0:
        town_id = int(tmp[-1])
        route_id = town_id * 100 + int(route.strip().split('_')[-1])
    else:
        route_id = int(route.strip().split('_')[1])
    # route = route[:-2] + str(route_id)
    if int(route_id) < 1000:
        save_path = train_path  
    else:
        save_path = test_path
    route = route + conf.amount_key

    _topdown = Image.fromarray(tick_data['topdown'])
    _rgb = Image.fromarray(tick_data['rgb'])
    _rgb_seg = Image.fromarray(tick_data['rgb_seg'])
    _rgb_left = Image.fromarray(tick_data['rgb_left'])
    _rgb_left_seg = Image.fromarray(tick_data['rgb_left_seg'])
    _rgb_right = Image.fromarray(tick_data['rgb_right'])
    _rgb_right_seg = Image.fromarray(tick_data['rgb_right_seg'])

    save_path = os.path.join(save_path, route)
    center_cam_path = os.path.join(save_path, 'center_cam')
    center_cam_seg_path = os.path.join(save_path, 'center_cam_seg')
    left_cam_path = os.path.join(save_path, 'left_cam')
    left_cam_seg_path = os.path.join(save_path, 'left_cam_seg')
    right_cam_path = os.path.join(save_path, 'right_cam')
    right_cam_seg_path = os.path.join(save_path, 'right_cam_seg')
    topdown_seg_path = os.path.join(save_path, 'topdown_seg')
    lidar_path = os.path.join(save_path, 'lidar')
    cloudpoint_path = os.path.join(save_path, 'cloudpoint')
    measurements_path = os.path.join(save_path, 'measurements')
    routes_path = os.path.join(save_path, 'routes')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(center_cam_path)
        os.mkdir(center_cam_seg_path)
        os.mkdir(left_cam_path)
        os.mkdir(left_cam_seg_path)
        os.mkdir(right_cam_path)
        os.mkdir(right_cam_seg_path)
        os.mkdir(topdown_seg_path)
        os.mkdir(lidar_path)
        os.mkdir(cloudpoint_path)
        os.mkdir(routes_path)
        if pos is not None:
            os.mkdir(measurements_path)

    _rgb.save(os.path.join(center_cam_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))
    _rgb_seg.save(os.path.join(center_cam_seg_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))
    _rgb_left.save(os.path.join(left_cam_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))
    _rgb_left_seg.save(os.path.join(left_cam_seg_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))
    _rgb_right.save(os.path.join(right_cam_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))
    _rgb_right_seg.save(os.path.join(right_cam_seg_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))
    Image.fromarray(tick_data['topdown_seg']).save(os.path.join(topdown_seg_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))

    # topdown = tick_data['topdown_seg']
    # left_line = 56
    # right_line = 200
    # topdown = topdown[:, left_line:right_line]

    route_fig = Image.fromarray(tick_data['route_fig'])

    # route_fig = tick_data['route_fig'] / np.max(tick_data['route_fig'])
    # route_fig = np.array(route_fig * (vehicle_speed / self.max_speed), dtype=np.float32)
    # route_fig = route_fig.swapaxes(0, 1)
    # route_fig = np.expand_dims(route_fig, 0)
    route_fig.save(os.path.join(routes_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))

    lidar_fig = Image.fromarray(tick_data['lidar'])
    lidar_fig.save(os.path.join(lidar_path, ('%s_%03d_%06d.png' % (xml_route_id, repeat_id, step))))
    # df_lidar = open(os.path.join(lidar_path, ('%02d_%04d.png' % (repeat_id, step))), 'wb')
    # pickle.dump(tick_data['lidar'], df_lidar)
    # df_lidar.close()
    
    # df_cloudpoint = open(os.path.join(pointcloud_path, ('%02d_%04d.pkl' % (repeat_id, step))), 'wb')
    # pickle.dump(tick_data['point_cloud'], df_cloudpoint)
    # df_cloudpoint.close()

    if pos is not None:
        theta = tick_data['compass']
        speed = tick_data['speed']
        data = {
            'x': pos[0],
            'y': pos[1],
            'theta': theta,
            'speed': speed,
            'target_speed': target_speed,
            'x_command': far_node[0],
            'y_command': far_node[1],
            'command': near_command.value,
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            'light_state': light_state,
            'light_distance': light_distance,
            'weather': weather
        }
        df_measurements = open(os.path.join(measurements_path, ('%s_%03d_%06d.json' % (xml_route_id, repeat_id, step))), 'wb')
        pickle.dump(data, df_measurements)
        df_measurements.close()


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


class VaeAgent(BaseVaeAgent):
    def __init__(self, path_to_conf_file, routes, dataset_config):
        super(VaeAgent, self).__init__(path_to_conf_file)

        self.dataset_config = dataset_config
        self.routes = routes
        # Load agent
        module_name = os.path.basename(self.dataset_config).split('.')[0]
        sys.path.insert(0, os.path.dirname(self.dataset_config))
        self.cur_dataset_config = importlib.import_module(module_name)
        cur_dataset_conf = self.cur_dataset_config.CONF()
        # from config_files.config_data_agent import CONF
        # conf = CONF()

        self.root_path = cur_dataset_conf.root_path
        self.noise = cur_dataset_conf.noise
        self.train_path = os.path.join(self.root_path, 'train')
        self.test_path = os.path.join(self.root_path, 'test')
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)
        self.DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
        cur_dataset_conf.log_info()
        self.cur_dataset_conf = cur_dataset_conf

        self.last_img = None
        self.last_light_state = -2

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self.debug_step = -1
        self.last_event_timestamp = 0

    def set_repeat_id(self, repeat_id):
        self.repeat_id = repeat_id

    def _init(self):
        super()._init()
        self.begin = True
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _get_control(self, target, far_target, tick_data, _draw=None):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4 if should_slow else 7.0

        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        # _draw.text((5, 90), 'Speed: %.3f' % speed)
        # _draw.text((5, 110), 'Target: %.3f' % target_speed)
        # _draw.text((5, 130), 'Angle: %.3f' % angle_unnorm)
        # _draw.text((5, 150), 'Angle Far: %.3f' % angle_far_unnorm)

        return steer, throttle, brake, target_speed

    def get_traffic(self, new_event_list):
        light_distance = -1
        light_state = -1
        for event in new_event_list:
            if event.get_type() == TrafficEventType.APPROACH_LIGHT:
                light_distance = 10.0
                if event.get_dict():
                    light_distance = 1.0 * event.get_dict()['distance']
                    light_state = int(event.get_dict()['state'])
                    light_id = event.get_dict()['id']
        if light_state == -1:
            has_traffic_light = False
        else:
            has_traffic_light = True
        return light_state, light_distance, has_traffic_light

    def draw_route(self, tick_data, route_list, size_x=144, size_y=256, pixels_per_meter=3.66,
                   color=(255, 0, 0)):
        topdown = tick_data['topdown_seg']
        # left_line = 56
        # right_line = 200
        # topdown = topdown[:, left_line:right_line]

        route_fig = np.zeros((topdown.shape[0], topdown.shape[1]), dtype=np.uint8)
        route_fig = Image.fromarray(route_fig)
        # color = 50 * (light_state + 1)
        color = 255
        route_draw = ImageDraw.Draw(route_fig)
        compass = tick_data['compass']
        compass = 0.0 if np.isnan(compass) else compass
        compass = compass + np.pi / 2
        R = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)],
        ])
        pos = self._get_position(tick_data)
        far_node = None
        for i in range(1, len(route_list)):
            cur_node = route_list[i]
            pre_node = route_list[i - 1]
            pre_x, pre_y = pixels_per_meter * (R.T.dot(pre_node - pos))
            pre_x += size_x / 2
            pre_y += size_y / 2

            cur_x, cur_y = pixels_per_meter * (R.T.dot(cur_node - pos))
            cur_x += size_x / 2
            cur_y += size_y / 2
            route_draw.line((pre_x, pre_y, cur_x, cur_y), color, width=15)
            if abs(route_list[i][0] - route_list[0][0]) + abs(
                    route_list[i][1] - route_list[0][1]) > 1e-3 and far_node is None:
                far_node = route_list[i]

        route_fig = np.array(route_fig)
        tick_data['route_fig'] = route_fig

        vehicle_fig = np.zeros_like(route_fig, dtype=np.uint8)
        vehicle_fig[topdown == 4] = (255)
        vehicle_fig[topdown == 10] = (255)
        tick_data['vehicle_fig'] = vehicle_fig

        road_fig = np.zeros_like(route_fig, dtype=np.uint8)
        road_fig[topdown == 7] = (255)
        tick_data['road_fig'] = road_fig
        return tick_data

    def _is_done(self, new_event_list, speed, max_block_time=400):
        done = 0
        target_reached = False
        if self.begin is False:
            for event in new_event_list:
                print(self.step, event.get_type())
                if event.get_type() == TrafficEventType.COLLISION_STATIC:
                    done = 1
                elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN or event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                    done = 1
                elif event.get_type() == TrafficEventType.VEHICLE_BLOCKED:
                    done = 1
                elif event.get_type() == TrafficEventType.ROUTE_DEVIATION:
                    done = 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETED:
                    target_reached = True
                    done = 1
                elif event.get_type() == TrafficEventType.ROUTE_COMPLETION:
                    if not target_reached:
                        if event.get_dict():
                            score_route = event.get_dict()['route_completed']
                        else:
                            score_route = 0
                        print('score_route:', score_route)
                    done = 1
                # # todo: remove traffic light
                # elif event.get_type() == TrafficEventType.APPROACH_LIGHT:
                #     red_light_distance = -1
                #     if event.get_dict():
                #         red_light_distance = 1.0 * event.get_dict()['distance']
                #         light_state = event.get_dict()['state']


        else:
            self.begin = False

        # ==================== speed_reward control : [-1, 1] =============================
        project_reward = 0


        if speed < 0.1 and (self.step - self.last_event_timestamp) > max_block_time:
            print('vehicle block')
            self.last_event_timestamp = self.step
            # if self.training:
            done = 1

        if len(new_event_list) > 0:
            self.last_event_timestamp = self.step

        return done

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        if self.cur_dataset_conf.benchmark == 'nocrash':
            weather_index = np.random.choice(self.cur_dataset_conf.weather)
            weather = PRESET_WEATHERS[weather_index]
            # print('Current step %d and weather %s' % (self.step, weather))
            self._world.set_weather(weather)
        else:
            weather_index = np.random.choice([i for i in range(len(PRESET_WEATHERS))])
            weather = PRESET_WEATHERS[weather_index + 1]
            # print('Current step %d and weather %s' % (self.step, weather))
            self._world.set_weather(weather)

        data = self.tick(input_data)
        topdown = data['topdown_seg']
        rgb = np.hstack((data['rgb_left'], data['rgb'], data['rgb_right']))

        rgb_seg = data['rgb_seg']

        gps = self._get_position(data)

        near_node, near_command, route_list = self._waypoint_planner.run_step(gps)
        far_node, far_command, _ = self._command_planner.run_step(gps)
        target_steer, target_throttle, target_brake, target_speed = self._get_control(near_node, far_node, data)

        data = self.draw_route(data, route_list)

        # todo: random control
        control = carla.VehicleControl()
        speed = data['speed']

        if self.cur_dataset_conf.collect_collision:
            ### only for collison data, add speed and correct steer and brake 
            control.steer = target_steer + 1e-2 * np.random.randn()
            if target_steer > -0.1 and target_steer < 0.1:
                control.throttle = 0.80
                control.brake = float(False)
        elif self.cur_dataset_conf.collect_traffic_lights:
            steer = target_steer + 1e-2 * np.random.randn()
            control.steer = steer
            control.throttle = target_throttle
            control.brake = float(target_brake)

        else:
            if self.cur_dataset_conf.random_control:
                steer = 2 * np.random.random() -1
                # desired_speed = 8
                # delta = np.clip(desired_speed - speed, 0.0, 0.25)
                # throttle = self._speed_controller.step(delta)
                # throttle = np.clip(throttle, 0.0, 0.75)
                # brake = 0.0
                control.steer = steer
                control.throttle = target_throttle
                control.brake = float(target_brake)
            else:
                prob = np.random.random()
                if prob > 0.3:
                    # (2 * np.random.random() -1) is [-1, 1]
                    steer_noise = self.noise * (2 * np.random.random() -1)
                    steer = target_steer + steer_noise
                    steer = np.clip(steer, -1.0, 1.0)
                    steer = round(steer, 3)     
                else:
                    steer = target_steer + 1e-2 * np.random.randn()

                throttle = target_throttle
                brake = target_brake

                # not for light collect
                # if target_throttle < 0.2:
                #     prob = np.random.random()
                #     if prob > 0.4:
                #         throttle = 0.75
                #         brake = False

                # control.steer = steer + 1e-2 * np.random.randn()

                control.steer = steer
                control.throttle = throttle
                control.brake = float(brake)

        target_diff = data['target_diff']
        new_event_list = self.get_event()
        light_state, light_distance, has_traffic_lights = self.get_traffic(new_event_list)

        self.cur_img = data['rgb']
        if self.last_img is None:
            cur_rgb_change = np.mean(self.cur_img)
        else:
            cur_rgb_change = np.absolute(np.mean(self.cur_img - self.last_img))

        if self.DEBUG:
            _rgb = Image.fromarray(rgb)
            _rgb_seg = Image.fromarray(COLOR[CONVERTER[rgb_seg]])
            _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
            _draw = ImageDraw.Draw(_topdown)
            _topdown.thumbnail((256, 256))
            _rgb = _rgb.resize((int(256 / _rgb.size[1] * _rgb.size[0]), 256))
            _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
            _draw = ImageDraw.Draw(_combined)
            _draw.text((5, 10), 'FPS: %.3f' % (self.step / (time.time() - self.wall_start)))
            _draw.text((5, 30), 'Steer: %.3f' % control.steer)
            _draw.text((5, 50), 'Throttle: %.3f' % control.throttle)
            _draw.text((5, 70), 'Brake: %s' % control.brake)
            _draw.text((5, 90), 'Light_state: %s' % light_state)
            cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
            cv2.imshow('rgb_seg', cv2.cvtColor(np.array(_rgb_seg), cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            rgb_topdown = data['topdown']
            cv2.imshow('topdown', cv2.cvtColor(np.array(rgb_topdown), cv2.COLOR_BGR2RGB))
            lidar = Image.fromarray(data['lidar'])
            cv2.imshow('lidar', cv2.cvtColor(np.array(lidar), cv2.COLOR_BGR2RGB))
            route_fig = Image.fromarray(data['route_fig'])
            cv2.imshow('route_fig', cv2.cvtColor(np.array(route_fig), cv2.COLOR_BGR2RGB))

        obstacle_distance = data['obstacle']

        cur_weather = weather_index + 1
        xml_route_id = self.xml_route_id

        if self.cur_dataset_conf.collect_collision:
            num_vehicle = np.count_nonzero(rgb_seg == 10)
            ratio_vehicle = num_vehicle / rgb_seg.size

            num_people = np.count_nonzero(rgb_seg == 4)
            ratio_people = num_people / rgb_seg.size

            if (ratio_people > 1/20 or ratio_vehicle > 1/8) and np.array(data['lidar']).shape[0] > 10 \
            and (target_diff >= 20 or has_traffic_lights):
                self.debug_step += 1
                debug_save(self.cur_dataset_conf, self.routes, self.train_path, self.test_path, data, 
                            self.repeat_id, self.debug_step, light_state, light_distance, 
                            far_node, near_command, target_steer, target_throttle,
                            target_brake, target_speed, gps, weather=cur_weather, xml_route_id=xml_route_id)
        elif self.cur_dataset_conf.collect_traffic_lights:
            num_light = np.count_nonzero(rgb_seg == 18)
            ratio_light = num_light / rgb_seg.size

            
            if has_traffic_lights and (light_state != self.last_light_state):
                self.debug_step += 1
                debug_save(self.cur_dataset_conf, self.routes, self.train_path, self.test_path, data, 
                            self.repeat_id, self.debug_step, light_state, light_distance, 
                            far_node, near_command, target_steer, target_throttle,
                            target_brake, target_speed, gps, weather=cur_weather, xml_route_id=xml_route_id)

            if has_traffic_lights and num_light >= 15 and cur_rgb_change > 75:
                self.debug_step += 1
                debug_save(self.cur_dataset_conf, self.routes, self.train_path, self.test_path, data, 
                            self.repeat_id, self.debug_step, light_state, light_distance, 
                            far_node, near_command, target_steer, target_throttle,
                            target_brake, target_speed, gps, weather=cur_weather, xml_route_id=xml_route_id)
        else:
            if np.array(data['lidar']).shape[0] > 10 and (target_diff >= 20 or has_traffic_lights):
                self.debug_step += 1
                debug_save(self.cur_dataset_conf, self.routes, self.train_path, self.test_path, data, 
                            self.repeat_id, self.debug_step, light_state, light_distance, 
                            far_node, near_command, target_steer, target_throttle,
                            target_brake, target_speed, gps, weather=cur_weather, xml_route_id=xml_route_id)
        done = self._is_done(new_event_list, speed)

        self.last_light_state = light_state
        self.last_img = data['rgb']

        return control, done

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))

        return any(x is not None for x in [vehicle, light, walker])

    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        # if not DEBUG:
        #     return
    
        return 

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0] + v[0], p[1] + v[1], z)
        color = carla.Color(*color)

        self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        self._draw_line(p1, v1, z + 2.5, (0, 0, 255))

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            self._draw_line(p2, v2, z + 2.5)

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None

    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
        v1_hat = o1
        v1 = s1 * v1_hat

        self._draw_line(p1, v1, z + 2.5, (255, 0, 0))

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            self._draw_line(p2, v2, z + 2.5, (255, 0, 0))

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None
