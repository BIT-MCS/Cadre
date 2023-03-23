import os
from collections import deque
from agents.navigation.local_planner import RoadOption
import copy

import numpy as np

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size, title=""):
        self.size = size
        self.clear()
        if title == "":
            self.title = str(self.size)
        else:
            self.title = title

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x - r, y - r, x + r, y + r), color)

    def line(self, pos, pre_node, cur_node, color=(255, 255, 255)):
        pre_x, pre_y = 5.5 * (pos - pre_node)
        pre_x += self.size / 2
        pre_y += self.size / 2

        cur_x, cur_y = 5.5 * (pos - cur_node)
        cur_x += self.size / 2
        cur_y += self.size / 2

        self.draw.line((pre_x, pre_y, cur_x, cur_y), color, width=5)

    def show(self):
        if not DEBUG:
            return

        import cv2
        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


# class RoutePlanner(object):
#     def __init__(self, min_distance, max_distance, debug_size=256):
#         self.route = deque()
#         self.min_distance = min_distance
#         self.max_distance = max_distance
#
#         self.mean = np.array([49.0, 8.0])
#         self.scale = np.array([111324.60662786, 73032.1570362])
#
#         self.debug = Plotter(debug_size)
#
#     def set_route(self, global_plan, gps=False):
#         self.route.clear()
#         for pos, cmd in global_plan:
#             if gps:
#                 pos = np.array([pos['lat'], pos['lon']])
#                 pos -= self.mean
#                 pos *= self.scale
#             else:
#                 pos = np.array([pos.location.x, pos.location.y])
#                 pos -= self.mean
#             self.route.append((pos, cmd))
#
#     def run_step(self, gps):
#         self.debug.clear()
#
#         if len(self.route) == 1:
#             return self.route[0]
#
#         to_pop = 0
#         farthest_in_range = -np.inf
#         cumulative_distance = 0.0
#         route_list = [self.route[0][0]]
#
#         for i in range(1, len(self.route)):
#             if cumulative_distance > self.max_distance:
#                 break
#
#             cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
#             distance = np.linalg.norm(self.route[i][0] - gps)
#
#             if distance <= self.min_distance and distance > farthest_in_range:
#                 farthest_in_range = distance
#                 to_pop = i
#
#             r = 255 * int(distance > self.min_distance)
#             g = 255 * int(self.route[i][1].value == 4)
#             b = 255
#             # self.debug.dot(gps, self.route[i][0], (r, g, b))
#             self.debug.line(gps, self.route[i][0], self.route[i-1][0], (r, g, b))
#             route_list.append(self.route[i][0])
#
#         for _ in range(to_pop):
#             if len(self.route) > 2:
#                 self.route.popleft()
#
#         self.debug.dot(gps, self.route[0][0], (0, 255, 0))
#         self.debug.dot(gps, self.route[1][0], (255, 0, 0))
#         self.debug.dot(gps, gps, (0, 0, 255))
#         self.debug.show()
#
#         return self.route[1][0], self.route[1][1], route_list
#
# class DynamicRoutePlanner(object):
#     def __init__(self, min_distance, max_distance, debug_size=256):
#         self.route = deque()
#         self.ori_min_distance = min_distance
#         self.min_distance = min_distance
#         self.ori_max_distance = max_distance
#         self.max_distance = max_distance
#
#         # self.mean = np.array([49.0, 8.0])
#         # self.scale = np.array([111324.60662786, 73032.1570362])
#         self.mean = np.array([49.0, 49.0])
#         self.scale = np.array([111324.60662786, 111324.60662786])
#         self.debug = Plotter(debug_size)
#         self.turn_distance = []
#         self.turn_command = []
#         self._turn_index = 0
#         self.pre_cmd = RoadOption.LANEFOLLOW
#
#     def count_distance(self, global_plan):
#         pre_cmd = RoadOption.LANEFOLLOW
#         sum_distance = 0
#         turn_command = [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]
#         pre_node = [global_plan[0][0]['lat'], global_plan[0][0]['lon']]
#         pre_node -= self.mean
#         pre_node *= self.scale
#         for pos, cmd in global_plan:
#             node = [pos['lat'], pos['lon']]
#             node -= self.mean
#             node *= self.scale
#             if cmd in turn_command:
#                 sum_distance += self.get_dis(node, pre_node)
#             elif cmd == RoadOption.LANEFOLLOW and pre_cmd in turn_command:
#                 sum_distance += self.get_dis(node, pre_node)
#                 # if sum_distance > 10:
#                 self.turn_distance.append(sum_distance)
#                 self.turn_command.append(pre_cmd)
#                 sum_distance = 0
#             pre_cmd = cmd
#             pre_node = node
#         if sum_distance > 0:
#             self.turn_distance.append(sum_distance)
#             self.turn_command.append(pre_cmd)
#
#     def set_route(self, global_plan, gps=False):
#         self.route.clear()
#         self.count_distance(global_plan)
#         # if len(self.turn_distance) == 0:
#         #     self.ori_min_distance = 9
#         # else:
#         #     if np.max(self.turn_distance) > 23:
#         #         self.ori_min_distance = 9
#         #     else:
#         #         self.ori_min_distance = 14
#         #
#         # if self.turn_distance[0] >= 30:
#         #     self.min_distance = 3
#         # else:
#         #     self.min_distance = self.ori_min_distance
#
#         for pos, cmd in global_plan:
#             if gps:
#                 pos = np.array([pos['lat'], pos['lon']])
#                 pos -= self.mean
#                 pos *= self.scale
#             else:
#                 pos = np.array([pos.location.x, pos.location.y])
#                 pos -= self.mean
#             self.route.append((pos, cmd))
#
#     def get_dis(self, node_1, node_2):
#         return np.sqrt((node_1[0] - node_2[0]) ** 2 + (node_1[1] - node_2[1]) ** 2)
#
#     def run_step(self, gps):
#         turn_command = [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]
#         if DEBUG:
#             self.debug.clear()
#
#         to_pop = 0
#         farthest_in_range = -np.inf
#         cumulative_distance = 0.0
#         route_list = [self.route[0][0]]
#
#         if len(self.route) == 1:
#             return self.route[0][0], self.route[0][1], route_list
#
#         for i in range(1, len(self.route)):
#             if cumulative_distance > self.max_distance:
#                 break
#
#             # cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
#             cumulative_distance += self.get_dis(self.route[i][0], self.route[i - 1][0])
#
#             # distance = np.linalg.norm(self.route[i][0] - gps)
#             distance = self.get_dis(self.route[i][0], gps)
#             if distance <= self.min_distance and distance > farthest_in_range:
#                 farthest_in_range = distance
#                 to_pop = i
#             route_list.append(self.route[i][0])
#
#             if DEBUG:
#                 r = 255 * int(distance > self.min_distance)
#                 g = 255 * int(self.route[i][1].value == 4)
#                 b = 255
#                 # self.debug.dot(gps, self.route[i][0], (r, g, b))
#                 self.debug.line(gps, self.route[i][0], self.route[i - 1][0], (r, g, b))
#
#         # to_pop = max(0, to_pop-1)
#         for _ in range(to_pop):
#             if len(self.route) > 2:
#                 # if self.route[1][1] == RoadOption.LANEFOLLOW and self.route[0][1] in turn_command:
#                 #     self._turn_index += 1
#                 #     self.min_distance = self.ori_min_distance
#                 #     if self._turn_index < len(self.turn_distance):
#                 #         if self.turn_distance[self._turn_index] > 30:
#                 #             self.min_distance = 3
#                 self.route.popleft()
#                 del route_list[0]
#
#         # if self.route[1][1] == RoadOption.LANEFOLLOW and self.pre_cmd in turn_command:
#         #     self._turn_index += 1
#         self.pre_cmd = self.route[1][1]
#         return self.route[1][0], self.route[1][1], route_list


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0])
        # self.scale = np.array([111324.60662786, 73032.1570362])
        self.mean = np.array([49.0, 49.0])
        self.scale = np.array([111324.60662786, 111324.60662786])
        self.debug = Plotter(debug_size)
        self.turn_distance = []
        self._turn_index = 0
        self.pre_cmd = RoadOption.LANEFOLLOW

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean
            self.route.append((pos, cmd))

        # print('in planner:', len(self.route), self.route[-1])

    # def run_step(self, gps):
    #     self.debug.clear()
    #
    #     if len(self.route) == 1:
    #         return self.route[0]
    #
    #     to_pop = 0
    #     farthest_in_range = -np.inf
    #     cumulative_distance = 0.0
    #     route_list = [self.route[0][0]]
    #
    #     for i in range(1, len(self.route)):
    #         if cumulative_distance > self.max_distance:
    #             break
    #
    #         cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
    #         distance = np.linalg.norm(self.route[i][0] - gps)
    #         route_list.append(self.route[i][0])
    #         if distance <= self.min_distance and distance > farthest_in_range:
    #             farthest_in_range = distance
    #             to_pop = i
    #         if DEBUG:
    #             r = 255 * int(distance > self.min_distance)
    #             g = 255 * int(self.route[i][1].value == 4)
    #             b = 255
    #             # self.debug.dot(gps, self.route[i][0], (r, g, b))
    #             self.debug.line(gps, self.route[i][0], self.route[i-1][0], (r, g, b))
    #
    #     to_pop = max(0, to_pop-1)
    #     for _ in range(to_pop):
    #         if len(self.route) > 2:
    #             self.route.popleft()
    #
    #     self.debug.dot(gps, self.route[0][0], (0, 255, 0))
    #     self.debug.dot(gps, self.route[1][0], (255, 0, 0))
    #     self.debug.dot(gps, gps, (0, 0, 255))
    #     self.debug.show()
    #
    #     return self.route[1][0], self.route[1][1], route_list
    def get_dis(self, node_1, node_2):
        return np.sqrt((node_1[0] - node_2[0]) ** 2 + (node_1[1] - node_2[1]) ** 2)

    def run_step(self, gps):
        if DEBUG:
            self.debug.clear()

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0
        route_list = [self.route[0][0]]

        if len(self.route) == 1:
            return self.route[0][0], self.route[0][1], route_list

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            # cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            cumulative_distance += self.get_dis(self.route[i][0], self.route[i - 1][0])

            # distance = np.linalg.norm(self.route[i][0] - gps)
            distance = self.get_dis(self.route[i][0], gps)
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i
            route_list.append(self.route[i][0])

            if DEBUG:
                r = 255 * int(distance > self.min_distance)
                g = 255 * int(self.route[i][1].value == 4)
                b = 255
                # self.debug.dot(gps, self.route[i][0], (r, g, b))
                self.debug.line(gps, self.route[i][0], self.route[i - 1][0], (r, g, b))

        # to_pop = max(0, to_pop-1)
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()
                del route_list[0]
        if DEBUG:
            self.debug.dot(gps, self.route[0][0], (0, 255, 0))
            self.debug.dot(gps, self.route[1][0], (255, 0, 0))
            self.debug.dot(gps, gps, (0, 0, 255))
            self.debug.show()
        return self.route[1][0], self.route[1][1], route_list
