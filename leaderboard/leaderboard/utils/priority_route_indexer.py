from collections import OrderedDict
from dictor import dictor
import numpy as np
from queue import PriorityQueue
from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration
import copy
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.checkpoint_tools import fetch_dict, create_default_json_msg, save_dict


class PriorityRouteIndexer():
    def __init__(self, routes_file, scenarios_file, vehicle_num):
        self._routes_file = routes_file
        self._scenarios_file = scenarios_file
        self._configs_list = []
        self._index = 0
        if vehicle_num is None:
            vehicle_num = [None, None]
        # retrieve routes
        route_descriptions_list = RouteParser.parse_routes_file(self._routes_file,self._scenarios_file, False)
        self.n_routes = 2 * len(route_descriptions_list)

        self.completion_ratio = np.zeros(self.n_routes)
        self.route_priority = 100 * np.ones(self.n_routes)
        cnt = 0
        for i, config in enumerate(route_descriptions_list):
            config.index = cnt
            config.vehicle_num = vehicle_num[0]
            config.walker_num = vehicle_num[1]
            self._configs_list.append(copy.copy(config))
            cnt += 1

            config.index = cnt
            config.vehicle_num = 0
            config.walker_num = 0
            self._configs_list.append(copy.copy(config))
            cnt += 1

    def peek(self):
        return True

    def update_route(self, route_id, route_completion, st_waypoint):
        # self._configs_list[route_id].st = None
        if route_completion == 100:
            self._configs_list[route_id].st = None
        else:
            self._configs_list[route_id].st = st_waypoint
        self.completion_ratio[route_id] = route_completion
        self.route_priority[route_id] = 100 - route_completion

    def next(self):
        eps = np.random.random()
        if eps > 0.8:
            route_index = np.random.randint(0, self.n_routes)
        else:
            if np.sum(self.route_priority) == 0:
                route_index = np.random.randint(0, self.n_routes)
            else:
                p = np.exp(self.route_priority) / np.sum(np.exp(self.route_priority))
                route_index = np.random.choice(self.n_routes, 1, p=p)[0]
        return self._configs_list[route_index]

