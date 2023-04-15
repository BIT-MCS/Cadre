import numpy as np
import copy
from leaderboard.utils.route_parser import RouteParser


class RouteIndexer():
    def __init__(self, routes_file, scenarios_file, vehicle_num):
        self._routes_file = routes_file
        self._scenarios_file = scenarios_file
        self._configs_list = []
        self._index = 0
        if vehicle_num is None:
            vehicle_num = [None, None]
        # retrieve routes
        route_descriptions_list = RouteParser.parse_routes_file(self._routes_file,self._scenarios_file, False)
        self.n_routes = len(route_descriptions_list)

        self.completion_ratio = np.zeros(self.n_routes)
        cnt = 0
        for i, config in enumerate(route_descriptions_list):
            config.index = cnt
            config.vehicle_num = vehicle_num[0]
            config.walker_num = vehicle_num[1]
            self._configs_list.append(copy.copy(config))
            cnt += 1
        self._route_index = 0

    def peek(self):
        return True

    def update_route(self, route_id, route_completion, st_waypoint):
        print('update new route')
        self._configs_list[route_id].st = None
        self.completion_ratio[route_id] = route_completion

    def next(self):
        route_index = self._route_index
        self._route_index += 1
        self._route_index %= self.n_routes
        return self._configs_list[route_index]

