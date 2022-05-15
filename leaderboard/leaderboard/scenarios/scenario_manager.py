#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider
import csv
import numpy as np
# from line_profiler import LineProfiler


class ScenarioManager(object):
    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, rank, result_file_path, log_file_path, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None
        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        timeout = 300
        self._timeout = float(timeout)
        self.rank = rank
        # Used to detect if the simulation is down
        # watchdog_timeout = max(5, self._timeout - 2)
        watchdog_timeout = max(0, self._timeout - 2)
        # self._watchdog = Watchdog(watchdog_timeout, rank)

        # # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        # self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None
        self.route_completion_ratio = 0
        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)
        self.result_file_path = result_file_path
        self.log_file_path = log_file_path
        if result_file_path:
            file = open(result_file_path, 'a', newline='')
            writer = csv.writer(file)
            writer.writerow(
                ['RouteCompletion', 'OutsideRouteLanes', 'Collision', 'RunningRedLight', 'RunningStop', 'InRoute',
                 'AgentBlocked'])
            file.close()

        self.reload = -1
        self.error_message = ""
        self.waypoint = None
        self.total_scenario_num = [0, 0]

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        print(self.rank, "Catch interrupt!!")
        # self._running = False
        # self.reload = 2
        self.reload = 1

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None
        self.reload = -1

    def load_scenario(self, scenario, agent):
        """
        Load a new scenario
        """
        # with open(self.log_file_path, 'a') as f:
        #     f.write('in load_scenario\n')
        self.error_message = ""
        GameTime.restart()
        # print(GameTime.get_time())
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        # todo: add
        self.list_scenarios = scenario.list_scenarios

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)
        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)
        if self._agent is not None:
            self._agent.setup_scenarios(self.scenario)

    def get_scenario_message(self):
        total_scenario_num = [0, 0]
        success_scenario_num = [0, 0]
        list_scenario_name = [scenario.name for scenario in self.list_scenarios]
        list_scenario_status = [scenario.status[0] for scenario in self.list_scenarios]
        for name, status in zip(list_scenario_name, list_scenario_status):
            if name == 'FollowVehicle':
                if status == 'success':
                    total_scenario_num[0] += 1
                    success_scenario_num[0] += 1
                elif status == 'running':
                    total_scenario_num[0] += 1
            elif name == 'DynamicObjectCrossing':
                if status == 'success':
                    total_scenario_num[1] += 1
                    success_scenario_num[1] += 1
                elif status == 'running':
                    total_scenario_num[1] += 1
        scenario_num = success_scenario_num + total_scenario_num + [self._agent._agent.intersection_num,
                                                                    self._agent._agent.success_intersection_num]
        return scenario_num

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        # with open(self.log_file_path, 'a') as f:
        #     f.write('in scenario_manager.run_scenario')
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        # self._watchdog.start()
        self._running = True
        reload = 0
        self.reload = -1
        # lp = LineProfiler()
        # lp_wrapper = lp(self._agent._agent.run_step)
        lp_wrapper = None
        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
                # else:
                #     with open(self.log_file_path, 'a') as f:
                #         f.write('timestamp is none\n')
            # else:
            #     with open(self.log_file_path, 'a') as f:
            #         f.write(' world is none\n')
            if timestamp:
                reload = self._tick_scenario(timestamp, lp_wrapper)
                # reload = lp_wrapper(timestamp, None)
                if reload is not None:
                    if reload:
                        break
                if self.reload > -1:
                    break
        # lp.print_stats()
        # if reload == 1:
        #     self._running = False
        if self.reload > -1:
            reload = self.reload
        # print("reload: ", reload, " running", self._running, " self.reload", self.reload)
        self.waypoint = self.get_cur_waypoint()
        return reload

    def _tick_scenario(self, timestamp, lp_wrapper=None):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        reload = None
        # with open(self.log_file_path, 'a') as f:
        #     f.write('in _tick_scenario\n')
        # return reload
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            # self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                # todo: add reload
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        self._agent.update_completion(criterion.actual_value)

                ego_action, reload, error_message = self._agent(lp_wrapper)
                self.error_message = error_message

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            # todo: change
            # if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            if self.scenario_tree.status == py_trees.common.Status.FAILURE:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        # list_scenario_status = [scenario.status[0] for scenario in self.list_scenarios]
        # list_scenario_name = [scenario.name for scenario in self.list_scenarios]
        # print('list_scenario_status', list_scenario_status)
        # if self._running and self.get_running_status():
        # with open(self.log_file_path, 'a') as f:
        #     f.write('before CarlaDataProvider.get_world().tick(self._timeout)\n')
        #     f.write(str(self._running)+'\n')
        if self._running:
            CarlaDataProvider.get_world().tick(self._timeout)

        # with open(self.log_file_path, 'a') as f:
        #     f.write('after CarlaDataProvider.get_world().tick(self._timeout)\n')
        return reload

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        # return self._watchdog.get_status()
        return True

    def get_cur_waypoint(self):
        location = self.ego_vehicles[0].get_transform().location
        min_index = 0
        min_distance = 999999
        for index, transform in enumerate(self.scenario_class.route):
            transform = transform[0].location
            distance = (location.x - transform.x) ** 2 + (location.y - transform.y) ** 2
            if distance < min_distance:
                min_index = index
                min_distance = distance
        if min_index > 0:
            st_index = min_index - 1
            sum_distance = 0
            pre_location = location
            while st_index >= 0:
                transform = self.scenario_class.route[st_index][0].location
                distance = (pre_location.x - transform.x) ** 2 + (pre_location.y - transform.y) ** 2
                sum_distance += distance
                st_index -=1
                pre_location = transform
                if sum_distance > 10:
                    break
            st_index += 1
        else:
            st_index = 0
        transform = self.scenario_class.route[st_index][0].location
        waypoint = carla.Location(x=float(transform.x), y=float(transform.y), z=float(transform.z))
        return waypoint

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        # self._watchdog.stop()
        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                # print("in stop scenario")
                self._agent.cleanup()
                self._agent = None
            # todo: change
            self.repetition_number = None

            self.analyze_scenario()

            self._agent = None
            self.scenario_class = None
            self.scenario = None
            self.scenario_tree = None
            self.ego_vehicles = None
            self.other_actors = None
            self.repetition_number = None

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m' + 'SUCCESS' + '\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.name == "RouteCompletionTest":
                self.route_completion_ratio = criterion.actual_value

            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        # todo: output only after success
        # if "SUCCESS" in global_result:
        ResultOutputProvider(self, global_result, self.result_file_path)
