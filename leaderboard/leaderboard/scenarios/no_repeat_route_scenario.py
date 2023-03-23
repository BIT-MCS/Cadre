#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import xml.etree.ElementTree as ET
import numpy.random as random

import py_trees

import carla

from agents.navigation.local_planner import RoadOption

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.basic_scenario import Scenario
from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRoute
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.junction_crossing_route import SignalJunctionCrossingRoute, NoSignalJunctionCrossingRoute
import srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions as conditions

from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)

from leaderboard.utils.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from leaderboard.utils.route_manipulation import interpolate_trajectory
import copy

ROUTESCENARIO = ["RouteScenario"]

SECONDS_GIVEN_PER_METERS = 0.8
INITIAL_SECONDS_DELAY = 5.0

NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    "Scenario2": FollowLeadingVehicle,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    "Scenario5": OtherLeadingVehicle,
    "Scenario6": ManeuverOppositeDirection,
    "Scenario7": SignalJunctionCrossingRoute,
    "Scenario8": SignalJunctionCrossingRoute,
    "Scenario9": SignalJunctionCrossingRoute,
    "Scenario10": NoSignalJunctionCrossingRoute
}


def oneshot_behavior(name, variable_name, behaviour):
    """
    This is taken from py_trees.idiom.oneshot.
    """
    # Initialize the variables
    blackboard = py_trees.blackboard.Blackboard()
    _ = blackboard.set(variable_name, False)

    # Wait until the scenario has ended
    subtree_root = py_trees.composites.Selector(name=name)
    check_flag = py_trees.blackboard.CheckBlackboardVariable(
        name=variable_name + " Done?",
        variable_name=variable_name,
        expected_value=True,
        clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE
    )
    set_flag = py_trees.blackboard.SetBlackboardVariable(
        name="Mark Done",
        variable_name=variable_name,
        variable_value=True
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="OneShot")
        sequence.add_children([behaviour, set_flag])

    subtree_root.add_children([check_flag, sequence])
    return subtree_root


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """

    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


class NoRepeatRouteScenario(object):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"

    def __init__(self, town, route_num):
        """
        Setup all relevant parameters and create scenarios along route
        """
        # self.config = config
        # self.route = None
        # todo: debug
        self.ego_vehicles = None
        self.gps_route = None
        # self.sampled_scenarios_definitions = None
        self.sampled_scenarios_definitions = [None for _ in range(route_num)]
        self.list_scenarios_list = [None for _ in range(route_num)]
        self.gps_route_list = [None for _ in range(route_num)]
        self.route_list = [None for _ in range(route_num)]
        self.list_scenarios = None

        self.origin_possible_scenarios = None

        # vehicle_amount = config.vehicle_num
        # pedestrian_amount = config.walker_num
        # vehicle_amount, pedestrian_amount = config.actor_amount
        # self.vehicle_amount = vehicle_amount
        # self.pedestrian_amount = pedestrian_amount
        # ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017',
        #                                                   random_location=True,
        #                                                   rolename='hero')
        # self.ego_vehicles = [ego_vehicle]

        self.other_actors = []
        self.world_actors_num = 0
        self.terminate_on_failure = False
        self._initialize_actors(town)

    def _update(self, st, ed, world, config, debug_mode=0):
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self._update_route)
        #
        # lp_wrapper(st, ed, world, config, debug_mode > 0)
        # lp.print_stats()
        self._update_route(st, ed, world, config, debug_mode > 0)

        ego_vehicle = self._update_ego_vehicle()
        if not ego_vehicle:
            return False

        if self.list_scenarios_list[config.index] is None:
            self.list_scenarios = self._build_scenario_instances(world,
                                                                 ego_vehicle,
                                                                 self.sampled_scenarios_definitions[config.index],
                                                                 scenarios_per_tick=10,
                                                                 timeout=self.timeout,
                                                                 debug_mode=debug_mode > 1)
            self.list_scenarios_list[config.index] = self.list_scenarios
            for scenario in self.list_scenarios_list[config.index]:
                self.other_actors.extend(scenario.other_actors)
        else:
            self.list_scenarios = self.list_scenarios_list[config.index]
        print('totally generated ', len(self.list_scenarios), ' scenarios!')
        # self.list_scenarios = self._build_scenario_instances(world,
        #                                                      ego_vehicle,
        #                                                      self.sampled_scenarios_definitions,
        #                                                      scenarios_per_tick=10,
        #                                                      timeout=self.timeout,
        #                                                      debug_mode=debug_mode > 1)
        # lp.print_stats()

        self.status = ["initialized"]
        self.scenario = None
        self.name = config.name
        self.config = config

        behavior = self._create_behavior(config.index)
        criteria = self._create_test_criteria()
        behavior_seq = py_trees.composites.Sequence()
        trigger_behavior = self._setup_scenario_trigger(config)
        if trigger_behavior:
            behavior_seq.add_child(trigger_behavior)
        if behavior is not None:
            behavior_seq.add_child(behavior)
            behavior_seq.name = behavior.name
        end_behavior = self._setup_scenario_end(config)
        if end_behavior:
            behavior_seq.add_child(end_behavior)
        self.scenario = Scenario(behavior_seq, criteria, self.name, self.timeout, self.terminate_on_failure)

        if not self.timeout:  # pylint: disable=access-member-before-definition
            self.timeout = 60  # If no timeout was provided, set it to 60 seconds
        # Add all the actors of the specific scenarios to self.other_actors

        print('len(self.other_actors', len(self.other_actors))
        if CarlaDataProvider.is_sync_mode():
            world.tick()
        else:
            world.wait_for_tick()
        return True

    def _update_route(self, st, ed, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """
        route_id = config.index

        if self.sampled_scenarios_definitions[route_id] is None:
            # Transform the scenario file into a dictionary
            world_annotations = RouteParser.parse_annotations_file(config.scenario_file)

            # prepare route's trajectory (interpolate and add the GPS route)
            # todo: change to resume start point
            gps_route, route = interpolate_trajectory(config.st, ed, world, config.trajectory)
            # gps_route, route = interpolate_trajectory(st, ed, world, config.trajectory)

            potential_scenarios_definitions, _, origin_possible_scenarios = RouteParser.scan_route_for_scenarios(
                config.town, route, world_annotations)
            self.gps_route_list[route_id] = gps_route
            self.route_list[route_id] = route
            self.sampled_scenarios_definitions[route_id] = self._scenario_sampling(potential_scenarios_definitions)
            # if self.origin_possible_scenarios is None:
            #     self.origin_possible_scenarios = origin_possible_scenarios
            # else:
            #     for scenario in origin_possible_scenarios:
            #         scenario_type = scenario['scenario_type']
            #         for i, origin_scenario in enumerate(self.origin_possible_scenarios):
            #             if scenario_type == origin_scenario['scenario_type']:
            #                 self.origin_possible_scenarios[i]['available_event_configurations'] += scenario[
            #                     'available_event_configurations']
        # else:
        #     import os
        #     import json
        #     file_name = "challenge_scenarios/filter_scenarios/dense_town01.json"
        #     origin_possible_scenarios = {"available_scenarios": [{config.town: self.origin_possible_scenarios}]}
        #     if not os.path.exists(file_name):
        #         with open(file_name, "a") as f:
        #             json.dump(origin_possible_scenarios, f)
        #
        #     else:
        #         with open(file_name, "w") as f:
        #             json.dump(origin_possible_scenarios, f)

        self.route = self.route_list[route_id]
        self.gps_route = self.gps_route_list[route_id]

        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        config.agent.set_global_plan(self.gps_route, self.route)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout()

    def _setup_scenario_trigger(self, config):
        """
        This function creates a trigger maneuver, that has to be finished before the real scenario starts.
        This implementation focuses on the first available ego vehicle.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        start_location = None
        if config.trigger_points and config.trigger_points[0]:
            start_location = config.trigger_points[0].location  # start location of the scenario

        ego_vehicle_route = CarlaDataProvider.get_ego_vehicle_route()

        if start_location:
            if ego_vehicle_route:
                if config.route_var_name is None:  # pylint: disable=no-else-return
                    return conditions.InTriggerDistanceToLocationAlongRoute(self.ego_vehicles[0],
                                                                            ego_vehicle_route,
                                                                            start_location,
                                                                            5)
                else:
                    check_name = "WaitForBlackboardVariable: {}".format(config.route_var_name)
                    return conditions.WaitForBlackboardVariable(name=check_name,
                                                                variable_name=config.route_var_name,
                                                                variable_value=True,
                                                                var_init_value=False)
            return conditions.InTimeToArrivalToLocation(self.ego_vehicles[0],
                                                        2.0,
                                                        start_location)

        return None

    def _setup_scenario_end(self, config):
        """
        This function adds and additional behavior to the scenario, which is triggered
        after it has ended.

        The function can be overloaded by a user implementation inside the user-defined scenario class.
        """
        ego_vehicle_route = CarlaDataProvider.get_ego_vehicle_route()
        if ego_vehicle_route:
            if config.route_var_name is not None:
                set_name = "Reset Blackboard Variable: {} ".format(config.route_var_name)
                return py_trees.blackboard.SetBlackboardVariable(name=set_name,
                                                                 variable_name=config.route_var_name,
                                                                 variable_value=False)
        return None

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]
        # elevate_transform.location.z += 0.5
        elevate_transform.location.z = 0.5

        if self.ego_vehicles is None or self.ego_vehicles[0] is None:
            ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017',
                                                              elevate_transform,
                                                              rolename='hero')
            if ego_vehicle is None:
                return ego_vehicle
            self.ego_vehicles = [ego_vehicle]
            # wait for the actor to be spawned properly before we do anything
        else:
            self.ego_vehicles[0].set_transform(elevate_transform)
            self.ego_vehicles[0].set_target_velocity(carla.Vector3D(0, 0, 0))
            if CarlaDataProvider.is_sync_mode():
                CarlaDataProvider._world.tick()
            else:
                CarlaDataProvider._world.wait_for_tick()

        if self.ego_vehicles[0] is not None:
            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

        return self.ego_vehicles[0]

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length + INITIAL_SECONDS_DELAY)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """
        # fix the random seed for reproducibility
        rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        def select_scenario(list_scenarios):
            # priority to the scenarios with higher number: 10 has priority over 9, etc.
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1

                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario

            return selected_scenario

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = select_scenario(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rgn.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)
        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []

        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'],
                                     scenario['trigger_position']['y'],
                                     scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                        color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

        for scenario_number, definition in enumerate(scenario_definitions):
            # Get the class possibilities for this scenario number
            scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]
            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                          ego_vehicle.get_transform(),
                                                                          'hero')]
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name
            # try:
            scenario_instance = None
            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration,
                                                   criteria_enable=False, timeout=timeout)
            except Exception as e:
                if definition['name'] == 'Scenario3':
                    del scenario_instance
                    scenario_instance = None
                    definition['name'] = 'Scenario2'
                    scenario_class = NUMBER_CLASS_TRANSLATION['Scenario2']
                    try:
                        scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration,
                                                           criteria_enable=False, timeout=timeout)
                    except Exception as e:
                        print('error in exception ', e)
                else:
                    print("error in generating scenario ", definition['name'])
            # Do a tick every once in a while to avoid spawning everything at the same time
            if scenario_number % scenarios_per_tick == 0:

                if CarlaDataProvider.is_sync_mode():
                    world.tick()
                else:
                    world.wait_for_tick()

                # except Exception as e:
                # print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue
            if scenario_instance:
                scenario_instance_vec.append(scenario_instance)
        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    # pylint: enable=no-self-use

    def _initialize_actors(self, town):
        """
        Set other_actors to the superset of all scenario actors
        """
        # Create the background activity of the route

        town_amount = {
            # 'Town01': 120,
            'Town01': 0,
            'Town02': 100,
            'Town03': 120,
            'Town04': 200,
            'Town05': 120,
            'Town06': 150,
            'Town07': 110,
            'Town08': 180,
            'Town09': 300,
            'Town10': 120,
        }

        # if self.vehicle_amount > 0:
        self.vehicle_amount = town_amount.get(town, 0)

        # amount = town_amount[config.town] if config.town in town_amount else 0
        new_cars = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                              self.vehicle_amount,
                                                              carla.Transform(),
                                                              autopilot=True,
                                                              random_location=True,
                                                              rolename='background')

        # new_walkers = CarlaDataProvider.request_new_batch_walkers('walker.pedestrian.*',
        #                                                         self.pedestrian_amount,
        #                                                         carla.Transform(),
        #                                                         autopilot=True,
        #                                                         random_location=True,
        #                                                         rolename='background')

        new_walkers = []
        new_actors = new_cars + new_walkers
        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)
        self.world_actors_num = len(self.other_actors)

    def _create_behavior(self, route_id):
        """
        Basic behavior do nothing, i.e. Idle
        """
        # todo: change scenario_trigger_distance
        # scenario_trigger_distance = 1.5  # Max trigger distance between route and scenario
        scenario_trigger_distance = 15  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        subbehavior = py_trees.composites.Parallel(name="Behavior",
                                                   policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        scenario_behaviors = []
        blackboard_list = []

        for i, scenario in enumerate(self.list_scenarios_list[route_id]):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name

                if route_var_name is not None:
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name,
                                            scenario.config.trigger_points[0].location])
                else:
                    name = "{} - {}".format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = oneshot_behavior(
                        name=name,
                        variable_name=name,
                        behaviour=scenario.scenario.behavior)
                    scenario_behaviors.append(oneshot_idiom)

        # Add behavior that manages the scenarios trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0],
            self.route,
            blackboard_list,
            scenario_trigger_distance,
            repeat_scenarios=False
        )

        subbehavior.add_child(scenario_triggerer)  # make ScenarioTriggerer the first thing to be checked
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        behavior.add_child(subbehavior)
        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria = []
        route = convert_transform_to_location(self.route)

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)

        route_criterion = InRouteTest(self.ego_vehicles[0],
                                      route=route,
                                      offroad_max=30,
                                      terminate_on_failure=True)

        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)

        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)

        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])

        stop_criterion = RunningStopTest(self.ego_vehicles[0])

        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0],
                                                         speed_threshold=0.1,
                                                         below_threshold_max_time=180.0,
                                                         terminate_on_failure=True,
                                                         name="AgentBlockedTest")

        criteria.append(completion_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(collision_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)
        criteria.append(route_criterion)
        criteria.append(blocked_criterion)

        return criteria

    def _remove_scenario_actors(self):
        actors_to_be_removed = len(self.other_actors) - self.world_actors_num
        for actor in self.other_actors[::-1]:
            if actor is not None:
                if CarlaDataProvider.actor_id_exists(actor.id):
                    CarlaDataProvider.remove_actor_by_id(actor.id)
                self.other_actors.remove(actor)
            actors_to_be_removed -= 1
            if actors_to_be_removed <= 0:
                break

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        del self.route_list
        del self.gps_route_list
        del self.list_scenarios_list
        self.remove_all_actors()

    def remove_all_actors(self):
        """
        Remove all actors
        """
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []
