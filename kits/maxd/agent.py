import sys

import numpy as np
from logger import logger
from lux.utils import direction_to

TH_MANHATTON_DISTANCE = 4
TH_MAXD_DISTANCE = 2
TH_RANDOM_DIRECTION = 18


def calc_distance_maxd(pos1, pos2):
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def find_nearist_relic_node(unit_pos, relic_node_positions):
    min_manhattan_distance = 999999
    max_manhattan_distance = 0
    nearest_relic_node_position = relic_node_positions[0]
    farthest_relic_node_position = relic_node_positions[0]

    # sort
    for relic_node_position in relic_node_positions:
        manhattan_distance = calc_distance_maxd(unit_pos, relic_node_position)
        if manhattan_distance < min_manhattan_distance:
            min_manhattan_distance = manhattan_distance
            nearest_relic_node_position = relic_node_position
        if manhattan_distance > max_manhattan_distance:
            max_manhattan_distance = manhattan_distance
            farthest_relic_node_position = relic_node_position
    return (
        min_manhattan_distance,
        nearest_relic_node_position,
        farthest_relic_node_position,
    )


# not exceed 5*5 realm
#  5 directions (center, up, right, down, left)
def get_safe_move(unit_pos, relic_pos):
    old_distance = calc_distance_maxd(unit_pos, relic_pos)

    for i in range(100):
        random_direction = np.random.randint(0, 5)
        if old_distance < TH_MAXD_DISTANCE:
            return [random_direction, 0, 0]

        new_pos = unit_pos.copy()
        if random_direction == 1:
            new_pos[1] -= 1
        elif random_direction == 2:
            new_pos[0] += 1
        elif random_direction == 3:
            new_pos[1] += 1
        elif random_direction == 4:
            new_pos[0] -= 1
        new_distance = calc_distance_maxd(new_pos, relic_pos)

        if new_distance <= TH_MAXD_DISTANCE:
            return [random_direction, 0, 0]
    return [0, 0, 0]


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        #
        self.score_gained = []
        self.score_gained_sum = []

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        # shape (max_units, )
        unit_mask = np.array(obs["units_mask"][self.team_id])
        # shape (max_units, 2)
        unit_positions = np.array(obs["units"]["position"][self.team_id])
        # shape (max_units, 1)
        unit_energys = np.array(obs["units"]["energy"][self.team_id])
        # shape (max_relic_nodes, 2)
        observed_relic_node_positions = np.array(obs["relic_nodes"])
        # shape (max_relic_nodes, )
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])

        # points of each team, team_points[self.team_id] is the points of the your team
        team_points = np.array(obs["team_points"])
        # new round
        if step == 0:
            self.score_gained = []
            self.score_gained_sum = []
        sum_points = team_points[self.team_id]
        if len(self.score_gained_sum) > 0:
            self.score_gained.append(sum_points - self.score_gained_sum[-1])
        else:
            self.score_gained.append(sum_points)
        self.score_gained_sum.append(sum_points)

        logger.info(
            f"test step={step} team_points={sum_points} new_points={self.score_gained[-1]} "
            + f" relic={self.relic_node_positions} "
        )

        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match

        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])

        # TODO 充分探索，绕开障碍（保存历史障碍信息），全局分配移动方向，避免重叠
        # TODO attack

        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]

            if len(self.relic_node_positions) > 0:
                # calculate nearest
                (
                    manhattan_distance,
                    nearest_relic_node_position,
                    farthest_relic_node_position,
                ) = find_nearist_relic_node(
                    unit_pos=unit_pos,
                    relic_node_positions=self.relic_node_positions,
                )

                # nearest_relic_node_position = self.relic_node_positions[0]
                # manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0] ) + abs(unit_pos[1] - nearest_relic_node_position[1])

                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= TH_MAXD_DISTANCE:
                    # save energy
                    if np.random.randint(0, 2) == 0:
                        actions[unit_id] = get_safe_move(
                            unit_pos, nearest_relic_node_position
                        )
                else:
                    # TODO detect blocked, and backward

                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [
                        direction_to(unit_pos, nearest_relic_node_position),
                        0,
                        0,
                    ]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if (
                    step % TH_RANDOM_DIRECTION == 0
                    or unit_id not in self.unit_explore_locations
                ):
                    rand_loc = (
                        np.random.randint(0, self.env_cfg["map_width"]),
                        np.random.randint(0, self.env_cfg["map_height"]),
                    )
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [
                    direction_to(unit_pos, self.unit_explore_locations[unit_id]),
                    0,
                    0,
                ]
        return actions
