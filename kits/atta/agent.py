import sys

import numpy as np
from logger import logger
from lux.utils import *

TH_MANHATTON_DISTANCE = 4
TH_MAXD_DISTANCE = 2
TH_RANDOM_DIRECTION = 24


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
# TODO move to more energy
def get_safe_move(unit_pos, relic_pos):
    old_distance = calc_distance_maxd(unit_pos, relic_pos)

    for i in range(1):
        random_direction = np.random.randint(0, 5)
        if old_distance < TH_MAXD_DISTANCE:
            return random_direction

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
            return random_direction
    return 0


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        logger.info(self.env_cfg)

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
        # logger.debug(obs)

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

        energy_map = np.array(obs["map_features"]["energy"])
        tile_map = np.array(obs["map_features"]["tile_type"])
        # logger.error(energy_map.shape)

        enemy_mask = np.array(obs["units_mask"][1 - self.team_id])
        enemy_positions = np.array(obs["units"]["position"][1 - self.team_id])
        available_enemy_ids = np.where(enemy_mask)[0]

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

        logger.debug(
            f"test step={step} team_points={sum_points} new_points={self.score_gained[-1]} "
            + f" relic={self.relic_node_positions} "
        )
        logger.debug(f"unit pos={unit_positions} mask={unit_mask}")
        logger.debug(f"enemy pos={enemy_positions} mask={enemy_mask}")

        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        scoring_positions = []

        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match

        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])

        # TODO 充分探索，绕开障碍（保存历史障碍信息），全局分配移动方向，避免重叠
        # TODO 移动避开能量陷阱
        # TODO 预测得分点，减少损失移动
        # TODO 预测地图移动
        # TODO 能量低的守着得分，能量高的去探索

        attack_cost = self.env_cfg.get("unit_sap_cost", 10) * self.env_cfg.get(
            "unit_sap_dropoff_factor", 0.5
        )
        attacked_enemy_ids = set()

        # at least 1 detector with max energy when availables>8
        max_unit_energy = 0
        max_unit_energy_id = -1
        if len(available_unit_ids) > 8:
            for i_unit, unit_id in enumerate(available_unit_ids):
                unit_energy = unit_energys[unit_id]
                if unit_energy > max_unit_energy:
                    max_unit_energy = unit_energy
                    max_unit_energy_id = unit_id

        # unit ids range from 0 to max_units - 1
        for i_unit, unit_id in enumerate(available_unit_ids):
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]

            # sap/attack/hit enemy: attack within <= params.unit_sap_range, energy>unit_sap_cost, only enemy in range
            # TODO 攻击需要预测对手移动可能
            take_attack = False
            for id_enemy in available_enemy_ids:
                if id_enemy in attacked_enemy_ids:
                    continue

                enemy_pos = enemy_positions[id_enemy]
                dist_unit_enemy = calc_distance_maxd(unit_pos, enemy_pos)

                friends = filter_realm_points(
                    center=enemy_pos,
                    rect_r=1,
                    points=unit_positions[available_unit_ids],
                )

                if (
                    dist_unit_enemy <= self.env_cfg.get("unit_sap_range", 4)
                    and unit_energys[unit_id] > attack_cost
                    # and len(friends) < 1
                ):
                    actions[unit_id] = [
                        5,
                        enemy_pos[0] - unit_pos[0],
                        enemy_pos[1] - unit_pos[1],
                    ]
                    take_attack = True
                    attacked_enemy_ids.add(id_enemy)
                    break
            if take_attack:
                continue

            if (
                len(self.relic_node_positions) > 0
                # and i_unit < len(available_unit_ids) * 0.70 # at least 30% random detect
                and unit_id != max_unit_energy_id
            ):
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
                # manhattan_distance = calc_distance_manhattan(unit_pos, nearest_relic_node_position)

                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= TH_MAXD_DISTANCE:
                    # save energy
                    if True or np.random.rand() <= 0.7:
                        to_move = get_safe_move(unit_pos, nearest_relic_node_position)
                        # 得分区域尽量分散
                        target_pos = tuple(calc_target_position(unit_pos, to_move))
                        if target_pos not in scoring_positions:
                            actions[unit_id] = [to_move, 0, 0]
                            scoring_positions.append(target_pos)
                else:
                    # TODO detect blocked, and refind way

                    # otherwise we want to move towards the relic node
                    to_move = direction_to_untrap(
                        unit_pos,
                        nearest_relic_node_position,
                        energy_map=energy_map,
                        tile_map=tile_map,
                    )
                    actions[unit_id] = [to_move, 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                # TODO optimize
                if (
                    step % TH_RANDOM_DIRECTION == 0
                    or unit_id not in self.unit_explore_locations
                ):
                    while True:
                        rand_loc = (
                            np.random.randint(0, self.env_cfg["map_width"]),
                            np.random.randint(0, self.env_cfg["map_height"]),
                        )
                        if (
                            calc_distance_manhattan(unit_pos, rand_loc)
                            > (self.env_cfg["map_width"] + self.env_cfg["map_height"])
                            / 4
                        ):
                            break
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [
                    direction_to_untrap(
                        unit_pos,
                        self.unit_explore_locations[unit_id],
                        energy_map=energy_map,
                        tile_map=tile_map,
                    ),
                    0,
                    0,
                ]
        return actions
