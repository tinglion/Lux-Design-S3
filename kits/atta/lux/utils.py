import numpy as np
from logger import logger


def calc_distance_maxd(pos1, pos2):
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))


def calc_distance_manhattan(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0

    use_x = True
    #  use random, to avoid blocked
    if abs(dx) > abs(dy):
        # 25% use y
        # may block abs(dy) == 0 or
        use_x = np.random.randint(0, 4) != 0
    elif abs(dx) < abs(dy):
        # 25% use x
        # abs(dx) == 0 or
        use_x = np.random.randint(0, 4) == 0
    elif abs(dx) == abs(dy):
        # 50%
        use_x = np.random.randint(0, 2) == 0

    if use_x:
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
# src & dst: x,y
def direction_to_untrap(src, target, energy_map, tile_map):
    distance_src = calc_distance_manhattan(src, target)
    energy_src = energy_map[src[0], src[1]]
    WEIGHT_DISTANCE = 0.9
    WEIGHT_ENERGY = 0.1

    possible_step = {}
    for i in range(5):
        move = calc_target_position(src, i)
        if (
            move[0] < 0
            or move[1] < 0
            or move[0] >= energy_map.shape[0] - 1
            or move[1] >= energy_map.shape[1] - 1
        ):
            continue
        if tile_map[move[0], move[1]] == 2:
            continue
        distance_move = calc_distance_manhattan(move, target)
        energy_move = energy_map[move[0], move[1]]
        possible_step[i] = (
            0
            + (energy_src - energy_move) * WEIGHT_ENERGY
            + (distance_src - distance_move) * WEIGHT_DISTANCE
        )

    sorted_list = sorted(possible_step.items(), key=lambda item: item[1], reverse=True)
    logger.info(sorted_list)
    return sorted_list[0][0] if len(sorted_list) > 0 else 0


def filter_realm_points(center=[1, 1], points=[], rect_r=1):
    result = []
    for point in points:
        if abs(point[0] - center[0]) <= rect_r and abs(point[1] - center[1]) <= rect_r:
            result.append(point)
    return result


def calc_target_position(src, move):
    target = src
    if move == 1:
        target = src + np.array([0, -1])
    elif move == 2:
        target = src + np.array([1, 0])
    elif move == 3:
        target = src + np.array([0, 1])
    elif move == 4:
        target = src + np.array([-1, 0])
    return target
