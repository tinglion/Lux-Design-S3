import numpy as np


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
        use_x = abs(dy) == 0 or np.random.randint(0, 4) != 0
    elif abs(dx) < abs(dy):
        # 25% use x
        use_x = abs(dx) == 0 or np.random.randint(0, 4) == 0
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
