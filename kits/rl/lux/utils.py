import os
import re


def find_max_in_filename(folder_path, pattern_str=r".*_(\d+)\..*"):
    max_number = float("-inf")
    max_file = None
    # 使用正则表达式提取数字
    pattern = re.compile(pattern_str)
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        # print(match)
        if match:
            num = int(match.group(1))
            if num > max_number:
                max_number = num
                max_file = filename
    return max_number, max_file


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1
