import logging
import os


# 创建一个函数来获取当前文件的相对路径
def get_relative_path(file_path):
    # 假设你想要相对于当前工作目录的路径
    return os.path.relpath(file_path, os.getcwd())


# 由于Formatter没有内置的relative_path选项，我们需要通过一个小技巧来添加它
# 我们可以使用Filter来实现这一点
class RelativePathFilter(logging.Filter):
    def filter(self, record):
        record.relative_path = get_relative_path(record.pathname)
        return True


# 创建日志记录器
logger = logging.getLogger(__name__)
logger.handlers = []

# 设置日志级别为DEBUG，这样所有级别的日志都会被处理
logger.setLevel(logging.DEBUG)

# 添加过滤器到日志记录器
logger.addFilter(RelativePathFilter())

# 创建一个流处理器（console handler）并设置级别为DEBUG
# ch = logging.StreamHandler()
ch = logging.FileHandler('op.log')
ch.setLevel(logging.DEBUG)

# 创建一个日志格式器并添加到处理器
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(relative_path)s %(funcName)s:%(lineno)d - %(message)s"
)
ch.setFormatter(formatter)

# 把处理器添加到日志记录器
logger.addHandler(ch)

# 记录一条日志
# logger.debug(12)
