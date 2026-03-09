from tetris.TetrisEnv.tetris_env import EnvModule
from tetris.Statistics.statistics import StatisticsModule


class Manager:
    """
    This class holds instances of various high-level modules of this project
    and manages the overall execution-flow between these modules
    """

    def __init__(self, env_module: EnvModule, statistics_module: StatisticsModule):
        self.env = env_module
        self.statistics = statistics_module

    # get snapshot

    # push snapshot to 'statistics' to update the statistics

    # save visualizations of statistics

    # implement full game-loop

    #
