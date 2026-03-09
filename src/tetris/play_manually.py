from tetris.Manager.manager import Manager
from tetris.TetrisEnv.tetris_env import TetrisEnv
from tetris.Statistics.statistics import Statistics


def run(env_class, statistics_class, manager_class, **kwargs):
    env = env_class(**kwargs)
    statistics = statistics_class()
    manager = manager_class(env_instance=env, statistics_instance=statistics)

    manager.play()


if __name__ == "__main__":
    run(
        env_class=TetrisEnv,
        statistics_class=Statistics,
        manager_class=Manager,
        field_height=18,
        field_width=10,
        len_tiles_queue=3,
    )
