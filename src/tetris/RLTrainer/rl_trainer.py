
from tetris.ClassInterfaces.class_interfaces import EnvModule, StatisticsModule, Algorithm


class RLTrainer:
    """
    This class is the overarching class over all elements, which are part of the
    process of training an Reinforcement Learning agent to successfully act in
    an environment
    """
    def __init__(self, env_instance: Env,
                    statistics_instance: Statistics,
                    algorithm: Algorithm,
                    policy: Policy
                    buffer: ExperienceBuffer):
        