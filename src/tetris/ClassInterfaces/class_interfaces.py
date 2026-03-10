"""
This file comprises all class-interfaces resp. abstract base classes of this project.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch

from tetris.ClassInterfaces.class_interfaces import CurrentStatsSnapshot


class EnvModule(ABC):
    @abstractmethod
    def take_snapshot(self) -> CurrentStatsSnapshot:
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError


class StatisticsModule(ABC):
    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def _update_statistics(self):
        raise NotImplementedError

    @abstractmethod
    def _update_data(self, data_snapshot: CurrentStatsSnapshot):
        raise NotImplementedError


class PolicyModule(ABC):
    @abstractmethod
    def forward(state: torch.Tensor):
        raise NotImplementedError


class AlgorithmModule(ABC):
    @abstractmethod
    def train(policy: PolicyModule):
        raise NotImplementedError


class ExperienceBuffer(ABC):
    pass


@dataclass(frozen=True)
class CurrentStatsSnapshot:
    """
    This class is an abstraction.
    It defines, which data needs to be contained in
    a snapshot of current (i.e. "online") gameplay data.

    It is set as immutable (due to 'frozen=True'), which means
    that after an instance of this dataclass is created,
    its data held cannot be changed anymore, only read.
    """

    n_games_played: int
    n_timesteps_conducted: int
    n_rows_cleared: int
    number_of_rows_cleared_at_once: dict
