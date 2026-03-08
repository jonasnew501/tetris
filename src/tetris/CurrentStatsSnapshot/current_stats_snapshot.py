from dataclasses import dataclass
from abc import ABC, abstractmethod


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
