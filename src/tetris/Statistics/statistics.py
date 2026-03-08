import numpy as np
from abc import ABC, abstractmethod
from collections import deque

from src.tetris.CurrentStatsSnapshot.current_stats_snapshot import CurrentStatsSnapshot


class StatisticsModule(ABC):
    def update(self):
        self._update_data()
        self._update_statistics()

    @abstractmethod
    def _update_statistics(self):
        raise NotImplementedError
    
    @abstractmethod
    def _update_data(self, data_snapshot: CurrentStatsSnapshot):
        raise NotImplementedError


class Statistics(StatisticsModule):
    """
    This class is responsible for holding and calculating "long-term" statistics,
    i.e. statistics which span over multiple snapshots of current stats.
    """
    def __init__(self):
        ### data-storage ###
        self.n_games_played = deque(maxlen=5000)
        self.n_timesteps_conducted_latest: int = 0
        self.n_rows_cleared = deque(maxlen=5000)
        self.number_of_rows_cleared_at_once = deque(maxlen=5000)
        ###

        ### statistics ###
        self.means = {
            "n_games_played_last_50_snapshots": deque(maxlen=5000),
            "n_rows_cleared_last_50_snapshots": deque(maxlen=5000),
            "number_of_rows_cleared_at_once_last_50_snapshots": deque(maxlen=5000)
        }

        self.std_devs = {
            "n_games_played_last_50_snapshots": deque(maxlen=5000),
            "n_rows_cleared_last_50_snapshots": deque(maxlen=5000),
            "number_of_rows_cleared_at_once_last_50_snapshots": deque(maxlen=5000)
        }
        ###

    

    def _update_data(self, data_snapshot: CurrentStatsSnapshot):
        self.n_games_played.append(data_snapshot.n_games_played)
        self.n_timesteps_conducted_latest = data_snapshot.n_timesteps_conducted
        self.n_rows_cleared.append(data_snapshot.n_rows_cleared)
        self.number_of_rows_cleared_at_once.append(data_snapshot.number_of_rows_cleared_at_once)
    
    def _update_statistics(self):
        pass
    
    def _update_means(self):
        self.means["n_games_played_last_50_snapshots"].append(np.mean(list(self.n_games_played)[-50]))
        