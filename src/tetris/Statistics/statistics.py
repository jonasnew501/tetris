import numpy as np
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from typing import Callable

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
        #TODO: Can be removed
        # self.means = {
        #     "n_games_played_last_50_snapshots": deque(maxlen=5000),
        #     "n_rows_cleared_last_50_snapshots": deque(maxlen=5000),
        #     "number_of_rows_cleared_at_once_last_50_snapshots": deque(maxlen=5000),
        # }

        # self.std_devs = {
        #     "n_games_played_last_50_snapshots": deque(maxlen=5000),
        #     "n_rows_cleared_last_50_snapshots": deque(maxlen=5000),
        #     "number_of_rows_cleared_at_once_last_50_snapshots": deque(maxlen=5000),
        # }
        self.means = defaultdict(deque(maxlen=5000))
        self.std_devs = defaultdict(deque(maxlen=5000))
        ###

    def _update_data(self, data_snapshot: CurrentStatsSnapshot):
        self.n_games_played.append(data_snapshot.n_games_played)
        self.n_timesteps_conducted_latest = data_snapshot.n_timesteps_conducted
        self.n_rows_cleared.append(data_snapshot.n_rows_cleared)
        self.number_of_rows_cleared_at_once.append(
            data_snapshot.number_of_rows_cleared_at_once
        )

    def _update_statistics(self):
        self._update_means()
        self._update_std_devs()

    #TODO: Just write one single, generic method for calculating the statistics and pass the method to use (np.mean or np.std) and the dict to work on in,
    #      instead of writing two separate functions, which produce code-repetition.
    def _update_means(self):
        self.means["n_games_played_last_50_snapshots"].append(
            np.mean(list(self.n_games_played)[-50:])
        )
        self.means["n_rows_cleared_last_50_snapshots"].append(
            np.mean(list(self.n_rows_cleared)[-50:])
        )
        self.means["number_of_rows_cleared_at_once_last_50_snapshots"].append(
            {
                key: np.mean(
                    [d[key] for d in list(self.number_of_rows_cleared_at_once)[-50:]],
                    dtype=np.float32,
                )
                for key in self.number_of_rows_cleared_at_once[0]
            }
        )
    
    def _update_std_devs(self):
        self.std_devs["n_games_played_last_50_snapshots"].append(
            np.std(list(self.n_games_played)[-50:])
        )
        self.std_devs["n_rows_cleared_last_50_snapshots"].append(
            np.std(list(self.n_rows_cleared)[-50:])
        )
        self.std_devs["number_of_rows_cleared_at_once_last_50_snapshots"].append(
            {
                key: np.std(
                    [d[key] for d in list(self.number_of_rows_cleared_at_once)[-50:]],
                    dtype=np.float32,
                )
                for key in self.number_of_rows_cleared_at_once[0]
            }
        )
    

    def _update_descriptive_statistic(self, target_dict: defaultdict, aggregation_function: Callable, period_to_aggregate: int):
        """
        This is a generic helper-function to calculate the 'aggregation_function'
        passed for every key of specific 'target_dict's.

        It is important to be aware that this function is designed to only work
        for target_dicts, which are meant to contain descriptive statistics.

        The target_dict needs to be a defaultdict with either a deque(with maxlen-parameter set)
        as default value or with another container (such as a list).
        However, due to performance using a deque with maxlen-parameter as a default-value
        is highly adivsed.

        The keys created (resp. updated) by this function are:
        "
            "n_games_played_last_{period_to_aggregate}_snapshots",
            "n_rows_cleared_last_{period_to_aggregate}_snapshots"
            "number_of_rows_cleared_at_once_last_{period_to_aggregate}_snapshots"
        "

        """
        target_dict[f"n_games_played_last_{period_to_aggregate}_snapshots"].append(
            aggregation_function(list(self.n_games_played)[-period_to_aggregate:])
        )
        target_dict[f"n_rows_cleared_last_{period_to_aggregate}_snapshots"].append(
            aggregation_function(list(self.n_rows_cleared)[-period_to_aggregate:])
        )
        target_dict[f"number_of_rows_cleared_at_once_last_{period_to_aggregate}_snapshots"].append(
            {
                key: aggregation_function(
                    [d[key] for d in list(self.number_of_rows_cleared_at_once)[-period_to_aggregate:]],
                    dtype=np.float32,
                )
                for key in self.number_of_rows_cleared_at_once[0]
            }
        )
