import sys
import pygame
import time

from tetris.TetrisEnv.tetris_env import EnvModule, TetrisEnv
from tetris.Statistics.statistics import StatisticsModule

from tetris.TetrisEnv.tetris_env_domain_specific_exceptions import GamewiseLogicalError


class Manager:
    """
    This class holds instances of various high-level modules of this project
    and manages the overall execution-flow between these modules
    """

    def __init__(self, env_instance: EnvModule, statistics_instance: StatisticsModule):
        self.env = env_instance
        self.statistics = statistics_instance

        # Initialize pygame and its display
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)  # Tiny invisible window
        # pygame.display.iconify()  # Minimize immediately

        # Optional: Disable audio init warnings if not needed
        pygame.mixer.quit()

    def play(self):
        self.env.launch_tile()
        while True:
            self.env.render()

            action = self._get_human_action(seconds_to_select_action=1)

            obs, reward, done = self.env.step(action)

            current_stats_snapshot = self.env.take_snapshot()

            self.statistics.update(data_snapshot=current_stats_snapshot)

            # save visualizations of statistics

    def _get_human_action(self, seconds_to_select_action: float) -> tuple[TetrisEnv.PossibleActions, float]:
        start_time = time.perf_counter()
        action_selected = None

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action_selected = self.env.PossibleActions.move_left
                        break
                    elif event.key == pygame.K_RIGHT:
                        action_selected = self.env.PossibleActions.move_right
                        break
                    elif event.key == pygame.K_UP:
                        action_selected = self.env.PossibleActions.rotate
                        break
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            
            now = time.perf_counter()

            if action_selected is not None:
                return (action_selected, self._calculate_remaining_waiting_time(seconds_to_select_action=seconds_to_select_action, start_time=start_time, now=now))

            elif (now - start_time) > seconds_to_select_action:
                return (self.env.PossibleActions.do_nothing, 0)
    
    def _calculate_remaining_waiting_time(self, seconds_to_select_action: float, start_time: float, now: float):
        """
        This helper-function simply calculates and returns the difference
        between the seconds the user had to select an action (see function '_get_human_action')
        and the time it took them to actually select an action
        """
        action_selection_duration = now - start_time
        if action_selection_duration > seconds_to_select_action:
            raise GamewiseLogicalError("In this function at hand, the time it took a player to select an action must logically be smaller than the time allowed to take an action. However, this was not the case here!")
        
        return seconds_to_select_action - action_selection_duration
