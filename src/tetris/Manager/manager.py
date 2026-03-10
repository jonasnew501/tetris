import sys
import pygame
import time

from tetris.TetrisEnv.tetris_env import EnvModule, TetrisEnv
from tetris.Statistics.statistics import StatisticsModule


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

    def _get_human_action(self, seconds_to_select_action: float):
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
                return action_selected

            elif (now - start_time) > seconds_to_select_action:
                return self.env.PossibleActions.do_nothing
