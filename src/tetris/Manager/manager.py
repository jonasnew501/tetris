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

    def __init__(self, env_module: EnvModule, statistics_module: StatisticsModule):
        self.env = env_module
        self.statistics = statistics_module

        # Initialize pygame and its display
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)  # Tiny invisible window
        # pygame.display.iconify()  # Minimize immediately

        # Optional: Disable audio init warnings if not needed
        pygame.mixer.quit()


    def play(self):
        # action = 
        pass

    # get snapshot

    # push snapshot to 'statistics' to update the statistics

    # save visualizations of statistics

    # implement full game-loop

    #

    def get_human_action(self, seconds_to_select_action: float) -> TetrisEnv.PossibleActions:
        start_time = time.perf_counter()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        return TetrisEnv.PossibleActions.move_left
                    elif event.key == pygame.K_RIGHT:
                        return TetrisEnv.PossibleActions.move_right
                    elif event.key == pygame.K_UP:
                        return TetrisEnv.PossibleActions.rotate
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
            
            now = time.perf_counter()

            if now - start_time > seconds_to_select_action:
                break

