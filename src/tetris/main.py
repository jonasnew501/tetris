import time
import sys
import pygame

from tetris.tetris_env import TetrisEnv


def main():
    # Initialize pygame and its display
    pygame.init()
    pygame.display.set_mode((1, 1), pygame.NOFRAME)  # Tiny invisible window
    pygame.display.iconify()  # Minimize immediately

    # Optional: Disable audio init warnings if not needed
    pygame.mixer.quit()

    env = TetrisEnv(field_height=18, field_width=10, len_tiles_queue=3)

    # ---
    drop_conducted = False

    # ---
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        env.handle_action(action=env.PossibleActions.move_left)
                        print("Moved left")
                        break
                    elif event.key == pygame.K_RIGHT:
                        env.handle_action(action=env.PossibleActions.move_right)
                        print("Moved right")
                        break
                    elif event.key == pygame.K_UP:
                        env.handle_action(action=env.PossibleActions.rotate)
                        print("Rotated")
                        break
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            # print("------------------------------")
            if not drop_conducted:
                env.launch_tile()

            if env.game_over:
                env.reset()
                env.launch_tile()

            # dropping the current tile by one row
            if drop_possible := env._drop_possible():
                env.drop_current_tile(drop_possible=drop_possible)
                drop_conducted = True
            else:
                drop_conducted = False

            print(env.field)
            print()

            time.sleep(0.5)

    except KeyboardInterrupt:
        pygame.quit()
        print("Exited gracefully.")


if __name__ == "__main__":
    main()
