import time
import sys
import pygame

from TetrisEnv import TetrisEnv



def main():
    # Initialize pygame and its display
    pygame.init()
    pygame.display.set_mode((1, 1), pygame.NOFRAME)  # Tiny invisible window
    pygame.display.iconify()  # Minimize immediately

    # Optional: Disable audio init warnings if not needed
    pygame.mixer.quit() 

    env = TetrisEnv(field_height=18, field_width=10, len_tiles_queue=3)

    #---
    drop_conducted = False

    #---
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        env.current_action = 1
                        env.handle_action(action=1)
                        print("Moved left")
                        break
                    elif event.key == pygame.K_RIGHT:
                        env.current_action = 2
                        env.handle_action(action=2)
                        print("Moved right")
                        break
                    elif event.key == pygame.K_UP:
                        env.current_action = 3
                        env.handle_action(action=3)
                        print("Rotated")
                        break
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            # print("------------------------------")
            if not drop_conducted:
                env.launch_tile()
            
            #TODO: Remove again
            # if not launched:
            #     env.launch_tile()
            #     for _ in range (3):
            #         _ = env.drop()
            #     launched = not launched

            # dropping the current tile by one row
            drop_conducted = env.drop()

            
            print(env.field)
            # print(f"current_tile: {env.current_tile}")
            # print(f"current_tile_positionInField: {env.current_tile_positionInField}")
            # print(f"current_action: {env.current_action}")
            print()
            



            time.sleep(0.5)
    
    except KeyboardInterrupt:
        pygame.quit()
        print("Exited gracefully.")






if __name__ == '__main__':
    main()