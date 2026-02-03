import pytest
from collections import deque
import numpy as np


from tetris.tetris_env import (
    TetrisEnv
)
from tetris.tetris_env_domain_specific_exceptions import (
    EmptyContainerError,
    NoneTypeError,
    WrongDatatypeError,
    OutOfBoundsError,
    GamewiseLogicalError,
    UnsupportedParameterValue
)



class TestTetrisEnv:
    
    # -----unittests for helper-functions-----------------------------------------------

    # ----------------------------------------------------------------------------------



    '''
    Central functions:
    - launch_tile
    - drop_current_tile
    - remove_full_rows
    - handle_action
    - move
    - rotate
    - reset
    '''

    # -----unittests for the happy-path-------------------------------------------------
    @staticmethod
    @pytest.fixture
    def env_setup_empty_field() -> TetrisEnv:
        env = TetrisEnv(field_height=7, field_width=10, len_tiles_queue=3)
        return env
    
    @staticmethod
    @pytest.mark.parametrize("tiles_queue, expected_field_after_put, current_tile_positionInField, game_over",
                             [
                                 (deque([["L", np.array([[1, 0], [1, 0], [1, 1]]), 0]]),
                                  np.array([[0,0,0,0,0,1,0,0,0,0],
                                            [0,0,0,0,0,1,0,0,0,0],
                                            [0,0,0,0,0,1,1,0,0,0],
                                            [0,0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0,0]]),
                                 [[0,0,1,1,2,2], [5,6,5,6,5,6]],
                                 False),
                             ],
                             )
    def test_lauch_tile_empty_field_happy_path(env_setup_empty_field: TetrisEnv, tiles_queue, expected_field_after_put, current_tile_positionInField, game_over):
        assert len(env_setup_empty_field.tiles_queue) == 3

        #overwriting the tiles_queue
        env_setup_empty_field.tiles_queue = tiles_queue

        assert len(env_setup_empty_field.tiles_queue) == 1

        env_setup_empty_field.launch_tile()

        #asserting field-correctness
        np.testing.assert_array_equal(env_setup_empty_field.field, expected_field_after_put)

        assert env_setup_empty_field.current_tile_positionInField == current_tile_positionInField

        assert env_setup_empty_field.game_over == game_over



        # self.tiles = {
        #     "I": np.ones((4, 1)),
        #     "O": np.ones((2, 2)),
        #     "S": np.array([[0, 1], [1, 1], [1, 0]]),
        #     "S_inv": np.array([[1, 0], [1, 1], [0, 1]]),
        #     "L": np.array([[1, 0], [1, 0], [1, 1]]),
        #     "L_inv": np.array([[0, 1], [0, 1], [1, 1]]),
        #     "T": np.array([[0, 1, 0], [1, 1, 1]]),
        # }



    # ----------------------------------------------------------------------------------


     # -----unittests for the unhappy-paths---------------------------------------------

     # ---------------------------------------------------------------------------------
