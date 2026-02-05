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

    @staticmethod
    @pytest.fixture
    def env_setup_empty_field() -> TetrisEnv:
        env = TetrisEnv(field_height=7, field_width=10, len_tiles_queue=3)
        return env

    @staticmethod
    @pytest.fixture
    def env_setup_occupied_field() -> TetrisEnv:
        env = TetrisEnv(field_height=7, field_width=10, len_tiles_queue=3)
        env.field = np.array([[0,0,0,0,0,0,0,0,0,0],
                              [0,0,1,1,0,0,0,0,0,0],
                              [0,0,1,1,1,0,1,0,0,0],
                              [0,0,1,1,1,0,1,0,0,0],
                              [0,0,0,1,1,1,1,0,0,0],
                              [0,0,0,0,1,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0]], dtype=np.int8)
        return env

    # -----unittests for the happy-path-------------------------------------------------
    
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
                                            [0,0,0,0,0,0,0,0,0,0]], dtype=np.int8),
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
    

    @staticmethod
    @pytest.mark.parametrize("tiles_queue, expected_field_after_put, current_tile_positionInField, game_over",
                             [
                                 (deque([["I", np.ones((4, 1)), 0]]),
                                  np.array([[0,0,0,0,0,1,0,0,0,0],
                                            [0,0,1,1,0,1,0,0,0,0],
                                            [0,0,1,1,1,1,1,0,0,0],
                                            [0,0,1,1,1,1,1,0,0,0],
                                            [0,0,0,1,1,1,1,0,0,0],
                                            [0,0,0,0,1,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0,0]], dtype=np.int8),
                                 [[0,1,2,3], [5,5,5,5]],
                                 False),
                             ],
                             )
    def test_launch_tile_occupied_field_happy_path(env_setup_occupied_field: TetrisEnv, tiles_queue, expected_field_after_put, current_tile_positionInField, game_over):
        assert len(env_setup_occupied_field.tiles_queue) == 3

        #overwriting the tiles_queue
        env_setup_occupied_field.tiles_queue = tiles_queue

        assert len(env_setup_occupied_field.tiles_queue) == 1

        env_setup_occupied_field.launch_tile()

        #asserting field-correctness
        np.testing.assert_array_equal(env_setup_occupied_field.field, expected_field_after_put)

        assert env_setup_occupied_field.current_tile_positionInField == current_tile_positionInField

        assert env_setup_occupied_field.game_over == game_over


    # ----------------------------------------------------------------------------------


    # -----unittests for the unhappy-paths---------------------------------------------
    
    @staticmethod
    @pytest.mark.parametrize("tiles_queue, expected_field, game_over",
                             [
                                 (deque([["I", np.ones((4, 1)), 0]]),
                                  np.array([[0,0,0,0,0,0,0,0,0,0],
                                            [0,0,1,1,0,0,0,0,0,0],
                                            [0,0,1,1,1,0,1,0,0,0],
                                            [0,0,1,1,1,0,1,0,0,0],
                                            [0,0,0,1,1,1,1,0,0,0],
                                            [0,0,0,0,1,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0,0]], dtype=np.int8),
                                 True),
                             ],
                             )
    def test_launch_tile_occupied_field_unhappy_path_overlap(env_setup_occupied_field: TetrisEnv, tiles_queue, expected_field, game_over):
        env_setup_occupied_field.launch_position = [0, 4]

        assert len(env_setup_occupied_field.tiles_queue) == 3

        #overwriting the tiles_queue
        env_setup_occupied_field.tiles_queue = tiles_queue

        assert len(env_setup_occupied_field.tiles_queue) == 1

        env_setup_occupied_field.launch_tile()

        #The field is expected to be exactly the same as the initial field of 'env_setup_occupied_field'
        #set in the fixture, because a put must not happen when an overlap is detected.
        #asserting field-correctness
        np.testing.assert_array_equal(env_setup_occupied_field.field, expected_field)

        assert env_setup_occupied_field.game_over == game_over


    @staticmethod
    @pytest.mark.parametrize("tiles_queue, exception",
                             [
                                 (deque([["I", np.ones((4, 1)), 0]]),
                                 OutOfBoundsError),
                             ],
                             )
    def test_launch_tile_occupied_field_unhappy_path_out_of_bounds(env_setup_occupied_field: TetrisEnv, tiles_queue, exception):
        env_setup_occupied_field.launch_position = [0, env_setup_occupied_field.field_width]

        assert len(env_setup_occupied_field.tiles_queue) == 3

        #overwriting the tiles_queue
        env_setup_occupied_field.tiles_queue = tiles_queue

        assert len(env_setup_occupied_field.tiles_queue) == 1

        with pytest.raises(expected_exception=exception):
            env_setup_occupied_field.launch_tile()





     # ---------------------------------------------------------------------------------
