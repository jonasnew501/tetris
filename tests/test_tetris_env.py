import pytest
from collections import deque
import numpy as np
from typing import List, Tuple


from tetris.tetris_env import TetrisEnv
from tetris.tetris_env_domain_specific_exceptions import (
    EmptyContainerError,
    NoneTypeError,
    WrongDatatypeError,
    OutOfBoundsError,
    GamewiseLogicalError,
    UnsupportedParameterValue,
)


"""
    Central functions:
    - launch_tile --> finished (potentially add some test-function for corner-cases)
    - drop_current_tile
    - remove_full_rows
    - handle_action
    - move
    - rotate
    - reset

    Helper functions:
    - drop_possible --> finished
"""


class TestTetrisEnv:

    # -----fixtures---------------------------------------------------------------------
    @staticmethod
    @pytest.fixture
    def env_setup_empty_field() -> TetrisEnv:
        env = TetrisEnv(field_height=7, field_width=10, len_tiles_queue=3)
        return env

    @staticmethod
    @pytest.fixture
    def env_setup_occupied_field() -> TetrisEnv:
        env = TetrisEnv(field_height=7, field_width=10, len_tiles_queue=3)
        env.field = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        return env

    # ----------------------------------------------------------------------------------

    # -----unittests for helper-functions-----------------------------------------------
    @staticmethod
    @pytest.mark.parametrize(
        "tiles_queue, field, current_tile_positionInField, expected_return_value",
        [
            (
                deque([["L", np.array([[1, 0], [1, 0], [1, 1]]), 0]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                        [1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                ),
                [[4, 4, 5, 5, 6, 6], [0, 1, 0, 1, 0, 1]],
                False,
            ),
            (
                deque([["L", np.array([[1, 0], [1, 0], [1, 1]]), 0]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                ),
                [[1, 1, 2, 2, 3, 3], [7, 8, 7, 8, 7, 8]],
                True,
            ),
            (
                deque([["L", np.array([[1, 1], [0, 1], [0, 1]]), 2]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    ],
                    dtype=np.int8,
                ),
                [[3, 3, 4, 4, 5, 5], [7, 8, 7, 8, 7, 8]],
                True,
            ),
            (
                deque([["L", np.array([[1, 1], [0, 1], [0, 1]]), 2]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                    ],
                    dtype=np.int8,
                ),
                [[3, 3, 4, 4, 5, 5], [7, 8, 7, 8, 7, 8]],
                False,
            ),
            (
                deque([["L", np.array([[1, 1], [0, 1], [0, 1]]), 2]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                    ],
                    dtype=np.int8,
                ),
                [[3, 3, 4, 4, 5, 5], [7, 8, 7, 8, 7, 8]],
                False,
            ),
            (
                deque([["I", np.ones((4, 1)), 0]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                    ],
                    dtype=np.int8,
                ),
                [[0, 1, 2, 3], [9, 9, 9, 9]],
                True,
            ),
        ],
    )
    def test__drop_possible_empty_field_happy_path(
        env_setup_empty_field: TetrisEnv,
        tiles_queue: deque,
        field: np.ndarray | None,
        current_tile_positionInField: list[list[int], list[int]],
        expected_return_value: bool,
    ):
        if field is not None:
            env_setup_empty_field.field = field

        env_setup_empty_field.tiles_queue = tiles_queue
        env_setup_empty_field.current_tile = tiles_queue.popleft()

        env_setup_empty_field.current_tile_positionInField = (
            current_tile_positionInField
        )

        env_setup_empty_field.top_left_corner_current_tile_in_field = (
            env_setup_empty_field._get_top_left_corner_of_current_tile_in_field()
        )
        env_setup_empty_field.current_tile_occupied_cells_in_field = (
            env_setup_empty_field._get_current_tile_occupied_cells_in_field()
        )

        assert env_setup_empty_field._drop_possible() == expected_return_value

    # -----unittests for the happy-path-------------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "tiles_queue, expected_field_after_put, current_tile_positionInField, top_left_corner_current_tile_in_field, current_tile_occupied_cells_in_field, game_over",
        [
            (
                deque([["L", np.array([[1, 0], [1, 0], [1, 1]]), 0]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                ),
                [[0, 0, 1, 1, 2, 2], [5, 6, 5, 6, 5, 6]],
                (0, 5),
                [[0, 1, 2, 2], [5, 5, 5, 6]],
                False,
            ),
        ],
    )
    def test_launch_tile_empty_field_happy_path(
        env_setup_empty_field: TetrisEnv,
        tiles_queue: deque,
        expected_field_after_put,
        current_tile_positionInField,
        top_left_corner_current_tile_in_field,
        current_tile_occupied_cells_in_field,
        game_over,
    ):
        assert len(env_setup_empty_field.tiles_queue) == 3

        # overwriting the tiles_queue
        env_setup_empty_field.tiles_queue = tiles_queue

        assert len(env_setup_empty_field.tiles_queue) == 1

        env_setup_empty_field.launch_tile()

        # asserting field-correctness
        np.testing.assert_array_equal(
            env_setup_empty_field.field, expected_field_after_put
        )

        assert (
            env_setup_empty_field.current_tile_positionInField
            == current_tile_positionInField
        )
        assert (
            env_setup_empty_field.top_left_corner_current_tile_in_field
            == top_left_corner_current_tile_in_field
        )
        assert (
            env_setup_empty_field.current_tile_occupied_cells_in_field
            == current_tile_occupied_cells_in_field
        )

        assert env_setup_empty_field.game_over == game_over

    @staticmethod
    @pytest.mark.parametrize(
        "tiles_queue, expected_field_after_put, current_tile_positionInField, top_left_corner_current_tile_in_field, current_tile_occupied_cells_in_field, game_over",
        [
            (
                deque([["I", np.ones((4, 1)), 0]]),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                ),
                [[0, 1, 2, 3], [5, 5, 5, 5]],
                (0, 5),
                [[0, 1, 2, 3], [5, 5, 5, 5]],
                False,
            ),
        ],
    )
    def test_launch_tile_occupied_field_happy_path(
        env_setup_occupied_field: TetrisEnv,
        tiles_queue: deque,
        expected_field_after_put,
        current_tile_positionInField,
        top_left_corner_current_tile_in_field,
        current_tile_occupied_cells_in_field,
        game_over,
    ):
        assert len(env_setup_occupied_field.tiles_queue) == 3

        # overwriting the tiles_queue
        env_setup_occupied_field.tiles_queue = tiles_queue

        assert len(env_setup_occupied_field.tiles_queue) == 1

        env_setup_occupied_field.launch_tile()

        # asserting field-correctness
        np.testing.assert_array_equal(
            env_setup_occupied_field.field, expected_field_after_put
        )

        assert (
            env_setup_occupied_field.current_tile_positionInField
            == current_tile_positionInField
        )
        assert (
            env_setup_occupied_field.top_left_corner_current_tile_in_field
            == top_left_corner_current_tile_in_field
        )
        assert (
            env_setup_occupied_field.current_tile_occupied_cells_in_field
            == current_tile_occupied_cells_in_field
        )

        assert env_setup_occupied_field.game_over == game_over

    @staticmethod
    @pytest.mark.parametrize(
        "tiles_queue, launch_position, expected_field, game_over",
        [
            (
                deque([["I", np.ones((4, 1)), 0]]),
                [0, 4],
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                ),
                True,
            ),
        ],
    )
    def test_launch_tile_occupied_field_happy_path_overlap(
        env_setup_occupied_field: TetrisEnv,
        tiles_queue: deque,
        launch_position: List,
        expected_field,
        game_over,
    ):
        env_setup_occupied_field.launch_position = launch_position

        assert len(env_setup_occupied_field.tiles_queue) == 3

        # overwriting the tiles_queue
        env_setup_occupied_field.tiles_queue = tiles_queue

        assert len(env_setup_occupied_field.tiles_queue) == 1

        env_setup_occupied_field.launch_tile()

        # The field is expected to be exactly the same as the initial field of 'env_setup_occupied_field'
        # set in the fixture, because a put must not happen when an overlap is detected.
        # asserting field-correctness
        np.testing.assert_array_equal(env_setup_occupied_field.field, expected_field)

        assert env_setup_occupied_field.current_tile_positionInField == [[], []]
        assert env_setup_occupied_field.top_left_corner_current_tile_in_field == ()
        assert env_setup_occupied_field.current_tile_occupied_cells_in_field == [[], []]

        assert env_setup_occupied_field.game_over == game_over

    @staticmethod
    @pytest.mark.parametrize(
        "tiles_queue, launch_position, current_tile_positionInField_after_drop, current_tile_occupied_cells_in_field_after_drop, top_left_corner_current_tile_in_field_after_drop, expected_field_after_drop",
        [
            (
                deque([["L", np.array([[1, 0], [1, 0], [1, 1]]), 0]]),
                [0, 7],
                [[1, 1, 2, 2, 3, 3], [7, 8, 7, 8, 7, 8]],
                [[1, 2, 3, 3], [7, 7, 7, 8]],
                (1, 7),
                np.array(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    dtype=np.int8,
                ),
            )
        ],
    )
    def test_drop_current_tile_drop_in_empty_place(
        env_setup_occupied_field: TetrisEnv,
        tiles_queue: deque,
        launch_position: List[int],
        current_tile_positionInField_after_drop: List[List[int]],
        current_tile_occupied_cells_in_field_after_drop: List[List[int]],
        top_left_corner_current_tile_in_field_after_drop: Tuple[int, int],
        expected_field_after_drop: np.ndarray,
    ):
        env_setup_occupied_field.tiles_queue = tiles_queue
        env_setup_occupied_field.launch_position = launch_position

        env_setup_occupied_field.launch_tile()
        env_setup_occupied_field.drop_current_tile()

        assert isinstance(
            env_setup_occupied_field.current_tile_positionInField[0], list
        )

        assert (
            env_setup_occupied_field.current_tile_positionInField
            == current_tile_positionInField_after_drop
        )
        assert (
            env_setup_occupied_field.current_tile_occupied_cells_in_field
            == current_tile_occupied_cells_in_field_after_drop
        )
        assert (
            env_setup_occupied_field.top_left_corner_current_tile_in_field
            == top_left_corner_current_tile_in_field_after_drop
        )
        np.testing.assert_array_equal(
            env_setup_occupied_field.field, expected_field_after_drop
        )

    # ----------------------------------------------------------------------------------

    # -----unittests for the unhappy-paths---------------------------------------------

    @staticmethod
    @pytest.mark.parametrize(
        "tiles_queue, exception",
        [
            (deque([["I", np.ones((4, 1)), 0]]), OutOfBoundsError),
        ],
    )
    def test_launch_tile_occupied_field_unhappy_path_out_of_bounds(
        env_setup_occupied_field: TetrisEnv, tiles_queue: deque, exception
    ):
        env_setup_occupied_field.launch_position = [
            0,
            env_setup_occupied_field.field_width,
        ]

        assert len(env_setup_occupied_field.tiles_queue) == 3

        # overwriting the tiles_queue
        env_setup_occupied_field.tiles_queue = tiles_queue

        assert len(env_setup_occupied_field.tiles_queue) == 1

        assert env_setup_occupied_field.current_tile_positionInField == [[], []]
        assert env_setup_occupied_field.top_left_corner_current_tile_in_field == ()
        assert env_setup_occupied_field.current_tile_occupied_cells_in_field == [[], []]

        with pytest.raises(expected_exception=exception):
            env_setup_occupied_field.launch_tile()

    @staticmethod
    @pytest.mark.parametrize(
        "tiles_queue, current_tile_positionInField, top_left_corner_current_tile_in_field, current_tile_occupied_cells_in_field, exception, match",
        [
            (
                deque([["O", np.ones((2, 2)), 0]]),
                [[0, 0, 1, 1], [5, 6, 5, 6]],
                (0, 5),
                [[0, 0, 1, 1], [5, 6, 5, 6]],
                GamewiseLogicalError,
                "'drop_possible' must always be True at this point.",
            ),
        ],
    )
    def test_drop_current_tile_drop_not_possible(
        env_setup_occupied_field: TetrisEnv,
        tiles_queue: deque,
        current_tile_positionInField,
        top_left_corner_current_tile_in_field,
        current_tile_occupied_cells_in_field,
        exception: Exception,
        match: str,
    ):
        assert len(env_setup_occupied_field.tiles_queue) == 3
        env_setup_occupied_field.tiles_queue = tiles_queue
        assert len(env_setup_occupied_field.tiles_queue) == 1

        env_setup_occupied_field.launch_tile()

        assert (
            env_setup_occupied_field.current_tile_positionInField
            == current_tile_positionInField
        )
        assert (
            env_setup_occupied_field.top_left_corner_current_tile_in_field
            == top_left_corner_current_tile_in_field
        )
        assert (
            env_setup_occupied_field.current_tile_occupied_cells_in_field
            == current_tile_occupied_cells_in_field
        )

        expected_field = np.array(
            [
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        )

        np.testing.assert_array_equal(env_setup_occupied_field.field, expected_field)

        with pytest.raises(exception, match=match):
            env_setup_occupied_field.drop_current_tile()

    # ---------------------------------------------------------------------------------
