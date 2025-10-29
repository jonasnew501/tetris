import numpy as np
import math
import matplotlib.pyplot as plt
import random
from collections import deque
from itertools import product
from enum import Enum
from typing import Union, Literal

from tetris.tetris_env_domain_specific_exceptions import (
    EmptyContainerError,
    NoneTypeError,
    WrongDatatypeError,
    OutOfBoundsError,
    GamewiseLogicalError,
    UnsupportedParameterValue,
)

plt.ion()


class TetrisEnv:
    """
    This class implements the whole Tetris environment.

    This class implements the whole Tetris environment.
    It's purpose is to implement the Tetris field/-grid, and all related functionalities.
    Amongst those are the launching of new Tetrominos, the drop of Tetrominos, clearing
    full rows, rotation of tetrominos etc.
    This class is meant to be instanciated in a 'main'-/entry-point-script, where
    either a human user or a Reinforcement Learning-Agent has access to specific
    functions of this class, such as moving or rotating Tetrominos.

    Attributes:
        field_height (int): Height of the Tetris playing field (number of rows).
        field_width (int): Width of the Tetris playing field (number of columns).
        len_tiles_queue (int): Number of upcoming tiles that are pre-selected and visible.
        launch_position (list[int]): Starting position [row, column] where new tiles appear.
        field (np.ndarray): 2D array representing the current state of the Tetris grid.
        tiles (dict[str, np.ndarray]): Dictionary mapping tile names to their shapes.
        current_action (int): The current or most recently executed action.
        tiles_queue (deque[list[str, np.ndarray, int]]): The tiles that are about to be launched next.
                                                         The first tile in the deque is the very next one
                                                         to be launched.
                                                         For every tile, holds a list of
                                                         "[Name of the tile, the tiles' array, the tiles' rotation]"

                                                         Explanation regarding the rotation of the tile:
                                                         The roation can take four values:
                                                           * "0" = not rotated (i.e. in the initial launch position)
                                                           * "1" = rotated by 90 degrees to the right
                                                           * "2" = rotated by 180 degrees
                                                           * "3" = rotated by 270 degrees (to the right)
        current_tile (list[str, np.ndarray, int]): The currently active tile in the field.
                                                   Holds a list of "[Name of the tile, the tiles' array, the tiles' rotation]"

        current_tile_positionInField (list[list[int], list[int]]): Holds the coordinates of the position of the current tile in the field.
                                                                   Every part/cell of the tile is represented.
                                                                   The list at index 0 holds the row-indices
                                                                   and the list at index 1 holds the column-indices.
                                                                   By zipping both lists element-wise the excact locations of every
                                                                   part/cell of the current tile are obtained.



    Methods:
        populate_tiles_queue():
            Populates 'self.tiles_queue' with tiles randomly selected
            from all tiles available.
            'self.tiles_queue' is filled up until it's length matches
            the desired length (='self.len_tiles_queue').

        #TODO: Complete the list of function here.
    """

    def __init__(self, field_height, field_width, len_tiles_queue):
        self.field_height = field_height
        self.field_width = field_width
        self.len_tiles_queue = len_tiles_queue

        self.launch_position = [0, math.floor(field_width / 2)]

        self.field = self._create_empty_field(
            field_height=field_height, field_width=field_width
        )

        self.tiles = {
            "I": np.ones((4, 1)),
            "O": np.ones((2, 2)),
            "S": np.array([[0, 1], [1, 1], [1, 0]]),
            "S_inv": np.array([[1, 0], [1, 1], [0, 1]]),
            "L": np.array([[1, 0], [1, 0], [1, 1]]),
            "L_inv": np.array([[0, 1], [0, 1], [1, 1]]),
            "T": np.array([[0, 1, 0], [1, 1, 1]]),
        }

        self.current_action = None

        self.tiles_queue = deque()
        self._populate_tiles_queue()

        self.current_tile = None

        self.current_tile_positionInField = [
            [],
            [],
        ]

    # -----ENUMs------------------------------------------------------------------------
    class PossibleActions(Enum):
        """
        Possible actions that can be taken in the game.
        """

        do_nothing = 0
        move_left = 1
        move_right = 2
        rotate = 3

    # ----------------------------------------------------------------------------------

    # -----central functions------------------------------------------------------------
    def launch_tile(self) -> bool:
        """
        Launches the next tile to come from 'self.tiles_queue' into the field.

        Side-effects are:
            - re-populate 'self.tiles_queue' after the pop of the leftmost tile
              from that queue was done.
            - tries to put 'self.current_tile' into the field at 'self.launch_position'.
            - updates 'self.current_tile_positionInField'

        Returns:
            (bool): True, if the tile could successfully be launched into the field,
                    False, if the tile could not be launched into the field, because
                    there was an overlap with the field at the cells where the tile
                    was tried to be launched.
        """
        self.current_tile = self._tiles_queue_pop_left()

        # since now one tile was removed from the tiles_queue,
        # the tiles queue is populated again
        self._populate_tiles_queue()

        put_successful = self._put_tile_into_field(
            self.current_tile[1], position=self.launch_position
        )

        if not put_successful:  # i.e. game over
            return False
        else:
            self._set_current_tile_position_in_field_at_launch(self.current_tile)
            return True

    def drop_current_tile(self, drop_possible: bool):
        """
        Drops the current tile in the field by one row.

        Expects a drop of the current tile to be possible in the current
        state of the field.

        Args:
            drop_possible (bool): Indicating whether a drop of the current
                                  tile is currently possible or not.

        Raises:
            GamewiseLogicalError: If 'drop_possible' is False
        """
        if drop_possible is False:
            raise GamewiseLogicalError

        current_tile_number_of_rows = self._current_tile_number_of_rows()
        current_tile_number_of_columns = self._current_tile_number_of_columns()

        # retaining the old 'current_tile_positionInField'-variable before it is updated below
        current_tile_positionInField_old = self.current_tile_positionInField.copy()

        # Increasing all row-numbers by one (i.e. the tile moves downward by one row)
        self.current_tile_positionInField[0] += np.ones(
            shape=(current_tile_number_of_columns,), dtype=np.int8
        )

        # Updating the tile in the field (i.e. doing the actual dropping)
        # dropping the current tile by merging it with the new place of the tile after the
        # drop by bitwise OR
        current_tile = self.current_tile[1]

        self.field[
            min(self.current_tile_positionInField[0]) : min(
                self.current_tile_positionInField[0]
            )
            + current_tile_number_of_rows,
            min(self.current_tile_positionInField[1]) : min(
                self.current_tile_positionInField[1]
            )
            + current_tile_number_of_columns,
        ] |= current_tile

        # clearning the topmost-row of the former current_tile_positionInField because the tile now
        # moved down by one row
        self.field[
            min(current_tile_positionInField_old[0]),
            min(self.current_tile_positionInField[1]) : min(
                self.current_tile_positionInField[1]
            )
            + current_tile_number_of_columns,
        ] = np.int8(0)

    def remove_full_rows(self, full_rows_indices: Union[list, np.ndarray]) -> int:
        """
        Removes all rows given by 'full_rows_indices' from the field.
        After that, all tiles above the removed rows are moved down
        by the number of previously removed rows.

        Args:
            full_rows_indices (list or np.ndarray): A 1D-container holding the indices
                                                    of full rows.
                                                    If empty, there are no rows to be
                                                    removed.

        Returns:
            Rows removed (int): The number of rows which were full and were thus removed.
        """
        if not full_rows_indices:
            return 0

        # removing the full rows from the field
        field_rows_deleted = np.delete(arr=self.field, obj=full_rows_indices, axis=0)

        # adding as many new (i.e. empty) rows to the top of the field as were just deleted
        self.field = np.vstack(
            (
                np.zeros(
                    shape=(len(full_rows_indices), self.field_width), dtype=np.int8
                ),
                field_rows_deleted,
            )
        )

        return len(full_rows_indices)

    def handle_action(self, action: PossibleActions):
        """
        Handles all possible actions that can be taken in the game.

        Args:
            action (PossibleActions): The action to be handled.

        Raises:
            ValueError: If the value passed to 'action'
                        is not listed in the Enum
                        'PossibleActions'
        """
        if action == self.PossibleActions.do_nothing:
            pass
        if action == self.PossibleActions.move_left:
            self.move(direction=action)
        elif action == self.PossibleActions.move_right:
            self.move(direction=action)
        elif action == self.PossibleActions.rotate:
            self.rotate()
        else:
            raise ValueError("Unknown action: {action}")

        self._set_current_action(action=action)

    def move(self, direction: PossibleActions):
        """
        Moves the current tile in the field either to the left or to the right
        by one column.

        Args:
            direction (PossibleActions): Possible values are 'move_left' and 'move_right'.
        """
        current_tile_number_of_rows = self._current_tile_number_of_rows()
        current_tile_number_of_columns = self._current_tile_number_of_columns()

        # retaining the old 'current_tile_positionInField'-variable before it is updated below
        current_tile_positionInField_old = self.current_tile_positionInField.copy()

        # move to the left
        if direction == self.PossibleActions.move_left:
            if not self._move_possible(direction=direction):
                return

            # updating 'self.current_tile_positionInField'
            self.current_tile_positionInField[1] -= np.ones(
                shape=(len(self.current_tile_positionInField[1]),), dtype=np.int8
            )

            # Updating the tile in the field (i.e. doing the actual movement)
            current_tile = self.current_tile[1]

            self.field[
                min(self.current_tile_positionInField[0]) : min(
                    self.current_tile_positionInField[0]
                )
                + current_tile_number_of_rows,
                min(self.current_tile_positionInField[1]) : min(
                    self.current_tile_positionInField[1]
                )
                + current_tile_number_of_columns,
            ] |= current_tile

            # emptying the rightmost column of the previous position of the current tile, because that column not got empty due to the move
            self.field[
                list(set(self.current_tile_positionInField[0])),
                max(current_tile_positionInField_old[1]),
            ] = np.int8(0)

        # move to the right
        elif direction == self.PossibleActions.move_right:
            if not self._move_possible(direction=direction):
                return

            # Updating 'self.current_tile_positionInField
            self.current_tile_positionInField[1] += np.ones(
                shape=(len(self.current_tile_positionInField[1]),), dtype=np.int8
            )

            # Updating the tile in the field (i.e. doing the actual movement)
            current_tile = self.current_tile[1]

            self.field[
                min(self.current_tile_positionInField[0]) : min(
                    self.current_tile_positionInField[0]
                )
                + current_tile_number_of_rows,
                min(self.current_tile_positionInField[1]) : min(
                    self.current_tile_positionInField[1]
                )
                + current_tile_number_of_columns,
            ] |= current_tile

            # emptying the leftmost column of the previous position of the current tile, because that column not got empty due to the move
            self.field[
                list(set(self.current_tile_positionInField[0])),
                min(current_tile_positionInField_old[1]),
            ] = np.int8(0)

        else:
            raise UnsupportedParameterValue(
                f"The value ({direction!r}) passed to the parameter 'direction' of this function is not supported in this function.\n \
                                            Only 'move_left' and 'move_right' are supported!"
            )

    def rotate(self):
        """
        Rotates the current tile by 90 degrees clockwise.

        This method expects that a rotation of 'self.current_tile' is currently possible.
        """
        current_tile_positionInField_before_rotation = (
            self.current_tile_positionInField.copy()
        )

        self.current_tile_positionInField = (
            self._get_current_tile_positionInField_after_rotation()
        )

        self.current_tile[2] = self._update_rotation_value()

        current_tile_rotated = self._rotate_tile(tile_to_rotate=self.current_tile[1])

        top_left_corner_of_current_tile_rotated_in_field = sorted(
            list(zip(*self.current_tile_positionInField.copy())),
            key=lambda tup: sum(tup),
        )[0]

        self._put_tile_into_field(
            tile_to_put_into_field=current_tile_rotated,
            position=list(top_left_corner_of_current_tile_rotated_in_field),
        )

        self._clear_unoccupied_cells_in_field_after_rotation(
            current_tile_positionInField_before_rotation=current_tile_positionInField_before_rotation,
            current_tile_positionInField_after_rotation=self.current_tile_positionInField.copy(),
        )

    def reset(self):
        """
        Resets the environment to an initial state.

        The initial state means the start of a new game.
        That means:
            - The field is emptied.
            - 'self.current_tile' is set to None
            - 'self.current_tile_positionInField' is emptied
            - 'self.current_action' is set to None
            - 'self.tiles_queue' is emptied and freshly populated
            - Points achieved are set to zero.

        """
        self.field = self._create_empty_field(
            field_height=self.field_height, field_width=self.field_width
        )

        self.current_tile = None

        self.current_tile_positionInField[0] = self._empty_list(
            list_to_empty=self.current_tile_positionInField[0]
        )
        self.current_tile_positionInField[1] = self._empty_list(
            list_to_empty=self.current_tile_positionInField[1]
        )

        self.current_action = None

        self.tiles_queue = deque()
        self._populate_tiles_queue()

        # TODO:
        # Set game-points achieved to zero.

    # ----------------------------------------------------------------------------------

    # -----Helper-functions-------------------------------------------------------------
    def _tiles_queue_pop_left(self) -> list[str, np.ndarray, int]:
        """
        Pops the first resp. leftmost element of 'self.tiles_queue'
        and returns it.

        Returns:
            (list): The leftmost element of 'self.tiles_queue'
                    holding "[Name of the tile, the tiles' array, the tiles' rotation]"

        Raises:
            EmptyContainerError: If 'self.tiles_queue' is empty
                                 before attempting to retrieve
                                 the leftmost element.
        """
        if not self.tiles_queue:
            raise EmptyContainerError

        return self.tiles_queue.popleft()

    def _populate_tiles_queue(self):
        """
        Populates 'self.tiles_queue' with tiles randomly selected
        from all tiles available defined in 'self.tiles'.

        'self.tiles_queue' is filled up until it's length matches
        the desired length (='self.len_tiles_queue').
        """
        while len(self.tiles_queue) < self.len_tiles_queue:
            self.tiles_queue.append([*random.choice(list(self.tiles.items())), 0])

    def _put_tile_into_field(
        self, tile_to_put_into_field: np.ndarray, position: list[int, int]
    ):
        """
        Puts 'tile_to_put_into_field' into the field at 'position' via a bitwise OR-operation.

        This function expects the put-operation to succeed. I.e. tests whether the put-operation
        at 'position' of 'tile_to_put_into_field' are not done by this function.

        Args:
            tile_to_put_into_field (np.ndarray): The tile to put into the field.
                                                 Needs to be a two-dimensional array.
            position (list[int, int]): The position in the field in form of "[row_index, column_index]"
                                       where the tile is tried to be put into the field.
                                       The tile will be put into the field so that its top-left corner
                                       (i.e. row 0, column 0 of the tile) is located at 'position'

        Raises:
            UnsupportedParameterValue: When the dimensionality of 'tile_to_put_into_field' is not 2.
            WrongDatatypeError:
                - When 'tile_to_put_into_field' is not of type 'np.ndarray'
                - When 'position' is not of type 'list[int, int]'
        """
        if not isinstance(tile_to_put_into_field, np.ndarray):
            raise WrongDatatypeError(
                f"'tile_to_put_into_field' is of type {type(tile_to_put_into_field)}, however needs to be of type 'np.ndarray'."
            )

        if (
            (not isinstance(position, list))
            or (not isinstance(position[0], int))
            or (not isinstance(position[1], int))
        ):
            raise WrongDatatypeError(
                f"'position' needs to be of type 'list', and its contents both need to be of type 'int'. One or both of these requirements were violated."
            )

        if (
            dim := self._get_dimensionality_of_ndarray(ndarray=tile_to_put_into_field)
            != 2
        ):
            raise UnsupportedParameterValue(
                f"The dimensionality of 'tile_to_put_into_field' is {dim}, however needs to be 2."
            )

        current_tile_number_of_rows = self._current_tile_number_of_rows()
        current_tile_number_of_columns = self._current_tile_number_of_columns()

        self.field[
            position[0] : position[0] + current_tile_number_of_rows,
            position[1] : position[1] + current_tile_number_of_columns,
        ] |= tile_to_put_into_field

    def _set_current_tile_position_in_field_at_launch(
        self, tile_to_put_into_field: np.ndarray
    ):
        """
        Updates 'self.current_tile_positionInField'.
        """
        # creating the row-indices
        tile_n_rows, tile_n_columns = tile_to_put_into_field.shape

        # For every row, there are as many row-indices as there are columns:
        # E.g. for a 3*2-array/-tile it would be "[0, 0, 1, 1, 2, 2]"
        row_indices = np.repeat(np.arange(tile_n_rows), repeats=tile_n_columns)

        # For every row, all column indices are listed, i.e. the column-indices
        # repeat n_row-times.
        # E.g. for a 3*2-array/-tile, it would be "[0, 1, 0, 1, 0, 1]"
        column_indices = np.tile(np.arange(tile_n_columns), reps=tile_n_rows)

        # Adding the launch-position offsets to get the actual positions of
        # the row- and column-indices in the field
        row_indices += self.launch_position[0]
        column_indices += self.launch_position[1]

        self.current_tile_positionInField = [list(row_indices), list(column_indices)]

    def _drop_possible(self) -> bool:
        """
        Checks whether a drop of the current tile is possible.

        Information about the function logic/implementation:
        A drop is only possible if the sum of the individual cells in the current lowest
        row of the tile in all columns and the respective cells in the field one row
        below that row is at most 1.
        Explanation: If the sum was 2, that would mean that both in a cell of the current
                     lowest row and the cell below that (i.e. the cell in the field) are
                     both 1s, i.e. both those cells are occupied already. Thus a drop
                     is not possible. However, if the sum of both cells is 0 or 1,
                     that means that either none of the cells is occupied, or only one
                     of them, which means that a drop is possible.

        Returns:
            (bool): True, if a drop is possible;
                    False otherwise.
        """
        if self._check_tile_at_edge(
            edge="bottom", tile_positionInField=self.current_tile_positionInField
        ):
            return False
        else:
            # loop-based approach
            # return all(self.field[max(self.current_tile_positionInField_old[0]), column]
            #     + self.field[max(self.current_tile_positionInField_old[0]) + 1, column]
            #     in [0, 1]
            #     for column in range(
            #         min(self.current_tile_positionInField_old[1]),
            #         max(self.current_tile_positionInField_old[1]) + 1,
            #         1,
            #     )
            # )

            # vectorized approach
            columns_of_current_tile = list(set(self.current_tile_positionInField[1]))

            lowest_row_current_tile = self.field[
                max(self.current_tile_positionInField[0]), columns_of_current_tile
            ]
            row_below_lowest_row_current_tile = self.field[
                max(self.current_tile_positionInField[0]) + 1, columns_of_current_tile
            ]

            sum_of_both_rows = (
                lowest_row_current_tile + row_below_lowest_row_current_tile
            )

            return all(sum_of_both_rows in [0, 1])

    def _check_tile_at_edge(
        self,
        edge: Literal["left", "right", "bottom"],
        tile_positionInField: list[list[int], list[int]],
    ) -> bool:
        """
        Checks whether the (current) tile (at least one column of it)
        is currently located on the leftmost, rightmost, or bottommost
        edge of the field.

        The check can aim at the current tile 'self.current_tile',
        but also on any other (fictional) tile in the field.
        This is possible the parameter 'tile_positionInField' does not
        necessarily need to be 'self.current_tile_positionInField', but can
        represent resp. hold row and column indices of any (fictional)
        tile.

        Args:
            edge (Literal["left", "right", "bottom"]): Which edge to check for.
            tile_positionInField (list[list[int], list[int]]): The row and column indices
                                                               of the (fictional) tile to
                                                               check.
                                                               This variable must follow the
                                                               principle of the attribute
                                                               'self.current_tile_positionInField'
                                                               of this class at hand (description
                                                               see in __init__-function of this class).

        Returns:
            bool: True, if the (current) tile is currently located
                  on the edge to check of the field;
                  False otherwise.
        """
        if edge == "left":
            pos_leftmost_column_current_tile = min(tile_positionInField[1])
            if pos_leftmost_column_current_tile == 0:
                return True
            else:
                return False

        elif edge == "right":
            pos_rightmost_column_current_tile = max(tile_positionInField[1])
            if pos_rightmost_column_current_tile == self.field_width - 1:
                return True
            else:
                return False

        elif edge == "bottom":
            pos_bottommost_column_current_tile = max(tile_positionInField[0])
            if pos_bottommost_column_current_tile == self.field_height - 1:
                return True
            else:
                return False

        else:
            raise ValueError(
                f"Invalid edge value: {edge!r}. Must be 'left', 'right' or 'bottom'."
            )

    def _current_tile_number_of_rows(self) -> int:
        """
        Returns:
            int: the number of rows of 'self.current_tile'.

        Raises:
            NoneTypeError: If 'self.current_tile' is None.
        """
        if self.current_tile is None:
            raise NoneTypeError

        return self.current_tile[1].shape[0]

    def _current_tile_number_of_columns(self) -> int:
        """
        Returns:
            int: the number of columns of 'self.current_tile'.

        Raises:
            NoneTypeError: If 'self.current_tile' is None.
        """
        if self.current_tile is None:
            raise NoneTypeError

        return self.current_tile[1].shape[1]

    def _check_for_full_rows(self, drop_possible: bool) -> np.ndarray:
        """
        Returns a np.ndarray containing the indices of rows
        in the field, that contain full rows.

        A full row is a row which only contains occupied cells.

        It is required that the check for full rows can only be
        conducted after a drop of the current tile is not
        possible anymore. However a new tile must also not
        already be launched yet.

        Args:
            drop_possible (bool): Indicating whether a drop of the current_tile
                                  is currently possible or not.

        Returns:
            Indices of full rows (np.ndarray): The indices of the rows
                                               that contain full rows.
                                               An empty array means that
                                               there are no full rows.

        Raises:
            GamewiseLogicalError: When 'drop_possible' is True.
                                  Reason: See description above.
        """
        if drop_possible:
            raise GamewiseLogicalError

        full_rows_bool = np.all(self.field == 1, axis=1)
        full_rows_indices = np.where(full_rows_bool)[0]

        return full_rows_indices

    def _set_current_action(self, action: PossibleActions):
        """
        Assigns 'action' to the attribute 'self.current_action'.

        Args:
            action (PossibleActions): The action to be assigned

        Raises:
            ValueError: If the value passed to 'action'
                        is not listed in the Enum
                        'PossibleActions'
        """
        if not isinstance(action, self.PossibleActions):
            raise ValueError(
                f"Invalid action: {action}. Must be a member of PossibleActions."
            )

        self.current_action = action

    def _move_possible(self, direction: PossibleActions) -> bool:
        """
        Checks if moving 'self.current_tile' to the desired 'direction'
        by one column is currently possible or not.

        Args:
            direction (PossibleActions): Possible values are 'move_left' and 'move_right'.

        Returns:
            (bool): True, if the move checked is possible;
                    False otherwise.
        """
        if direction == self.PossibleActions.move_left:
            tile_at_edge = self._check_tile_at_edge(edge="left")
            if tile_at_edge:
                return False

            # Checking whether the sums of the individual cells
            # of the leftmost column of the current tile in all rows
            # and the column in the field left of this leftmost column
            # are at most 1
            rows_of_current_tile = list(set(self.current_tile_positionInField[0]))

            leftmost_column_current_tile = self.field[
                rows_of_current_tile, min(self.current_tile_positionInField[1])
            ]
            column_left_of_leftmost_column_current_tile = self.field[
                rows_of_current_tile, min(self.current_tile_positionInField[1]) - 1
            ]

            sum_of_both_columns = (
                leftmost_column_current_tile
                + column_left_of_leftmost_column_current_tile
            )

            return all(sum_of_both_columns in [0, 1])

        if direction == self.PossibleActions.move_right:
            tile_at_edge = self._check_tile_at_edge(edge="right")
            if tile_at_edge:
                return False

            rows_of_current_tile = list(set(self.current_tile_positionInField[0]))

            rightmost_column_current_tile = self.field[
                rows_of_current_tile, max(self.current_tile_positionInField[1])
            ]
            column_right_of_rightmost_column_current_tile = self.field[
                rows_of_current_tile, max(self.current_tile_positionInField[1]) + 1
            ]

            sum_of_both_columns = (
                rightmost_column_current_tile
                + column_right_of_rightmost_column_current_tile
            )

            return all(sum_of_both_columns in [0, 1])

    def _get_current_tile_positionInField_after_rotation(
        self,
    ) -> list[list[int], list[int]]:
        """
        Creates a variable of the same principle as 'self.current_tile_positionInField'
        holding the row and column indices it would have after a rotation of
        'self.current_tile' by 90 degrees clockwise would have been done.

        I.e. an actual rotation, i.e. update of 'self.field' or
        'self.current_tile_positionInField' or of other attributes
        is not done, but only this variable described above is created and returned.

        Returns:
            list[list[int], list[int]]: The variable of the form/principle of 'self.current_tile_positionInField'
                                        after a simulated rotation by 90 degrees clockwise.
        """
        current_tile_positionInField_copy = self.current_tile_positionInField.copy()

        current_tile_shape_after_rotation = (
            self._get_shape_of_current_tile_after_rotation()
        )

        diff_in_rows, diff_in_columns = (
            self._get_diff_in_rows_and_columns_of_current_tile_after_rotation()
        )

        # calculation of new row and column indices
        # rows
        if diff_in_rows < 0:
            new_row_indices_unique = list(set(current_tile_positionInField_copy[0]))[
                :diff_in_rows
            ]
        elif diff_in_rows > 0:
            new_row_indices_unique = range(
                min(current_tile_positionInField_copy[0]),
                max(current_tile_positionInField_copy[0]) + diff_in_rows + 1,
                1,
            )

        # columns
        if diff_in_columns < 0:
            new_column_indices_unique = list(set(current_tile_positionInField_copy[1]))[
                :diff_in_columns
            ]
        elif diff_in_rows > 0:
            new_column_indices_unique = range(
                min(current_tile_positionInField_copy[1]),
                max(current_tile_positionInField_copy[1]) + diff_in_columns + 1,
                1,
            )

        # creating the lists of row and column indices as held in 'self.current_tile_positionInField' in principle
        new_row_indices = list(
            np.repeat(
                new_row_indices_unique, repeats=current_tile_shape_after_rotation[1]
            )
        )
        new_column_indices = list(
            np.tile(
                new_column_indices_unique, reps=current_tile_shape_after_rotation[0]
            )
        )

        return list(new_row_indices, new_column_indices)

    def _update_rotation_value(self) -> int:
        """
        Updates the rotation-value held in
        'self.current_tile[2]' by one rotation
        by 90 degrees clockwise.

        'self.current_tile[2]' is not altered
        by this function.

        Returns:
            new_rotation_value (int): The updated rotation value.
        """
        current_rotation_value = self.current_tile[2]
        new_rotation_value = (
            (current_rotation_value + 1) if (current_rotation_value + 1) < 4 else 0
        )  # because a rotation of "4" would just mean it is at rotation 0 (i.e. initial position) again

        return new_rotation_value

    def _rotate_tile(self, tile_to_rotate: np.ndarray) -> np.ndarray:
        """
        Rotates 'tile_to_rotate' by 90 degrees clockwise.

        Returns:
            rotated_tile (np.ndarray): The rotated tile.
        """
        return np.rot90(tile_to_rotate, k=1, axes=(1, 0))

    def _clear_unoccupied_cells_in_field_after_rotation(
        self,
        current_tile_positionInField_before_rotation: list[list[int], list[int]],
        current_tile_positionInField_after_rotation: list[list[int], list[int]],
    ):
        """
        After a tile having a different number of rows and columns is rotated,
        there are cells in the field, which are then not occupied by the
        current tile anymore.

        This function sets those cells in 'self.field' to zero.

        Args:
            current_tile_positionInField_before_rotation (list[list[int], list[int]]):
            The value held in 'self.current_tile_positionInField' before a rotation of
            'self.current_tile' was conducted resp. before this attribute was updated to the
            state after the rotation.
            current_tile_positionInField_after_rotation (list[list[int], list[int]]):
            The value held in 'self.current_tile_positionInField' after a rotation
            of 'self.current_tile' was conducted resp. after this attribute was updated to the
            state after the rotation.
        """
        coordinates_current_tile_positionInField_before_rotation = list(
            zip(*current_tile_positionInField_before_rotation)
        )
        coordinates_current_tile_positionInField_after_rotation = list(
            zip(*current_tile_positionInField_after_rotation)
        )

        unoccupied_cells_after_rotation = [
            tup
            for tup in coordinates_current_tile_positionInField_before_rotation
            if tup not in coordinates_current_tile_positionInField_after_rotation
        ]

        rows_unoccupied_cells = [row for row, _ in unoccupied_cells_after_rotation]
        columns_unoccupied_cells = [col for _, col in unoccupied_cells_after_rotation]

        self.field[
            min(rows_unoccupied_cells) : max(rows_unoccupied_cells) + 1,
            min(columns_unoccupied_cells) : max(columns_unoccupied_cells) + 1,
        ] = np.int(8)

    def _check_rotation_possible(self) -> bool:
        """
        Checks whether rotating 'self.current_tile' at its current
        position in the field by 90 degrees to the right is possible.

        Returns:
            bool: True, if a rotation by 90 degrees to the right is possible;
                  False otherwise.
        """
        out_of_bounds_with_rotation = self._out_of_bounds_with_rotation()
        collision_with_rotation = self._collision_with_rotation()

        if (not out_of_bounds_with_rotation) and (not collision_with_rotation):
            return True
        else:
            return False

    def _out_of_bounds_with_rotation(self) -> bool:
        """
        Checks whether 'self.current_tile' would be out of bounds
        of the field when it would be rotated by 90 degrees
        clockwise.

        An actual rotation, i.e. an update of 'self.field' or
        'self.current_tile_positionInField' or of other attributes
        is not done, but only the check described above is
        done and the respective boolean is returned.

        Returns:
            (bool): True, if 'self.current_tile' would be
                    out-of-bounds when rotated by 90 degrees
                    clockwise;
                    False otherwise.
        """
        shape_of_current_tile = self._get_shape_of_current_tile()

        if shape_of_current_tile[0] == shape_of_current_tile[1]:
            return False

        current_tile_at_right_edge = self._check_tile_at_edge(edge="right")
        current_tile_at_bottom_edge = self._check_tile_at_edge(edge="bottom")

        # if the tile has more rows than columns and the tile is located
        # at the right edge, after a rotation the tile would reach out
        # of the right edge of the field.
        # Equivalently, if the tile has more columns than rows and the
        # tile is located at the bottom edge, after a rotation the tile
        # would reach out of the bottom edge of the field.
        if (
            (shape_of_current_tile[0] > shape_of_current_tile[1])
            and current_tile_at_right_edge
        ) or (
            (shape_of_current_tile[1] > shape_of_current_tile[0])
            and current_tile_at_bottom_edge
        ):
            return True

        return False

    def _collision_with_rotation(self) -> bool:
        """
        Checks whether with a rotation the area of the field,
        which the then rotated tile will cover, and which was previously not covered
        by the current tile, will collide with the respective
        part of the then rotated tile or not.

        A 'collision' means that the sum of individual cells is
        at most 1.

        If a collision would happen, a rotation is not possible.

        An actual rotation, i.e. an update of 'self.field' or
        'self.current_tile_positionInField' or of other attributes
        is not done, but only the check described above is
        done and the respective boolean is returned.

        Returns:
            bool: True, if a collision with the rotated tile
                  and the field as described above would happen;
                  False otherwise.
        """
        shape_of_current_tile = self._get_shape_of_current_tile()

        if shape_of_current_tile[0] == shape_of_current_tile[1]:
            return False

        # Determining the cells of the field which 'self.current_tile' doesn't occupy
        # now, but will occupy when it was rotated by 90 degrees clockwise.
        current_tile_positionInField = self.current_tile_positionInField.copy()
        current_tile_positionInField_after_rotation = (
            self._get_current_tile_positionInField_after_rotation()
        )

        current_tile_cells_occupied = list(
            zip(current_tile_positionInField[0], current_tile_positionInField[1])
        )
        current_tile_cells_occupied_after_rotation = list(
            zip(
                current_tile_positionInField_after_rotation[0],
                current_tile_positionInField_after_rotation[1],
            )
        )

        new_cells_occupied_after_rotation = [
            tup
            for tup in current_tile_cells_occupied_after_rotation
            if tup in current_tile_cells_occupied_after_rotation
            and tup not in current_tile_cells_occupied
        ]

        field_at_new_cells_occupied_after_rotation = (
            self._get_slice_of_field_from_coords(
                coords=new_cells_occupied_after_rotation
            )
        )

        current_tile_at_new_cells_occupied_after_rotation = (
            self._get_slice_of_current_tile_at_new_cells_occupied_after_rotation()
        )

        assert (
            field_at_new_cells_occupied_after_rotation.shape
            == current_tile_at_new_cells_occupied_after_rotation.shape
        ), "The two slices do not have the same shape, however, this is required."

        sum_of_both_slices = (
            field_at_new_cells_occupied_after_rotation
            + current_tile_at_new_cells_occupied_after_rotation
        )

        return not all(sum_of_both_slices in [0, 1])

    def _get_shape_of_current_tile(self) -> tuple:
        """
        From 'self.current_tile_positionInField', computes and returns
        the shape of the current tile (incorporating resp. considering
        it's current degree of rotation).

        Returns:
        Shape of the current tile (as a tuple)
        """
        n_rows = len(set(self.current_tile_positionInField[0].copy()))
        n_columns = len(set(self.current_tile_positionInField[1].copy()))

        return (n_rows, n_columns)

    def _get_slice_of_field_from_coords(self, coords: list[tuple]) -> np.ndarray:
        """
        Slices out the area defined by 'coords' from 'self.field'
        and returns that slice.

        'self.field' is not altered by this function.

        It is required that the coords are adjacent to each other, either
        horizontally or vertically (not diagonally). That means there
        must be no gap between one section in the field defined
        by the coordinates and another section in the field defined
        by the coordinates.

        Args:
            coords (list[tulple[int, int]]): A list of one or multiple tuples of the form
                                             (row-index, column-index).
                                             Every tuple defines a coordinate in 'self.field'

        Returns:
            field_slice (np.ndarray): The slice of 'self.field' defined by 'coords'.

        Raises:
            GamewiseLogicalError: If either the rows and/or the columns were identified as not
                                  being adjacent to each other, i.e. they have gaps in between.
        """
        # splitting the coordinates into rows and columns
        rows = [row for row, _ in coords]
        columns = [column for _, column in coords]

        # Checking for adjacency of both rows and columns
        adjacent_rows = list(range(min(rows), max(rows) + 1, 1))
        adjacent_columns = list(range(min(columns), max(columns) + 1, 1))

        check_adjacent_rows = sorted(rows) == adjacent_rows
        check_adjacent_columns = sorted(columns) == adjacent_columns

        if (not check_adjacent_rows) or (not check_adjacent_columns):
            raise GamewiseLogicalError(
                f"The rows and/or columns from the coords were identified as not being adjacent.\n \
                                       However, this is required by the logic of this function.\n \
                                       rows: {rows}, columns: {columns}."
            )

        field_slice = self.field[
            min(rows) : max(rows) + 1, min(columns) : max(columns) + 1
        ]

        return field_slice

    def _get_slice_of_current_tile_at_new_cells_occupied_after_rotation(
        self,
    ) -> np.ndarray:
        """
        From the rotated tile, slices that part of the tile which would now
        occupy new cells in the field (i.e. cells resp. an area of the field
        which the non-rotated tile did not occupy in the field yet, but
        the rotated tile will).
        The part of the tile which occupied the field before the rotation
        and which still occupies that same place in the field after the
        rotation is omitted, i.e. not incorporated the the slice created here.

        An update of 'self.field' or 'self.current_tile_positionInField'
        or of other attributes is not done,
        only the operations described above are
        done and the respective np.ndarray is returned.

        Returns:
            np.ndarray: The slice of the current_tile which will occupy new
                        cells in the field after a rotation.

        Raises:
            GamewiseLogicalError: If the current_tile has as many rows as it has
                                  columns.
                                  Reason: This function expects the current_tile
                                          to have a different number of rows and
                                          columns, because only then a new part
                                          of the field would be occupied with
                                          a rotation.
        """
        shape_of_current_tile = self._get_shape_of_current_tile()

        if shape_of_current_tile[0] == shape_of_current_tile[1]:
            raise GamewiseLogicalError(
                "'self.current_tile' has the same number of rows and columns.\n \
                                       However, this function expects current tile to have a different number of rows and columns.\n \
                                       Only call this function with a current_tile that has a different number of rows and columns."
            )

        current_tile_number_of_rows = self._current_tile_number_of_rows()
        current_tile_number_of_columns = self._current_tile_number_of_columns()

        # The following row- and column-indices of the current (non-rotated) tile
        # start from 0, i.e. are individual to the tile and not related to the
        # tiles' position in the field.
        current_tile_row_indices = list(range(0, current_tile_number_of_rows, 1))
        current_tile_column_indices = list(range(0, current_tile_number_of_columns, 1))

        current_tile_copy = self.current_tile[1].copy()
        current_tile_rotated = self._rotate_tile(tile_to_rotate=current_tile_copy)

        diff_in_rows, diff_in_columns = (
            self._get_diff_in_rows_and_columns_of_current_tile_after_rotation()
        )

        if (diff_in_rows > 0) and (diff_in_columns < 0):
            current_tile_rotated_new_cells_slice = current_tile_rotated[
                max(current_tile_row_indices) : max(current_tile_row_indices)
                + diff_in_rows
                + 1,
                min(current_tile_column_indices) : max(current_tile_column_indices)
                + diff_in_columns
                + 1,
            ]
        elif (diff_in_columns > 0) and (diff_in_rows < 0):
            current_tile_rotated_new_cells_slice = current_tile_rotated[
                max(current_tile_column_indices) : max(current_tile_column_indices)
                + diff_in_columns
                + 1,
                min(current_tile_row_indices) : max(current_tile_row_indices)
                + diff_in_rows
                + 1,
            ]

        return current_tile_rotated_new_cells_slice

    def _get_diff_in_rows_and_columns_of_current_tile_after_rotation(
        self,
    ) -> tuple[int, int]:
        """
        Calculates and returns the difference in rows and columns 'self.current_tile'
        would have when it would be rotated by 90 degrees clockwise.

        Example:
        Say the current_tile has 3 rows and 2 columns (i.e. shape=(3,2)).
        After a rotation by 90 degrees clockwise it would have 2 rows and 3 columns
        (i.e. shape=(2, 3)).
        Thus, the difference in rows is '-1' (because the rotated tile has one row
        less that the initial tile),
        and the difference in columns is '1' (because the rotated tile has one column
        more than the initial tile).

        An actual rotation, i.e. an update of 'self.field' or
        'self.current_tile_positionInField' or of other attributes
        is not done, but only the variable described above is
        created and returned.

        Returns:
            tuple[int, int]: A tuple containing two integers:
                - diff_in_rows (int): The difference in rows after a rotation by 90 degrees clockwise.
                - diff_in_columns (int): The difference in columns after a rotation by 90 degrees clockwise.
        """

        current_tile_shape = self._get_shape_of_current_tile()

        current_tile_shape_after_rotation = (
            self._get_shape_of_current_tile_after_rotation()
        )

        diff_in_rows = current_tile_shape_after_rotation[0] - current_tile_shape[0]
        diff_in_columns = current_tile_shape_after_rotation[1] - current_tile_shape[1]

        return (diff_in_rows, diff_in_columns)

    def _get_shape_of_current_tile_after_rotation(self) -> tuple[int, int]:
        """
        Calculates and returns the shape 'self.current_tile' would have
        after a rotation by 90 degrees clockwise would have been
        conducted.

        An actual rotation, i.e. an update of 'self.field' or
        'self.current_tile_positionInField' or of other attributes
        is not done, but only the variable described above is
        created and returned.
        """

        current_tile_shape = self._get_shape_of_current_tile()

        # First getting/computing the shape after a rotation would have been done
        # (i.e. just flipping rows and columns)
        current_tile_shape_after_rotation = tuple(reversed(current_tile_shape))

        return current_tile_shape_after_rotation

    def _create_empty_field(self, field_height: int, field_width: int) -> np.ndarray:
        """
        Creates an empty field with the shape (field_height, field_width).

        Returns:
            (np.ndarray): The empty field.

        Raises:
            WrongDatatypeError: When at least one of the values given
                                to the parameters doesn't match the
                                expected datatype of the parameter.
        """
        if not isinstance(field_height, int) or not isinstance(field_width, int):
            raise WrongDatatypeError

        return np.zeros(shape=(field_height, field_width), dtype=np.int8)

    def visualize_field(self):
        # TODO: Finish with constantly updating plot
        plt.figure(figsize=(10, 10))

        plt.scatter(x=self.field, color="red")

        plt.show(block=True)

    def _empty_list(self, list_to_empty: list) -> list:
        """
        Empties 'list_to_empty' and returns the emptied list.

        If 'list_to_empty' was already empty at time of this
        method-call, the unchanged 'list_to_empty' is returned.

        Returns:
            (list): The emptied 'list_to_empty'.

        Raises:
            WrongDatatypeError: When 'list_to_empty' is not of type 'list'.
        """
        if not isinstance(list_to_empty, list):
            raise WrongDatatypeError

        return list_to_empty.clear()

    def _get_dimensionality_of_ndarray(self, ndarray: np.ndarray) -> int:
        """
        Returns the dimensionality of 'ndarray'

        Returns:
            dimensionality (int): The dimensionality of 'ndarray'.

        Raises:
            WrongDatatypeError: When 'ndarray' is not of type 'np.ndarray'
        """
        if not isinstance(ndarray, np.ndarray):
            raise WrongDatatypeError

        return ndarray.ndim

    def _out_of_bounds_at_launch(self, tile_to_check: np.ndarray) -> bool:
        """
        Checks whether 'tile_to_check' would be out of bounds
        of the field when being put into the field.

        The core assumption is that the tile is always put into
        the field with its top-left corner being on the
        'self.launch_position'.

        Returns:
            (bool): True, if 'tile_to_check' would be out-of-bounds when launched
                    at 'self.launch_position' into the field;
                    False otherwise.
        """
        current_tile_number_of_rows = self._current_tile_number_of_rows()
        current_tile_number_of_columns = self._current_tile_number_of_columns()

        # Check for condition 1: Is any of the indices of the launch_position negative?
        negative_launch_pos_indices = (
            self.launch_position[0] < 0 or self.launch_position[1] < 0
        )

        # Check for condition 2: Does the tile reach out of the right hand side border of the field?
        field_rightmost_column_idx = self.field_width - 1
        out_of_bounds_right = (
            self.launch_position[1] + (current_tile_number_of_columns - 1)
            > field_rightmost_column_idx
        )

        # Check for condition 2: Does the tile reach out of the bottom border of the field?
        field_bottommost_row_idx = self.field_height - 1
        out_of_bounds_bottom = (
            self.launch_position[0] + (current_tile_number_of_rows - 1)
            > field_bottommost_row_idx
        )

        if negative_launch_pos_indices or out_of_bounds_right or out_of_bounds_bottom:
            return True
        else:
            return False

    def _overlap_at_launch(self, tile_to_put_into_field: np.ndarray) -> bool:
        """
        Checks whether 'tile_to_put_into_field' does not overlap/collide with
        the field when put into the field at self.launch_position.
        An overlap happens if at a specific coordinate, both the field
        and the tile (when put into the field) hold a "1".

        Returns:
            (bool): True, if 'tile_to_put_into_field' could successfully be put
                    into the field,
                    False, if the 'tile_to_put_into_field' collides with other
                    tiles in the field at 'self.launch_position'.
        """
        # Getting the section of the field where the tile would go
        current_tile_number_of_rows = self._current_tile_number_of_rows()
        current_tile_number_of_columns = self._current_tile_number_of_columns()

        field_section = self.field[
            self.launch_position[0] : self.launch_position[0]
            + current_tile_number_of_rows,
            self.launch_position[1] : self.launch_position[1]
            + current_tile_number_of_columns,
        ]

        # Checking if there would be an overlap with the field in any cell
        overlap = np.any(field_section & tile_to_put_into_field)

        return overlap
# ----------------------------------------------------------------------------------
