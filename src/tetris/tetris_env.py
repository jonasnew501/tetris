import numpy as np
import math
import matplotlib.pyplot as plt
import random
from collections import deque
from itertools import product
from enum import Enum

from tetris.tetris_env_domain_specific_exceptions import(
    EmptyContainerError,
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
        tiles_queue (deque[list[str, np.ndarray]]): The tiles that are about to be launched next.
                                                    The first tile in the deque is the very next one
                                                    to be launched.
                                                    For every tile, holds a list of
                                                    "[Name of the tile, the tiles' array]"
        current_tile (list[str, np.ndarray, int]): The currently active tile in the field.
                                                   Holds a list of "[Name of the tile, the tiles' array, the tiles' rotation]"
                                                   Explanation regarding the rotation of the tile:
                                                   The roation can take four values:
                                                      * "0" = not rotated (i.e. in the initial launch position)
                                                      * "1" = rotated by 90 degrees to the right
                                                      * "2" = rotated by 180 degrees
                                                      * "3" = rotated by 270 degrees (to the right)
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

        self.lauch_position = [0, math.floor(field_width / 2)]

        self.field = np.zeros(shape=(field_height, field_width), dtype=np.int8)

        self.tiles = {
            "I": np.ones((4, 1)),
            "O": np.ones((2, 2)),
            "S": np.array([[0, 1], [1, 1], [1, 0]]),
            "S_inv": np.array([[1, 0], [1, 1], [0, 1]]),
            "L": np.array([[1, 0], [1, 0], [1, 1]]),
            "L_inv": np.array([[0, 1], [0, 1], [1, 1]]),
            "T": np.array([[0, 1, 0], [1, 1, 1]]),
        }

        # TODO: Das Enum noch (richtig) verwenden
        class Possible_Actions(Enum):
            move_left = 1
            move_right = 2
            move_up = 3

        self.current_action = None

        self.tiles_queue = deque()
        self.populate_tiles_queue()

        self.current_tile = None

        self.current_tile_positionInField = [
            [],
            [],
        ]

    def populate_tiles_queue(self):
        """
        Populates 'self.tiles_queue' with tiles randomly selected
        from all tiles available.
        'self.tiles_queue' is filled up until it's length matches
        the desired length (='self.len_tiles_queue').
        """
        while len(self.tiles_queue) < self.len_tiles_queue:
            self.tiles_queue.append([*random.choice(list(self.tiles.items())), 0])

    def launch_tile(self):
        """
        Launches the next tile to come from 'self.tiles_queue' into the field.

        Does various other things too, which are necessary resp. a consequence
        when launching a new tile into the field.
        Those other things are:
            - Populating the tiles queue after the next tile to come was popped
              to fill it up again to the desired length.
            - Assigning the position of the new tile in the field to
              'self.current_tile_positionInField'.
        """
        # popping the first tile from the tiles_queue and assigning it as/to the current_tile
        self.current_tile = self.tiles_queue.popleft()

        # since now one tile was removed from the tiles_queue,
        # the tiles queue is populated again
        self.populate_tiles_queue()

        # defining the shape of the array 'current_tile_positionInField'
        # based on the current_tile
        # self.current_tile_positionInField = np.zeros(shape=(len(self.current_tile[1]), len(self.current_tile[1][0])))

        # cleaning the data (rows and columns) in 'current_tile_positionInField' at this point,
        # so it can be freshly assigned for the now launched tile in the loop below
        self.current_tile_positionInField[0].clear()
        self.current_tile_positionInField[1].clear()

        # putting the current_tile into the field
        for n_row in range(
            len(self.current_tile[1])
        ):  # iterating over the number of rows of the tile
            for n_column in range(
                len(self.current_tile[1][0])
            ):  # iterating over the number of columns of the tile
                self.field[
                    self.lauch_position[0] + n_row, self.lauch_position[1] + n_column
                ] = self.current_tile[1][n_row, n_column]

                # assigning the position of the current_tile in the field to 'current_tile_positionInField'
                self.current_tile_positionInField[0].append(
                    self.lauch_position[0] + n_row
                )  # adding the row
                self.current_tile_positionInField[1].append(
                    self.lauch_position[1] + n_column
                )  # adding the column

    def drop(self) -> bool:
        """
        Drops the current tile in the field by one row.

        Returns:
        A boolean indicating if a drop was possible and thus conducted or not.
        """

        # retaining the old 'current_tile_positionInField'-variable before it is updated below
        current_tile_positionInField_old = self.current_tile_positionInField.copy()

        # Checking if a drop is possible
        # NOTE: A drop is only possible if the sum of the individual cells in the current lowest
        #      row of the tile in all columns and the respective cells in the field one row
        #      below that row is at most 1.
        #      Explanation: If the sum was 2, that would mean that both in a cell of the current
        #                   lowest row and the cell below that (i.e. the cell in the field) are
        #                   both 1s, i.e. both those cells are occupied already. Thus a drop
        #                   is not possible. However, if the sum of both cells is 0 or 1,
        #                   that means that either none of the cells is occupied, or only one
        #                   of them, which means that a drop is possible.
        if (
            max(current_tile_positionInField_old[0]) + 1 == self.field_height
        ):  # the drop isn´t possible anymore because the tile currently is already at the lowest existing row in the field
            drop_possible = False
        else:
            drop_possible = all(
                self.field[max(current_tile_positionInField_old[0]), column]
                + self.field[max(current_tile_positionInField_old[0]) + 1, column]
                in [0, 1]
                for column in range(
                    min(current_tile_positionInField_old[1]),
                    max(current_tile_positionInField_old[1]) + 1,
                    1,
                )
            )

        if not drop_possible:
            # checking if there are full rows and if so, removing those
            # and updating the field accordingly
            rows_dropped = self.check_for_and_handle_full_row()
            # print(f"***** {rows_dropped} rows dropped!*****")
            return False

        # First updating the variable "current_tile_positionInField" by increasing
        # all row-numbers by one (i.e. the tiles moves downward by one row)
        self.current_tile_positionInField[0] = [
            row + 1 for row in self.current_tile_positionInField[0]
        ]

        # Updating the tile in the field (i.e. doing the actual dropping)
        for n_row_new in range(
            max(self.current_tile_positionInField[0]),
            min(self.current_tile_positionInField[0]) - 1,
            -1,
        ):  # iterating backwards over all the (new) rows where the dropped tile will be positioned
            for n_column in range(
                min(self.current_tile_positionInField[1]),
                max(self.current_tile_positionInField[1]) + 1,
                1,
            ):  # iterating over the columns in the field
                # assigning the correct number of the tile (0 or 1) to the respective cell,
                # depending on which number the tile has at that position of it´s grid
                # (i.e. in the field now still at one row up)
                # Rules for the result of a drop:
                # 1.: If a 1 drops into a 0, the 1 stays
                # 2.1: If a 0 drops into a 1, and both are part of the current tile, then the 0 stays.
                # 2.2: If a 0 drops into a 1, but the 1 belongs to the field already and not to the current tile, then the 1 stays.

                # getting the coordinates of the cells of the current tile in the field
                coord_current_tile = list(zip(*current_tile_positionInField_old))
                if (self.field[n_row_new - 1, n_column] == 1) and (
                    self.field[n_row_new, n_column] == 0
                ):
                    self.field[n_row_new, n_column] = np.int8(1)
                elif (
                    (self.field[n_row_new - 1, n_column] == 0)
                    and (self.field[n_row_new, n_column] == 1)
                    and ((n_row_new - 1, n_column) in coord_current_tile)
                    and ((n_row_new, n_column) in coord_current_tile)
                ):
                    self.field[n_row_new, n_column] = np.int8(0)
                elif (
                    (self.field[n_row_new - 1, n_column] == 0)
                    and (self.field[n_row_new, n_column] == 1)
                    and ((n_row_new - 1, n_column) in coord_current_tile)
                    and ((n_row_new, n_column) not in coord_current_tile)
                ):
                    self.field[n_row_new, n_column] = np.int8(1)

        # emptying (i.e. assigning 0s) to the topmost row of the old tile-position in the field,
        # because those cells now got empty because the tile dropped down by one row now
        for n_column in range(
            min(self.current_tile_positionInField[1]),
            max(self.current_tile_positionInField[1]) + 1,
            1,
        ):
            self.field[min(current_tile_positionInField_old[0]), n_column] = np.int8(0)

        return True

    def check_for_and_handle_full_row(self) -> int:
        """
        Checks whether one row or multiple rows is/are full,
        i.e. contain(s) only 1s.
        If so, this row/those rows is/are removed, and all tiles above move
        down by the number of full rows (one or more).

        Both the check whether one or multiple rows are full as well as
        the dropping of the other 1s in the field (except for the current tile)
        is handled by this function.

        Returns:
        An int, indicating how many rows were full and were thus removed.
        """
        # a list holding the indices of full rows
        indices_full_rows = []

        # Iterating through all the rows and saving the indices of full rows
        for i, row in enumerate(self.field):
            if all(row == 1):
                indices_full_rows.append(i)

        if len(indices_full_rows) == 0:
            return 0

        # removing the full rows from the field
        field_rows_deleted = np.delete(arr=self.field, obj=indices_full_rows, axis=0)

        # adding as many new (i.e. empty) rows to the top of the field as were just deleted
        self.field = np.vstack(
            (
                np.zeros(
                    shape=(len(indices_full_rows), self.field_width), dtype=np.int8
                ),
                field_rows_deleted,
            )
        )

        return len(indices_full_rows)

    def handle_action(self, action: int):
        if action == 0:  # i.e. do nothing
            pass
        if action == 1:  # i.e. move tile to the left
            self.move(direction=1)
        elif action == 2:  # i.e. move tile to the right
            self.move(direction=2)
        elif action == 3:  # i.e. rotate tile
            self.rotate()

    def move(self, direction: int) -> bool:
        """
        Moves the current tile in the field one column
        either to the left or to the right.

        Params:
        -'direction': 1=left, 2=right

        Returns:
        A boolean indicating if the desired movement was possible
        and thus conducted or not.
        """
        # retaining the old 'current_tile_positionInField'-variable before it is updated below
        current_tile_positionInField_old = self.current_tile_positionInField.copy()

        # Checking if a move into the desired direction is possible
        if direction == 1:
            if (
                min(current_tile_positionInField_old[1]) == 0
            ):  # the movement to the left isn´t possible anymore because the tile currently is already at the leftmost column in the field
                return False
            else:
                move_possible = all(
                    self.field[row, min(current_tile_positionInField_old[1]) - 1] == 0
                    for row in range(
                        min(current_tile_positionInField_old[0]),
                        max(current_tile_positionInField_old[0]) + 1,
                        1,
                    )
                )

        if direction == 2:
            if (
                max(current_tile_positionInField_old[1]) + 1 == self.field_width
            ):  # the movement to the right isn´t possible anymore because the tile currently is already at the right column in the field
                return False
            else:
                move_possible = all(
                    self.field[row, max(current_tile_positionInField_old[1]) + 1] == 0
                    for row in range(
                        min(current_tile_positionInField_old[0]),
                        max(current_tile_positionInField_old[0]) + 1,
                        1,
                    )
                )

        if not move_possible:
            return False

        # First updating the variable "current_tile_positionInField" by decreasing/increasing
        # all column-numbers by one (i.e. the tiles moves leftward or rightward by one column)
        if direction == 1:
            self.current_tile_positionInField[1] = [
                column - 1 for column in self.current_tile_positionInField[1]
            ]
        elif direction == 2:
            self.current_tile_positionInField[1] = [
                column + 1 for column in self.current_tile_positionInField[1]
            ]

        # Updating the tile in the field (i.e. doing the actual movement)
        if direction == 1:  # i.e. moving to the left
            for n_column_new in range(
                min(self.current_tile_positionInField[1]),
                max(self.current_tile_positionInField[1]) + 1,
                1,
            ):  # iterating forward over all columns where the left-moved tile will be positioned
                for n_row in range(
                    min(self.current_tile_positionInField[0]),
                    max(self.current_tile_positionInField[0]) + 1,
                    1,
                ):  # iterating forward over the rows in the field
                    # assigning the correct number of the tile (0 or 1) to the respective cell,
                    # depending on which number the tile has at that position of it´s grid
                    # (i.e. in the field now still at one column to the right)
                    self.field[n_row, n_column_new] = (
                        np.int8(0)
                        if self.field[n_row, n_column_new + 1] == 0
                        else np.int8(1)
                    )

            # emptying (i.e. assigning 0s) to the rightmost column of the old tile-position in the field,
            # because those cells now got empty because the tile moved to the left by one column now
            for n_row in range(
                min(self.current_tile_positionInField[0]),
                max(self.current_tile_positionInField[0]) + 1,
                1,
            ):
                self.field[n_row, max(current_tile_positionInField_old[1])] = np.int8(0)

        elif direction == 2:  # i.e. moving to the right
            for n_column_new in range(
                max(self.current_tile_positionInField[1]),
                min(self.current_tile_positionInField[1]) - 1,
                -1,
            ):  # iterating backwards over all columns where the right-moved tile will be positioned
                for n_row in range(
                    min(self.current_tile_positionInField[0]),
                    max(self.current_tile_positionInField[0]) + 1,
                    1,
                ):  # iterating forward over the rows in the field
                    # assigning the correct number of the tile (0 or 1) to the respective cell,
                    # depending on which number the tile has at that position of it´s grid
                    # (i.e. in the field now still at one column to the left)
                    self.field[n_row, n_column_new] = (
                        np.int8(0)
                        if self.field[n_row, n_column_new - 1] == 0
                        else np.int8(1)
                    )

            # emptying (i.e. assigning 0s) to the leftmost column of the old tile-position in the field,
            # because those cells now got empty because the tile moved to the right by one column now
            for n_row in range(
                min(self.current_tile_positionInField[0]),
                max(self.current_tile_positionInField[0]) + 1,
                1,
            ):
                self.field[n_row, min(current_tile_positionInField_old[1])] = np.int8(0)

        return True

    # TODO: Debug!
    def rotate(self) -> bool:
        """
        Rotates the current tile by 90 degrees to the right.

        Returns:
        A boolean indicating if the desired rotation was possible
        and thus conducted or not.
        """
        # checking if a rotation is possible in the field
        # (depends on how the rotated tile will be positioned afterwards
        # and if there is enough space to the right of the tile for the flip).

        # Getting the shape of the current tile
        current_shape = self._get_shape_of_current_tile()

        # First getting/computing the shape after a rotation would have been done
        # (i.e. just flipping rows and columns)
        shape_after_rotation = tuple(reversed(current_shape))
        diff_in_rows = shape_after_rotation[0] - current_shape[0]
        diff_in_columns = shape_after_rotation[1] - current_shape[1]

        # TODO: Add a test, whether the current tile is too close to the right-hand border of the field
        #      in order to conduct a rotation.

        if (
            self.current_tile[0] == "O"
        ):  # this code is redundant, however still included for reasons of completeness:
            # Rotating the 'O' tetromino is always possible but also completely without effect
            # since it is quadratic.
            return False  # in this case, False is returned, indicating that a rotation wasn't done (since it wouldn't have an effect anyway for the 'O' tetromino)

        # Checks for the case, that with a rotation the number of columns of the then rotated tile increases,
        # and thus the number of rows decreases.
        elif (
            diff_in_columns > 0
        ):  # i.e. the number of columns would increase with a rotation
            # checking whether the current tile is too close to the right-hand border of the field
            # in order to conduct a rotation.
            # TODO: Das scheint noch nicht zu funktionieren
            if (
                max(self.current_tile_positionInField[1]) + diff_in_columns
            ) > self.field_width:  # a rotation is not possible, an out-of-bounds-error would occur
                return False

            # checking whether in the field to the right of the current
            # tile there are enough empty cells, so a rotation is possible
            for column in range(
                max(self.current_tile_positionInField[1]) + 1,
                max(self.current_tile_positionInField[1]) + 1 + (diff_in_columns),
                1,
            ):
                for row in range(
                    min(self.current_tile_positionInField[0]),
                    max(self.current_tile_positionInField[0]) + 1 + (diff_in_rows),
                    1,
                ):
                    if not self.field[row, column] == 0:
                        return False

        # Checks for the case, that with a rotation the number of columns of the then rotated tile decreases,
        # and thus the number of rows increases.
        elif (
            diff_in_columns < 0
        ):  # i.e. the number of columns would decrease with a rotation (automatically meaning that the number of rows will increase)
            # checking whether the current tile is too close to the bottommost border of the field
            # in order to conduct a rotation.
            # TODO: Das scheint noch nicht zu funktionieren
            if (
                max(self.current_tile_positionInField[0]) + diff_in_rows
            ) > self.field_height:  # a rotation is not possible, an out-of-bounds-error would occur
                return False

            # checking whether in the field below the current tile there are
            # enough empty cells, so a rotation is possible.
            for row in range(
                max(self.current_tile_positionInField[0]) + 1,
                max(self.current_tile_positionInField[0]) + 1 + (diff_in_rows),
                1,
            ):
                for column in range(
                    min(self.current_tile_positionInField[1]),
                    max(self.current_tile_positionInField[1]) + 1 + (diff_in_columns),
                    1,
                ):
                    if not self.field[row, column] == 0:
                        return False

        # at this point it is known that a rotation of the current tile is possible

        # first updating the rotation of the current tile in 'self.current_tile'
        current_rotation = self.current_tile[2]
        new_rotation = (
            (current_rotation + 1) if (current_rotation + 1) < 4 else 0
        )  # because a rotation of "4" would just mean it is at rotation 0 (i.e. initial position) again
        self.current_tile[2] = new_rotation

        # changing the rows and columns in 'current_tile_positionInField' to sets and then to lists again, to make the following rotation-operation possible
        # also saving the sets for later use
        current_tile_positionInField_rows_set_old = list(
            set(self.current_tile_positionInField[0].copy())
        )
        current_tile_positionInField_columns_set_old = list(
            set(self.current_tile_positionInField[1].copy())
        )

        self.current_tile_positionInField[0] = (
            current_tile_positionInField_rows_set_old.copy()
        )
        self.current_tile_positionInField[1] = (
            current_tile_positionInField_columns_set_old.copy()
        )

        # rotating resp. modifying the data held in 'self.current_tile_positionInField'
        if (
            diff_in_columns > 0
        ):  # i.e. the number of columns would increase with a rotation
            self.current_tile_positionInField[0] = self.current_tile_positionInField[0][
                : -abs(diff_in_rows)
            ]
            self.current_tile_positionInField[1].extend(
                list(
                    range(
                        max(self.current_tile_positionInField[1]) + 1,
                        max(self.current_tile_positionInField[1])
                        + abs(diff_in_columns)
                        + 1,
                        1,
                    )
                )
            )

        elif (
            diff_in_columns < 0
        ):  # i.e. the number of columns would decrease with a rotation (automatically meaning that the number of rows will increase)
            self.current_tile_positionInField[0].extend(
                list(
                    range(
                        max(self.current_tile_positionInField[0]) + 1,
                        max(self.current_tile_positionInField[0])
                        + abs(diff_in_rows)
                        + 1,
                        1,
                    )
                )
            )
            self.current_tile_positionInField[1] = self.current_tile_positionInField[1][
                : -abs(diff_in_columns)
            ]

        # multiplying-out the set-like rows and columns in 'current_tile_positionInField' again,
        # so that the actual initial structure of this attribute is obtained again.
        # the "Multiplying-out" is actually called the "Cartesian product", which can be
        # obtained by using "product" from "itertools"
        # Cartesian product
        coords = list(
            product(
                self.current_tile_positionInField[0],
                self.current_tile_positionInField[1],
            )
        )
        # Transpose to get two lists: rows and columns
        rows_full, cols_full = zip(*coords)
        # Assembling the result back together
        self.current_tile_positionInField = [list(rows_full), list(cols_full)]

        # conducting the actual 90 degree rotation to the right
        # 1.: Swapping the rows of the current tile in the field
        tile_swapped = np.flip(
            self.field.copy()[
                min(current_tile_positionInField_rows_set_old) : max(
                    current_tile_positionInField_rows_set_old
                )
                + 1,
                min(current_tile_positionInField_columns_set_old) : max(
                    current_tile_positionInField_columns_set_old
                )
                + 1,
            ],
            axis=0,
        )

        # 2.: Transposing that swapped tile (now the tile (still not in the field yet) is actually rotated by 90 degrees to the right)
        tile_t = tile_swapped.T

        # 3.: Deleting the current (non-rotated) tile from the field
        self.field[
            min(current_tile_positionInField_rows_set_old) : max(
                current_tile_positionInField_rows_set_old
            )
            + 1,
            min(current_tile_positionInField_columns_set_old) : max(
                current_tile_positionInField_columns_set_old
            )
            + 1,
        ] = 0

        # 4.: From the already updated data in 'self.current_tile_positionInField', again getting the sets
        current_tile_positionInField_rows_set_new = list(
            set(self.current_tile_positionInField[0].copy())
        )
        current_tile_positionInField_columns_set_new = list(
            set(self.current_tile_positionInField[1].copy())
        )

        # 5.: Putting the swapped tile into the correct position in the field
        self.field[
            min(current_tile_positionInField_rows_set_new) : max(
                current_tile_positionInField_rows_set_new
            )
            + 1,
            min(current_tile_positionInField_columns_set_new) : max(
                current_tile_positionInField_columns_set_new
            )
            + 1,
        ] = tile_t

        # TODO:
        # asserting that the current tile in the field now actually has the correct shape

        print("-----Rotate fully executed-----")
        return True

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

    def visualize_field(self):
        # TODO: Finish with constantly updating plot
        plt.figure(figsize=(10, 10))

        plt.scatter(x=self.field, color="red")

        plt.show(block=True)




    def _tiles_queue_pop_left(self) -> list[str, np.ndarray]:
        """
        Pops the first resp. leftmost element of 'self.tiles_queue'
        and returns it.

        Returns:
            (list): The leftmost element of 'self.tiles_queue'
                    holding "[Name of the tile, the tiles' array]"
        
        Raises:
            EmptyContainerError: If 'self.tiles_queue' is empty
                                 before attempting to retrieve
                                 the leftmost element.
        """
        if not self.tiles_queue:
            raise EmptyContainerError

        return self.tiles_queue.popleft()