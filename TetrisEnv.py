import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import os
import sys
from collections import deque

plt.ion()




class TetrisEnv():
    def __init__(self, field_height, field_width, len_tiles_queue):
        self.field_height = field_height
        self.field_width = field_width
        self.len_tiles_queue = len_tiles_queue #the number of tiles, which will come next and which are already known/selected

        self.lauch_position = [0, math.floor(field_width / 2)]

        self.field = np.zeros(shape=(field_height, field_width), dtype=np.int8)

        self.tiles = {"I": np.ones((4, 1)),
                      "O": np.ones((2, 2)),
                      "S": np.array([[0, 1], [1, 1], [1, 0]]),
                      "S_inv": np.array([[1, 0], [1, 1], [0, 1]]),
                      "L": np.array([[1, 0], [1, 0], [1, 1]]),
                      "L_inv": np.array([[0, 1], [0, 1], [1, 1]]),
                      "T": np.array([[0, 1, 0], [1, 1, 1]])
                      }
        
        #holding the tile(s), that are about to come next
        #for every tile, holds a list of "[Name of the tile, the tiles´ array]"
        self.tiles_queue = deque()
        #initially populating the tiles_queue
        self.populate_tiles_queue()

        self.current_tile = None #hold the currently launched tile (holds a list of "[Name of the tile, the tiles´ array, the tiles´ rotation]")
                                 #Explanation regarding the rotation of the tile:
                                 #The roation can be four values:
                                 #* "0" = not rotated (i.e. in the initial launch position)
                                 #* "1" = rotated by 90 degrees to the right
                                 #* "2" = rotated by 180 degrees (to the right)
                                 #* "3" = rotated by 270 degrees (to the right)

        self.current_tile_positionInField = [[], []] #a list of two elements: A list holding the row-indices
                                                     #                        and a list holding the column indices
                                                     #NOTE: zipping the two lists (i.e. pairing them element-wise)
                                                     #      gives the exact coordinates of the cells in the field,
                                                     #      where the tile is currently.
        
    
    def populate_tiles_queue(self):
        while len(self.tiles_queue) < self.len_tiles_queue:
            self.tiles_queue.append([*random.choice(list(self.tiles.items())), 0])
    
    def launch_tile(self):
        #popping the first tile from the tiles_queue and assigning it as/to the current_tile
        self.current_tile = self.tiles_queue.popleft()

        #since now one tile was removed from the tiles_queue,
        #the tiles queue is populated again
        self.populate_tiles_queue()

        #defining the shape of the array 'current_tile_positionInField'
        #based on the current_tile
        # self.current_tile_positionInField = np.zeros(shape=(len(self.current_tile[1]), len(self.current_tile[1][0])))

        #cleaning the data (rows and columns) in 'current_tile_positionInField' at this point,
        #so it can be freshly assigned for the now launched tile in the loop below
        self.current_tile_positionInField[0].clear()
        self.current_tile_positionInField[1].clear()

        #putting the current_tile into the field
        for n_row in range(len(self.current_tile[1])): #iterating over the number of rows of the tile
            for n_column in range(len(self.current_tile[1][0])): #iterating over the number of columns of the tile
                self.field[self.lauch_position[0] + n_row, self.lauch_position[1] + n_column] = self.current_tile[1][n_row, n_column]

                #assigning the position of the current_tile in the field to 'current_tile_positionInField'
                self.current_tile_positionInField[0].append(self.lauch_position[0] + n_row) #adding the row
                self.current_tile_positionInField[1].append(self.lauch_position[1] + n_column) #adding the column


    def drop(self) -> bool:
        '''
        Drops the current tile in the field by one row.

        Returns:
            -A boolean indicating if a drop was possible and thus conducted or not.
        '''
        #retaining the old 'current_tile_positionInField'-variable before it is updated below
        current_tile_positionInField_old = self.current_tile_positionInField.copy()

        #Checking if a drop is possible
        #NOTE: A drop is only possible if in the row below the current lowest row of the tile
        #      in all columns of the tile there is space in the field (i.e. there are only zeros)
        #      Easily said: This is the row, where the tile will drop to/in, and this has to be
        #                   empty in order for a drop to be possible.
        #TODO: Das Droppen funktioniert bisher nur beim allerersten tile, das nächste wird dann nicht mehr (korrekt) gedroppt.

        if max(current_tile_positionInField_old[0]) + 1 == self.field_height: #the drop isn´t possible anymore because the tile currently is already at the lowest existing row in the field
            return False
        else:
            drop_possible = all(self.field[max(current_tile_positionInField_old[0])+1, column] == 0 for column in range(min(current_tile_positionInField_old[1]), max(current_tile_positionInField_old[1])+1, 1))

        if not drop_possible:
            return False


        #First updating the variable "current_tile_positionInField" by increasing
        #all row-numbers by one (i.e. the tiles moves downward by one row)
        self.current_tile_positionInField[0] = [row+1 for row in self.current_tile_positionInField[0]]

        


        #Updating the tile in the field (i.e. doing the actual dropping)
        # #creating coordinates of the old tile-position by zipping the rows and columns
        # tile_coordinates_old = list(zip(current_tile_positionInField_old[0], current_tile_positionInField_old[1]))
        # #creating coordinates of the new tile-position by zipping the rows and columns
        # tile_coordinates_new = list(zip(self.current_tile_positionInField[0], self.current_tile_positionInField[1]))

        for n_row_new in range(max(self.current_tile_positionInField[0]), min(self.current_tile_positionInField[0])-1, -1): #iterating backwards over all the (new) rows where the dropped tile will be positioned
            for n_column in range(min(self.current_tile_positionInField[1]), max(self.current_tile_positionInField[1])+1, 1): #iterating over the columns in the field
                #assigning the correct number of the tile (0 or 1) to the respective cell,
                #depending on which number the tile has at that position of it´s grid
                #(i.e. in the field now still at one row up)
                self.field[n_row_new, n_column] = np.int8(0) if self.field[n_row_new-1, n_column] == 0 else np.int8(1)
        
        #emptying (i.e. assigning 0s) the topmost row of the old tile-position in the field,
        #because those cells now got empty because the tile dropped down by one row now
        for n_column in range(min(self.current_tile_positionInField[1]), max(self.current_tile_positionInField[1])+1, 1):
            self.field[min(current_tile_positionInField_old[0]), n_column] = np.int8(0)
        
        return True
 
    def check_for_and_handle_full_row(self) -> int:
        '''
        Checks whether one row or multiple rows is/are full,
        i.e. contain(s) only 1s.
        If so, this row/those rows is/are removed, and all tiles above move
        down by the number of full rows (one or more).
        
        Both the check whether one or multiple rows are full as well as
        the dropping of the other 1s in the field (except for the current tile)
        is handles by this function.

        Returns:
            An int, indicating how many rows were full and were thus removed.
        '''
        #a list holding the indices of full rows
        indices_full_rows = []
        #Iterating through all the rows and saving the indices of full rows
        for i, row in enumerate(self.field):
            if all(row == 1):
                indices_full_rows.append(i)
        
        if len(indices_full_rows) == 0:
            return 0

        #removing the full rows from the field
        self.field = np.delete(arr=self.field, obj=indices_full_rows, axis=0)

        #adding as many new (i.e. empty) rows to the top of the field as were just deleted
        self.field = np.vstack((np.zeros(shape=(len(indices_full_rows), self.field_width), dtype=np.int8), self.field))

        return len(indices_full_rows)





    def visualize_field(self):
        #TODO: Finish with constantly updating plot
        plt.figure(figsize=(10, 10))

        plt.scatter(x=self.field, color='red')

        plt.show(block=True)
    

    def rotate(self):
        pass
