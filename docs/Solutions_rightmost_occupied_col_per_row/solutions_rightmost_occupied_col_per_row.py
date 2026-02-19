import numpy as np
import pytest
from collections import defaultdict


"""
In this file, I practiced different ways to implement the problem described in
'Solutions_rightmost_occupied_col_per_row.docx' and learned various things
along the way, which I documented under "Learned" below.
"""


"""
Learned:
1.: Going from x = [(1,6), (2,8)] to [(1,2), (6,8)]:
    --> rows, cols = zip(*x) (basically it's like a Matrix-Transpose)

2.: Creating an array in the form "np.array([[1,2], [6,8]])" from x = [(1,6), (2,8)]:
    -->np.array(x).T
    -->np.stack(x, axis=1)
        -->https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        -->What does np.stack do?:
           - It takes a Series multiple array-like containers (held in one container)
             (e.g. a list containing two tuples, each containing two numbers).
             The array-like containers all need to have the same shape (the tuples
             in this example both had shape (2,)).
           - Unter the hood, np.stack then converts all containers to a numpy-array:
             "arrays = [np.asarray(x) for x in coords]"
           - When then doing "np.stack(arrays, axis=x)", the different containers (now np.ndarrays)
             all held in the list "arrays" are stacked along the specified axis.
             -->The result is one new array.

3.: Creating a list of tuples in the form "[(1,6), (2,8)]" from "x = np.array([[1,2], [6,8]])"
    (i.e. simply reversing the operation of 2.: above).
    -->"list(map(tuple, x.T))".
        - How does "map" work?:
            *Formal definition:
                '
                map(function, iterable)
                '
                
                >Takes one callable
                >Takes one or more iterables
                >Returns an iterator
                >Applies the callable to each element produced by the iterable(s)
                
                Conceptually:
                '
                map(f, [x1, x2, x3])  →  f(x1), f(x2), f(x3)
                '
                
                Which is equivalent to:
                '(f(x) for x in iterable)'
        
        - Why can we pass tuple to map (I wondered, because "tuple" is a class, and not a function)?
            *Yes—tuple is a class, but more importantly:
                >Classes are callables in Python
                
                When you write:

                '
                tuple(x)
                '
                
                you are calling the constructor, exactly like:
                
                '
                int("42")
                list([1, 2, 3])
                np.array([1, 2, 3])
                '

4.: np.unique():
    -->https://numpy.org/doc/2.4/reference/generated/numpy.unique.html
    -->Finds the unique elements of an array (+3 optional outputs)
    -->returns a np.ndarray containing the unique values
"""





rows = [1,2,2,2]
cols = [6,6,7,8]

#Goal: Find the rightmost occupied cells per row and represent them both as NumPy-array and as tuples in a list.
#      The solution is:
#       - np.array([[1,2],[6,8]])
#       - [(1,6), (2,8)]


#---Solution 1: Pure Python--------------

#create a defaultdict to avoid KeyErrors
d = defaultdict(int)

#looping over the single elements (pairwise):
for row, col in zip(rows, cols):
    #grouping based on individual rows; calculating a running max of the column-indices per row/group
    d[row] = max(d[row], col)

#obtaining the key-value-pairs from 'd' and creating a list of tuples from it
list_of_tuples_1 = list(d.items())

#creating an array of the desired shape from the list of tuples
array_1 = np.stack(list_of_tuples_1, axis=1)
#----------------------------------------

#---Solution 2.1: As much NumPy as possible
#General idea: using a mask to obtain the values per row/group from cols.

#Since we`re working with NumPy heavily, we first create numpy-arrays from the initial lists.
rows = np.asarray(rows)
cols = np.asarray(cols)

#Obtaining the unique values from row --> i.e. the unique groups
unique_rows = np.unique(rows)

#looping over the individual rows
max_cols = []
for row in unique_rows:
    #creating a mask to get the indices of "rows" where "row" stands
    mask = row == rows
    #retrieving the corresponding values from cols
    c = cols[mask]
    #getting the maximum column-value from "c"
    max_c = max(c)
    #appending max_c to "max_cols"
    max_cols.append(max_c)

#stacking the unique rows and the corresponding max_cols into one array along axis 0, i.e. along the rows
array_2_1 = np.stack((unique_rows, max_cols), axis=0)

#creating the coordinate version of the result
list_of_tuples_2_1 = list(map(tuple, array_2_1.T)) #or 'list_of_tuples_2 = [tuple(np.int8(x)) for x in array_2.T]'
#----------------------------------------

#---Solution 2.1: As much NumPy as possible
#General idea: using a mask to obtain the values per row/group from cols.

#Since we`re working with NumPy heavily, we first create numpy-arrays from the initial lists.
rows = np.asarray(rows)
cols = np.asarray(cols)

#Obtaining the unique values from row --> i.e. the unique groups
unique_rows = np.unique(rows)

#creating an array containing the maximum column-value per row (i.e. per group)
#by using a mask in combination with a list-comprehension, which is directly
#embedded into the np.array
max_cols = np.array([cols[row==rows].max() for row in unique_rows])

#stacking the unique rows and the corresponding max_cols along axis 0
array_2_2 = np.stack([unique_rows, max_cols], axis=0) #'array_2_2 = np.vstack([unique_rows, max_cols])'

#creating the coordinate version of the result
list_of_tuples_2_1 = list(map(tuple, array_2_1.T)) #or 'list_of_tuples_2 = [tuple(np.int8(x)) for x in array_2.T]'



