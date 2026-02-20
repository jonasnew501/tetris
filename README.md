## About this project
**This project comprises a full implementation of the game 'Tetris'**.  

Emphasis is put on adhering to **clean-code** practices as taught in the book 'Clean Code - A Handbook of Agile Software Craftsmanship' by Robert C. Martin.  

Furthermore, the **Reinforcement Learning** algorithm 'PPO' ('Proximal Policy Optimization') is planned to be implemented and an RL-agent shall be trained to master the game.  

For more complicated functions, a **unittest-suite** was implemented using *'pytest'* and *'pytest-mock'*.
Automatic **CI** ('Continuous Integration') via GitHub Actions is implemented (for the unittests and auto-checking for correct code formatting).

The purpose of this project is to practice the use of NumPy and PyTorch, and practice building a more large-scale application.


## Structure of this project
### Content
In the following a short overview about specific files is given:  
- *'src/tetris/**tetris_env.py**'*: The class *TetrisEnv* fully implements the Tetris-environment.  
This comprises all components of the game such as the playing field, all tetrominos/tiles,  
as well as all logic of the game.
<br>

- *'src/tetris/**main.py**'*: This file acts as the entry-point of the application by bringing  
together all the logic-functionality of *TetrisEnv* in a sensible manner.  
Furthermore, it enables the collection and processing of keyboard-input from a user via the library *Pygame*.
<br>

- *'src/tetris/**tetris_env_domain_specific_exceptions.py**'*: Contains various custom exceptions used in the class *TetrisEnv*.  
Creating and using custom, domain-specific exceptions is a clean-code practice.  

The folder **'github'** contains yaml-files for *actions* and *workflows* on ***GitHub Actions***.  

### Folder-structure
The folderstructure of this project follows a **src-based structure/-layout**.  
The general structure looks like this:
```cpp
project-root/
├── src/
│   └── package-name/
│       ├── __init__.py
│       └── hyperparameters.py
│       └── main.py
│       └── tetris_env_domain_specific_exceptions.py
│       └── tetris_env.py

├── tests/
│       └── test_tetris_env.py
├── dist/
│   └── tetris.exe
├── pyproject.toml
└── README.md
└── requirements.txt
└── .gitignore
└── .github/
```

The 'src'-folder contains the source-code, the 'tests'-folder contains the testing-code.  


## Status overview
![Unittests](https://github.com/jonasnew501/tetris/actions/workflows/unittests.yml/badge.svg)
![Check for correct code-format](https://github.com/jonasnew501/tetris/actions/workflows/formatting_and_linting.yml/badge.svg)

## Tech stack
**Language**  
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)  

**Programming Paradigm**  
![OOP](https://img.shields.io/badge/OOP-Object%20Oriented%20Programming-4CAF50)  

**Data Science & Machine Learning**  
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing%2C%20Vectorized%20Operations-013243?logo=numpy&logoColor=white)  
<!--
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
-->

**Tools & Workflow**  
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git&logoColor=white)  
![GitHub](https://img.shields.io/badge/GitHub-Repos-181717?logo=github&logoColor=white)  

**Testing & CI**  
![PyTest](https://img.shields.io/badge/Testing-PyTest-46375B?logo=pytest&logoColor=white)  
![Pytest-Mock](https://img.shields.io/badge/Testing-Pytest--Mock-6A5ACD?logo=pytest&logoColor=white)  
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)  

<!--
**Packaging & Deployment**  
![PyInstaller](https://img.shields.io/badge/Build%20Tool-PyInstaller-3776AB?logo=python&logoColor=white)  
-->

**Code Formatting**  
![Black](https://img.shields.io/badge/Auto%20Formatting-Black-000000?logo=python&logoColor=white)

**Other Topics Studied**  
![Clean Code](https://img.shields.io/badge/Reading-Clean%20Code-000000)  





## What I Learned

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
        
        - Why can we pass a tuple to map (I wondered, because "tuple" is a class, and not a function)?
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

5.: Copying the content of a file to another file, which is created newly too.
    Example: Copying the contents of file "A.py" to "B.py" (and first creating "B.py"):
    
    -->(On Windows): "Copy-Item path/to/A.py path/to/B.py"
        -->If "B.py" exists already and shall be overwritten: "Copy-Item path/to/A.py path/to/B.py -Force"
    -->(On Linux): "cp path/to/A.py path/to/B.py"

6.: Deleting the complete content of a fiel, such that the file still exists, but is completely empty.
    Example: Deleting all contents of file "A.py":

    -->(On Windows): "Clear-Content path/to/A.py"
    -->(On Linux) (there are multiple different ways to achieve the same result): "echo -n > path/to/A.py"

7.: Get the absolute number of elements in an array (i.e. all axes taken together, as if the array was flattened):
    -->"array.size"
    -->The same as "np.prod(array.shape)"
    -->https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html

8.: Passing multiple positional arguments to the Python builtIn-function "max":
    -->If multiple positional arguments to the Python builtin-function "max" are passed,
       the largest of them is returned.
       I.e. max does not only take a container (like a list for example) are returns
       the maximum element on that container.
    
    -->If two or more iterable with ordered elements are passed to "max",
       the iterables are compared lexicographically (exactly like words in a dictionary).

       That means:
       E.g.: Two lists are compared:

            >element by element
            >from left to right
            >the first differing element decides
            >if all elements match so far, the longer list wins
        
        -->E.g.:
            '
            x = [1,2,3,4]
            y = [4,4,2]

            z = max(x, y)

            print(z)
            '
        
            -->Result: "[4,4,2]"
                -->Why? (One could think "x" would be determined to be larger, because x is longer.):
                    
                    Comparison process:
                    Position:	x:	y:	Result:
                    index 0	    1	4	4 > 1 → y wins immediately

                    Python stops here.
                    It never looks at 2 vs 4, 3 vs 2, or list length.
        
        -->However, in order to compare the iterables according to another rule than lexicographically,
           a "key" must be passed, which needs to be a callable. A callable is something that can be called like a function with one argument."
            -->e.g.:
                >"max(x, y, key=sum)" --> The iterable having the bigger sum is determined to be bigger.
                >"max(x, y, key=len)" --> The iterable containing more elements is determined to be bigger.
            
            -->e.g.:
                >x = [1, 2, 3, 4]
                 y = [4, 4, 2]

                 # Compare by last element in the list
                 z = max(x, y, key=lambda lst: lst[-1])
                 print(z)  # Output: [1, 2, 3, 4] (last element 4 > 2)
            
                >x = [1, 2, 3, 4]
                 y = [4, 4, 2]

                 def product(lst):
                     result = 1
                     for v in lst:
                         result *= v
                     return result

                 z = max(x, y, key=product)
                 print(z)  # Output: [1, 2, 3, 4] (product = 24 > 32)

9.: "defaultdict":
    -->A defaultdict is a subclass of the standard Python dict.
    -->imported via "from collections import defaultdict"
    -->The idea is that a defaultdict takes a "default_factory", which needs to be a callable expecting no arguments (which is expected to return a value).
       When a key is accessed, which is not present in that defaultdict, instead of raising a "KeyError" as a standard "dict", the default_factory-callable
       is called, that key is created and the value retrned by the default_factory will is assigned as the value of that key.
    

10.: Number 9 above leads to the question: What is a "callable" in Python?
     - object "x" is a callable when "callable(x) == True".
        --> But when is "callable(x) True"?
            -->More precisely:
               Something (some object) is a callable, when this Object implements the/a "call protocol".
               A "call protocol" can be implemented/satisfied via two different ways:
                    1.: The Object itself implements a/the "__call__()"-method.
                        -->E.g.:
                           '
                           class F:
                              def __call__(self):
                                 return 42
                        
                           f = F()
                           f() #calls f.__call__()
                           '
                           (--> Here, the instance is callable. <- explained later)
                    
                    2.: The object is a class.
                        -->Classes are callables "automatically", also when they themselves don't define "__call__()".
                            -->Why?: Because classes in Python automatically inherit from the metaclass "type", which
                                     in turn does implement "__call__()".

                    
                    -->In general:
                       - Instances are callables if their class defines "__call__()".
                       - Classes are callables because their metaclass("type") defines "__call__()".
                    
                    -->Question: Why are def-functions and lambda-functions callables?
                        -->Because both def-functions and lambda-functions are instances of the class "function",
                           which in turn implements the call protocol in C.
        
        -->Summary/Overview of what is a callable in Python:
            An object is a callable if Python knows how to execute obj().

                -->This happens if:

                    Object kind	    |    Why it's callable

                    Function	    |    Function type implements call protocol
                    Lambda	        |    Same as function
                    Class	        |    Metaclass (type) implements __call__
                    Instance	    |    Its class defines __call__
