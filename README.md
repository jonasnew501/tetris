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
*This section will be written soon.*
<!--
XXXXXX
-->
