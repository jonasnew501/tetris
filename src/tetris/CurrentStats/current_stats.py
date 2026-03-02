
class CurrentStats():
    """
    This class holds data and functionalities which are related to a running
    tetris session.
    A tetris session does not only refer to one single game, but to
    a series of tetris games played in one session, i.e. in one
    execution of the program.
    """
    def __init__(self):
        self.n_games_played = 0
        self.n_timesteps_conducted = 0
        self.n_rows_cleared = 0

        self.number_of_rows_cleared_at_once = {1: 0,
                                               2: 0,
                                               3: 0,
                                               4: 0}
    
    def increment_n_games_played(self):
        self.n_games_played += 1
    
    def increment_n_timesteps_conducted(self):
        self.n_timesteps_conducted += 1
    
    def update_n_rows_cleared(self, rows_cleared: int):
        self.n_rows_cleared += rows_cleared
        self.update_number_of_rows_cleared_at_once(rows_cleared=rows_cleared)
    
    def update_number_of_rows_cleared_at_once(self, rows_cleared):
        self.number_of_rows_cleared_at_once[rows_cleared] += 1
