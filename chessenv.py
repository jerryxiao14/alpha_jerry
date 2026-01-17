
import chess

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
    
    def copy(self):
        env = ChessEnv()
        env.board = self.board.copy(stack=False)
        return env
    def legal_moves(self):
        return list(self.board.legal_moves)
    
    def push(self,move):
        self.board.push(move)
    
    def is_terminal(self):
        return self.board.is_game_over()
    
    def result(self):
        if not self.board.is_game_over():
            return None 
        outcome = self.board.outcome()
        #print(f'outcome is {outcome} and outcome winner is {outcome.winner}')
        if outcome.winner is None:
            #print(f'returning 0')
            return 0
        
        # in python chess true refers to white false refers to black
        return 1 if outcome.winner else -1

