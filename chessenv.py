
import chess
import numpy as np

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()
    
    def copy(self):
        env = ChessEnv()
        env.board = self.board.copy(stack = True)
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

    # encodes current position from perspective of current player
    def encode(self):
        '''
        AlphaZero style encoding

        Returns:
            planes: np.ndarray of shape (21, 8 ,8)
        
        :param self: Description
        '''
        
        planes = np.zeros((21,8,8), dtype = np.float32)

        # Piece planes 
        # Planes 0-5: p1 pieces
        # Planes 6-11: p2 pieces
        # Piece type enums directly correspond to plane
        for square, piece in self.board.piece_map().items():
            row = 7-chess.square_rank(square)
            col = chess.square_file(square)

            if piece.color == self.board.turn:
                planes[piece.piece_type-1, row, col]=1.0
            else:
                planes[piece.piece_type+5,row,col]=1.0

        # repetition planes 12-13
        # plane 12 represents repeated once 
        # plane 13 represents repeated twice

        if self.board.is_repetition(1):
            planes[12, :, :] = 1.0
        if self.board.is_repetition(2):
            planes[13, :, :] = 1.0

        
        # plane 14: all 1s if its white
        if self.board.turn:
            planes[14,:,:]=1.0
        
        # plane 15: move count normalized by /100
        planes[15,:,:] = min(1,self.board.fullmove_number/100)

        # plane 16-17: current player has castling rights
        if self.board.has_kingside_castling_rights(self.board.turn):
            planes[16,:,:]=1.0
        if self.board.has_queenside_castling_rights(self.board.turn):
            planes[17,:,:]=1.0
        
        # plane 18-19: opponent has castling rights
        opponent = not self.board.turn
        if self.board.has_kingside_castling_rights(opponent):
            planes[18,:,:]=1.0
        if self.board.has_queenside_castling_rights(opponent):
            planes[19,:,:]=1.0

        # plane 20: halfmove clock normalized by /100

        planes[20,:,:]= min(1,self.board.halfmove_clock/100)

        return planes
        

        



            
            

            
