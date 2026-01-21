import chess 
import numpy as np 
import config

# Directions for sliding pieces

PROMOTION_PIECES = [chess.ROOK, chess.KNIGHT, chess.BISHOP]
DIRS = [
    (1,0), (-1,0), (0,1), (0,-1),
    (1,1), (1,-1), (-1,1), (-1,-1)
]

KNIGHT_DIRS = [
    (2,1), (2,-1), (-2,1), (-2,-1),
    (1,2), (1,-2), (-1,2), (-1,-2)
]

def sign(x):
    if x>0:
        return 1 
    elif x<0:
        return -1 
    else:
        return 0



def move_to_index(move: chess.Move) -> int:
    # maps a chess move to index in [0,4671]
    from_sq = move.from_square 
    to_sq = move.to_square 

    fx,fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    tx,ty = chess.square_file(to_sq), chess.square_rank(to_sq) 

    dx = tx-fx 
    dy = ty-fy 

    if move.promotion and move.promotion != chess.QUEEN: 
        direction = dx+1 
        promo_idx = PROMOTION_PIECES.index(move.promotion)
        plane = 64+direction*3+promo_idx 
    elif (dx,dy) in KNIGHT_DIRS:
        plane = 56 + KNIGHT_DIRS.index((dx,dy))
    else:
        step_dx = 0 if dx==0 else dx//abs(dx)
        step_dy = 0 if dy==0 else dy//abs(dy)

        direction = DIRS.index((step_dx,step_dy))
        dist = max(abs(dx),abs(dy))
        plane = direction*7 + (dist-1)

    return from_sq * 73 + plane 

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    from_sq = index // 73 
    plane = index %73 

    #print(f'plane is {plane} square is {from_sq} which is {chess.square_name(from_sq)}')

    piece = board.piece_at(from_sq)
    if piece is None:
        #print(f'No piece at from_sq {chess.square_name(from_sq)}')
        return None
    #print(f'current piece at from_sq is {piece} color is {("W" if piece and piece.color==chess.WHITE else "B" if piece else "None")}')
    fx,fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    
    if plane>=64:
        direction = (plane-64)//3 
        promo_idx = (plane-64)%3
        #print(f'underpromotion direction is {direction} promo idx is {promo_idx} which corresponds to {PROMOTION_PIECES[promo_idx]}')
        dx = direction-1 
        dy = 1 if piece.color == chess.WHITE else -1

        to_sq = chess.square(fx+dx,fy + dy)
        #print(f'dx is {dx} dy is {dy} to_sq is {fx+dx},{fy + dy} which is {chess.square_name(chess.square(fx+dx,fy + dy))}')
        promotion = PROMOTION_PIECES[promo_idx]
        if fx+dx<0 or fx+dx>7 or fy+dy<0 or fy+dy>7 or piece is None or piece.piece_type != 1:
            #print(f'out of bounds or invalid')
            return None
        return chess.Move(from_sq, to_sq, promotion=promotion)
    elif plane>=56:
        knight_idx = plane-56 
        dx,dy = KNIGHT_DIRS[knight_idx]
        to_sq = chess.square(fx+dx,fy+dy)
        if fx+dx<0 or fx+dx>7 or fy+dy<0 or fy+dy>7:
            return None
        return chess.Move(from_sq,to_sq)
    else:
        direction = plane//7
        dist = (plane%7)+1

        step_dx,step_dy = DIRS[direction]
        dx = step_dx * dist
        dy = step_dy * dist

        #print(f'direction is {direction} which is {step_dx}, {step_dy} dist is {dist} resulting in dx {dx} dy {dy}')

        to_rank = fy+dy 
        #print(f'to_rank is {to_rank} piece is {piece} piecetype is {piece.piece_type if piece else "None"} which corresponds to {chess.piece_name(piece.piece_type) if piece else "None"}')
        if piece is not None and piece.piece_type == 1 and (to_rank==0 or to_rank==7):
            promotion = chess.QUEEN 
        else:
            promotion = None 
        #print(f'promotion is {promotion}')

        # check if last rank and pawn move for queen promotion
        to_sq = chess.square(fx+dx,fy+dy)
        if fx+dx<0 or fx+dx>7 or fy+dy<0 or fy+dy>7:
            return None
        return chess.Move(from_sq,to_sq,promotion=promotion)
    

