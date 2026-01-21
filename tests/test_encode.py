import pytest
import chess
import random
from decoder import move_to_index, index_to_move, PROMOTION_PIECES, KNIGHT_DIRS, DIRS

# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def empty_board():
    board = chess.Board()
    board.clear_board()
    return board

@pytest.fixture
def starting_board():
    return chess.Board()

# ----------------------------
# Parametrized Tests
# ----------------------------

@pytest.mark.parametrize("uci", [
    "e2e4", "d2d4", "g1f3", "b1c3", "f2f4", "c2c3"
])
def test_normal_moves(starting_board, uci):
    move = chess.Move.from_uci(uci)
    index = move_to_index(move)
    decoded = index_to_move(index, starting_board)
    assert decoded == move

@pytest.mark.parametrize("uci", [
    "e7e8q", "e7e8r", "e7e8b", "e7e8n",
    "a2a1q", "h2h1r", "c7c8b", "g7g8n"
])
def test_promotions(empty_board, uci):
    # Place pawns for promotion
    board = empty_board
    
    from_sq = chess.parse_square(uci[:2])
    to_sq = chess.parse_square(uci[2:4])
    board.set_piece_at(from_sq, chess.Piece(chess.PAWN, chess.WHITE if from_sq>=48 else chess.BLACK))
    board.turn = chess.WHITE if from_sq >= 48 else chess.BLACK
    move = chess.Move.from_uci(uci)
    print(f'Testing on board:\n{board}')
    print(f'Testing promotion move: {move}')    
    index = move_to_index(move)
    decoded = index_to_move(index, board)
    print(f'index is {index}, decoded move is {decoded}')
    assert decoded == move

@pytest.mark.parametrize("uci", [
    "f7f8q", "f7f8r", "f7f8b", "f7f8n",
    "f7e8q", "f7e8r", "f7e8b", "f7e8n",
    "f7g8q", "f7g8r", "f7g8b", "f7g8n",
    "e2e1q", "e2e1r", "e2e1b", "e2e1n",
    "e2d1q", "e2d1r", "e2d1b", "e2d1n",
    "e2f1q", "e2f1r", "e2f1b", "e2f1n"
])
def test_custum_promotions(uci):
    board = chess.Board("4b1b1/5P2/8/4K3/2k5/8/4p3/3B1B2 w - - 0 1")
    move = chess.Move.from_uci(uci)
    print(f'testing on board:\n{board}')
    print(f'Testing promotion move: {move}')  
    index = move_to_index(move)
    decoded = index_to_move(index, board)
    print(f'index is {index}, decoded move is {decoded}')
    assert decoded == move

@pytest.mark.parametrize("uci", [
    "g1f3", "b1c3", "g8f6", "b8c6"
])
def test_knight_moves(starting_board, uci):
    move = chess.Move.from_uci(uci)
    index = move_to_index(move)
    decoded = index_to_move(index, starting_board)
    assert decoded == move

@pytest.mark.parametrize("uci", [
    "e1g1", "e1c1", "e8g8", "e8c8"
])
def test_castling_moves(starting_board, uci):
    move = chess.Move.from_uci(uci)
    index = move_to_index(move)
    decoded = index_to_move(index, starting_board)
    assert decoded.from_square == move.from_square
    assert decoded.to_square == move.to_square

@pytest.mark.parametrize("dx,dy", DIRS)
def test_sliding_moves_empty_board(empty_board, dx, dy):
    # place a rook in the center
    board = empty_board
    board.set_piece_at(chess.D4, chess.Piece(chess.ROOK, chess.WHITE))
    fx, fy = 3,3
    for dist in range(1,8):
        to_x = fx + dx*dist
        to_y = fy + dy*dist
        if 0 <= to_x <= 7 and 0 <= to_y <= 7:
            from_sq = chess.square(fx, fy)
            to_sq = chess.square(to_x, to_y)
            move = chess.Move(from_sq, to_sq)
            index = move_to_index(move)
            decoded = index_to_move(index, board)
            assert decoded.from_square == move.from_square
            assert decoded.to_square == move.to_square

@pytest.mark.parametrize("dx,dy", KNIGHT_DIRS)
def test_knight_moves_empty_board(empty_board, dx, dy):
    board = empty_board
    fx, fy = 4,4
    board.set_piece_at(chess.square(fx,fy), chess.Piece(chess.KNIGHT, chess.WHITE))
    to_x = fx+dx
    to_y = fy+dy
    if 0 <= to_x <= 7 and 0 <= to_y <= 7:
        from_sq = chess.square(fx,fy)
        to_sq = chess.square(to_x,to_y)
        move = chess.Move(from_sq,to_sq)
        index = move_to_index(move)
        decoded = index_to_move(index, board)
        assert decoded.from_square == move.from_square
        assert decoded.to_square == move.to_square

def test_off_board_index_returns_none(empty_board):
    board = empty_board
    index = 8*73 + 1000  # deliberately invalid
    result = index_to_move(index, board)
    assert result is None

# ----------------------------
# Fuzz Test
# ----------------------------

def test_random_legal_moves_reversibility():
    board = chess.Board()
    for _ in range(100):
        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)
        index = move_to_index(move)
        decoded = index_to_move(index, board)
        if decoded is not None:
            assert decoded.from_square == move.from_square
            assert decoded.to_square == move.to_square
            # Promotions handled if present
            if move.promotion:
                assert decoded.promotion == move.promotion
        board.push(move)
