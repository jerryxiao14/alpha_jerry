import tkinter as tk
import chess
from PIL import Image, ImageTk
import os

SQUARE_SIZE = 64

class ChessBoard(tk.Canvas):
    def __init__(self, parent, game, piece_folder="assets"):
        super().__init__(parent, width=8*SQUARE_SIZE, height=8*SQUARE_SIZE)
        self.game = game
        self.selected_square = None
        self.highlighted_squares = []

        # Load piece images
        self.piece_images = self.load_piece_images(piece_folder)

        self.bind("<Button-1>", self.on_click)
        self.draw()

    def load_piece_images(self, folder):
        images = {}
        for color in ["white", "black"]:
            for piece in ["king", "queen", "rook", "bishop", "knight", "pawn"]:
                filename = f"{color}_{piece}.png"
                path = os.path.join(folder, filename)
                
                img = Image.open(path).convert("RGBA")  # <-- keep transparency
                img = img.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
                
                images[f"{color}_{piece}"] = ImageTk.PhotoImage(img)
        return images

    def draw(self):
        """Draw board and pieces."""
        self.delete("all")
        board = self.game.env.board

        for rank in range(8):
            for file in range(8):
                x1 = file * SQUARE_SIZE
                y1 = rank * SQUARE_SIZE
                x2 = x1 + SQUARE_SIZE
                y2 = y1 + SQUARE_SIZE

                # Board colors
                color = "#f0d9b5" if (rank + file) % 2 == 0 else "#b58863"
                self.create_rectangle(x1, y1, x2, y2, fill=color)

        # Highlight possible moves if a piece is selected
        for sq in self.highlighted_squares:
            rank = 7 - chess.square_rank(sq)
            file = chess.square_file(sq)
            x1 = file * SQUARE_SIZE
            y1 = rank * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            self.create_rectangle(x1, y1, x2, y2, outline="yellow", width=3)

        # Draw pieces
        for square, piece in board.piece_map().items():
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)
            key = f"{'white' if piece.color else 'black'}_{chess.piece_name(piece.piece_type)}"
            self.create_image(file*SQUARE_SIZE, rank*SQUARE_SIZE, image=self.piece_images[key], anchor="nw")

    def on_click(self, event):
        file = event.x // SQUARE_SIZE
        rank = 7 - (event.y // SQUARE_SIZE)
        square = chess.square(file, rank)
        board = self.game.env.board

        if self.selected_square is None:
            # Select a piece
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                self.selected_square = square
                # Highlight legal moves
                self.highlighted_squares = list(board.attacks(square)) + [square]
        else:
            # Attempt to move
            move = chess.Move(self.selected_square, square)
            if move in board.legal_moves:
                self.game.human_move(move)
            # Reset selection
            self.selected_square = None
            self.highlighted_squares = []

        self.draw()
