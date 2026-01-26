import tkinter as tk
import torch
import sys 
import os
sys.path.insert(0,"../")
print(os.path.abspath("../"))
from chessenv import ChessEnv
from model import ChessModel
from mcts import AlphaMCTS
from board import ChessBoard



class ChessGame:
    def __init__(self, vs_ai):
        self.vs_ai = vs_ai
        self.env = ChessEnv()

        self.device = "cpu"
        self.model = None
        self.mcts = None

        if vs_ai:
            self.load_ai()

        self.root = tk.Tk()
        self.root.title("Chess")

        self.board = ChessBoard(self.root, self)
        self.board.pack()

        self.root.mainloop()

    def load_ai(self):
        self.model = ChessModel().to(self.device)
        self.model.load_state_dict(torch.load("model_0.pt", map_location=self.device))
        self.model.eval()
        self.mcts = AlphaMCTS(self.model, self.device)

    def human_move(self, move):
        self.env.push(move)
        self.board.draw()

        if self.vs_ai and not self.env.is_terminal():
            self.root.after(200, self.ai_move)

    def ai_move(self):
        pi = self.mcts.search(self.env)
        move = max(pi, key=pi.get)
        self.env.push(move)
        self.board.draw()
