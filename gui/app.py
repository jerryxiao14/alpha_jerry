import tkinter as tk
from game import ChessGame

class MainMenu:
    def __init__(self, root):
        self.root = root
        root.title("AlphaZero Chess")

        tk.Label(root, text="AlphaZero Chess", font=("Arial", 20)).pack(pady=20)

        tk.Button(root, text="Two Player", width=20,
                  command=self.two_player).pack(pady=10)

        tk.Button(root, text="Play vs AI", width=20,
                  command=self.vs_ai).pack(pady=10)

    def two_player(self):
        self.root.destroy()
        ChessGame(vs_ai=False)

    def vs_ai(self):
        self.root.destroy()
        ChessGame(vs_ai=True)

if __name__ == "__main__":
    root = tk.Tk()
    MainMenu(root)
    root.mainloop()
