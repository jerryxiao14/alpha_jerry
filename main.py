from model import ChessModel 

from train import AlphaZero 
import torch
import config
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force = True)
    
    model = ChessModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

    alpha_jerry = AlphaZero(model, optimizer = optimizer)

    alpha_jerry.learn()

