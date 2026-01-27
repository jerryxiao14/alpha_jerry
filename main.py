# train_chess.py
import torch
import config
import os
from model import ChessModel
from train import AlphaZero, run_parallel_self_play
import multiprocessing as mp

def main():
    # Use spawn for multiprocessing (safe for Windows)
    mp.set_start_method("spawn", force=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and optimizer
    model = ChessModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Initialize AlphaZero trainer
    alpha_trainer = AlphaZero(model, optimizer)

    # Training loop
    for iteration in range(config.TRAIN_ITERATIONS):
        print(f"\n=== Iteration {iteration} ===")

        # --- Self-play phase ---
        print("Self-play phase...")
        model.eval()  # make sure model is in eval mode during self-play

        memory = run_parallel_self_play(
            model=model,
            device=device.type,  # pass 'cpu' or 'cuda' string
        )
        print(f"Collected {len(memory)} examples from self-play.")

        # --- Training phase ---
        print("Training phase...")
        model.train()  # switch model to training mode

        for epoch in range(config.NUM_EPOCHES):
            stats = alpha_trainer.train(memory)
            print(
                f"Epoch {epoch+1}/{config.NUM_EPOCHES} | "
                f"Loss {stats['loss']:.4f} | "
                f"Policy {stats['policy']:.4f} | "
                f"Value {stats['value']:.4f} | "
                f"Entropy {stats['entropy']:.4f}"
            )

        # --- Save model and optimizer ---
        model_path = f"model_iter_{iteration}.pt"
        optimizer_path = f"optimizer_iter_{iteration}.pt"
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        print(f"Saved model to {model_path} and optimizer to {optimizer_path}")

if __name__ == "__main__":
    main()
