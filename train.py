
import config 
import torch 
import numpy as np 
import chess
from mcts import AlphaMCTS
from chessenv import ChessEnv
from model import ChessModel
import torch.nn.functional as F
from decoder import move_to_index


import multiprocessing as mp 
import os 


def self_play_worker(
    worker_id,
    model_state_dict,
    device,
    num_games,
    out_queue,
):
    np.random.seed(os.getpid())
    torch.manual_seed(os.getpid())

    device = torch.device(device)

    model = ChessModel().to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    az = AlphaZero(model, optimizer = None)
    
    worker_memory = []
    try:
        for _ in range(num_games):
            worker_memory.extend(az.selfPlay())

        out_queue.put(worker_memory)
    except Exception as e:
        out_queue.put(e)


def run_parallel_self_play(model, device):
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()

    num_workers = min(config.SELF_PLAY_ITERATIONS,max(1,os.cpu_count()-1))

    games_per_worker = config.SELF_PLAY_ITERATIONS // num_workers 
    remainder = config.SELF_PLAY_ITERATIONS % num_workers 

    model_state = model.state_dict()
    processes = []
    
    for worker_id in range(num_workers):
        games = games_per_worker + (1 if worker_id<remainder else 0)

        p = ctx.Process(
            target=self_play_worker,
            args = (
                worker_id,
                model_state,
                device,
                games,
                out_queue,
            ),
        )

        p.start()
        processes.append(p)

    memory = []

    for _ in range(num_workers):
        result = out_queue.get()
        if isinstance(result, Exception):
            raise result 
        memory.extend(result)
    
    for p in processes:
        p.join() 
    
    return memory 



class AlphaZero:
    def __init__(self,model,optimizer):
        self.model = model 
        self.optimizer = optimizer 
        self.mcts = AlphaMCTS(model)
        self.device = next(model.parameters()).device
    
    @staticmethod
    def pi_to_vector(pi_dict):
        target = np.zeros(config.ACTION_SIZE,dtype=np.float32)
        for move, prob in pi_dict.items():
            index = move_to_index(move)
            target[index]=prob 
        return target
    
    # lets try multiprocessing for self play 

    def selfPlay(self):
        memory = []
        env = ChessEnv() 
        move_count = 0

        while not env.is_terminal() and move_count<config.MAX_GAME_LENGTH:
            # mcts 
            pi = self.mcts.search(env)
            memory.append([env.encode(),self.pi_to_vector(pi)])

            # temperature 
            tau = 1 if move_count<15 else 0

            if tau==0:
                action = max(pi, key = pi.get)
            else:
                moves = list(pi.keys())
                probs = np.array([pi[m] for m in moves])
                probs = probs / probs.sum()
                action = np.random.choice(moves, p = probs)
            
            env.push(action)
            self.mcts.advance_tree(action)
            move_count+=1 
        z = env.result()
        if z is None:
            z = 0

        for entry in memory:
            entry.append(z)
            z = -z 
        return memory
    def train(self,memory):
        np.random.shuffle(memory)
        
        total_loss = 0.0 
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for startInd in range(0,len(memory),config.BATCH_SIZE):
            sample = memory[startInd:min(startInd+config.BATCH_SIZE,len(memory))]
            states, policy_targets, value_targets = zip(*sample)

            states, policy_targets, value_targets = np.array(states), np.array(policy_targets), np.array(value_targets)

            states = torch.tensor(states, dtype=torch.float32).to(self.device)
            policy_targets = torch.tensor(policy_targets, dtype = torch.float32).to(self.device)
            value_targets = torch.tensor(value_targets,dtype = torch.float32).unsqueeze(1).to(self.device)

            out_policy,out_value = self.model(states)

            legal_mask = (policy_targets>0).float()
            out_policy = out_policy * legal_mask 
            out_policy = out_policy / (out_policy.sum(dim=1,keepdim=True)+1e-10)

            policy_loss = -torch.mean(
                torch.sum(policy_targets * torch.log(out_policy+1e-8),dim=1)
            )
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss 

            entropy = -torch.mean(
                torch.sum(out_policy * torch.log(out_policy+1e-8),dim=1)
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0)
            self.optimizer.step()

            total_loss+=loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

            num_batches +=1 
        
        return {
            "loss": total_loss/num_batches,
            "policy": total_policy_loss/num_batches,
            "value": total_value_loss/num_batches,
            "entropy": total_entropy/num_batches
        }

    def learn(self):
        for iteration in range(config.TRAIN_ITERATIONS):
            print(f"\n=== Iteration {iteration}")
            memory = []
            
            self.model.eval()

            # self play 
            # try multiprocessing

            memory = run_parallel_self_play(
                model=self.model,
                device = self.device,
            )
            #for selfplay_iter in range(config.SELF_PLAY_ITERATIONS):
                #memory+=self.selfPlay()
            

            
            # train 
            self.model.train()

            for epoch in range(config.NUM_EPOCHES):
                stats = self.train(memory)
                print(f'------- Epoch {epoch+1}: --------------')
                print(
                    f"Loss {stats['loss']:.4f} | "
                    f"Policy {stats['policy']:.4f} | "
                    f"Value {stats['value']:.4f} | "
                    f"Entropy {stats['entropy']:.4f}"
                )
            
            torch.save(self.model.state_dict(), f'model_{iteration}.pt')
            torch.save(self.optimizer.state_dict(), f'optimizer_{iteration}.pt')