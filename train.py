
import config 
import torch 
import numpy as np 
import chess
from mcts import AlphaMCTS
from chessenv import ChessEnv
from model import ChessModel
import torch.nn.functional as F
from decoder import move_to_index
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
    
    def selfPlay(self):
        memory = []
        env = ChessEnv() 
        move_count = 0

        while not env.is_terminal():
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
            move_count+=1 
        z = env.result()

        for entry in memory:
            entry.append(z)
            z = -z 
        return memory
    def train(self,memory):
        np.random.shuffle(memory)
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
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0)
            self.optimizer.step()

    def learn(self):
        for iteration in range(config.TRAIN_ITERATIONS):
            memory = []
            
            self.model.eval()

            # self play 
            for selfplay_iter in range(config.SELF_PLAY_ITERATIONS):
                memory+=self.selfPlay()
            
            # train 
            self.model.train()

            for epoch in range(config.NUM_EPOCHES):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f'model_{iteration}.pt')
            torch.save(self.optimizer.state_dict(), f'optimizer_{iteration}.pt')