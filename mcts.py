import chessenv
import numpy as np
from chessenv import ChessEnv
import config

import torch 

from decoder import *

class Node:
    def __init__(self,env:ChessEnv, parent: "Node" = None, parent_move = None, prior = 0):
        self.env = env 
        self.parent = parent 
        self.parent_move = parent_move 

        self.children = {}
        self.untried_moves = env.legal_moves()

        np.random.shuffle(self.untried_moves)

        self.N = 0 
        self.W = 0.0
        self.prior = prior
        


    
    def is_fully_expanded(self):
        return len(self.untried_moves)==0 and len(self.children)>0
    
    '''
    def get_ucb(self, child):
        if child.N == 0:
            return float('inf')  
        q_value = 1 - ((child.W/child.N)+1)/2 
        return q_value + config.UCB_C * np.sqrt(np.log(self.N) / child.N)
    '''
    def get_ucb(self, child):
        if child.N == 0:
            return float('inf')
        return (child.W / child.N) + config.UCB_C * np.sqrt(np.log(self.N) / child.N)

    def get_puct(self, child):
        q_value = 1 - ((child.W/child.N)+1)/2 
        u_value = config.PUCT_C * child.prior * (np.sqrt(self.N) / (1+child.N)) 
        return q_value + u_value
    
    def expand_random(self):
        action = self.untried_moves[-1]
        self.untried_moves.pop()
        
        child_state = self.env.copy()
        child_state.push(action)
        child = Node(child_state, parent = self, parent_move = action)
        self.children[action] = child 
        return child 
    
    def expand(self, model, device):
        planes = self.env.encode() 
        x = torch.tensor(planes, dtype = torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            policy, value = model(x)
        policy = policy.squeeze(0).cpu().numpy()
        value = value.item()
        legal_moves = self.env.legal_moves()

        priors = []
        total_prior = 0.0

        for move in legal_moves:
            idx = move_to_index(move)
            priors.append(policy[idx])
        priors = np.array(priors)
        priors = np.maximum(priors,1e-10)
        priors /= np.sum(priors)
        
        for move,prior in zip(legal_moves,priors):
            next_env = self.env.copy()
            next_env.push(move)
            self.children[move] = Node(next_env,parent=self,parent_move=move,prior=prior)
        return value
    
    def select_random(self):
        best_child = None 
        best_ucb = -np.inf
        
        for child_action, child in self.children.items():
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child 
                best_ucb = ucb 
        return best_child 
    def select(self):
        best_child = None 
        best_puct = -np.inf 
        for child_action, child in self.children.items():
            puct = self.get_puct(child)
            if puct > best_puct:
                best_child = child 
                best_puct = puct 
        return best_child 
    
    def simulate(self):
        winner = self.env.result() 
        
        rollout_state = self.env.copy()
        initial_player = 1 if self.env.board.turn else -1
        #print(f'initial player is initally {initial_player} board is \n {rollout_state.board}')

        val = None
        
        while True:
            if winner is not None:
                #print(f'simulate board is now\n {rollout_state.board} winner is {winner}')
                if winner==initial_player:
                    val= 1
                elif winner==0:
                    val= 0
                else:
                    val = -1
                #print(f'returning simulate function {val}')
                return val
            valid_moves=  rollout_state.legal_moves()
            action = np.random.choice(valid_moves)
            rollout_state.push(action)
            winner = rollout_state.result()
            
    
    def backpropagate(self, value):
        self.W+=value 
        self.N+=1 
        #print(f'propagating state {self.env.board} with val {value}')
        if self.parent is not None:
            self.parent.backpropagate(value*-1)
    

class MCTS:
    def __init__(self,model = None, device = 'cpu'):
        self.model = model 
        self.device = device
        pass 

    def search(self,state):
        root = Node(state)

        for _ in range(config.NUM_SEARCHES):
            node = root 

            while node.is_fully_expanded():
                node = node.select_random()
        
            val = node.env.result()
            if val is None:
                if node.untried_moves:
                    node = node.expand_random()
                    val = node.simulate()
            else:
                pass
            node.backpropagate(val)
        
        action_probs = {}
        total = 0
        for child_action, child in root.children.items():
            action_probs[child_action]=child.N
            total+=child.N 
        
        for key in action_probs:
            action_probs[key]/=total 
        
        return action_probs 


class AlphaMCTS:
    def __init__(self,model, device = 'cpu'):
        self.model = model 
        self.device = device 

    def search(self, state:ChessEnv):
        root = Node(state)

        for _ in range(config.NUM_SEARCHES):
            node = root 

            while node.is_fully_expanded():
                node = node.select()
            value = node.env.result()
            if value is None:
                value = node.expand(self.model,self.device)
            node.backpropagate(value)
        action_probs = {}
        total_visits = sum(child.N for child in root.children.values())

        for child_action, child in root.children.items():
            action_probs[child_action] = child.N/total_visits 
        
        return action_probs 

            





        
    
    



