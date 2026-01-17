import chessenv
import numpy as np
from chessenv import ChessEnv
import config

class Node:
    def __init__(self,env:ChessEnv, parent: "Node" = None, parent_move = None):
        self.env = env 
        self.parent = parent 
        self.parent_move = parent_move 

        self.children = {}
        self.untried_moves = env.legal_moves()

        np.random.shuffle(self.untried_moves)

        self.N = 0 
        self.W = 0.0
        


    
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

    
    def expand(self):
        action = self.untried_moves[-1]
        self.untried_moves.pop()
        
        child_state = self.env.copy()
        child_state.push(action)
        child = Node(child_state, parent = self, parent_move = action)
        self.children[action] = child 
        return child 
    def select(self):
        best_child = None 
        best_ucb = -np.inf
        
        for child_action, child in self.children.items():
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child 
                best_ucb = ucb 
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
    def __init__(self):
        pass 

    def search(self,state):
        root = Node(state)

        for _ in range(config.NUM_SEARCHES):
            node = root 

            while node.is_fully_expanded():
                node = node.select()
        
            val = node.env.result()
            if val is None:
                if node.untried_moves:
                    node = node.expand()
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


        
            

            





        
    
    



