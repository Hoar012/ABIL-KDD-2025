import torch
import torch.nn as nn

from hacl.p.kfac.minigrid.bc_policy import BCPolicyNetwork
from hacl.p.kfac.minigrid.dt_policy import DTPolicyNetwork

class BCModel(nn.Module):
    def __init__(self, task, domain, ACTIONS, PREDICATES):
        super(BCModel, self).__init__()
        self.domain = domain
        self.task = task
        self.actions = ACTIONS

        for i in range(len(ACTIONS)):
            ACTIONS[i] = domain.operators[ACTIONS[i]]('r')
        self.policy = BCPolicyNetwork(domain, action_space=ACTIONS, predicates=PREDICATES)
        
    def make_action(self, state, goal):
        prob = self.policy.forward_state(state, goal)
        action_index = prob.argmax(dim=0)
        return self.actions[action_index]
        
    def forward(self, robot_tensors, object_tensors, goal_tensors, actions, filt_obj_tensors=None):
        return self.policy.bc(robot_tensors, object_tensors, goal_tensors, actions)
    
    def process_data(self, states, actions, goal, succ = 1):
        return self.policy.encode(states, actions, goal, succ)
    
class DTModel(nn.Module):
    def __init__(self, task, domain, ACTIONS, PREDICATES):
        super(DTModel, self).__init__()
        self.domain = domain
        self.task = task
        self.actions = ACTIONS
        self.his_states = []
        self.his_actions = []
        
        for i in range(len(ACTIONS)):
            ACTIONS[i] = domain.operators[ACTIONS[i]]('r')
        self.policy = DTPolicyNetwork(domain, action_space=ACTIONS, predicates=PREDICATES)
        
    def make_action(self, state, goal):
        self.his_states.append(state)
        prob = self.policy.forward_state(self.his_states[-2:], self.his_actions[-1:], goal, 1)
        action_index = prob.argmax(dim=0)
        self.his_actions.append(self.actions[action_index])
        return self.actions[action_index]
        
    def forward(self, robot_tensors, object_tensors, goal_tensor, actions, filt_obj_tensors=None):
        return self.policy.dt(robot_tensors, object_tensors, goal_tensor, actions)
    
    def process_data(self, states, actions, goal, succ = 1):
        return self.policy.encode(states, actions, goal, succ)