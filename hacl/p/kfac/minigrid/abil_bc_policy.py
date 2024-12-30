import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
from typing import List
from hacl.pdsketch.interface.v2.state import State, BatchState
from hacl.pdsketch.interface.v2.domain import Domain
from hacl.p.kfac.nn.int_embedding import IntEmbedding, FallThroughEmbedding, ConcatIntEmbedding
from einops import rearrange

__all__ = ['ABIL_BCPolicyNetwork']


class ABIL_BCPolicyNetwork(nn.Module):
    def __init__(self, domain: Domain, action_space, predicates: List[str], goal_augment = True):
        super().__init__()
        self.domain = domain
        self.action_space = action_space
        self.goal_augment = goal_augment

        robot_embedding_dim = 6
        item_embedding_dim = 5

        self.goal_embedding = nn.Embedding(len(predicates) + 1, 32, padding_idx=0)
        
        if self.goal_augment:
            self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim + 64, item_embedding_dim], [96, 64], 'mlp', 256, activation='relu'
        )
            self.obj_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim, item_embedding_dim], [32, 32], 'mlp', 128, activation='relu'
        )
            self.action_classifier = nn.Linear(96 + 32, len(action_space))
        else:
            self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim + 64, item_embedding_dim], [128, 64], 'mlp', 256, activation='relu'
        )
            self.action_classifier = nn.Linear(128, len(action_space))
        
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0
        self.action2index = {get_action_desc(action): i for i, action in enumerate(action_space)}
        print(self.action2index,action_space)
        self.loss = nn.CrossEntropyLoss()

    def forward_state(self, state: State, goal, filt_expr, succ: int = 1):
        states = [state]
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()
        robot_tensors.append(batched_states['robot-feature'].tensor[:,0])
        robot_tensors.append(batched_states['holding'].tensor[:,0])
        robot_obj_tensor = torch.cat(robot_tensors, dim=-1)
        
        goal_tensor = jactorch.add_dim(self.forward_goal(goal).flatten(), 0, len(states))
        goal_tensor=self.goal_embedding(goal_tensor).flatten().unsqueeze(0)
        robot_tensors.append(goal_tensor)
    
        object_tensors=batched_states['item-feature'].tensor
        
        filt = self.domain.forward_expr(batched_states, {}, self.domain.parse(filt_expr)).tensor
        mask = (filt > 0.8)
        n = sum(mask[0])
        if n > 0:
            filt_object_tensors = object_tensors[mask].reshape([1, n, 5])
            pad_filt_object_tensors = torch.zeros_like(object_tensors)
            pad_filt_object_tensors[:, :n, :] = filt_object_tensors
        else:
            filt_object_tensors = object_tensors
            pad_filt_object_tensors = object_tensors
            
        
        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = object_tensors
        
        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature = self.obj_encoder([robot_obj_tensor, filt_object_tensors])[0]
        
        output = self.action_classifier(torch.cat([state_feature, obj_feature], dim=-1))
        return output[0]

    def forward_goal(self, goal):
        predicate_names = list()
        try:
            for predicate in goal.expr.arguments:
                if predicate.feature_def.name in self.predicate2index:
                    predicate_names.append(predicate.feature_def.name)
        except:
            predicate_names = ['nextto']
        assert len(predicate_names) in (0, 1, 2)
        predicate_names = [self.predicate2index[n] for n in predicate_names]
        if len(predicate_names) == 0:
            predicate_names += [0, 0]
        if len(predicate_names) == 1:
            predicate_names.append(0)
            
        return torch.tensor(predicate_names, dtype=torch.long, device=self.goal_embedding.weight.device)

    def forward_actions(self, actions):
        actions = [self.action2index[get_action_desc(action)] for action in actions]
        return torch.tensor(actions, dtype=torch.long, device=self.action_classifier.weight.device)

    def bc(self, robot_tensors, object_tensors, goal_tensors, actions, filt_obj_tensors):
        if len(actions) <= 0:
            return 0, {'loss': 0}, {}

        batch_size = len(actions)
        goal_tensor = self.goal_embedding(goal_tensors.cuda())
        goal_tensor = rearrange(goal_tensor, "b t a -> b (t a)")
        robot_obj_tensors = robot_tensors= robot_tensors.cuda()
        
        robot_tensor = torch.cat([robot_tensors, goal_tensor],dim=-1)
        object_tensor = object_tensors.cuda()
        filt_obj_tensors = filt_obj_tensors.cuda()

        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature = self.obj_encoder([robot_obj_tensors, filt_obj_tensors])[0]
        
        output = self.action_classifier(torch.cat([state_feature, obj_feature],dim=-1))

        loss = self.loss(output, actions.cuda())
        return loss, {'loss': loss}, output

    
    def encode(self, states: List[State], actions, goal, filt_expr):

        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()

        robot_tensors.append(batched_states['robot-feature'].tensor[:, 0])
        robot_tensors.append(batched_states['holding'].tensor[:,0])
        goal_tensor = jactorch.add_dim(self.forward_goal(self.domain.parse(goal)).flatten(), 0, len(states))
        
        object_tensors = batched_states['item-feature'].tensor
        
        filt = self.domain.forward_expr(batched_states, {}, self.domain.parse(filt_expr)).tensor
        mask = (filt > 0.8)
        n = sum(mask[0])
        batch_size = object_tensors.size()[0]
        if n > 0:
            filt_object_tensors = object_tensors[mask].reshape([batch_size, n, 5])
        else:
            filt_object_tensors = object_tensors
        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = object_tensors
        
        return robot_tensor, object_tensor, goal_tensor, self.forward_actions(actions), filt_object_tensors
        
    
def get_action_desc(action):
    return '{}({})'.format(action.name, ', '.join(map(str, action.arguments)))