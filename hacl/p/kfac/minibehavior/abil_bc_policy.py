import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
from typing import List
from hacl.pdsketch.interface.v2.state import State, BatchState
from hacl.pdsketch.interface.v2.domain import Domain
from hacl.p.kfac.nn.int_embedding import IntEmbedding, FallThroughEmbedding, ConcatIntEmbedding

__all__ = ['ABIL_BCPolicyNetwork']


class ABIL_BCPolicyNetwork(nn.Module):
    def __init__(self, domain: Domain, action_space, predicates: List[str], goal_augment=False, state_dim = 64, obj_dim = 64):
        super().__init__()
        self.domain = domain
        self.action_space = action_space
        self.goal_augment = goal_augment
        
        self.robot_feature = ConcatIntEmbedding({
            2: FallThroughEmbedding(input_dim=2),
            1: IntEmbedding(16, input_dim=1, value_range=4, concat_input=True),
            3: FallThroughEmbedding(input_dim=3)
        })
        robot_embedding_dim = self.robot_feature.output_dim
        self.item_feature = ConcatIntEmbedding({
            3: IntEmbedding(64, input_dim=3, value_range=(0, 64)),
            2: FallThroughEmbedding(input_dim=2),
            1: FallThroughEmbedding(input_dim=1)
        })
        item_embedding_dim = self.item_feature.output_dim

        self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim, item_embedding_dim], [state_dim, 64], 'mlp', 256, activation='relu'
        )
        self.goal_embedding = None
        
        self.obj_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim, item_embedding_dim], [obj_dim, 64], 'mlp', 256, activation='relu'
        )
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0
        self.action2index = {get_action_desc(action): i for i, action in enumerate(action_space)}
        print(self.action2index,action_space)
        self.action_classifier = nn.Linear(128, len(action_space))
        self.loss = nn.CrossEntropyLoss()

    def forward_actions(self, actions):
        actions = [self.action2index[get_action_desc(action)] for action in actions]
        return torch.tensor(actions, dtype=torch.long, device=self.action_classifier.weight.device)

    def bc(self, robot_tensors, object_tensors, actions, filt_obj_tensors):
        if len(actions) <= 0:
            return 0, {'loss': 0}, {}

        batch_size = len(actions)

        object_tensor = self.item_feature(object_tensors.cuda())
        robot_tensor = self.robot_feature(robot_tensors.cuda())
        filt_obj_tensor = self.item_feature(filt_obj_tensors.cuda())

        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature = self.obj_encoder([robot_tensor, filt_obj_tensor])[0]
        
        output = self.action_classifier(torch.cat([state_feature, obj_feature],dim=-1))
        loss = self.loss(output, actions.cuda())
        return loss, {'loss': loss}, output
    
    def forward_state(self, state: State, goal, succ: int = 1, goal_predicates = None):
        states = [state]
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()

        robot_tensors = torch.cat([batched_states['robot-feature'].tensor[:, 0], batched_states['holding'].tensor[:,0]],dim=1)
        robot_tensor = self.robot_feature(robot_tensors)

        object_tensors=batched_states['item-feature'].tensor
        object_tensor = self.item_feature(object_tensors)
        filt = self.domain.forward_expr(batched_states,{},self.domain.parse(goal)).tensor
        mask = (filt > 0.9)
        n = sum(mask[0])
        batch_size = object_tensor.size()[0]

        filt_object_tensor = object_tensor[mask].reshape([batch_size, n, 67])
        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature =  self.obj_encoder([robot_tensor, filt_object_tensor])[0]

        output = torch.cat([state_feature, obj_feature], dim = -1)
        output = self.action_classifier(output)
        
        return output[0]

    def forward_state_single(self, state: State, filt_expr, succ: int = 1, goal_predicates = None):
        states = [state]
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()

        robot_tensors = torch.cat([batched_states['robot-feature'].tensor[:, 0], batched_states['holding'].tensor[:,0]],dim=1)
        robot_tensor = self.robot_feature(robot_tensors)

        object_tensors=batched_states['item-feature'].tensor
        object_tensor = self.item_feature(object_tensors)
        filt = self.domain.forward_expr(batched_states,{},self.domain.parse(filt_expr)).tensor
        mask = (filt > 0.9)
        n = sum(mask[0])
        batch_size = object_tensor.size()[0]

        object_tensor = filt_object_tensor = object_tensor[mask].reshape([batch_size, n, 67])[:,-1:,:]

        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature =  self.obj_encoder([robot_tensor, filt_object_tensor])[0]
        output = torch.cat([state_feature, obj_feature], dim = -1)
        output = self.action_classifier(output)
        
        return output[0]

    def forward_state_two(self, state: State, filt_expr1, filt_expr2, succ: int = 1):
        states = [state]
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()

        robot_tensors = torch.cat([batched_states['robot-feature'].tensor[:, 0], batched_states['holding'].tensor[:,0]],dim=1)
        robot_tensor = self.robot_feature(robot_tensors)

        object_tensors=batched_states['item-feature'].tensor
        object_tensor = self.item_feature(object_tensors)
        filt1 = self.domain.forward_expr(batched_states,{},self.domain.parse(filt_expr1)).tensor
        mask1 = (filt1 > 0.9)
        n = sum(mask1[0])
        batch_size = object_tensor.size()[0]

        filt_object_tensor = object_tensor[mask1].reshape([batch_size, n, 67])[:,-1:,:]

        filt2 = self.domain.forward_expr(batched_states,{},self.domain.parse(filt_expr2)).tensor
        mask2 = (filt2 > 0.9)
        n = sum(mask2[0])
        env_tensor = object_tensor[mask2].reshape([batch_size, n, 67])

        object_tensor = torch.cat([filt_object_tensor, env_tensor],dim=1)

        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature =  self.obj_encoder([robot_tensor, filt_object_tensor])[0]
        output = torch.cat([state_feature, obj_feature], dim = -1)
        output = self.action_classifier(output)
        
        return output[0]
    
    def encode(self, states: List[State], actions, filt_expr):
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()

        robot_tensors.append(batched_states['robot-feature'].tensor[:, 0])
        robot_tensors.append(batched_states['holding'].tensor[:,0])
        object_tensors = batched_states['item-feature'].tensor
        
        filt = self.domain.forward_expr(batched_states, {}, self.domain.parse(filt_expr)).tensor
        mask = (filt > 0.9)
        n = sum(mask[0])
        batch_size = object_tensors.size()[0]
        filt_object_tensors = object_tensors[mask].reshape([batch_size, n, 6])
        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = object_tensors
        
        return robot_tensor, object_tensor, self.forward_actions(actions), filt_object_tensors
    
def get_action_desc(action):
    return '{}({})'.format(action.name, ', '.join(map(str, action.arguments)))

