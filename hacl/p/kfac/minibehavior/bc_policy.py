import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
from typing import List
from hacl.pdsketch.interface.v2.state import State, BatchState
from hacl.pdsketch.interface.v2.domain import Domain
from hacl.p.kfac.nn.int_embedding import IntEmbedding, FallThroughEmbedding, ConcatIntEmbedding

__all__ = ['BCPolicyNetwork']


class BCPolicyNetwork(nn.Module):
    def __init__(self, domain: Domain, action_space, predicates: List[str], state_dim=128):
        super().__init__()
        self.domain = domain
        self.action_space = action_space

        self.robot_feature = ConcatIntEmbedding({
            2: FallThroughEmbedding(input_dim=2),
            1: IntEmbedding(16, input_dim=1, value_range=4, concat_input=True),
            3: FallThroughEmbedding(input_dim=3),
        })
        robot_embedding_dim = self.robot_feature.output_dim
        self.item_feature = ConcatIntEmbedding({
            3: IntEmbedding(64, input_dim=3, value_range=(0, 64)),
            2: FallThroughEmbedding(input_dim=2),
            1: FallThroughEmbedding(input_dim=1)
        })
        item_embedding_dim = self.item_feature.output_dim

        self.goal_embedding = nn.Embedding(len(predicates) + 1, 32, padding_idx=0)
        self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim, item_embedding_dim], [state_dim, 64], 'mlp', 256, activation='relu'
        )
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0
        self.action2index = {get_action_desc(action): i for i, action in enumerate(action_space)}
        print(self.action2index,action_space)
        self.action_classifier = nn.Linear(128, len(action_space))
        self.loss = nn.CrossEntropyLoss()
        
    def bc(self, robot_tensors, object_tensors, actions):
        if len(actions) <= 0:
            return 0, {'loss': 0}, {}
        
        object_tensor = self.item_feature(object_tensors.cuda())
        robot_tensor = self.robot_feature(robot_tensors.cuda())
        
        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        
        output = self.action_classifier(state_feature)
        loss = self.loss(output, actions.cuda())

        return loss, {'loss': loss}, output

    def forward_state(self, state: State, goal, succ: int = 1):
        states = [state]
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()
        
        robot_tensors = torch.cat([batched_states['robot-feature'].tensor[:, 0], batched_states['holding'].tensor[:,0]],dim=1)
        robot_tensor = self.robot_feature(robot_tensors)
        object_tensors=batched_states['item-feature'].tensor
        object_tensor =self.item_feature(object_tensors)
        
        output = self.state_encoder([robot_tensor, object_tensor])[0]
        output = self.action_classifier(output)
        
        return output[0]
    
    def encode(self, states: List[State], actions):
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()

        robot_tensors.append(batched_states['robot-feature'].tensor[:, 0])
        robot_tensors.append(batched_states['holding'].tensor[:,0])
        object_tensors = batched_states['item-feature'].tensor
        
        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = object_tensors
        
        return robot_tensor, object_tensor, self.forward_actions(actions)

    def forward_actions(self, actions):
        actions = [self.action2index[get_action_desc(action)] for action in actions]
        return torch.tensor(actions, dtype=torch.long, device=self.action_classifier.weight.device)

    
def get_action_desc(action):
    return '{}({})'.format(action.name, ', '.join(map(str, action.arguments)))

