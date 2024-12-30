import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
from typing import List
from hacl.pdsketch.interface.v2.state import State, BatchState
from hacl.pdsketch.interface.v2.domain import Domain
from hacl.p.kfac.nn.int_embedding import IntEmbedding, FallThroughEmbedding, ConcatIntEmbedding
from einops import rearrange

__all__ = ['DTPolicyNetwork']


class DTPolicyNetwork(nn.Module):
    def __init__(self, domain: Domain, action_space, predicates: List[str], updown=False):
        super().__init__()
        self.domain = domain
        self.action_space = action_space
        self.updown = updown

        robot_embedding_dim = 6
        item_embedding_dim = 5

        if self.updown:
            self.succ_embedding = nn.Embedding(2, 32)
            robot_embedding_dim += 32
        else:
            self.add_module('succ_embedding', None)

        self.goal_embedding = nn.Embedding(len(predicates) + 1, 32, padding_idx=0)
        self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim + 64, item_embedding_dim], [128, 64], 'mlp', 256, activation='relu'
        )
        self.transformer_model = nn.Transformer(d_model=128, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, batch_first=True, dropout=0.03)
        self.time_embedding = nn.Embedding(20, 128, padding_idx=0)
        self.action_embedding = nn.Embedding(len(action_space) + 1, 128, padding_idx=0)
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0
        self.action2index = {get_action_desc(action): i for i, action in enumerate(action_space)}
        print(self.action2index,action_space)
        self.action_classifier = nn.Linear(128, len(action_space))
        self.positional_encoding = PositionalEncoding(128, dropout=0)
        self.loss = nn.CrossEntropyLoss()

        self.state_eocode = nn.Linear(128, 32)

    def forward_state(self, states: State, actions, goal, succ: int = 1):

        target = torch.cat([torch.tensor([0]), self.forward_actions(actions) + 1])

        t = len(states)
        
        target = self.action_embedding(target)
        robot_tensors = list()
        object_tensors = list()
        batched_states = BatchState.from_states(self.domain, states)

        robot_tensors.append(batched_states['robot-feature'].tensor[:, 0])
        robot_tensors.append(batched_states['holding'].tensor[:,0])
        goal_tensor = jactorch.add_dim(self.goal_embedding(torch.tensor(self.forward_goal(goal))).flatten(), 0, len(states))
        robot_tensors.append(goal_tensor)
        object_tensors.append(batched_states['item-feature'].tensor)

        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = torch.cat(object_tensors, dim=-1)

        state_embedding = self.state_encoder([robot_tensor, object_tensor])[0].unsqueeze(0)
        action_embedding = target.unsqueeze(0)

        # state_embedding = self.positional_encoding(state_embedding)
        # action_embedding = self.positional_encoding(action_embedding)

        output = self.transformer_model(state_embedding, action_embedding)
        output = self.action_classifier(output[-1])
        
        return output[-1]

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
        return predicate_names

    def forward_actions(self, actions):
        actions = [self.action2index[get_action_desc(action)] for action in actions]
        return torch.tensor(actions, dtype=torch.long, device=self.action_classifier.weight.device)

    def dt(self, robot_tensors, object_tensors, goal_tensor, actions):

        t = actions[0].shape[0]
        batch_size = len(actions)
        actions = torch.stack(actions)
        zero = torch.zeros(len(actions),dtype=torch.long).reshape(len(actions),1)
        target = torch.cat([zero,actions + 1],dim=1).cuda()
        
        robot_tensor = []
        target = self.action_embedding(target)
        goal_tensors = self.goal_embedding(torch.cat(goal_tensor).cuda())
        goal_tensors = rearrange(goal_tensors, "b t a -> b (t a)")

        robot_tensor.append(torch.cat(robot_tensors, dim=0).cuda())
        robot_tensor.append(goal_tensors)
        robot_tensor = torch.cat(robot_tensor,dim=1)
        object_tensor = torch.cat(object_tensors, dim=0).cuda()

        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]

        action_embedding = target
        state_embedding = state_feature.reshape(batch_size, 2, 128)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t+1).cuda()
        src_mask = nn.Transformer.generate_square_subsequent_mask(t).cuda()
        
        # state_embedding = self.positional_encoding(state_embedding)
        # action_embedding = self.positional_encoding(action_embedding)

        output1 = self.transformer_model(state_embedding, action_embedding, src_mask=src_mask, tgt_mask=tgt_mask)
        
        output = self.action_classifier(output1[:,:-1,:])
        output = rearrange(output, "b t a -> (b t) a")
        loss = self.loss(output, rearrange(actions, "b t -> (b t)").cuda())
        return loss, {'loss': loss}, output
    
    def encode(self, states: List[State], actions, goal, succ = 1):
  
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = list()
        object_tensors = list()

        robot_tensors.append(batched_states['robot-feature'].tensor[:, 0])
        robot_tensors.append(batched_states['holding'].tensor[:, 0])

        goal_tensor = torch.tensor(self.forward_goal(goal)).repeat(2,1)
        object_tensors.append(batched_states['item-feature'].tensor)

        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = torch.cat(object_tensors, dim=-1)
        
        return robot_tensor, object_tensor, goal_tensor, self.forward_actions(actions)
    
def get_action_desc(action):
    return '{}({})'.format(action.name, ', '.join(map(str, action.arguments)))


import math
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
