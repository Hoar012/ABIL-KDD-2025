import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
from typing import List
from hacl.pdsketch.interface.v2.state import State, BatchState
from hacl.pdsketch.interface.v2.domain import Domain
from hacl.p.kfac.nn.int_embedding import IntEmbedding, FallThroughEmbedding, ConcatIntEmbedding
from einops import rearrange

__all__ = ['ABIL_DTPolicyNetwork']


class ABIL_DTPolicyNetwork(nn.Module):
    def __init__(self, domain: Domain, action_space, predicates: List[str], updown=False, state_dim = 64, obj_dim = 64):
        super().__init__()
        self.domain = domain
        self.action_space = action_space
        self.updown = updown

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

        if self.updown:
            self.succ_embedding = nn.Embedding(2, 32)
            robot_embedding_dim += 32
        else:
            self.add_module('succ_embedding', None)

        self.goal_embedding = nn.Embedding(len(predicates) + 1, 32, padding_idx=0)
        self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim, item_embedding_dim], [state_dim, 64], 'mlp', 258, activation='relu'
        )
        self.obj_encoder = jacnn.NeuralLogicMachine(
            2, 1, [robot_embedding_dim, item_embedding_dim], [obj_dim, 64], 'mlp', 256, activation='relu'
        )
        self.transformer_model = nn.Transformer(d_model=128, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, batch_first=True)
        self.time_embedding = nn.Embedding(20, 128, padding_idx=0)
        self.action_embedding = nn.Embedding(len(action_space) + 1, 128, padding_idx=0)
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0
        self.action2index = {get_action_desc(action): i for i, action in enumerate(action_space)}
        print(self.action2index,action_space)
        self.action_classifier = nn.Linear(128, len(action_space))
        self.positional_encoding = PositionalEncoding(128, dropout=0)
        self.loss = nn.CrossEntropyLoss()


    def forward_state(self, states: State, actions, goal, succ: int = 1):
        target = torch.cat([torch.tensor([0]), self.forward_actions(actions) + 1])
        target = self.action_embedding(target)
        
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = torch.cat([batched_states['robot-feature'].tensor[:, 0], batched_states['holding'].tensor[:,0]],dim=1)
        robot_tensor = self.robot_feature(robot_tensors)
        object_tensor = self.item_feature(batched_states['item-feature'].tensor)
        filt = self.domain.forward_expr(batched_states, {}, self.domain.parse(goal)).tensor
        mask = filt > 0.9
        n = sum(mask[0])
        seq_size = object_tensor.size()[0]
        filt_object_tensor = object_tensor[mask].reshape([seq_size, n, 67])
        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature =  self.obj_encoder([robot_tensor, filt_object_tensor])[0]
        
        state_embedding = torch.cat([state_feature, obj_feature], dim = -1)        
        action_embedding = target

        output = self.transformer_model(state_embedding, action_embedding)
        output = self.action_classifier(output[-1])
        
        return output
    
    def forward_state_single(self, states: State, actions, goal, succ: int = 1):

        target = torch.cat([torch.tensor([0]), self.forward_actions(actions) + 1])
        target = self.action_embedding(target)
        
        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = torch.cat([batched_states['robot-feature'].tensor[:, 0], batched_states['holding'].tensor[:,0]],dim=1)
        robot_tensor = self.robot_feature(robot_tensors)
        object_tensor = self.item_feature(batched_states['item-feature'].tensor)
        filt = self.domain.forward_expr(batched_states, {}, self.domain.parse(goal)).tensor
        mask = filt > 0.9
        n = sum(mask[0])
        seq_size = object_tensor.size()[0]
        object_tensor = filt_object_tensor = object_tensor[mask].reshape([seq_size, n, 67])[:,:1,:]

        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature =  self.obj_encoder([robot_tensor, filt_object_tensor])[0]
        
        state_embedding = torch.cat([state_feature, obj_feature], dim = -1)        
        action_embedding = target

        output = self.transformer_model(state_embedding, action_embedding)
        output = self.action_classifier(output[-1])
        
        return output

    def forward_state_two(self, states: State, actions, filt_expr1, filt_expr2, succ: int = 1):

        target = torch.cat([torch.tensor([0]), self.forward_actions(actions) + 1])
        target = self.action_embedding(target)

        batched_states = BatchState.from_states(self.domain, states)
        robot_tensors = torch.cat([batched_states['robot-feature'].tensor[:, 0], batched_states['holding'].tensor[:,0]],dim=1)
        robot_tensor = self.robot_feature(robot_tensors)
        object_tensor = self.item_feature(batched_states['item-feature'].tensor)
        filt1 = self.domain.forward_expr(batched_states,{},self.domain.parse(filt_expr1)).tensor
        mask1 = (filt1 > 0.9)
        n = sum(mask1[0])
        seq_size = object_tensor.size()[0]
        filt_object_tensor = object_tensor[mask1].reshape([seq_size, n, 67])[:,-1:,:]
        
        filt2 = self.domain.forward_expr(batched_states,{},self.domain.parse(filt_expr2)).tensor
        mask2 = (filt2 > 0.9)
        n = sum(mask2[0])
        env_tensor = object_tensor[mask2].reshape([seq_size, n, 67])

        object_tensor = torch.cat([filt_object_tensor, env_tensor],dim=1)

        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature =  self.obj_encoder([robot_tensor, filt_object_tensor])[0]
        
        state_embedding = torch.cat([state_feature, obj_feature], dim = -1)        
        action_embedding = target

        output = self.transformer_model(state_embedding, action_embedding)
        output = self.action_classifier(output[-1])
        
        return output

    def forward_actions(self, actions):
        actions = [self.action2index[get_action_desc(action)] for action in actions]
        return torch.tensor(actions, dtype=torch.long, device=self.action_classifier.weight.device)
    
    def dt(self, robot_tensors, object_tensors, actions, filt_object_tensors):

        if len(actions) == 0:
            print(robot_tensors)
        t = actions[0].shape[0]
        
        batch_size = len(actions)
        actions = torch.stack(actions)
        zero = torch.zeros(len(actions),dtype=torch.long).reshape(len(actions),1).cuda()
        target = torch.cat([zero,actions+1],dim=1)
        target = self.action_embedding(target)

        robot_tensor = self.robot_feature(torch.cat(robot_tensors, dim=0).cuda())
        object_tensor = self.item_feature(torch.cat(object_tensors, dim=0).cuda())
        filt_object_tensor = self.item_feature(torch.cat(filt_object_tensors, dim=0).cuda())
        
        state_feature = self.state_encoder([robot_tensor, object_tensor])[0]
        obj_feature =  self.obj_encoder([robot_tensor, filt_object_tensor])[0]
        
        action_embedding = target
        state_embedding = torch.cat([state_feature, obj_feature], dim = -1).reshape(batch_size, 2, 128)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t+1).cuda()
        src_mask = nn.Transformer.generate_square_subsequent_mask(t).cuda()
        
        # state_embedding = self.positional_encoding(state_embedding)
        # action_embedding = self.positional_encoding(action_embedding)

        output1 = self.transformer_model(state_embedding, action_embedding, src_mask=src_mask, tgt_mask=tgt_mask)
        
        output = self.action_classifier(output1[:,:-1,:])
        output = rearrange(output, "b t a -> (b t) a")
        loss = self.loss(output, rearrange(actions, "b t -> (b t)").cuda())

        return loss, {'loss': loss}, output
    
    def encode(self, states: List[State], actions, filt_expr, succ = 1):

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
        if n > 0:
            filt_object_tensors = object_tensors[mask].reshape([batch_size, n, 6])
        else:
            filt_object_tensors = object_tensors
        robot_tensor = torch.cat(robot_tensors, dim=-1)
        object_tensor = object_tensors
        
        return robot_tensor, object_tensor, self.forward_actions(actions), filt_object_tensors
    
def get_action_desc(action):
    return '{}({})'.format(action.name, ', '.join(map(str, action.arguments)))


import math
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
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
