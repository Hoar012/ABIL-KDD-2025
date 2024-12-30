import torch
import torch.nn as nn
import torch.nn.functional as F
import jactorch
import jactorch.nn as jacnn
from typing import List
from hacl.pdsketch.interface.v2.state import State, BatchState
from hacl.pdsketch.interface.v2.domain import Domain
from einops import rearrange
from hacl.nn.lenet import LeNetRGB32

__all__ = ['BCPolicyNetwork']


class BCPolicyNetwork(nn.Module):
    def __init__(self, predicates: List[str]):
        super().__init__()
        
        self.item_feature = LeNetRGB32(64)
        item_embedding_dim = 67

        self.goal_embedding = nn.Embedding(len(predicates) + 1, 16, padding_idx=0)
        goal_embedding_dim = 32
        self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [goal_embedding_dim, item_embedding_dim], [128, 64], 'mlp', 128, activation='relu'
        )
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0

        self.action_decoder = nn.Linear(128, 6)
        self.loss = nn.MSELoss()
        
    def bc(self, images, poses, actions, goal):
        if len(actions) <= 0:
            return 0, {'loss': 0}, {}
        
        goal_tensor = self.goal_embedding(goal.cuda())
        goal_tensor = rearrange(goal_tensor, "b t a -> b (t a)")
        
        object_features = []
        for image in images:
            object_features.append(self.item_feature(image.cuda()))
        object_features = torch.stack(object_features)
        object_features = torch.cat([object_features, poses.cuda()], dim=-1)
        
        state_feature = self.state_encoder([goal_tensor, object_features])[0]
        
        output = self.action_decoder(state_feature)
        loss = self.loss(output, actions.float().cuda())
        return loss, {'loss': loss}, output
    
    def forward_goal(self, goal):
        predicate_names = goal

        assert len(predicate_names) in (0, 1, 2)
        predicate_names = [self.predicate2index[n] for n in predicate_names]
        if len(predicate_names) == 0:
            predicate_names.extend([0, 0])
        elif len(predicate_names) == 1:
            predicate_names.append(0)
        return torch.tensor(predicate_names, dtype=torch.long, device=self.goal_embedding.weight.device)

    def forward_state(self, images, poses, goal):        
        goal_tensor = self.goal_embedding(self.forward_goal(goal))
        goal_tensor = rearrange(goal_tensor, "t a -> (t a)").unsqueeze(0)
        
        object_features = self.item_feature(images)
        object_features = torch.cat([object_features, poses], dim=-1).unsqueeze(0)
        state_feature = self.state_encoder([goal_tensor, object_features])[0]
        
        output = self.action_decoder(state_feature)
        return output[0]