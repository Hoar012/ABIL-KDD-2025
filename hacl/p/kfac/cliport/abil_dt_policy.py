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

__all__ = ['ABIL_DTPolicyNetwork']

class ABIL_DTPolicyNetwork(nn.Module):
    def __init__(self, domain, predicates: List[str]):
        super().__init__()
        self.domain = domain
        self.item_feature = LeNetRGB32(64)
        item_embedding_dim = 67

        self.goal_embedding = nn.Embedding(len(predicates) + 1, 16, padding_idx=0)
        goal_embedding_dim = 32

        self.state_encoder = jacnn.NeuralLogicMachine(
            2, 1, [goal_embedding_dim, item_embedding_dim], [64, 64], 'mlp', 128, activation='relu'
        )
        self.obj_encoder = jacnn.NeuralLogicMachine(
            2, 1, [goal_embedding_dim, item_embedding_dim], [64, 64], 'mlp', 128, activation='relu'
        )
        self.predicate2index = {predicate: i + 1 for i, predicate in enumerate(predicates)}
        self.predicate2index['<PAD>'] = 0

        self.transformer_model = nn.Transformer(d_model=128, nhead=1, num_encoder_layers=1, num_decoder_layers=1, batch_first=True, dim_feedforward=64, dropout=0.03)
        self.action_encoder = nn.Linear(6, 128)
        self.action_decoder = nn.Linear(128, 6)
        self.loss = nn.MSELoss()
        
        
    def dt(self, images, poses, action, goal, filt_obj_images, filt_obj_poses):
        t = 1
        batch_size = 1
        zero = torch.zeros([batch_size, t, 128], dtype=torch.float)
        action = action.float().cuda().unsqueeze(0)

        target = torch.cat([zero.cuda(), self.action_encoder(action.unsqueeze(0))], dim=1)
        goal_tensor = self.goal_embedding(goal.cuda()).unsqueeze(0)
        goal_tensor = rearrange(goal_tensor, "b t a -> b (t a)")
        
        
        object_features = self.item_feature(images.cuda())
        object_features = torch.cat([object_features, poses.cuda()], dim=-1).unsqueeze(0)
        state_feature = self.state_encoder([goal_tensor, object_features])[0]

        filt_obj_features = self.item_feature(filt_obj_images.cuda())
        filt_obj_features = torch.cat([filt_obj_features, filt_obj_poses.cuda()], dim=-1).unsqueeze(0)
        obj_feature = self.state_encoder([goal_tensor, filt_obj_features])[0]

        output = torch.cat([state_feature, obj_feature], dim = -1)

        state_embedding = output.unsqueeze(0)
        action_embedding = target

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(t+1).cuda()
        src_mask = nn.Transformer.generate_square_subsequent_mask(t).cuda()
        
        output1 = self.transformer_model(state_embedding, action_embedding, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.action_decoder(output1[:,0,:])
        loss = self.loss(output, action.float().cuda())

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

    def forward_state(self, images, poses, goal, filt_obj_images, filt_obj_poses):
        zero = torch.zeros([1, 128],dtype=torch.float)
    
        target = zero
        goal_tensor = self.goal_embedding(self.forward_goal(goal))
        goal_tensor = rearrange(goal_tensor, "t a -> (t a)").unsqueeze(0)
        
        object_features = self.item_feature(images)
        object_features = torch.cat([object_features, poses], dim=-1).unsqueeze(0)
        
        state_feature = self.state_encoder([goal_tensor, object_features])[0]
        
        filt_obj_features = self.item_feature(filt_obj_images)
        filt_obj_features = torch.cat([filt_obj_features, filt_obj_poses], dim=-1).unsqueeze(0)
        obj_feature = self.state_encoder([goal_tensor, filt_obj_features])[0]
        
        state_embedding = torch.cat([state_feature, obj_feature], dim=-1)

        output1 = self.transformer_model(state_embedding.unsqueeze(0), zero.unsqueeze(0))[-1]

        action = self.action_decoder(output1)
        return action[0]
    
    def get_obj(self, states: List[State], filt_expr, is_container = False):
        batched_states = BatchState.from_states(self.domain, states)
        batched_states = states[0]
        if is_container:
            object_images = batched_states['container-image'].tensor
            object_poses = batched_states['container-pose'].tensor
            if filt_expr is None:
                return [object_images, object_poses]
        else:
            object_images = batched_states['item-image'].tensor
            object_poses = batched_states['item-pose'].tensor

        filt = self.domain.forward_expr(batched_states, {}, self.domain.parse(filt_expr)).tensor
        mask = (filt > 0.5)
        n = sum(mask)

        if n > 0:
            filt_obj_images = object_images[mask].reshape([n, 3*24*24])
            filt_obj_poses = object_poses[mask].reshape([n, 3])
        else:
            filt_obj_images = object_images
            filt_obj_poses = object_poses
        
        return [filt_obj_images, filt_obj_poses]