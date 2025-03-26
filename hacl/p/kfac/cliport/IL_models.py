import torch
import torch.nn as nn
import numpy as np
import random

from hacl.p.kfac.cliport.bc_policy import BCPolicyNetwork
from hacl.p.kfac.cliport.abil_bc_policy import ABIL_BCPolicyNetwork
from hacl.p.kfac.cliport.dt_policy import DTPolicyNetwork
from hacl.p.kfac.cliport.abil_dt_policy import ABIL_DTPolicyNetwork

from cliport.utils import utils
import pybullet as p

def rotate(pose1, pose2, info):
    # Compute the rotation parameter based on object-centric info
    pick_pose = (np.asarray(pose1), np.asarray((0, 0, 0, 1)))

    pick_id = place_id = None
    for id, obj_info in info.items():
        if id != "lang_goal":
            if (np.abs(obj_info[0] - pose1) <= 0.001).all():
                pick_id = id
            if (np.abs(obj_info[0] - pose2) <= 0.001).all():
                place_id = id

    obj_pose = p.getBasePositionAndOrientation(pick_id)
    targ_pose = p.getBasePositionAndOrientation(place_id)
    
    sixdof = True
    if sixdof:
        obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
        obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
        obj_pose = (obj_pose[0], obj_quat)
    world_to_pick = utils.invert(pick_pose)
    obj_to_pick = utils.multiply(world_to_pick, obj_pose)
    pick_to_obj = utils.invert(obj_to_pick)
    place_pose = utils.multiply(targ_pose, pick_to_obj)

    place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))
    return place_pose

class BCModel(nn.Module):
    def __init__(self, task, domain, predicates):
        super(BCModel, self).__init__()
        self.task = task
        self.policy = BCPolicyNetwork(predicates=predicates)

    def make_action(self, images, poses, goal, domain_gr, state, task, info):
        action = self.policy.forward_state(images, poses, goal)
        
        if task in ["packing-shapes", "packing-5shapes"]:
            pose1 = poses[torch.norm(poses-action[:3],1,dim=-1).argmin()]
            pose2 = poses[torch.norm(poses-action[3:],1,dim=-1).argmin()]
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act
        
        elif task in ["put-block-in-bowl-seen-colors", "place-red-in-green", "put-block-in-bowl-composed-colors"]:
            item_poses = state["item-pose"].tensor
            container_poses = state["container-pose"].tensor
            pose1 = item_poses[torch.norm(item_poses-action[:3],1,dim=-1).argmin()].numpy()
            pose2 = container_poses[torch.norm(container_poses-action[3:],1,dim=-1).argmin()].numpy()
        
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act

        elif task in ["separating-piles-seen-colors", "separating-20piles", "separating-10piles"]:
            pose1 = action[:3].detach()
            pose2 = action[3:].detach()
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act
      
    def forward(self, images, poses, action, goal, filt_obj_images = None, filt_obj_poses = None):
        return self.policy.bc(images, poses, action, goal)

class DTModel(nn.Module):
    def __init__(self, task, domain, predicates):
        super(DTModel, self).__init__()
        self.task = task
        self.policy = DTPolicyNetwork(predicates=predicates)

    def make_action(self, images, poses, goal, domain_gr, state, task, info):
        action = self.policy.forward_state(images, poses, goal)
        
        if task in ["packing-shapes", "packing-5shapes"]:
            pose1 = poses[torch.norm(poses-action[:3],1,dim=-1).argmin()]
            pose2 = poses[torch.norm(poses-action[3:],1,dim=-1).argmin()]
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act
        
        elif task in ["put-block-in-bowl-seen-colors", "place-red-in-green", "put-block-in-bowl-composed-colors"]:
            item_poses = state["item-pose"].tensor
            container_poses = state["container-pose"].tensor
            pose1 = item_poses[torch.norm(item_poses-action[:3],1,dim=-1).argmin()].numpy()
            pose2 = container_poses[torch.norm(container_poses-action[3:],1,dim=-1).argmin()].numpy()
        
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act
        
        elif task in ["separating-piles-seen-colors", "separating-20piles", "separating-10piles"]:
            pose1 = action[:3].detach()
            pose2 = action[3:].detach()
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act
        

    def forward(self, images, poses, action, goal, filt_obj_images = None, filt_obj_poses = None):
        return self.policy.dt(images, poses, action, goal)

class ABIL_BCModel(nn.Module):
    def __init__(self, task, domain, predicates):
        super(ABIL_BCModel, self).__init__()
        self.domain = domain
        self.task = task
        self.policy = ABIL_BCPolicyNetwork(domain, predicates=predicates)

    def make_action(self, images, poses, predicates, domain_gr, state, task, info):
        if task in ["packing-shapes", "packing-5shapes"]:
            shape = predicates[0]
            item_filt_expr = f"(foreach (?o - item) (is-{shape} ?o))"
            item_images, item_poses = self.policy.get_obj([state], item_filt_expr)
            container_images, container_poses = self.policy.get_obj([state], None, is_container=True)
            filt_obj_images = torch.cat([item_images, container_images], dim = 0)
            filt_obj_poses = torch.cat([item_poses, container_poses], dim = 0)
            action = self.policy.forward_state(images, poses, predicates, filt_obj_images, filt_obj_poses)

            pose1 = poses[torch.norm(poses-action[:3],1,dim=-1).argmin()]
            pose2 = poses[torch.norm(poses-action[3:],1,dim=-1).argmin()]
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act
        
        elif task in ["put-block-in-bowl-seen-colors", "place-red-in-green", "put-block-in-bowl-composed-colors"]:
            block_color = predicates[0]
            bowl_color = predicates[1]

            item_filt_expr = f"(foreach (?o - item) (is-{block_color} ?o))"
            item_images, item_poses = self.policy.get_obj([state], item_filt_expr)
            container_filt_expr = f"(foreach (?o - container) (c-is-{bowl_color} ?o))"
            container_images, container_poses = self.policy.get_obj([state], container_filt_expr, is_container = True)

            filt_obj_images = torch.cat([item_images, container_images], dim = 0)
            filt_obj_poses = torch.cat([item_poses, container_poses], dim = 0)
            
            action = self.policy.forward_state(images, poses, predicates, filt_obj_images, filt_obj_poses)
            
            pose1 = item_poses[torch.norm(item_poses-action[:3],1,dim=-1).argmin()]
            pose2 = container_poses[torch.norm(container_poses-action[3:],1,dim=-1).argmin()]
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}

            return act

        elif task in ["separating-piles-seen-colors", "separating-20piles", "separating-10piles"]:
            block_color = predicates[0]
            zone_color = predicates[1]

            item_filt_expr = f"(foreach (?o - item) (not(exists (?t - container)(is-in ?o ?t))))"
            item_images, item_poses = self.policy.get_obj([state], item_filt_expr)
            container_filt_expr = f"(foreach (?o - container) (c-is-{zone_color} ?o))"
            container_images, container_poses = self.policy.get_obj([state], container_filt_expr, is_container = True)

            filt_obj_images = container_images
            filt_obj_poses = container_poses
            
            action = self.policy.forward_state(images, poses, predicates, filt_obj_images, filt_obj_poses)

            pose1 = action[:3].detach()
            pose2 = action[3:].detach()
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act
        
    def forward(self, images, poses, action, goal, filt_obj_images, filt_obj_poses):
        return self.policy.bc(images, poses, action, goal, filt_obj_images, filt_obj_poses)
    

class ABIL_DTModel(nn.Module):
    def __init__(self, task, domain, predicates):
        super(ABIL_DTModel, self).__init__()
        self.domain = domain
        self.task = task
        self.policy = ABIL_DTPolicyNetwork(domain, predicates=predicates)

    def make_action(self, images, poses, predicates, domain_gr, state, task, info):
        if task in ["packing-shapes", "packing-5shapes"]:
            shape = predicates[0]
            item_filt_expr = f"(foreach (?o - item) (is-{shape} ?o))"
            item_images, item_poses = self.policy.get_obj([state], item_filt_expr)
            container_images, container_poses = self.policy.get_obj([state], None, is_container=True)
            filt_obj_images = torch.cat([item_images, container_images], dim = 0)
            filt_obj_poses = torch.cat([item_poses, container_poses], dim = 0)
            action = self.policy.forward_state(images, poses, predicates, filt_obj_images, filt_obj_poses)

            pose1 = poses[torch.norm(poses-action[:3],1,dim=-1).argmin()]
            pose2 = poses[torch.norm(poses-action[3:],1,dim=-1).argmin()]
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act

        elif task in ["put-block-in-bowl-seen-colors", "place-red-in-green", "put-block-in-bowl-composed-colors"]:
            block_color = predicates[0]
            bowl_color = predicates[1]

            item_filt_expr = f"(foreach (?o - item) (is-{block_color} ?o))"
            item_images, item_poses = self.policy.get_obj([state], item_filt_expr)
            container_filt_expr = f"(foreach (?o - container) (c-is-{bowl_color} ?o))"
            container_images, container_poses = self.policy.get_obj([state], container_filt_expr, is_container = True)

            filt_obj_images = torch.cat([item_images, container_images], dim = 0)
            filt_obj_poses = torch.cat([item_poses, container_poses], dim = 0)
            
            action = self.policy.forward_state(images, poses, predicates, filt_obj_images, filt_obj_poses)

            # item_poses, item_images = state["item-pose"].tensor, state["item-image"].tensor
            # container_poses, container_images = state["container-pose"].tensor, state["container-image"].tensor
            pose1 = item_poses[torch.norm(item_poses-action[:3],1,dim=-1).argmin()]
            pose2 = container_poses[torch.norm(container_poses-action[3:],1,dim=-1).argmin()]
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}

            return act

        elif task in ["separating-piles-seen-colors", "separating-20piles", "separating-10piles"]:
            block_color = predicates[0]
            zone_color = predicates[1]

            item_filt_expr = f"(foreach (?o - item) (not(exists (?t - container)(is-in ?o ?t))))"
            item_images, item_poses = self.policy.get_obj([state], item_filt_expr)
            container_filt_expr = f"(foreach (?o - container) (c-is-{zone_color} ?o))"
            container_images, container_poses = self.policy.get_obj([state], container_filt_expr, is_container = True)

            filt_obj_images = container_images
            filt_obj_poses = container_poses
            
            action = self.policy.forward_state(images, poses, predicates, filt_obj_images, filt_obj_poses)

            pose1 = action[:3].detach()
            pose2 = action[3:].detach()
            act = {"pose0": (pose1, [0, 0, 0, 1]), "pose1": (pose2, [0, 0, 0, 1])}
            return act

    def forward(self, images, poses, action, goal, filt_obj_images, filt_obj_poses):
        return self.policy.dt(images, poses, action, goal, filt_obj_images, filt_obj_poses)