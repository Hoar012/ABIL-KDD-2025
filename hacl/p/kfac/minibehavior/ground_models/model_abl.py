import torch
import torch.nn as nn
import jactorch.nn as jacnn
import hacl.pdsketch as pds
from hacl.p.kfac.nn.int_embedding import IntEmbedding

__all__ = ['ModelABL']


class ModelABL(pds.mini_nn_abl.ABILModel):
    def init_networks(self, domain):
        self.register_buffer('empty_pose', torch.tensor([-1, -1], dtype=torch.float32))
        self.register_buffer('dir_to_vec', torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=torch.float32))

        domain.register_external_function_implementation('feature::empty-pose', self.empty_pose_fn)
        domain.register_external_function_implementation('feature::facing', self.facing)
        domain.register_external_function_implementation('type::pose::equal', self.pose_equal)
        domain.register_external_function_implementation('type::direction::equal', self.pose_equal)
        domain.register_external_function_implementation('type::height::equal', self.height_equal)
        
        domain.register_external_function_implementation('feature::bottom', self.bottom_height_fn)
        domain.register_external_function_implementation('feature::middle', self.middle_height_fn)
        domain.register_external_function_implementation('feature::top', self.top_height_fn)

        domain.register_external_function_implementation('generator::gen-direction', self.gen_direction)
        domain.register_external_function_implementation('generator::gen-pose-neq', self.gen_pose_neq)
        domain.register_external_function_implementation('generator::gen-facing-robot', self.gen_facing_robot)

        if 'feature::direction-left' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-left', self.direction_left)
        if 'feature::direction-right' in domain.external_functions:
            domain.register_external_function_implementation('feature::direction-right', self.direction_right)

        for k in ['item-type', 'item-color', 'item-state']:
            module = IntEmbedding(32,value_range=(0, 64))
            self.add_module(k, module)
            domain.register_external_function_implementation('feature::' + k + '::f', module)

        for k in ['has-seen', 'is-furniture', 'is-printer', 'is-table', 'is-package','is-shelf', 'toggleon', 'is-open', 'is-book',
                  "is-hamburger", "is-ashcan", "is-countertop", "is-carton", "is-door", "is-plywood", "is-collect",
                  "is-plate", "is-cabinet", "is-box", "is-sink", "is-plant", "is-bucket", "is-tool", "is-car", "is-dusty",
                  "is-stove", "is-marker", "is-pan", "is-electric_refrigerator", "is-brush", "is-teapot", "is-teabag"]:
            identifier = 'derived::' + k + '::f'
            if identifier in domain.external_functions:
                module = nn.Sequential(
                    pds.babyai_nn.UnpackValue(), nn.Linear(32, 1), pds.babyai_nn.Squeeze(-1),
                    nn.Sigmoid()
                )
                self.add_module(k, module)
                domain.register_external_function_implementation(identifier, module)
    
        identifier = 'derived::' + 'nextto' + '::f'
        if identifier in domain.external_functions:
            module = pds.babyai_nn.AutoBatchWrapper(nn.Sequential(
            jacnn.MLPLayer(4, 1, [16], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)
            self.add_module(identifier, module)
            domain.register_external_function_implementation(identifier, module)
            
        identifier = 'derived::' + 'ontop' + '::f'
        if identifier in domain.external_functions:
            module = pds.babyai_nn.AutoBatchWrapper(nn.Sequential(
            jacnn.MLPLayer(2, 1, [8], flatten=False),
            nn.Sigmoid()
        ), squeeze=-1)
            self.add_module(identifier, module)
            domain.register_external_function_implementation(identifier, module)
        # identifier = 'derived::' + "ontop" + '::f'
        # domain.register_external_function_implementation(identifier, self.gen_forward_relation())

        identifier = 'action::toggle::f'
        if identifier in domain.external_functions:
            module = pds.babyai_nn.AutoBatchWrapper( nn.Linear(96, 32) )
            self.add_module(identifier, module)
            domain.register_external_function_implementation(identifier, module)
            
        identifier = 'action::open::f'
        if identifier in domain.external_functions:
            module = pds.babyai_nn.AutoBatchWrapper( nn.Linear(96, 32) )
            self.add_module(identifier, module)
            domain.register_external_function_implementation(identifier, module)

    def empty_pose_fn(self):
        return self.empty_pose

    def dir_to_vec_fn(self, d):
        return self.dir_to_vec[d.flatten()].reshape(d.shape[:-1] + (2, ))

    def facing(self, p, d):
        return p.tensor + self.dir_to_vec_fn(d.tensor)

    def pose_equal(self, p1, p2):
        return (torch.abs(p1.tensor - p2.tensor) < 0.5).all(dim=-1)
    
    def height_equal(self, p1, p2):
        p = p2.tensor[0]
        return (torch.abs(p1.tensor - p) < 0.5).all(dim=-1)

    def gen_direction(self):
        return torch.randint(4, size=(1, )),

    def gen_pose_neq(self, pose1):
        return torch.zeros_like(pose1.tensor) + 3,

    def gen_facing_robot(self, target):
        i = torch.randint(4, size=(1, ))[0]
        return target.tensor + self.dir_to_vec[i], ((i + 2) % 4).unsqueeze(0)

    def direction_left(self, d):
        return (d.tensor - 1) % 4

    def direction_right(self, d):
        return (d.tensor + 1) % 4
    
    def bottom_height_fn(self):
        return torch.tensor([0], dtype=torch.float32)
    
    def middle_height_fn(self):
        return torch.tensor([1], dtype=torch.float32)
    
    def top_height_fn(self):
        return torch.tensor([2], dtype=torch.float32)

    def forward_ontop(self, height1, height2):
        return (height1.tensor > height2.tensor).all(dim=-1)

    def gen_forward_relation(self):
        def func(height1, height2):
            return self.forward_ontop(height1, height2)
        return func