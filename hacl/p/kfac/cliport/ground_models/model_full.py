import os.path as osp
import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
import hacl.pdsketch as pds
from hacl.nn.lenet import LeNetRGB32
from hacl.nn.quantization.vector_quantizer import VectorQuantizer
from hacl.p.kfac.nn.int_embedding import IntEmbedding, ConcatIntEmbedding

__all__ = ['ModelFull', 'load_domain']


def load_domain():
    return pds.load_domain_file(osp.join(osp.dirname(__file__), 'cliport-v2024.pdsketch'))

class ModelFull(pds.cliport_nn_abl.PDSModel):
    USE_GT_CLASSIFIER = False

    def set_debug_options(self, use_gt_classifier=None):
        if use_gt_classifier is not None:
            type(self).USE_GT_CLASSIFIER = use_gt_classifier

    def init_networks(self, domain):
        self.options['bptt'] = True

        dimensions = 64
        self.functions.item_feature = nn.Sequential(pds.babyai_nn.UnpackValue(), LeNetRGB32(dimensions))
        # jactorch.mark_freezed(self.functions.item_feature)
        domain.register_external_function_implementation('feature::item-feature::f', self.forward_item_feature)
        self.functions.container_feature = nn.Sequential(pds.babyai_nn.UnpackValue(), LeNetRGB32(dimensions))
        domain.register_external_function_implementation('feature::container-feature::f', self.forward_container_feature)

        for k in ['is-blue', 'is-red', 'is-green', 'is-orange', 'is-yellow', 'is-purple', 'is-pink', 'is-cyan', 'is-brown', 'is-white', 'is-gray', 'is-bowl',
                  'c-is-blue', 'c-is-red', 'c-is-green', 'c-is-orange', 'c-is-yellow', 'c-is-purple', 'c-is-pink', 'c-is-cyan', 'c-is-brown', 'c-is-white', 'c-is-gray']:
            identifier = 'derived::' + k + '::f'
            if identifier in domain.external_functions:
                module = pds.babyai_nn.AutoBatchWrapper(
                    nn.Sequential(jacnn.MLPLayer(64, 1, [64], activation='relu'), nn.Sigmoid()),
                    squeeze=-1
                )
                self.functions.add_module(k, module)
                # if k != 'is-target':
                #     jactorch.mark_freezed(module)
                domain.register_external_function_implementation(identifier, module)


        for k in ['is-letter_R', 'is-letter_A', 'is-triangle', 'is-square', 'is-plus', 'is-letter_T', 'is-diamond',
                  'is-pentagon', 'is-rectangle', 'is-flower', 'is-star', 'is-circle', 'is-letter_G', 'is-letter_V',
                  'is-letter_E', 'is-letter_L', 'is-ring', 'is-hexagon', 'is-heart', 'is-letter_M']:
            identifier = 'derived::' + k + '::f'
            if identifier in domain.external_functions:
                module = pds.babyai_nn.AutoBatchWrapper(
                    nn.Sequential(jacnn.MLPLayer(64, 1, [64], activation='relu'), nn.Sigmoid()),
                    squeeze=-1
                )
                self.functions.add_module(k, module)
                domain.register_external_function_implementation(identifier, module)
        
        for k in ['is-same-shape']:
            identifier = 'derived::' + k + '::f'
            if identifier in domain.external_functions:
                module = pds.babyai_nn.AutoBatchWrapper(
                    nn.Sequential(jacnn.MLPLayer(128, 1, [64], activation='relu'), nn.Sigmoid()),
                    squeeze=-1
                )
                self.functions.add_module(k, module)
                domain.register_external_function_implementation(identifier, module)

        for k in ['is-in']:
            identifier = 'derived::' + k + '::f'
            self.functions.register_parameter(identifier, nn.Parameter(torch.zeros(3)))
            identifier = 'derived::' + k + '::thresh'
            self.functions.register_parameter(identifier, nn.Parameter(torch.tensor(0.05, dtype=torch.float32)))
            identifier = 'derived::' + k + '::gamma'
            self.functions.register_parameter(identifier, nn.Parameter(torch.tensor(10.0, dtype=torch.float32)))

            identifier = 'derived::' + k + '::f'
            domain.register_external_function_implementation(identifier, self.gen_forward_relation(k))
            

    def forward_relation(self, k, pose1, pose2):
        pose1 = pose1.tensor
        pose2 = pose2.tensor
        param = getattr(self.functions, 'derived::' + k + '::f')
        thresh = getattr(self.functions, 'derived::' + k + '::thresh')
        gamma = getattr(self.functions, 'derived::' + k + '::gamma')
        norm = torch.norm((pose2 + param) - pose1, p=2, dim=-1)
        out = torch.sigmoid((thresh - norm) * gamma)

        if k == 'is-in':
            return (torch.norm(pose1 - pose2, p=1, dim=-1) < 0.08).float()
        return out

    def gen_forward_relation(self, k):
        def func(pose1, pose2, k=k):
            return self.forward_relation(k, pose1, pose2)
        return func

    def forward_item_feature(self, images):
        return pds.babyai_nn.AutoBatchWrapper(self.functions.item_feature[1])(images)

    def forward_container_feature(self, images):
        return pds.babyai_nn.AutoBatchWrapper(self.functions.container_feature[1])(images)

