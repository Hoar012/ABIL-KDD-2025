import collections
import torch
import torch.nn as nn
import jactorch
import jactorch.nn as jacnn
import jactorch.nn.functional as jacf

from typing import Sequence
from tabulate import tabulate
from jacinle.logging import get_logger
from jacinle.utils.enum import JacEnum
from jactorch.graph.context import ForwardContext, get_forward_context
from hacl.nn.quantization.gumbel_quantizer import GumbelQuantizer
from hacl.nn.quantization.vector_quantizer import VectorQuantizer

from .value import BOOL, VectorValueType, Value, QuantizedTensorValue
from .state import BatchState, concat_batch_states
from .expr import QINDEX, FunctionDef, ExpressionExecutionContext, ConstantExpression, AssignOp
from .domain import AugmentedFeatureStage, Domain, Operator, OperatorApplier

logger = get_logger(__file__)

__all__ = ['UnpackValue', 'Concat', 'Squeeze', 'AutoBatchWrapper', 'QuantizerMode', 'SimpleQuantizedEncodingModule', 'SimpleQuantizationWrapper',
    'PDSketchMultiStageModel', 'ABILModel']


class UnpackValue(nn.Module):
    """A simple module that unpacks PyTorch tensors from Value objects."""

    def forward(self, *tensors):
        if len(tensors) == 1:
            return tensors[0].tensor
        return tuple([t.tensor for t in tensors])


class Concat(nn.Module):
    """Concatenate all inputs along the last axis."""
    def forward(self, *tensors):
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-1)


class Squeeze(nn.Module):
    """Squeeze a give axis."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.squeeze(self.dim)


class AutoBatchWrapper(nn.Module):
    def __init__(self, wrapped, concat=True, squeeze=None):
        super().__init__()
        self.wrapped = wrapped
        self.concat = concat
        self.squeeze = squeeze

    def forward(self, *values: Value):
        v = values[0]
        total_batch_dims = v.batch_dims + len(v.batch_variables)
        tensors = tuple(v.tensor for v in values)

        if total_batch_dims == 0:
            tensors = tuple(t.unsqueeze(0) for t in tensors)
        else:
            tensors = tuple(t.reshape((-1,) + t.shape[total_batch_dims:]) for t in tensors)

        if self.concat:
            if len(tensors) == 1:
                input_tensor = tensors[0]
            else:
                input_tensor = torch.cat(tensors, dim=-1)
            rv = self.wrapped(input_tensor)
        else:
            rv = self.wrapped(*tensors)

        if total_batch_dims == 0:
            rv = rv.squeeze(0)
        else:
            rv = rv.reshape(v.tensor.shape[:total_batch_dims] + rv.shape[1:])

        if self.squeeze is not None:
            assert self.squeeze == -1, 'Only last-dim squeezing is implemented.'
            rv = rv.squeeze(self.squeeze)
        return rv


class QuantizerMode(JacEnum):
    NONE = 'none'
    VQ = 'vq'
    VQ_MULTISTAGE = 'vq-multistage'
    GUMBEL = 'gumbel'


class SimpleQuantizedEncodingModule(nn.Module):
    """A simple feature extractor that supports quantization."""
    def __init__(self, input_dim, hidden_dim, nr_quantizer_states, quantizer_mode='vq-multistage', forward_key=None, q_lambda=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nr_quantizer_states = nr_quantizer_states
        self.quantizer_mode = QuantizerMode.from_string(quantizer_mode)
        self.q_lambda = q_lambda
        self.forward_key = forward_key

        if self.quantizer_mode is QuantizerMode.NONE:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [], activation='tanh', flatten=False, last_activation=True)
            self.add_module('quantizer', None)
        elif self.quantizer_mode is QuantizerMode.VQ:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [hidden_dim], activation='tanh', flatten=False, last_activation=False)
            self.quantizer = VectorQuantizer(self.nr_quantizer_states, self.hidden_dim, beta=1.0)
        elif self.quantizer_mode is QuantizerMode.VQ_MULTISTAGE:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [hidden_dim], activation='tanh', flatten=False, last_activation=False)
            self.quantizer = VectorQuantizer(self.nr_quantizer_states, self.hidden_dim, beta=1.0)
        elif self.quantizer_mode is QuantizerMode.GUMBEL:
            self.mlp = jacnn.MLPLayer(input_dim, hidden_dim, [], activation='tanh', flatten=False, last_activation=True)
            self.quantizer = GumbelQuantizer(self.hidden_dim, self.nr_quantizer_states, self.hidden_dim, hard=False)
        else:
            raise ValueError('Unknown quantizer mode: {}.'.format(self.quantizer_mode))

        self.vq_stage = 1

    def set_vq_stage(self, stage: int):
        assert stage in (1, 2, 3)
        self.vq_stage = stage

    def extra_state_dict(self):
        return {'vq_stage': self.vq_stage}

    def load_extra_state_dict(self, state_dict):
        self.vq_stage = state_dict.pop('vq_stage', 1)

    def embedding_weight(self):
        if self.quantizer is not None:
            return self.quantizer.embedding_weight
        return None

    def quantize(self, z, loss_key):
        ctx = get_forward_context()
        if loss_key is not None:
            ctx.monitor_rms({f'q/{loss_key}/0': z})

        if self.quantizer_mode is QuantizerMode.NONE:
            return z, None
        elif self.quantizer_mode in (QuantizerMode.VQ, QuantizerMode.VQ_MULTISTAGE):
            if self.vq_stage == 1:
                return z, None
            elif self.vq_stage == 2:
                self.quantizer.save_tensor(z)
                return z, None
            else:
                assert self.vq_stage == 3

                if self.training:
                    z, z_id, loss = self.quantizer(z)
                    ctx.add_loss(loss * self.q_lambda, f'loss/q/{loss_key}')
                else:
                    z, z_id = self.quantizer(z)

                ctx.monitor_rms({f'q/{loss_key}/1': z})
                return z, z_id
        elif self.quantizer_mode is QuantizerMode.GUMBEL:
            z, z_id = self.quantizer(z)
            return z, z_id.argmax(dim=-1)
        else:
            raise ValueError('Unknown quantizer mode: {}.'.format(self.quantizer_mode))

    def forward(self, x, loss_key=None):
        if isinstance(x, Value):
            x = x.tensor
        if loss_key is None:
            loss_key = self.forward_key
        if loss_key is None:
            raise ValueError('Should specify either self.forward_key or loss_key when calling the function.')

        z = self.mlp(x)
        z, z_id = self.quantize(z, loss_key)
        return QuantizedTensorValue(z, z_id)


class SimpleQuantizationWrapper(nn.Module):
    """A simple wrapper for mapping functions that supports quantization.

    This function assumes that all inputs to the module are quantized vectors.
    Thus, during training, the actual inputs to the inner module is a list of
    vectors, and the output is either a Boolean value, or a quantized vector.

    Usage:

        >>> module.train()
        >>> # train the module using the standard supervised learning.
        >>> module.eval()
        >>> module.enable_quantization_recorder()
        >>> for data in dataset:
        >>>     _ = module(data)
        >>> module.quantize_mapping_module()  # this ends the quantization encoder and creates the quantized mapping.
        >>> # now use the model in eval mode only, and use quantized values as its input.
    """
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        self.save_tensors = False
        self.saved_tensors = list()
        self.quantized_mapping = None

    def forward_quantized(self, *args):
        if self.quantized_mapping is not None:
            assert not self.training

            args = [arg.tensor for arg in args]
            output_shape = args[0].shape
            args = [arg.reshape(-1) for arg in args]

            output = list()
            for i in range(args[0].shape[0]):
                argv = tuple(int(arg[i]) for arg in args)
                if argv in self.quantized_mapping:
                    output.append( max(self.quantized_mapping[argv], key=self.quantized_mapping[argv].get) )
                else:
                    output.append(0)
            return torch.tensor(output, dtype=torch.int64).reshape(output_shape)
        raise ValueError('Run quantize_mapping_module first.')

    def forward(self, *args, **kwargs):
        ret = self.wrapped(*args, **kwargs)
        if self.save_tensors:
            self.saved_tensors.append((args, kwargs, ret))
        return ret

    def extra_state_dict(self):
        return {'quantized_mapping': self.quantized_mapping}

    def load_extra_state_dict(self, state_dict):
        if 'quantized_mapping' in state_dict:
            self.quantized_mapping = state_dict['quantized_mapping']

    def enable_quantization_recorder(self):
        assert self.quantized_mapping is None, 'Quantized mapping has already been created. Manually reset it to retrain.'
        self.save_tensors = True
        self.saved_tensors = list()

    def finalize_quantization_recorder(self):
        try:
            self.save_tensors = False
            return self.saved_tensors
        finally:
            self.saved_tensors = list()

    def quantize_mapping_module(self, function_def: FunctionDef):
        """Generate the quantized mapping given the recorded input output pairs.

        As described in the top-level docstring, this function assumes that

            - All arguments are vector-typed and quantized.
            - The output type is either Boolean or quantized vector.

        Based on the recorded IO pairs, it creates a mapping function, stored as a dictionary:
            Mapping[ Sequence[int], Mapping[Union[int, bool], int] ]

        It's a nested mapping. The top level key is the argument values. The second level key is
        the output value (thus, either an integer, or a Boolean value). The value is the number
        of occurrances of the (args, rv) pair.

        Args:
            function_def (FunctionDef): The corresponding function definition of this module.
        """
        saved_tensors = self.finalize_quantization_recorder()

        inputs_quantized = [list() for _ in range(len(function_def.arguments))]
        output_quantized = list()
        for inputs, _, output in saved_tensors:
            for i, (arg, arg_def) in enumerate(zip(inputs, function_def.arguments)):
                assert isinstance(arg_def, VectorValueType) and arg_def.quantized
                inputs_quantized[i].extend(arg.tensor_indices.flatten().tolist())
            if function_def.output_type == BOOL:
                output_quantized.extend((output > 0.5).flatten().tolist())
            else:
                assert isinstance(function_def.output_type, VectorValueType) and function_def.output_type.choices == 0
                output_quantized.extend(output.argmax(-1).flatten().tolist())

        mapping = collections.defaultdict(lambda: collections.defaultdict(int))
        for i in range(len(output_quantized)):
            args = tuple(arg[i] for arg in inputs_quantized)
            output = output_quantized[i]
            mapping[args][output] += 1

        self.quantized_mapping = mapping

    def print_quantized_mapping(self):
        """Prints the quantized mapping table."""
        rows = list()
        for key, values in self.quantized_mapping.items():
            row = list()
            row.extend(key)

            total_counts = sum(values.values())
            normalized = dict()
            for v, c in values.items():
                normalized[v] = c / total_counts
            row.append(' '.join([ f'{v}({c:.4f})' for v, c in sorted(normalized.items()) ]))
            rows.append(row)

        headers = list()
        headers.extend([f'arg{i}' for i in range(len(rows[0][:-1]))])
        headers.append('rv')
        print(tabulate(rows, headers=headers))


class ABILModel(nn.Module):
    DEFAULT_OPTIONS = {
        'bptt': False
    }

    def __init__(self, domain: Domain, goal_loss_weight=1.0, action_loss_weight=1.0, **options):
        super().__init__()
        self.domain = domain
        self.functions = nn.ModuleDict()
        self.bce = nn.BCELoss()
        self.xent = nn.CrossEntropyLoss()
        # self.mse = nn.MSELoss(reduction='sum')
        self.mse = nn.SmoothL1Loss(reduction='sum')

        self.goal_loss_weight = goal_loss_weight
        self.action_loss_weight = action_loss_weight
        self.options = options

        for key, value in type(self).DEFAULT_OPTIONS.items():
            self.options.setdefault(key, value)

        self.init_networks(domain)

    def init_networks(self, domain):
        raise NotImplementedError()

    def forward(self, feed_dict, task, count, forward_augmented=False):
        forward_ctx = ForwardContext(self.training)
        with forward_ctx.as_default():
            goal_expr = feed_dict['goal_expr']
            states, actions, done = feed_dict['states'], feed_dict['actions'], feed_dict['dones']

            assert forward_augmented
            if forward_augmented:
                for state in states:

                    self.domain.forward_augmented_features(state)

            batch_state = BatchState.from_states(self.domain, states)
            name_dict = batch_state._object_name2index[0]
            final_img = states[-1].features.tensor_dict["item-image"].tensor
            # print(batch_state)
            # print(batch_state._object_name2index[0])
            actions: Sequence[OperatorApplier]
            
            self.domain.forward_derived_features(batch_state)
            
            if goal_expr is not None:
                pred = self.domain.forward_expr(batch_state, [], goal_expr).tensor
                target = done
                if self.training:
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())

                    forward_ctx.add_loss(loss, 'goal', accumulate=self.goal_loss_weight)
                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal')
            
            for typ in ['door', 'key', 'ball', 'box']:
                expr = f"(not(is-{typ} wall:0))"
                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(expr)).tensor
                target = torch.ones(len(pred))
                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target)
                forward_ctx.add_loss(loss, 'prior', accumulate=self.goal_loss_weight)
                
            objs = []
            for key, val in name_dict.items():
                if key != "r" and key[0] != "w":
                    objs.append(key)
            
            # 加入知识约束
            # 1. 同一个物体的颜色type不变
            # 2. 不同谓词之间互斥
            # 3. 分段加入谓词约束：同一段上任务相关的谓词伪标注一致
            # 搜索出所有分段方法，对每个分段方式进行评分，评分最好的作为分段方式
            # for i in range(len())

            use_knowledge = count > 500

            if use_knowledge:
                for key in objs:
                    pred = []
                    target = []
                    max_p = 0
                    pseudo_label = None
                    for color in ['red', 'green', 'blue', 'purple', 'yellow', 'grey']:
                        knowledge1 = f"(is-{color} {key})"
                        pred1 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        
                        if sum(pred1) > max_p:
                            max_p = sum(pred1)
                            pseudo_label = color
                        pred.append(pred1)

                    for color in ['red', 'green', 'blue', 'purple', 'yellow', 'grey']:
                        if pseudo_label == color:
                            target.append(torch.ones(len(pred1)))
                        else:
                            target.append(torch.zeros(len(pred1)))

                    pred = torch.cat(pred)
                    target = torch.cat(target)

                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 0.1

                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    pred = []
                    target = []
                    max_p = 0
                    pseudo_label = None
                    for type in ['door', 'key', 'ball', 'box']:
                        knowledge2 = f"(is-{type} {key})"
                        pred2 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        if sum(pred2) > max_p:
                            max_p = sum(pred2)
                            pseudo_label = color
                        pred.append(pred2)

                    for type in ['door', 'key', 'ball', 'box']:
                        if pseudo_label == type:
                            target.append(torch.ones(len(pred2)))
                        else:
                            target.append(torch.zeros(len(pred2)))

                    pred = torch.cat(pred)
                    target = torch.cat(target)

                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 0.1
                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
            
            succ = done[-1]
            use_seg_knowledge = True
            if use_seg_knowledge:
                if task == "open":
                    open_door = None
                    for i, action in enumerate(actions):
                        if action.name == "toggle":
                            open_door = action.arguments[-1]

                    if succ: # correct door
                        pred = []
                        target = []
                        for color in ['red', 'green', 'blue', 'purple', 'yellow', 'grey']:
                            knowledge1 = f"(is-{color} {open_door})"
                            if str(goal_expr).find(color) > -1:
                                pred1 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                                target1 = torch.ones(len(pred1))
                            else:
                                pred1 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                                target1 = torch.zeros(len(pred1))
                            pred.append(pred1)
                            target.append(target1)
                        pred = torch.cat(pred)
                        target = torch.cat(target)
                        
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1

                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                    knowledge2 = f"(is-door {open_door})"
                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                    target = torch.ones(len(pred))
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                    knowledge3 = f"(exists (?o - item) (and (is-open ?o) ))"
                    
                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor[:-1]
                    target = torch.zeros(len(pred))
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
                elif task == "put":
                    if succ:
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse("(hands-free r)")).tensor
                    
                        pick_index = len(pred)
                        place_index = 0
                        for i, action in enumerate(actions):
                            if action.name == "pickup":
                                pick_index = i
                                pick_obj = action.arguments[-1]
                            elif action.name == "place":
                                place_index = i
                                
                        target = torch.ones(pick_index)
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred[:pick_index], target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        ball_pred = self.domain.forward_expr(batch_state, [], self.domain.parse(f"(is-ball {pick_obj})")).tensor[0]
                        box_pred = self.domain.forward_expr(batch_state, [], self.domain.parse(f"(is-box {pick_obj})")).tensor[0]
                        
                        if ball_pred >= box_pred:
                            pseudo_pick_label = "ball"
                        else:
                            pseudo_pick_label = "box"
                        
                        knowledge1 = f"(exists (?o - item)(and(robot-is-facing r ?o)(is-{pseudo_pick_label} ?o)))"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        
                        if pick_index > 0:
                            target = torch.cat([torch.zeros(pick_index), torch.ones(1)])
                        else:
                            target = torch.ones(1)
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred[:pick_index + 1].float(), target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        for obj in objs:
                            if obj != pick_obj:
                                if pseudo_pick_label == "ball":
                                    knowledge2 = f"(and (not (hands-free r))(and (nextto (item-pose {obj}) (robot-facing r) ) (is-box {obj}) ))"
                                else:
                                    knowledge2 = f"(and (not (hands-free r))(and (nextto (item-pose {obj}) (robot-facing r) ) (is-ball {obj}) ))"
            
                                break
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.cat([torch.zeros(place_index), torch.ones(1)])
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred[:place_index + 1].float(), target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    # if succ:
                    #     pred = self.domain.forward_expr(batch_state, [], self.domain.parse("(hands-free r)")).tensor
                    
                    #     pick_index = len(pred)
                    #     place_index = 0
                    #     for i, action in enumerate(actions):
                    #         if action.name == "pickup":
                    #             pick_index = i
                    #             pick_obj = action.arguments[-1]
                    #         elif action.name == "place":
                    #             place_index = i
                                
                    #     target = torch.ones(pick_index)
                    #     loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred[:pick_index], target) * 1
                    #     forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                    #     if True:
                    #         pseudo_pick_label = "ball"
                    #     else:
                    #         pseudo_pick_label = "box"
                        
                    #     knowledge1 = f"(exists (?o - item)(and(robot-is-facing r ?o)(is-{pseudo_pick_label} ?o)))"
                    #     pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        
                    #     if pick_index > 0:
                    #         target = torch.cat([torch.zeros(pick_index), torch.ones(1)])
                    #     else:
                    #         target = torch.ones(1)
                    #     loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred[:pick_index + 1].float(), target) * 1
                    #     forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                    #     for obj in objs:
                    #         if obj != pick_obj:
                    #             if pseudo_pick_label == "ball":
                    #                 knowledge2 = f"(and (not (hands-free r))(and (nextto (item-pose {obj}) (robot-facing r) ) (is-box {obj}) ))"
                    #                 knowledge3 = f"(is-box {obj})"
                    #             else:
                    #                 knowledge2 = f"(and (not (hands-free r))(and (nextto (item-pose {obj}) (robot-facing r) ) (is-ball {obj}) ))"
                    #                 knowledge3 = f"(is-ball {obj})"
            
                    #             break
                        
                    #     pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                    #     target = torch.cat([torch.zeros(place_index), torch.ones(1)])
                    #     loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred[:place_index + 1].float(), target) * 1
                    #     forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                    #     pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                    #     target = torch.ones(len(pred))
                    #     loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                    #     forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                elif task == "unlock":
                    if succ:
                        pick_objs = []
                        toggle_objs = []
                        pick_index = 0
                        toggle_index = 0
                        for i, action in enumerate(actions):
                            if action.name == "pickup":
                                pick_objs.append(action.arguments[-1])
                            elif action.name == "toggle-tool":
                                toggle_objs.append(action.arguments[-1])
                                toggle_index = i
                        
                        k2 = f"(is-key {pick_objs[-1]})(not(is-door {pick_objs[-1]}))(is-door {toggle_objs[-1]})(not(is-key {toggle_objs[-1]}))"
                        knowledge2 = f"(and {k2})"
                        pred2 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target2 = torch.ones(len(pred2))
                        
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred2, target2) * 0.2
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)\
                        
                        for color in ['red', 'green', 'blue', 'purple', 'yellow', 'grey']:
                            if str(goal_expr).find(color) > -1:
                                k2 = f"(is-{color} {pick_objs[-1]})(is-{color} {toggle_objs[-1]})"
                            else:
                                k2 = f"(not(is-{color} {pick_objs[-1]}))(not(is-{color} {toggle_objs[-1]}))"
                        
                            knowledge2 = f"(and {k2})"
                            pred2 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                            target2 = torch.ones(len(pred2))
                            
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred2, target2) * 0.2
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        knowledge3 = f"(is-open {toggle_objs[-1]})"
                        pred3 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                        target3 = torch.cat([torch.zeros(toggle_index + 1), torch.ones(len(pred3)-toggle_index-1)])
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred3, target3) * 0.2
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        k4 = ""
                        for obj in states[0]._object_names:
                            if obj != 'r' and obj not in toggle_objs:
                                k4 += f"(not(is-open {obj}))"
                                
                        knowledge4 = f"(and {k4})"
                        pred4 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                        target4 = torch.ones(len(pred4))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred4, target4) * 0.2
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

            if self.action_loss_weight > 0:
                for i, action in enumerate(actions):
                    state = states[i]
                    next_state_pred = action.apply_effect(state)
                    next_state_target = states[i + 1]

                    has_learnable_parameters = False
                    for eff in action.operator.effects:
                        feature_def = eff.unwrapped_assign_expr.feature.feature_def
                    # for feature_def in self.domain.features.values():
                        if feature_def.group != 'augmented':
                            continue

                        has_learnable_parameters = True
                        feature_name = feature_def.name

                        this_loss = self.mse(
                            input=next_state_pred[feature_name].tensor,
                            target=next_state_target[feature_name].tensor
                        )

                        forward_ctx.add_loss(this_loss, f'a', accumulate=False)
                        forward_ctx.add_loss(this_loss, f'a/{action.operator.name}/{feature_name}', accumulate=self.action_loss_weight)

                    if has_learnable_parameters and self.options['bptt']:
                        self.domain.forward_derived_features(next_state_pred)  # forward again the derived features.
                        if goal_expr is not None:
                            pred = self.domain.forward_expr(next_state_pred, [], goal_expr).tensor
                            target = done[i + 1]
                            loss = self.bce(pred, target.float())
                            forward_ctx.add_loss(loss, 'goal_bptt', accumulate=self.goal_loss_weight * 0.1)
                            forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal_bptt')


        return forward_ctx.finalize()
