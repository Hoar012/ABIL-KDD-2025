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

__all__ = ['UnpackValue', 'Concat', 'Squeeze', 'AutoBatchWrapper', 'QuantizerMode', 'SimpleQuantizedEncodingModule', 'SimpleQuantizationWrapper', 'PDSketchMultiStageModel', 'ABILModel']


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

    def forward(self, feed_dict, task = None, forward_augmented=False):
        forward_ctx = ForwardContext(self.training)
        with forward_ctx.as_default():
            goal_expr = feed_dict['goal_expr']
            states, actions, done = feed_dict['states'], feed_dict['actions'], feed_dict['dones']

            assert forward_augmented
            if forward_augmented:
                for state in states:
                    # print(state)
                    self.domain.forward_augmented_features(state)

            batch_state = BatchState.from_states(self.domain, states)
            name_dict = batch_state._object_name2index[0]
            final_img = states[-1].features.tensor_dict["item-image"].tensor
            # print(batch_state)
            # print(batch_state._object_name2index[0])
            actions: Sequence[OperatorApplier]
            
            self.domain.forward_derived_features(batch_state)
            # print(batch_state) 
            
            if goal_expr is not None:
                pred = self.domain.forward_expr(batch_state, [], goal_expr).tensor
                target = done
                if self.training:
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())
                    # loss = self.bce(pred, target.float())
                    forward_ctx.add_loss(loss, 'goal', accumulate=self.goal_loss_weight)
                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal')
                
            use_knowledge = True

            if use_knowledge:
                if done[-1]:
                    knowledge0 = f"(forall (?o - item) (has-seen ?o))"
                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge0)).tensor
                    target = torch.ones(len(pred))
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1

                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
                    if task in ["install-a-printer", "install-a-printer-multi"]:
                        toggle_objs = []
                        toggle_index = 0
                        drop_index = 0

                        for i, action in enumerate(actions):
                            if action.name in ["toggle"]:
                                if action.arguments[-1] not in toggle_objs:
                                    toggle_objs.append(action.arguments[-1])
                                    toggle_index = i
                            elif action.name in ["drop_2"]:
                                drop_index = i
                        
                        print(toggle_objs)
                        printer = toggle_objs[-1]

                        knowledge1 = f"(and (is-printer {printer})(not(is-table {printer})))"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)


                        for obj in states[0]._object_names:
                            if obj != 'r' and obj not in toggle_objs:

                                knowledge2 = f"(and (not(is-printer {obj}))(is-table {obj})(is-furniture {obj}))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                                target = torch.ones(len(pred))
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        knowledge3 = f"(toggleon {printer})"
                        pred3 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                        target3 = torch.cat([torch.zeros(toggle_index + 1), torch.ones(len(pred3)-toggle_index-1)])
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred3, target3) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        knowledge4 = f"(exists (?t - item)(and(is-table ?t)(ontop {printer} ?t)))"
                        pred4 = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                        target4 = torch.cat([torch.zeros(drop_index + 1), torch.ones(len(pred4)-drop_index-1)])
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred4, target4) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                    elif task == "opening_packages":
                        objs = []
                        for i, action in enumerate(actions):
                            if action.name == "open":
                                if action.arguments[-1] not in objs:
                                    objs.append([i, action.arguments[-1]])

                        for i, obj in objs:
                            knowledge1 = f"(and(is-package {obj}) ) "
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                            target = torch.ones(len(pred))
                            
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 0.2
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                            
                            knowledge2 = f"(and(is-open {obj}) ) "
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                            target = torch.cat([torch.zeros(i + 1),torch.ones(len(pred)-i-1)])
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 0.2
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                    elif task == "MovingBoxesToStorage":
                        objs = []
                        for action in actions:
                            if action.name == "pickup_0":
                                if action.arguments[-1] not in objs:
                                    objs.append(action.arguments[-1])
                                    
                        for obj in objs:
                            knowledge1 = f"(and(is-carton {obj}) ) "
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1

                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        k2 = ""
                        for obj in states[0]._object_names:
                            if obj != 'r':
                                if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(equal (robot-pose r)(item-pose {obj}))")).tensor.any():
                                    k2 += f"(is-door {obj})(not(is-carton {obj}))"
                                else:
                                    k2 += f"(not(is-door {obj}))"
                        knowledge2 = f"(and {k2})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1

                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        # k3 = ""
                        # for obj in states[0]._object_names:
                        #     if obj != 'r' and obj not in objs:
                        #         k3 += f"(is-furniture {obj})"
                        
                        # knowledge3 = f"(and {k3})"
                        # pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                        # target = torch.ones(len(pred))
                        # loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1

                        # forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                    elif task == "SortingBooks":
                        pick_objs = []
                        for obj in name_dict:
                            cond = f"(robot-holding r {obj})"
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(cond)).tensor.any():
                                pick_objs.append(obj)
                        print(pick_objs)
                        k1 = ""
                        for obj in pick_objs:
                            k1 += f"(is-book {obj})(not(is-shelf {obj}))(not(is-furniture {obj}))"
                        
                        knowledge1 = f"(and {k1})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        k2 = ""
                        for obj in states[0]._object_names:
                            if obj != 'r' and obj not in pick_objs:
                                k2 += f"(not(is-book {obj}))(is-furniture {obj})"
                        
                        knowledge2 = f"(and {k2})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        for i, action in enumerate(actions):
                            if action.name == "drop_2":
                                obj = action.arguments[-1]
                                
                                knowledge3 = f"(exists(?t - item)(and (robot-is-facing r ?t)(is-shelf ?t)(not (is-book ?t))))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor[i]
                                target = torch.ones(1)
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                                
                                knowledge4 = f"(exists(?t - item)(and (is-shelf ?t)(ontop {obj} ?t)))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                                
                                target = torch.cat([torch.zeros(i + 1), torch.ones(len(pred)-i-1)])
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                    elif task == "Throwing_away_leftovers":
                        objs = []
                        for obj in name_dict:
                            cond = f"(robot-holding r {obj})"
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(cond)).tensor.any():
                                objs.append(obj)
                        print(objs)
                        i = 0
                        for action in actions:
                            if action.name == "pickup_2":
                                break
                            i += 1
                        for obj in objs:
                            knowledge0 = f"(exists (?t - item) (and (is-countertop ?t) (ontop {obj} ?t)))"
                            
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge0)).tensor
                            
                            pred = pred[:i]
                            target = torch.ones(i)
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        for obj in objs:
                            knowledge1 = f"(and(is-hamburger {obj}) (not (is-ashcan {obj})) (not (is-countertop {obj})) )"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1

                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        knowledge2 = f"(and(is-countertop countertop_0) (not(is-countertop plate_0)) )"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        k_3 = ""
                        for obj in states[0]._object_names:
                            if obj not in objs:
                                k_3 += f"(is-hamburger {obj})"
                        knowledge3 = f"(or {k_3} )"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                        target = torch.zeros(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        drop_objs = []
                        for action in actions:
                            if action.name == "drop_in":
                                if action.arguments[-1] not in drop_objs:
                                    drop_objs.append(action.arguments[-1])
                        print(drop_objs)
                        k_4 = ""
                        for obj in drop_objs:
                            k_4 += f"(is-ashcan {obj})(not(is-countertop {obj}))"
                        knowledge4 = f"(and {k_4} )"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                
                    elif task == "LayingWoodFloors":
                        objs = []
                        
                        for obj in name_dict:
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(robot-holding r {obj})")).tensor.any():
                                if obj not in objs:
                                    objs.append(obj)
                                
                        print(objs)

                        k1 = ""
                        for obj in objs:
                            k1 += f"(is-plywood {obj})"
                        knowledge1 = f"(and {k1} )"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1

                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                    elif task == "CleaningACar":
                        objs = []
                        for obj in name_dict:
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(robot-holding r {obj})")).tensor.any():
                                if obj not in objs:
                                    objs.append(obj)
                        k1 = ""
                        for obj in objs:
                            k1 += f"(is-tool {obj})(not (is-car {obj}))(not (is-bucket {obj}))"
                        print(objs)
                        for obj in states[0]._object_names:
                            if obj != 'r' and obj not in objs:
                                k1 += f"(not(is-tool {obj}))(is-furniture {obj})"
                        
                        knowledge1 = f"(and {k1})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        clean_index = 0
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse("(and(is-dusty car_0))")).tensor
                        scores = [1]
                        for i in range(1, len(states) - 1):
                            target_i = torch.cat([torch.ones(i + 1), torch.zeros(len(pred) - i - 1)])
                            scores.append(((pred - target_i)**2).mean())
                        scores.append(1)
                        clean_index = torch.argmin(torch.tensor(scores))
                        target = torch.cat([torch.ones(clean_index + 1), torch.zeros(len(pred) - clean_index - 1)])
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        knowledge2 = f"(and (is-car car_0))"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'prior', accumulate=self.goal_loss_weight)

                        bucket_obj = []
                        for i, action in enumerate(actions):
                            if action.name in ["drop_in"]:
                                filt = self.domain.forward_expr(states[i], [], self.domain.parse("(foreach(?o - item)(equal(item-pose ?o) (robot-facing r)))")).tensor
                                for j, p in enumerate(filt):
                                    if p:
                                        bucket_obj.append(states[i]._object_names[j+1])
                        print(bucket_obj)
                        
                        k3 = ""
                        for buc in bucket_obj:
                            k3 += f"(is-bucket {buc})"
                        knowledge3 = f"(and {k3})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
                    elif task == "CleaningShoes":
                        objs = []
                        for obj in name_dict:
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(robot-holding r {obj})")).tensor.any():
                                if obj not in objs:
                                    objs.append(obj)
                        k1 = ""
                        for obj in objs:
                            k1 += f"(is-collect {obj})(not (is-sink {obj}))"
                        print(objs)
                        for obj in states[0]._object_names:
                            if obj != 'r' and obj not in objs:
                                k1 += f"(not(is-collect {obj}))"
                        
                        knowledge1 = f"(and {k1})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        toggle_objs = []

                        for i, action in enumerate(actions):
                            if action.name in ["toggle"]:
                                if action.arguments[-1] not in toggle_objs:
                                    toggle_objs.append(action.arguments[-1])
                                break
                        
                        print(toggle_objs)
                        k2 = ""
                        for obj in toggle_objs:
                            k2 += f"(is-sink {obj})"
                        knowledge2 = f"(and {k2}(not (is-sink bed_0)))"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
                    elif task == "MakingTea":
                        pick_num = 0
                        
                        objs = []
                        for i, action in enumerate(actions):
                            if action.name in ["pickup_0", "pickup_1", "pickup_2"]:
                                if action.arguments[-1] not in objs:
                                    obj = action.arguments[-1]
                                    objs.append(obj)
                                    if pick_num == 0:
                                        knowledge1 = f"(and(is-teapot {obj})(not (is-stove {obj}))(not (is-cabinet {obj})))"
                                    elif pick_num == 1:
                                        knowledge1 = f"(and(is-teabag {obj})(not (is-stove {obj}))(not (is-cabinet {obj})))"
                                    
                                    pick_num += 1
                                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                                    target = torch.ones(len(pred))
                                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                                    
                                    knowledge2 = f"(exists (?t - item)(and(is-cabinet ?t)(not(is-stove ?t))(inside {obj} ?t)))"
                                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                                    target = torch.cat([torch.ones(i + 1), torch.zeros(len(pred)-i-1)])
                                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        print(objs)
                        for obj in states[0]._object_names:
                            if obj not in objs:

                                knowledge1 = f"(and(not(is-teapot {obj}))(not(is-teabag {obj})))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                                target = torch.ones(len(pred))
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                            
                        for i, action in enumerate(actions):
                            if action.name in ["toggle"]:
                                obj = action.arguments[-1]
                                knowledge3 = f"(toggleon {obj})"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                                target = torch.cat([torch.zeros(i+1), torch.ones(len(pred)-i-1)])
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                                
                                knowledge4 = f"(and(not(is-cabinet {obj}))(is-stove {obj}))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                                target = torch.ones(len(pred))
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        knowledge5 = "(exists (?o - item) (and (is-electric_refrigerator ?o)(not(is-cabinet ?o))(not(is-stove ?o)) (inside lemon_0 ?o)))"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge5)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                    elif task == "CollectMisplacedItems":
                        objs = []
                        for obj in name_dict:
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(robot-holding r {obj})")).tensor.any():
                                if obj not in objs:
                                    objs.append(obj)
                        k1 = ""
                        for obj in objs:
                            k1 += f"(is-collect {obj})(not(is-table {obj}))"
                            
                        knowledge1 = f"(and {k1})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        k2 = ""
                        for obj in states[0]._object_names:
                            if obj not in objs:
                                k2 += f"(not(is-collect {obj}))(is-furniture {obj})"

                        knowledge2 = f"(and {k2})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        for i, action in enumerate(actions):
                            if action.name == "drop_2":
                                obj = action.arguments[-1]
                                
                                knowledge3 = f"(exists(?t - item)(and (robot-is-facing r ?t)(is-table ?t)(not (is-collect ?t))))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor[i]
                                target = torch.ones(1)
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                                
                                knowledge4 = f"(exists(?t - item)(and (is-table ?t)(ontop {obj} ?t)))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                                
                                target = torch.cat([torch.zeros(i + 1), torch.ones(len(pred)-i-1)])
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
                    elif task == "PuttingAwayDishesAfterCleaning":
                        objs = []
                        for obj in name_dict:
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(robot-holding r {obj})")).tensor.any():
                                if obj not in objs:
                                    objs.append(obj)
                        k1 = ""
                        for obj in objs:
                            k1 += f"(is-plate {obj})(not (is-cabinet {obj}))(not (is-countertop {obj}))"
                            
                        for obj in states[0]._object_names:
                            if obj != 'r' and obj not in objs:
                                k1 += f"(not(is-plate {obj}))"
                            
                        knowledge1 = f"(and {k1})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        open_objs = []
                        i = 0
                        for i,action in enumerate(actions):
                            if action.name in ["open"]:
                                if action.arguments[-1] not in open_objs:
                                    open_objs.append(action.arguments[-1])
                                break
                        k2 = ""
                        for obj in open_objs:
                            k2 += f"(is-cabinet {obj})(not(is-countertop {obj}))"
                        
                        knowledge2 = f"(and {k2})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        i = 0
                        for action in actions:
                            if action.name == "pickup_1":
                                break
                            i += 1
                        for obj in objs:
                            knowledge3 = f"(exists (?t - item) (and (is-countertop ?t) (ontop {obj} ?t)))"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                            pred = pred[:i]
                            target = torch.ones(i)
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
                    elif task == "WateringHouseplants":
                        objs = []
                        for obj in name_dict:
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(robot-holding r {obj})")).tensor.any():
                                if obj not in objs:
                                    objs.append(obj)
                        k1 = ""
                        for obj in objs:
                            k1 += f"(is-plant {obj})"
                        print(objs)
                        for obj in states[0]._object_names:
                            if obj != 'r' and obj not in objs:
                                k1 += f"(not(is-plant {obj}))"
                                
                        toggle_objs = []
                        i = 0
                        for i, action in enumerate(actions):
                            if action.name in ["toggle"]:
                                if action.arguments[-1] not in toggle_objs:
                                    toggle_objs.append(action.arguments[-1])
                                break
                        knowledge1 = f"(and {k1})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
        
                        k2 = ""
                        for obj in toggle_objs:
                            k2 += f"(is-sink {obj})"
                        print(toggle_objs)
                        knowledge2 = f"(and {k2})"

                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        k3 = ""
                        for obj in states[0]._object_names:
                            if obj != 'r':
                                if self.domain.forward_expr(batch_state, [], self.domain.parse(f"(equal (robot-pose r)(item-pose {obj}))")).tensor.any():
                                    k3 += f"(is-door {obj})"
                                else:
                                    k3 += f"(not(is-door {obj}))"
                        knowledge3 = f"(and {k3})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
                    elif task == "OrganizingFileCabinet":
                        objs = []
                        for obj in name_dict:
                            cond = f"(robot-holding r {obj})"
                            if self.domain.forward_expr(batch_state, [], self.domain.parse(cond)).tensor.any():
                                objs.append(obj)
                        k1 = ""
                        for obj in objs:
                            k1 += f"(is-collect {obj})(not (is-cabinet {obj}))"
                        print(objs)
                        
                        knowledge1 = f"(and {k1})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 3
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        k2 = ""
                        for obj in name_dict:
                            if obj not in objs:
                                k2 += f"(not(is-collect {obj}))(is-furniture {obj})"

                        knowledge2 = f"(and {k2})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                        open_objs = []
                            
                        for i, action in enumerate(actions):
                            if action.name in ["open"]:
                                obj = action.arguments[-1]
                                if obj not in open_objs:
                                    open_objs.append(obj)
                        
                        print(open_objs)

                        k4 = ""
                        for obj in open_objs:
                            k4 += f"(is-cabinet {obj})(not(is-table {obj}))"
                        for obj in objs:
                            k4 += f"(not(is-cabinet {obj}))"

                        knowledge4 = f"(and {k4})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                    
        return forward_ctx.finalize()
