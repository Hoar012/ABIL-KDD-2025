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

class PDSModel(nn.Module):
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

    def forward(self, feed_dict, task, forward_augmented=False):
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
            
            self.domain.forward_derived_features(batch_state)

            if goal_expr is not None:
                pred = self.domain.forward_expr(batch_state, [], goal_expr).tensor

                target = done
                if self.training:
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())
                    # loss = self.bce(pred, target.float())
                    forward_ctx.add_loss(loss, 'goal', accumulate=self.goal_loss_weight)
                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal')
                
        return forward_ctx.finalize()


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

    def forward(self, feed_dict, task, forward_augmented=False):
        forward_ctx = ForwardContext(self.training)
        with forward_ctx.as_default():
            goal_expr = feed_dict['goal_expr']
            states, actions, done = feed_dict['states'], feed_dict['actions'], feed_dict['dones']
            predicates = feed_dict['predicates']

            assert forward_augmented
            if forward_augmented:
                for state in states:
                    # print(state)
                    self.domain.forward_augmented_features(state)

            batch_state = BatchState.from_states(self.domain, states)
            name_dict = batch_state._object_name2index[0]
            final_img = states[-1].features.tensor_dict["item-image"].tensor
            # print(batch_state)
            # print(name_dict)
            
            self.domain.forward_derived_features(batch_state)

            
            if goal_expr is not None:
                pred = self.domain.forward_expr(batch_state, [], goal_expr).tensor

                # print(pred, done)
                target = done
                if self.training:
                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target.float())
                    # loss = self.bce(pred, target.float())
                    forward_ctx.add_loss(loss, 'goal', accumulate=self.goal_loss_weight)
                forward_ctx.add_accuracy(((pred > 0.5) == target).float().mean(), 'goal')
                # print((goal_expr, pred, target))
                
            succ = done[-1]

            if succ:
                if task == "packing-shapes":
                    objs = []

                    for obj, typ in name_dict.items():
                        if typ[0] == "item":
                            objs.append(obj)
                    
                    poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - item) (item-pose ?o))")).tensor[0]

                    dist = torch.norm(poses - actions[0][0][:3], 1, dim=1)
                    picked_obj = objs[dist.argmin()]
                    # print(picked_obj, goal_expr)
                    k = ""

                    for shape in ['is-letter_R', 'is-letter_A', 'is-triangle', 'is-square', 'is-plus', 'is-letter_T', 'is-diamond',
                  'is-pentagon', 'is-rectangle', 'is-flower', 'is-star', 'is-circle', 'is-letter_G', 'is-letter_V',
                  'is-letter_E', 'is-letter_L', 'is-ring', 'is-hexagon', 'is-heart', 'is-letter_M']:
                        k = ""
                        if str(goal_expr).find(shape) > -1:
                            k += f"({shape} {picked_obj})"
                            knowledge = k
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        else:
                            k += f"(not({shape} {picked_obj}))"
                            knowledge = k
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1 / 19
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        
                elif task == "packing-5shapes":
                    objs = []

                    for obj, typ in name_dict.items():
                        if typ[0] == "item":
                            objs.append(obj)
                    
                    poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - item) (item-pose ?o))")).tensor[0]

                    dist = torch.norm(poses - actions[0][0], 1, dim=1)
                    picked_obj = objs[dist.argmin()]
                    k = ""             

                    for shape in ['is-letter_R', 'is-letter_A', 'is-triangle', 'is-square', 'is-plus']:
                        k = ""
                        if str(goal_expr).find(shape) > -1:
                            k += f"({shape} {picked_obj})"
                        else:
                            k += f"(not({shape} {picked_obj}))"
                        knowledge = k
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                elif task == "place-red-in-green":

                    objs = []
                    containers = []
                    picked_objs = []
                    place_objs = []

                    for obj, typ in name_dict.items():
                        if typ[0] == "item":
                            objs.append(obj)
                        elif typ[0] == "container":
                            containers.append(obj)

                    for i in range(len(actions)-1):
                        item_poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - item) (item-pose ?o))")).tensor[i]
                        dist = torch.norm(item_poses - actions[i][0][:3], 1, dim=1)
                        picked_obj = objs[dist.argmin()]
                        if picked_obj not in picked_objs:
                            picked_objs.append(picked_obj)
                        
                            knowledge1 = f"(is-red {picked_obj})"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        container_poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - container) (container-pose ?o))")).tensor[i]
                        dist = torch.norm(container_poses - actions[i][1][:3], 1, dim=1)

                        place_obj = containers[dist.argmin()]
                        if place_obj not in place_objs:
                            place_objs.append(place_obj)
                            knowledge2 = f"(c-is-green {place_obj})"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor

                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)


                    for obj in objs:
                        if obj not in picked_objs:
                            knowledge3 = f"(not(is-red {obj}))"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                            
                    for container in containers:
                        if container not in place_objs:
                            knowledge4 = f"(not(c-is-green {container}))"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge4)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                            
                elif task in ["put-block-in-bowl-seen-colors", "put-block-in-bowl-composed-colors"]:
                    objs = []
                    containers = []
                    picked_objs = []
                    place_objs = []

                    for obj, typ in name_dict.items():
                        if typ[0] == "item":
                            objs.append(obj)
                        elif typ[0] == "container":
                            containers.append(obj)

                    for i in range(len(actions)-1):
                        item_poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - item) (item-pose ?o))")).tensor[i]
                        dist = torch.norm(item_poses - actions[i][0][:3], 1, dim=1)
                        picked_obj = objs[dist.argmin()]
                        if picked_obj not in picked_objs:
                            picked_objs.append(picked_obj)
                        
                            for color in ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']:
                                knowledge1 = ""

                                if color == predicates[0]:
                                    knowledge1 = f"(is-{color} {picked_obj})"
                                    
                                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                                    target = torch.ones(len(pred))
                                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 3
                                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                                else:
                                    knowledge1 = f"(not(is-{color} {picked_obj}))"

                                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge1)).tensor
                                    target = torch.ones(len(pred))
                                    loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 3 / 6
                                    forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                        container_poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - container) (container-pose ?o))")).tensor[i]
                        dist = torch.norm(container_poses - actions[i][1][:3], 1, dim=1)

                        place_obj = containers[dist.argmin()]
                        if place_obj not in place_objs:
                            place_objs.append(place_obj)

                            for color in ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']:
                                knowledge2 = ""

                                if color == predicates[1]:
                                    knowledge2 = f"(c-is-{color} {place_obj})"
                                else:
                                    knowledge2 = f"(not(c-is-{color} {place_obj}))"
                                try:
                                    pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge2)).tensor
                                except:
                                    from IPython import embed; embed()
                                    exit()
                                target = torch.ones(len(pred))
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                    color = predicates[0]
                    for obj in objs:
                        if obj not in picked_objs:
                            knowledge3 = f"(not(is-{color} {obj}))"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge3)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                
                elif task in ["separating-piles-seen-colors", "separating-20piles", "separating-10piles"]:
                    objs = []
                    containers = []
                    picked_objs = []
                    place_objs = []

                    for obj, typ in name_dict.items():
                        if typ[0] == "item":
                            objs.append(obj)
                        elif typ[0] == "container":
                            containers.append(obj)

                    scores = []
                    for zone in containers:
                        pred_expr = f"(foreach (?o - item)(is-in ?o {zone}))"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(pred_expr)).tensor[-1]
                        scores.append(torch.sum(pred))

                    # print(scores)
                    
                    goal_zone = containers[torch.tensor(scores).argmax()]
                    
                    for color in ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']:
                        knowledge = ""

                        if color == predicates[1]:
                            knowledge = f"(c-is-{color} {goal_zone})"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                            # print(pred)
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        else:
                            knowledge = f"(not(c-is-{color} {goal_zone}))"
                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                            # print(pred)
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 0.1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

                    for zone in containers:
                        if zone != goal_zone:
                            color = predicates[1]
                            knowledge = f"(not(c-is-{color} {zone}))"

                            pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                            target = torch.ones(len(pred))
                            loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 0.1
                            forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                            
                elif task == "assembling-kits":
                    objs = []
                    containers = []

                    for obj, typ in name_dict.items():
                        if typ[0] == "item":
                            objs.append(obj)
                        elif typ[0] == "container":
                            containers.append(obj)

                    for i in range(len(actions)-1):
                        item_poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - item) (item-pose ?o))")).tensor[i]
                        dist = torch.norm(item_poses - actions[i][0][:3], 1, dim=1)
                        picked_obj = objs[dist.argmin()]
                        
                        container_poses = self.domain.forward_expr(batch_state, [], self.domain.parse("(foreach (?o - container) (container-pose ?o))")).tensor[i]
                        dist = torch.norm(container_poses - actions[i][1][:3], 1, dim=1)

                        place_obj = containers[dist.argmin()]
                        
                        # print(picked_obj, place_obj)
                        
                        knowledge = f"(is-same-shape {picked_obj} {place_obj})"
                        pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                        target = torch.ones(len(pred))
                        loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 1
                        forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)
                        # print(pred)
                        for container in containers:
                            if container != place_obj:
                                knowledge = f"(not(is-same-shape {picked_obj} {container}))"
                                pred = self.domain.forward_expr(batch_state, [], self.domain.parse(knowledge)).tensor
                                target = torch.ones(len(pred))
                                loss = jacf.pn_balanced_binary_cross_entropy_with_probs(pred, target) * 0.2
                                forward_ctx.add_loss(loss, 'knowledge', accumulate=self.goal_loss_weight)

        return forward_ctx.finalize()
