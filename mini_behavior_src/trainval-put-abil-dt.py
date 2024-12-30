import os.path as osp
import time
from copy import deepcopy
import random
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import jacinle
import jacinle.io as io
from jactorch.io import load_state_dict, state_dict

import hacl.pdsketch as pds
import hacl.envs.mini_behavior.minibehavior_v2023 as mb

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

parser = jacinle.JacArgumentParser()
parser.add_argument('env', choices=['mini_behavior'])
parser.add_argument('task', choices=mb.SUPPORTED_TASKS)
parser.add_argument('--seed', type=int, default=33)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--update-interval', type=int, default=50)
parser.add_argument('--print-interval', type=int, default=50)
parser.add_argument('--evaluate-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=500)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--use-offline', type='bool', default=True)

parser.add_argument('--action-mode', type=str, default='envaction', choices=mb.SUPPORTED_ACTION_MODES)
parser.add_argument('--structure-mode', type=str, default='abl', choices=mb.SUPPORTED_STRUCTURE_MODES)
parser.add_argument('--load', type='checked_file', default=None)
parser.add_argument('--load_domain', type='checked_file', default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--generalize', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize-failure', action='store_true', help='visualize the tasks that the planner failed to solve.')
parser.add_argument('--evaluate-objects', type=int, default=4)

# append the results
parser.add_argument('--append-expr', action='store_true')
parser.add_argument('--append-result', action='store_true')

# debug options
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

if args.workers > 0:
    import ray
    ray.init()

if args.env == 'mini_behavior':
    lib = mb
    import hacl.p.kfac.minibehavior.ground_models as lib_models
    from hacl.p.kfac.minibehavior.data_generator import worker_eval
    from hacl.p.kfac.minibehavior.abil_dt_data_processor import worker_abil
    
    worker_inner = worker_abil(1)

else:
    raise ValueError(f'Unknown environment: {args.env}')

Model = lib_models.get_model(args)

assert args.task in lib.SUPPORTED_TASKS
assert args.action_mode in lib.SUPPORTED_ACTION_MODES
assert args.structure_mode in lib.SUPPORTED_STRUCTURE_MODES

logger = jacinle.get_logger(__file__)
log_timestr = time.strftime(time.strftime('%Y-%m-%d-%H-%M-%S'))
if args.load is not None:
    load_id = osp.basename(args.load).split('-')
    if load_id[1] == 'goto' and load_id[2] == 'single':
        load_id = 'gotosingle'
    else:
        load_id = load_id[1]
else:
    load_id = 'scratch'

args.id_string = f'{args.task}-{args.action_mode}-{args.structure_mode}-lr={args.lr}-load={load_id}-{log_timestr}'
args.log_filename = f'dumps/abil-dt-{args.id_string}.log'
args.json_filename = f'dumps/abil-dt-{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))

if not args.debug and args.append_expr:
    with open('./experiments.txt', 'a') as f:
        f.write(args.id_string + '\n')


def build_model(args, domain, **kwargs):
    return Model(domain, **kwargs)

def evaluate(env, model, domain_gr, visualize: bool = False):
    model.cpu()
    if visualize:
        # lib.visualize_planner(env, planner)
        pass
    else:
        jacinle.reset_global_seed(args.seed+1, True)
        succ, nr_expansions, total = 0, 0, 0
        nr_expansions_hist = list()
        scores = list()

        for i in jacinle.tqdm(100, desc='Evaluation'):
            is_valid = False
            while not is_valid:
                obs = env.reset()
                c_env = deepcopy(env)
                is_valid = worker_eval(args, model.domain, c_env)
            
            pddl_action = None
            his_action = []
            states = []
            model.actions = []
            model.states = []
            this_succ = False
            steps = 0
            while steps < MAX_STEPS:
                obs = env.compute_obs()
                state, goal = obs['state'], obs['mission']
                states.append(state)
                model.states.append(state)
                pddl_action = model.make_action(domain_gr, states[-2:], his_action[-1:], args.task)
                his_action.append(pddl_action)
                model.actions.append(pddl_action)
                # print(pddl_action)

                _, _, (done,score), _ = env.step(pddl_action)
                steps = steps + 1 
                if done:
                    break
            
            (done, score) = env.compute_done()
            print("Score:", score)
            scores.append(score)
            if done:
                this_succ = True
                nr_expansions += steps
                
            succ += int(this_succ)
            total += 1
            print(this_succ, steps)
            nr_expansions_hist.append({'succ': this_succ, 'expansions': steps})

        evaluate_key = f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects}'
        io.dump(f'./results/{evaluate_key}.json', nr_expansions_hist)
        print("avg_score =",sum(scores) / total)
        return succ / total, nr_expansions / max(succ,1)
    
    
def validate(model, val_set):
    model.eval()
    outputs = []
    targets = []
    i = 1
    
    while i < len(val_set):
        end = time.time()
        robot_tensors, object_tensors, action_tensors, filt_obj_tensors = val_set[i-1]
        loss, _, output = model.dt(robot_tensors, object_tensors, action_tensors, filt_obj_tensors)

        outputs.append(output)
        targets.append(torch.stack(action_tensors).flatten())

        i = i + 1
        if i > len(val_set):
            break
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    
    end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    try:
        val_acc = 100. * accuracy(labels, logits)
    except:
        val_acc = 0.
    
    print(f'{end} Val_ACC: {val_acc:.2f}')
    sys.stdout.flush()
    

if args.env == 'mini_behavior':
    lib = mb

from hacl.util import accuracy, goal_test
from hacl.p.kfac.minibehavior.abil_dt_policy import ABIL_DTPolicyNetwork

if args.task == 'SortingBooks':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2']
    # ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2']
    # ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'drop_2']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2']
    PREDICATES = ['is-book', 'is-hardback', 'ontop']
    SEQ_LENGTH = 4
    MAX_STEPS = 80
elif args.task == 'Throwing_away_leftovers':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_2', 'drop_in']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'drop_in', 'pickup_2']
    # ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_2']
    # ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'drop_in']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_2', 'drop_in']
    PREDICATES = ['is-ashcan', 'is-hamburger', 'inside', 'ontop']
    SEQ_LENGTH = 3
    MAX_STEPS = 80
elif args.task == 'PuttingAwayDishesAfterCleaning':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_1', 'drop_in']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_1', 'open', 'drop_in']
    # ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_1']
    # ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'open', 'drop_in']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_1', 'open', 'drop_in']
    PREDICATES = ['is-plate', 'is-cabinet']
    SEQ_LENGTH = 4
    MAX_STEPS = 100
elif args.task == 'BoxingBooksUpForStorage':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_in']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_in']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_in']
    PREDICATES = ['is-book', 'is-box', 'is-shelf', 'inside']
    SEQ_LENGTH = 5
    MAX_STEPS = 110
elif args.task == 'LayingWoodFloors':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_0']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_0']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_0']
    PREDICATES = ['nextto']
    SEQ_LENGTH = 6
    MAX_STEPS = 100
elif args.task == 'WateringHouseplants':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_in']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_in', 'toggle']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_in', 'toggle']
    PREDICATES = ['inside']
    SEQ_LENGTH = 2
    MAX_STEPS = 100
elif args.task == 'CollectMisplacedItems':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2']
    PREDICATES = ['is-collect', 'is-table', 'ontop']
    SEQ_LENGTH = 5
    MAX_STEPS = 120
elif args.task == 'CleaningShoes':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_1', 'drop_in', 'toggle']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_1', 'drop_in', 'toggle']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_1', 'drop_in', 'toggle']
    PREDICATES = ['is-clean']
    SEQ_LENGTH = 5
    MAX_STEPS = 100
elif args.task == 'MakingTea':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_2', 'open', 'drop_in']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_2', 'drop_in', 'open']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_2', 'drop_in', 'open', 'toggle']
    PREDICATES = ['ontop', 'toggleon']
    SEQ_LENGTH = 5
    MAX_STEPS = 80
elif args.task == 'OrganizingFileCabinet':
    ACTIONS_PICK = ['forward', 'lturn', 'rturn', 'open', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_in', 'drop_2']
    ACTIONS_PLACE = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2', 'drop_in']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2', 'drop_in', 'open']
    PREDICATES = ['inside', 'ontop']
    SEQ_LENGTH = 7
    MAX_STEPS = 160

class ABIL_DTModel(nn.Module):
    def __init__(self, task, domain):
        super(ABIL_DTModel, self).__init__()
        self.domain = domain
        self.task = task
        self.actions = []
        self.states = []
        self.state = 1

        for i in range(len(ACTIONS_PICK)):
            ACTIONS_PICK[i] = domain.operators[ACTIONS_PICK[i]]('r')
        self.pick_abs = ABIL_DTPolicyNetwork(domain, action_space = ACTIONS_PICK, predicates=PREDICATES)
        
        for i in range(len(ACTIONS_PLACE)):
            ACTIONS_PLACE[i] = domain.operators[ACTIONS_PLACE[i]]('r')
        self.place_abs = ABIL_DTPolicyNetwork(domain, action_space = ACTIONS_PLACE, predicates=PREDICATES)
    
    def reset_state(self, new_state):
        if self.state != new_state:
            self.actions = []
            self.states = [self.states[-1]]
            self.state = new_state
            
    def make_action(self, domain_gr, states, his_action, task):
        if task == "BoxingBooksUpForStorage":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item) (or(is-book ?o)(is-furniture ?o)))"
                prob = self.pick_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item) (or(is-book ?o)(is-furniture ?o)))"
                prob = self.place_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
            
        elif task == "SortingBooks":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item)(or(is-book ?o)))"
                prob = self.pick_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item)(or(is-book ?o)(is-shelf ?o)) )"
                prob = self.place_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
            
        elif task == "Throwing_away_leftovers":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item) (or(is-countertop ?o)(is-hamburger ?o)))"
                prob = self.pick_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item) (or(is-hamburger ?o)(is-ashcan ?o)))"
                prob = self.place_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
        
        elif task == "LayingWoodFloors":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item) (and(is-plywood ?o) ) )"
                prob = self.pick_abs.forward_state(states, his_action, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
            else:
                filt_expr = "(foreach (?o - item) (and(is-plywood ?o) ) )"
                prob = self.place_abs.forward_state(states, his_action, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
        
        elif task == "PuttingAwayDishesAfterCleaning":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                self.state = 1
                filt_expr = "(foreach (?o - item)(or(is-countertop ?o) (is-plate ?o) ) )"
                prob = self.pick_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                self.state = 2
                filt_expr = "(foreach (?t - item) (or(is-cabinet ?t)(is-plate ?t) ) )"
                prob = self.place_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]

        elif task == "WateringHouseplants":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item)(or (is-plant ?o) ))"
                prob = self.pick_abs.forward_state(states, his_action, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item)(or (is-plant ?o)(is-sink ?o) ))"
                prob = self.place_abs.forward_state(states, his_action, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
        
        elif task == "CollectMisplacedItems":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item)(or(is-collect ?o) (is-furniture ?o) ))"
                prob = self.pick_abs.forward_state(states, his_action, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item)(or(is-collect ?o) (is-table ?o) ))"
                prob = self.place_abs.forward_state(states, his_action, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
        
        elif task == "MakingTea":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(exists (?o - item)(and (is-stove ?o)(ontop teapot_0 ?o)(inside tea_bag_0 ?o)))")) > 0.9:
                return pds.rl.RLEnvAction("toggle")

            elif goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item)(or(is-teapot ?o) (is-teabag ?o) (is-cabinet ?o) ) )"
                prob = self.pick_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item)(or(is-teapot ?o) (is-teabag ?o) (is-stove ?o) ) )"
                prob = self.place_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
            
        elif task == "OrganizingFileCabinet":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item) (or(is-collect ?o) (is-furniture ?o)))"
                prob = self.pick_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item) (or(is-collect ?o) (is-furniture ?o)))"
                prob = self.place_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]
        
        elif task == "CleaningShoes":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(and(exists(?o - item)(and(is-sink ?o)(atSameLocation shoe_0 ?o)))(exists(?o - item)(and(is-sink ?o)(atSameLocation shoe_1 ?o))))")) > 0.9:
                return pds.rl.RLEnvAction("toggle")
            elif goal_test(domain_gr, states[-1], domain_gr.parse(f"(hands-free r)")) > 0.9:
                filt_expr = "(foreach (?o - item)(is-collect ?o ))"
                prob = self.pick_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PICK[action_index]
            else:
                filt_expr = "(foreach (?o - item)(or(is-collect ?o) (is-sink ?o) ))"
                prob = self.place_abs.forward_state(self.states[-2:], self.actions[-1:], filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PLACE[action_index]

    def forward(self, states, goal_tensors, actions, filt_obj_tensors):
        return 0
    
    def encod(self, states, actions, filt_expr):
        return self.pick_abs.encode(states, actions, filt_expr)
    
from hacl.p.kfac.minibehavior.ground_models import ModelABL

def main():
    jacinle.reset_global_seed(args.seed, True)
    lib.set_domain_mode(args.action_mode, args.structure_mode)

    logger.critical('Creating model...')
    domain_gr = lib.get_domain()
    
    model_g = ModelABL(
        domain_gr,
        goal_loss_weight=0,
        action_loss_weight=0
    )
    if args.load_domain is not None:
        try:
            load_state_dict(model_g, io.load(args.load_domain))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')
            
    model = ABIL_DTModel(args.task, domain_gr)
    model = model.cuda()
    
    if args.load is not None:
        try:
            load_state_dict(model, io.load(args.load))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')

    # It will be used for evaluation.
    logger.critical(f'Creating environment (task={args.task})...')

    if args.env == 'mini_behavior':
        env = lib.make(args.task, args.generalize)
    else:
        raise ValueError(f'Unknown environment: {args.env}.')
    env.env.seed(args.seed)

    if args.evaluate:
        args.iterations = 0

    logger.warning('Using single-threaded mode.')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    meters = jacinle.GroupMeters()
    
    pick_dataset = []
    place_dataset = []
    succ_num = 0

    for i in jacinle.tqdm(range(1, 30000)):
        if succ_num >= args.iterations:
            break
        end = time.time()
        states, actions, dones, filt_expr, succ, extra_monitors = worker_inner.gen_data(args=args, domain=domain_gr, env=env)
        if succ:

            for j in range(len(actions)):
                actions_set = [[], []]
                robot_set = [[], []]
                object_set = [[], []]
                filt_obj_set = [[], []]
                for seq in range(len(actions[j][0])):
                    robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.pick_abs.encode(states[j][0][seq], actions[j][0][seq], filt_expr[0])
                    robot_set[0].append(robot_tensor)
                    object_set[0].append(object_tensor)
                    actions_set[0].append(action_tensor)
                    filt_obj_set[0].append(filt_obj_tensor)
                if len(robot_set[0]) > 0:
                    pick_dataset.append([robot_set[0], object_set[0], actions_set[0], filt_obj_set[0]])
                
                for seq in range(len(actions[j][1])):
                    robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.place_abs.encode(states[j][1][seq], actions[j][1][seq], filt_expr[1])
                    robot_set[1].append(robot_tensor)
                    object_set[1].append(object_tensor)
                    actions_set[1].append(action_tensor)
                    filt_obj_set[1].append(filt_obj_tensor)
                if len(robot_set[1]) > 0:
                    place_dataset.append([robot_set[1], object_set[1], actions_set[1], filt_obj_set[1]])
                
            succ_num += 1
        
    print(f"Successful demo of all data:{succ_num}")
        
    # generate validation set
    val_pick_set = []
    val_place_set = []
    
    if not args.evaluate:
        succ_num = 0
        for i in jacinle.tqdm(range(1, 2000)):
            if succ_num >= args.iterations / 10:
                break
            end = time.time()

            states, actions, dones, filt_expr, succ, extra_monitors = worker_inner.gen_data(args=args, domain=domain_gr, env=env)
            if succ:

                for j in range(len(actions)):
                    actions_set = [[], []]
                    robot_set = [[], []]
                    object_set = [[], []]
                    filt_obj_set = [[], []]
                    for seq in range(len(actions[j][0])):
                        robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.pick_abs.encode(states[j][0][seq], actions[j][0][seq], filt_expr[0])
                        robot_set[0].append(robot_tensor)
                        object_set[0].append(object_tensor)
                        actions_set[0].append(action_tensor)
                        filt_obj_set[0].append(filt_obj_tensor)
                    
                    if len(robot_set[0]) > 0:
                        val_pick_set.append([robot_set[0], object_set[0], actions_set[0], filt_obj_set[0]])
                    
                    for seq in range(len(actions[j][1])):
                        robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.place_abs.encode(states[j][1][seq], actions[j][1][seq], filt_expr[1])
                        robot_set[1].append(robot_tensor)
                        object_set[1].append(object_tensor)
                        actions_set[1].append(action_tensor)
                        filt_obj_set[1].append(filt_obj_tensor)
                    
                    if len(robot_set[1]) > 0:
                        val_place_set.append([robot_set[1], object_set[1], actions_set[1], filt_obj_set[1]])                    
                succ_num += 1

    batch_size = args.batch_size
    
    if succ_num > 0:
        for epoch in range(50):
            random.shuffle(pick_dataset)
            outputs = []
            targets = []
            a_loss = []
            total_n, total_loss = 0, 0
            
            model.pick_abs.train()
            i = 1
            
            while i < len(pick_dataset):
                while total_n < batch_size:
                    end = time.time()
                    robot_tensors, object_tensors, action_tensors, filt_obj_tensors = pick_dataset[i-1]
                    loss, _, output = model.pick_abs.dt(robot_tensors, object_tensors, action_tensors, filt_obj_tensors)

                    outputs.append(output)
                    targets.append(torch.stack(action_tensors).flatten())
                    a_loss.append(loss)

                    n = len(action_tensors)
                    total_n += n
                    total_loss += loss * n

                    i = i + 1
                    if i > len(pick_dataset):
                        break

                if total_loss.requires_grad:
                    opt.zero_grad()
                    total_loss /= total_n
                    total_loss.backward()
                    opt.step()
                    
                total_n, total_loss = 0, 0
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            logits = outputs.softmax(-1).data.cpu().numpy()
            labels = targets.data.cpu().numpy()
            
            end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            try:
                train_acc = 100. * accuracy(labels, logits)
            except:
                train_acc = 0.
            
            print(f'{end} [Epoch {epoch}] loss: {torch.mean(torch.stack(a_loss)):.4f} pick_abs_ACC: {train_acc:.2f}')
            sys.stdout.flush()
            
            validate(model.pick_abs, val_pick_set)
            
            random.shuffle(place_dataset)
            outputs = []
            targets = []
            a_loss = []
            total_n, total_loss = 0, 0
            
            model.place_abs.train()
            i = 1
            
            while i < len(place_dataset):
                while total_n < 32:
                    end = time.time()
                    robot_tensors, object_tensors, action_tensors, filt_obj_tensors = place_dataset[i-1]
                    loss, _, output = model.place_abs.dt(robot_tensors, object_tensors, action_tensors, filt_obj_tensors)

                    outputs.append(output)
                    targets.append(torch.stack(action_tensors).flatten())
                    a_loss.append(loss)

                    n = len(action_tensors)
                    total_n += n
                    total_loss += loss * n

                    i = i + 1
                    if i > len(pick_dataset):
                        break

                if total_loss.requires_grad:
                    opt.zero_grad()
                    total_loss /= total_n
                    total_loss.backward()
                    opt.step()
                    
                total_n, total_loss = 0, 0
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            logits = outputs.softmax(-1).data.cpu().numpy()
            labels = targets.data.cpu().numpy()
            
            end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            try:
                train_acc = 100. * accuracy(labels, logits)
            except:
                train_acc = 0.
            
            print(f'{end} [Epoch {epoch}] loss: {torch.mean(torch.stack(a_loss)):.4f} place_abs_ACC: {train_acc:.2f}')
            sys.stdout.flush()
            
            validate(model.place_abs, val_place_set)

            if (epoch + 1) % 10 == 0:
                ckpt_name = f'dumps/seed{args.seed}/abil-dt-{args.task}-load={load_id}-epoch={epoch}.pth'
                logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
                io.dump(ckpt_name, state_dict(model))
            
    if args.evaluate:
        if args.env == 'mini_behavior':
            env.set_options(nr_objects=args.evaluate_objects)
            env.env.seed(args.seed+1)
        if not args.visualize:
            succ_rate, avg_expansions = evaluate(env, model, domain_gr, visualize=False)
            if args.append_result:
                with open('./experiments-eval.txt', 'a') as f:
                    print(f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects},{succ_rate},{avg_expansions}', file=f)
            print(f'succ_rate = {succ_rate}')
            print(f'avg_expansions = {avg_expansions}')
        else:
            evaluate(domain_gr, env, heuristic_model=None, visualize=True)
    else:
        ckpt_name = f'dumps/seed{args.seed}/abil-dt-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

    # from IPython import embed; embed()

if __name__ == '__main__':
    main()
