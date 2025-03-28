import os.path as osp
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import jacinle
import jacinle.io as io
import jactorch
from jactorch.io import load_state_dict, state_dict

import hacl.pdsketch as pds
import hacl.envs.gridworld.minigrid.minigrid_v2023 as mg

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

parser = jacinle.JacArgumentParser()
parser.add_argument('env', choices=['minigrid'])
parser.add_argument('task', choices=mg.SUPPORTED_TASKS)
parser.add_argument('--model', choices=['abil-bc'])
parser.add_argument('--seed', type=int, default=33)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--use-offline', type='bool', default=True)

parser.add_argument('--structure-mode', type=str, default='abl', choices=mg.SUPPORTED_STRUCTURE_MODES)
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

if args.env == 'minigrid':
    lib = mg
    import hacl.p.kfac.minigrid.ground_models as lib_models
    
    from hacl.p.kfac.minigrid.abil_bc_data_processor import worker_inst, worker_unlock, worker_put
    if args.task == "unlock":
        worker_inner = worker_unlock
    elif args.task == "put":
        worker_inner = worker_put
    else:
        worker_inner = worker_inst
else:
    raise ValueError(f'Unknown environment: {args.env}')

Model = lib_models.get_model(args)

assert args.task in lib.SUPPORTED_TASKS
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

args.id_string = f'{args.task}-{args.structure_mode}-lr={args.lr}-load={load_id}-{log_timestr}'
args.log_filename = f'dumps/abil-bc-{args.id_string}.log'
args.json_filename = f'dumps/abil-bc-{args.id_string}.jsonl'

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

    jacinle.reset_global_seed(args.seed+1, True)
    succ, nr_expansions, total = 0, 0, 0
    nr_expansions_hist = list()
    for i in jacinle.tqdm(100, desc='Evaluation'):
        obs = env.reset()
        env_copy = deepcopy(env)
        plan = []
        action = None
        this_succ = False
        steps = 0
        while steps < 30:
            obs = env.compute_obs()
            state, goal = obs['state'], obs['mission']
            rl_action = model.make_action(domain_gr, state, args.task, goal, env)
            plan.append(rl_action)

            _, _, done, _ = env.step(rl_action)
            steps = steps + 1 
            if done:
                break
        
        done = env.compute_done()
        if done:
            this_succ = True
                
        nr_expansions += steps
        succ += int(this_succ)
        total += 1

        print(this_succ, steps)
        nr_expansions_hist.append({'succ': this_succ, 'expansions': steps})
        
        if visualize:
            lib.visualize_plan(env_copy, plan)
            
    evaluate_key = f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects}'
    io.dump(f'./results/{evaluate_key}.json', nr_expansions_hist)
    return succ / total, nr_expansions / total


if args.env == 'minigrid':
    lib = mg

from hacl.util import accuracy, goal_test
from hacl.p.kfac.minigrid.abil_bc_policy import ABIL_BCPolicyNetwork

ACTIONS = ['forward', 'lturn', 'rturn']
PREDICATES = ['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box', 'is-door']

class BCModel(nn.Module):
    def __init__(self, task, domain):
        super(BCModel, self).__init__()
        self.domain = domain
        self.task = task

        for i in range(len(ACTIONS)):
            ACTIONS[i] = domain.operators[ACTIONS[i]]('r')
        self.goto_abs = ABIL_BCPolicyNetwork(domain, action_space = ACTIONS, predicates=PREDICATES, goal_augment=True)
        
    def make_action(self, domain_gr, state, task, goal, env):
        
        if task in ["gotosingle", "goto"]:
            goal_obj = env.goal_obj
            typ = goal_obj.type
            color = goal_obj.color
            filt_expr = f'(foreach (?o - item) (and (is-{typ} ?o)(is-{color} ?o)))'
            prob = self.goto_abs.forward_state(state, goal, filt_expr)
            action_index = prob.argmax(dim=0)
            return ACTIONS[action_index]
        
        elif task == "pickup":
            goal_obj = env.goal_obj
            typ = goal_obj.type
            color = goal_obj.color
            if goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (not (exists (?o - item) (and (robot-is-facing r ?o) (is-{typ} ?o) (is-{color} ?o))) ) )")) > 0.5:
                sub_goal = domain_gr.parse(
                f'(exists (?o - item) (and (is-{typ} ?o) (is-{color} ?o)))'
                )
                filt_expr = f'(foreach (?o - item) (and (is-{typ} ?o)(is-{color} ?o)))'
                prob = self.goto_abs.forward_state(state, sub_goal, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            elif goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (exists (?o - item) (and (robot-is-facing r ?o) (is-{typ} ?o) (is-{color} ?o)) ) )")) > 0.5:
                return pds.rl.RLEnvAction("pickup")
            else:
                return pds.rl.RLEnvAction("forward")
            
        elif task == "open":
            goal_obj = env.goal_obj
            color = goal_obj.color
            if goal_test(domain_gr, state, domain_gr.parse(f"(exists (?o - item) (and (robot-is-facing r ?o) (is-{color} ?o) (is-door ?o))) ")) < 0.5:
                sub_goal = domain_gr.parse(
                f'(exists (?o - item) (and (is-door ?o) (is-{color} ?o)))'
                )
                filt_expr = f'(foreach (?o - item) (and (is-door ?o)(is-{color} ?o)))'
                prob = self.goto_abs.forward_state(state, sub_goal, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
        
            elif goal_test(domain_gr, state, domain_gr.parse(f"(exists (?o - item) (and (robot-is-facing r ?o) (is-{color} ?o) (is-door ?o)) ) ")) > 0.5:
                return pds.rl.RLEnvAction("toggle")
            
        elif task == "unlock":
            color = env.goal_obj.color
        
            if goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (not (exists (?o - item) (and (robot-is-facing r ?o) (is-key ?o) (is-{color} ?o))) ) )")) > 0.5:
                sub_goal = domain_gr.parse(
                f'(exists (?o - item) (and (is-key ?o) (is-{color} ?o)))'
                )
                filt_expr = f'(foreach (?o - item) (and (is-key ?o)(is-{color} ?o )))'
                prob = self.goto_abs.forward_state(state, sub_goal, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            
            elif goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (exists (?o - item) (and (robot-is-facing r ?o) (is-key ?o) (is-{color} ?o)) ) )")) > 0.5:
                return pds.rl.RLEnvAction("pickup")
            
            elif goal_test(domain_gr, state, domain_gr.parse(f"(not (exists (?o - item) (and (robot-is-facing r ?o) (is-door ?o) (is-{color} ?o)) ) )")) > 0.5:
                sub_goal = domain_gr.parse(
                f'(exists (?o - item) (and (is-door ?o) (is-{color} ?o)))'
                )
                filt_expr = f'(foreach (?o - item) (and (is-door ?o)(is-{color} ?o)))'
                prob = self.goto_abs.forward_state(state, sub_goal, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            
            else:
                return pds.rl.RLEnvAction("toggle")
        
        elif task == "put":
            if goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (not (exists (?o - item) (and (robot-is-facing r ?o) (or(is-ball ?o)) )) ) )")) > 0.7:
                sub_goal = domain_gr.parse(
                f'(exists (?o - item) (and (is-ball ?o)))'
                )
                filt_expr = '(foreach (?o - item) (or (is-ball ?o)))'
                prob = self.goto_abs.forward_state(state, sub_goal, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            
            elif goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (exists (?o - item) (and (robot-is-facing r ?o) (or(is-ball ?o)) ) ) )")) > 0.7:
                return pds.rl.RLEnvAction("pickup")
            
            elif goal_test(domain_gr, state, domain_gr.parse(f"(not (exists (?o - item) (and (nextto (item-pose ?o) (robot-facing r) ) (or(is-box ?o)) )) )")) > 0.7:
                sub_goal = domain_gr.parse(
                f'(exists (?o - item) (and (is-box ?o)))'
                )
                filt_expr = '(foreach (?o - item) (or (is-box ?o)))'
                prob = self.goto_abs.forward_state(state, sub_goal, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            
            else:
                return pds.rl.RLEnvAction("place")
    
    def forward(self, states, goal_tensors, object_tensors, actions, goal_obj_tensors):
        return self.goto_abs.bc(states, goal_tensors, object_tensors, actions, goal_obj_tensors)
    
    def encod(self, states, actions, goal, filt_expr):
        return self.goto_abs.encode(states, actions, goal, filt_expr)
    
from hacl.p.kfac.minigrid.ground_models import ModelABL

def main():
    jacinle.reset_global_seed(args.seed, True)
    lib.set_domain_mode('envaction', args.structure_mode)
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
            
    logger.critical('Creating model...')
    
    model = BCModel(args.task, domain_gr)
    if args.load is not None:
        try:
            load_state_dict(model, io.load(args.load))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')
            
    # It will be used for evaluation.
    logger.critical(f'Creating environment (task={args.task})...')

    if args.env == 'minigrid':
        env = lib.make(args.task, args.generalize)
    else:
        raise ValueError(f'Unknown environment: {args.env}.')

    if args.evaluate:
        args.iterations = 0

    logger.warning('Using single-threaded mode.')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    succ_num = 0
    dataset = []
    
    for i in jacinle.tqdm(range(1, args.iterations + 1)):        
        end = time.time()
        states, actions, dones, goal, succ, filt_expr = worker_inner(args, domain_gr, env)
        if succ:
            for j in range(len(actions)):
                if len(actions[j]) > 0:
                    robot_tensor, object_tensor, goal_tensor, action_tensor, goal_obj_tensor = model.encod(states[j], actions[j], goal[j], filt_expr[j])
                    dataset.append([robot_tensor, object_tensor, goal_tensor, action_tensor, goal_obj_tensor])

            succ_num += 1
    
    print(f"Successful demo of all data:{succ_num}/{args.iterations}")
    print("length of dataset:", len(dataset))

    model = model.cuda()
    if args.iterations > 0:
        for epoch in range(80):
            outputs = []
            targets = []
            a_loss = []
            total_n, total_loss = 0, 0
            i = 1
            
            while i < len(dataset):
                while total_n < 32:
                    end = time.time()
                    robot_tensor, object_tensor, goal_tensor, action_tensor, goal_obj_tensor = dataset[i-1]
                    
                    loss, _, output = model.goto_abs.bc(robot_tensor, object_tensor, goal_tensor, action_tensor, goal_obj_tensor)
                    outputs.append(output)
                    targets.append(action_tensor)
                    a_loss.append(loss)

                    n = len(action_tensor)
                    total_n += n
                    total_loss += loss * n

                    i = i + 1
                    if i > len(dataset):
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
                
            print(f'{end} [Epoch {epoch}] loss: {torch.mean(torch.stack(a_loss)):.4f} Train_ACC: {train_acc:.2f}')

            if (epoch + 1) % 10 == 0:
                ckpt_name = f'dumps/seed{args.seed}/abil-bc-{args.task}-load={load_id}-epoch={epoch}.pth'
                logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
                io.dump(ckpt_name, state_dict(model))

    if args.evaluate:
        if args.env == 'minigrid':
            env.set_options(nr_objects=args.evaluate_objects)
        if not args.visualize:
            succ_rate, avg_expansions = evaluate(env, model, domain_gr, visualize=False)
            if args.append_result:
                with open('./experiments-eval.txt', 'a') as f:
                    print(f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects},{succ_rate},{avg_expansions}', file=f)
            print(f'succ_rate = {succ_rate}')
            print(f'avg_expansions = {avg_expansions}')
        else:
            evaluate(env, model=model, domain_gr=domain_gr, visualize=True)
    else:
        ckpt_name = f'dumps/seed{args.seed}/abil-bc-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

if __name__ == '__main__':
    main()
