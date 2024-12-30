import os.path as osp
import time
from copy import deepcopy
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
parser.add_argument('--use-offline', type='bool', default=False)

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
    from hacl.p.kfac.minibehavior.abil_bc_data_processor import worker_abil
    
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
    load_id = load_id[1]
else:
    load_id = 'scratch'

args.id_string = f'{args.task}-{args.action_mode}-{args.structure_mode}-lr={args.lr}-load={load_id}-{log_timestr}'
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
            
            rl_action = None
            this_succ = False
            steps = 0
            while steps < MAX_STEPS:
                obs = env.compute_obs()
                state, goal = obs['state'], obs['mission']
                rl_action = model.make_action(domain_gr, state, args.task, goal, env)
                
                # print(rl_action.name)
                _, _, (done,score), _ = env.step(rl_action)

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

if args.env == 'mini_behavior':
    lib = mb

from hacl.util import accuracy, goal_test
from hacl.p.kfac.minibehavior.abil_bc_policy import ABIL_BCPolicyNetwork
if args.env == 'mini_behavior':
    if args.task in ['install-a-printer', 'install-a-printer-multi']:
        PREDICATES = ['is-printer', 'is-table', 'toggleon', 'ontop']
        SEQ_LENGTH = 2
        MAX_STEPS = 30
    elif args.task == 'opening_packages':
        PREDICATES = ['is-package', 'is-open']
        SEQ_LENGTH = 2
        MAX_STEPS = 60
    elif args.task == 'MovingBoxesToStorage':
        PREDICATES = ['ontop']
        SEQ_LENGTH = 2
        MAX_STEPS = 60
    else:
        raise ValueError(f'Unknown task: {args.task}.')
    
ACTIONS = ['forward', 'lturn', 'rturn']

class ABIL_BCModel(nn.Module):
    def __init__(self, task, domain):
        super(ABIL_BCModel, self).__init__()
        self.domain = domain
        self.task = task
        for i in range(len(ACTIONS)):
            ACTIONS[i] = domain.operators[ACTIONS[i]]('r')

        self.goto_abs = ABIL_BCPolicyNetwork(domain, action_space = ACTIONS, predicates=PREDICATES)

    def make_action(self, domain_gr, state, task, goal, env):
        if task in ["install-a-printer", "install-a-printer-multi"]:
            if goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (not (robot-is-facing r printer_0) ) ) ")) > 0.9:
                filt_expr = "(foreach (?o - item) (or(is-printer ?o)))"
                prob = self.goto_abs.forward_state(state, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            elif goal_test(domain_gr, state, domain_gr.parse(f"(and (hands-free r) (robot-is-facing r printer_0)\
                                                            (not (exists(?t - item) (and (is-table ?t) (ontop printer_0 ?t)))))")) > 0.9:
                return pds.rl.RLEnvAction("pickup_0")
            elif goal_test(domain_gr, state, domain_gr.parse(f"(not (exists (?t - item) (and (is-table ?t) (robot-is-facing r ?t) )) )")) > 0.9:
                filt_expr = "(foreach (?o - item) (or(is-table ?o)))"
                prob = self.goto_abs.forward_state(state, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            elif goal_test(domain_gr, state, domain_gr.parse(f"(not (hands-free r))")) > 0.9:
                return pds.rl.RLEnvAction("drop_2")
            else:
                return pds.rl.RLEnvAction("toggle")
                
        elif task == "opening_packages":
            if goal_test(domain_gr, state, domain_gr.parse(f"(exists(?o - item) (and (robot-is-facing r ?o)(is-package ?o) (not (is-open ?o))))")) > 0.9:
                return pds.rl.RLEnvAction("open")
        
            elif goal_test(domain_gr, state, domain_gr.parse(f"(exists(?o - item) (and (not(robot-is-facing r ?o))(is-package ?o) (not (is-open ?o))))")) > 0.9:
                filt_expr = "(foreach (?o - item) (and(is-package ?o)))"
                prob = self.goto_abs.forward_state(state, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
        
        elif task == "MovingBoxesToStorage":
            if goal_test(domain_gr, state, domain_gr.parse(f"(not (exists(?o - item) (and (is-carton ?o) (robot-is-facing r ?o)) ))")) > 0.2:
                filt_expr = "(foreach (?o - item)(or (is-carton ?o) (is-door ?o)))"
                prob = self.goto_abs.forward_state(state, filt_expr)
                action_index = prob.argmax(dim=0)
                return ACTIONS[action_index]
            elif goal_test(domain_gr, state, domain_gr.parse(f"(hands-free r)")) > 0.9:
                return pds.rl.RLEnvAction("pickup_0")
            else:
                return pds.rl.RLEnvAction("drop_1")

    def forward(self, robot_tensors, object_tensors, actions, filt_obj_tensors):
        return self.goto_abs.bc(robot_tensors, object_tensors, actions, filt_obj_tensors)
    
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
            
    model = ABIL_BCModel(args.task, domain_gr)
    model.cuda()
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
    dataset = []
    
    robot_set = [[], []]
    object_set = [[], []]
    actions_set = [[], []]
    filt_obj_set = [[], []]
    
    succ_num = 0
    
    for i in jacinle.tqdm(range(1, 30000)):
        if succ_num >= args.iterations:
            break
        end = time.time()
        states, actions, dones, goal, succ, extra_monitors = worker_inner.gen_data(args=args, domain=domain_gr, env=env)
        
        if succ:
            for j in range(len(actions)):
                if len(actions[j]) > 0:
                    robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.goto_abs.encode(states[j], actions[j], goal[j]) 
                    robot_set[j].append(robot_tensor)
                    filt_obj_set[j].append(filt_obj_tensor)
                    object_set[j].append(object_tensor)
                    actions_set[j].append(action_tensor)
            succ_num += 1

    print(f"Successful demo of all data:{succ_num}")
    
    for i in range(len(robot_set)):
        if len(robot_set[i]) > 0:
            robot_set[i] = torch.cat(robot_set[i])
            filt_obj_set[i] = torch.cat(filt_obj_set[i])
            object_set[i] = torch.cat(object_set[i])
            actions_set[i] = torch.cat(actions_set[i])

    batch_size = args.batch_size
    model = model.cuda()
    if args.iterations > 0:
        for epoch in range(50):
            outputs = []
            targets = []
            a_loss = []

            for j in range(2):
                i = 0
                while i < len(robot_set[j]):
                    robot_tensors, object_tensors, actions = robot_set[j][i:i+batch_size], object_set[j][i:i+batch_size], actions_set[j][i:i+batch_size]
                    filt_obj_tensors = filt_obj_set[j][i:i+batch_size]
                    i = i + batch_size
        
                    opt.zero_grad()
                    loss, _, output = model(robot_tensors, object_tensors, actions, filt_obj_tensors)
                    outputs.append(output)

                    targets.append(actions)
                    loss.backward()
                    opt.step()
                    a_loss.append(loss)

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            logits = outputs.softmax(-1).data.cpu().numpy()
            labels = targets.data.cpu().numpy()
            
            end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            try:
                train_acc = 100. * accuracy(labels, logits)
            except:
                train_acc = 0.

            print(f'{end} [Epoch {epoch}] loss: {torch.mean(torch.stack(a_loss)):.4f} ACC: {train_acc:.2f}')

            if (epoch + 1) % 5 == 0:
                ckpt_name = f'dumps/seed{args.seed}/abil-bc-{args.task}-load={load_id}-epoch={epoch}.pth'
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
        ckpt_name = f'dumps/seed{args.seed}/abil-bc-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

    # from IPython import embed; embed()

if __name__ == '__main__':
    main()
