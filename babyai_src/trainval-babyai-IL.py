import os.path as osp
import time
from copy import deepcopy

import numpy as np
import torch
import jacinle
import jacinle.io as io
import jactorch
from jactorch.io import load_state_dict, state_dict
import torch.nn as nn

import hacl.pdsketch as pds
import hacl.envs.gridworld.minigrid.minigrid_v2023 as mg

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

parser = jacinle.JacArgumentParser()
parser.add_argument('env', choices=['minigrid'])
parser.add_argument('task', choices=mg.SUPPORTED_TASKS)
parser.add_argument('--model', choices=['bc', 'dt'])
parser.add_argument('--seed', type=int, default=33)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--use-offline', type='bool', default=True)

parser.add_argument('--structure-mode', type=str, default='full', choices=mg.SUPPORTED_STRUCTURE_MODES)
parser.add_argument('--load', type='checked_file', default=None)
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

if args.env == 'minigrid':
    lib = mg
    import hacl.p.kfac.minigrid.ground_models as lib_models
    from hacl.p.kfac.minigrid.data_generator import worker_il, worker_search
    
    from hacl.p.kfac.minigrid.IL_dataset import init_dataset, preprocess
    
    if args.use_offline:
        worker_inner = worker_il
    else:
        worker_inner = worker_search
else:
    raise ValueError(f'Unknown environment: {args.env}')

Model = lib_models.get_model(args)

assert args.task in lib.SUPPORTED_TASKS
assert args.model in ['bc', 'dt']

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

args.id_string = f'{args.task}-{args.model}-lr={args.lr}-load={load_id}-{log_timestr}'
args.log_filename = f'dumps/{args.id_string}.log'
args.json_filename = f'dumps/{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))

if not args.debug and args.append_expr:
    with open('./experiments.txt', 'a') as f:
        f.write(args.id_string + '\n')


def build_model(args, domain, **kwargs):
    return Model(domain, **kwargs)

def evaluate(env, model, task, visualize: bool = False):
    model.cpu()

    jacinle.reset_global_seed(args.seed+1, True)
    succ, nr_expansions, total = 0, 0, 0
    nr_expansions_hist = list()
    for i in jacinle.tqdm(100, desc='Evaluation'):
        obs = env.reset()
        env_copy = deepcopy(env)
        plan = []

        this_succ = False
        steps = 0
        while steps < 30:
            obs = env.compute_obs()
            state, goal = obs['state'], obs['mission']
            rl_action = model.make_action(model.domain, state, task, goal, env)

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

    evaluate_key = f'{args.model}-{args.task}-load={load_id}-objs={args.evaluate_objects}'
    io.dump(f'./results/{evaluate_key}.json', nr_expansions_hist)
    return succ / total, nr_expansions / succ


if args.env == 'minigrid':
    lib = mg
    if args.task in ('goto', 'goto2', 'gotosingle', 'gotodoor'):
        ACTIONS = ['forward', 'lturn', 'rturn']
        PREDICATES = ['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box', 'is-door']
    elif args.task == 'pickup':
        ACTIONS = ['pickup', 'forward', 'lturn', 'rturn']
        PREDICATES = ['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box']
    elif args.task == 'open':
        ACTIONS = ['forward', 'lturn', 'rturn', 'toggle']
        PREDICATES = ["is-open", 'is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey']
    elif args.task == 'unlock':
        ACTIONS = ['pickup', 'forward', 'lturn', 'rturn', 'toggle-tool']
        PREDICATES = ["is-open", 'is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey']
    elif args.task == 'put':
        ACTIONS = ['pickup', 'forward', 'lturn', 'rturn', 'place']
        PREDICATES = ['nextto']
    elif args.task == 'generalization':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup', 'toggle-tool']
        PREDICATES = ['is-red', 'is-green', 'is-blue', 'is-purple', 'is-yellow', 'is-grey', 'is-key', 'is-ball', 'is-box', 'is-door']
    else:
        raise ValueError(f'Unknown task: {args.task}.')

from hacl.util import accuracy
from hacl.p.kfac.minigrid.IL_models import BCModel, DTModel

def get_model(args):
    if args.model == "bc":
        return BCModel
    elif args.model == "dt":
        return DTModel
        
from torch.utils.data import DataLoader

def main():
    jacinle.reset_global_seed(args.seed, True)
    lib.set_domain_mode('envaction', args.structure_mode)
    domain = lib.get_domain()
    
    logger.critical('Creating model...')
    model = get_model(args)(args.task, domain, ACTIONS, PREDICATES)
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
    
    dataset, succ_num = init_dataset(args, worker_inner, env, domain, model)
    print(f"Successful demo of all data:{succ_num}/{args.iterations}")
    
    dataset = preprocess(args, dataset, domain, model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    model = model.cuda()
    if args.iterations > 0:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=0)
        for epoch in range(80):
            outputs = []
            targets = []
            a_loss = []
            
            for batch in dataloader:
                robot_tensors, object_tensors, goal_tensors, actions = batch
    
                opt.zero_grad()
                loss, _, output = model(robot_tensors, object_tensors, goal_tensors, actions)
                loss.backward()
                opt.step()
                a_loss.append(loss)
                
                outputs.append(output)
                if isinstance(actions, list):
                    actions = torch.stack(actions).flatten()
                targets.append(actions)

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
            
            if (epoch + 1) % 10 == 0:
                ckpt_name = f'dumps/seed{args.seed}/{args.model}-{args.task}-load={load_id}-epoch={epoch}.pth'
                logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
                io.dump(ckpt_name, state_dict(model))
            
    if args.evaluate:
        if args.env == 'minigrid':
            env.set_options(nr_objects=args.evaluate_objects)
        if not args.visualize:
            succ_rate, avg_expansions = evaluate(env, model, args.task, visualize=False)
            if args.append_result:
                with open('./experiments-eval.txt', 'a') as f:
                    print(f'{args.model}-{args.task}-load={load_id}-objs={args.evaluate_objects},{succ_rate},{avg_expansions}', file=f)
            print(f'succ_rate = {succ_rate}')
            print(f'avg_expansions = {avg_expansions}')
        else:
            evaluate(env, model=model, task=args.task, visualize=True)
    else:
        ckpt_name = f'dumps/seed{args.seed}/{args.model}-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

if __name__ == '__main__':
    main()