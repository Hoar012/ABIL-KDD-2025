import os.path as osp
import time
from copy import deepcopy
import sys
import pickle

import numpy as np
import torch
import jacinle
import jacinle.io as io
import jactorch
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
parser.add_argument('--structure-mode', type=str, default='full', choices=mb.SUPPORTED_STRUCTURE_MODES)
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

if args.workers > 0:
    import ray
    ray.init()

if args.env == 'mini_behavior':
    lib = mb
    import hacl.p.kfac.minibehavior.ground_models as lib_models

    from hacl.p.kfac.minibehavior.data_generator import worker_bc, worker_eval
    if args.use_offline:
        worker_inner = worker_bc

else:
    raise ValueError(f'Unknown environment: {args.env}')

if args.env == 'mini_behavior':
    if args.task in ['install-a-printer', 'install-a-printer-multi']:
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_2', 'toggle']
        PREDICATES = ['toggleon', 'ontop']
        MAX_STEPS = 30
    elif args.task in ['opening_packages', 'opening_packages1', 'opening_packages3']:
        ACTIONS = ['forward', 'lturn', 'rturn', 'open']
        PREDICATES = ['is-open']
        if args.task == 'opening_packages':
            MAX_STEPS = 60
        elif args.task == 'opening_packages1':
            MAX_STEPS = 30
        elif args.task == 'opening_packages3':
            MAX_STEPS = 80
    elif args.task == 'MovingBoxesToStorage':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_1']
        PREDICATES = ['ontop']
        MAX_STEPS = 60
    elif args.task in ['SortingBooks', 'SortingBooks-multi']:
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2']
        PREDICATES = ['is-book', 'is-hardback', 'ontop']
        MAX_STEPS = 80
    elif args.task in ['Throwing_away_leftovers', 'Throwing_away_leftovers1', 'Throwing_away_leftovers2']:
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_2', 'drop_in']
        PREDICATES = ['is-ashcan', 'inside']
        if args.task == 'Throwing_away_leftovers':
            MAX_STEPS = 80
        elif args.task == 'Throwing_away_leftovers1':
            MAX_STEPS = 30
        elif args.task == 'Throwing_away_leftovers2':
            MAX_STEPS = 60
    elif args.task == 'PuttingAwayDishesAfterCleaning':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_1', 'open', 'drop_in']
        PREDICATES = ['is-plate', 'is-cabinet']
        MAX_STEPS = 100
    elif args.task == 'BoxingBooksUpForStorage':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_in']
        PREDICATES = ['is-book', 'is-box', 'is-shelf', 'inside']
        MAX_STEPS = 100
    elif args.task == 'Setting_up_candles':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_2']
        PREDICATES = ['is-book', 'is-hardback', 'ontop']
    elif args.task == 'CleaningACar':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2', 'drop_in', 'toggle']
        PREDICATES = ['is-clean']
        MAX_STEPS = 90
    elif args.task == 'CleaningShoes':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_1', 'drop_in', 'toggle']
        PREDICATES = ['is-clean']
        MAX_STEPS = 100
    elif args.task in ['CollectMisplacedItems', 'CollectMisplacedItems-multi']:
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2']
        PREDICATES = ['ontop']
        MAX_STEPS = 120
    elif args.task == 'LayingWoodFloors':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_0']
        PREDICATES = ['nextto']
        MAX_STEPS = 100
    elif args.task == 'MakingTea':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_2', 'drop_in', 'open', 'toggle']
        PREDICATES = ['ontop', 'toggleon']
        MAX_STEPS = 80
    elif args.task == 'OrganizingFileCabinet':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'pickup_2', 'drop_2', 'drop_in', 'open']
        PREDICATES = ['inside', 'ontop']
        MAX_STEPS = 160
    elif args.task == 'Washing_pots_and_pans':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_in', 'drop_1', 'toggle', 'open']
        PREDICATES = ['inside']
        MAX_STEPS = 200
    elif args.task == 'WateringHouseplants':
        ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'drop_in', 'toggle']
        PREDICATES = ['inside']
        MAX_STEPS = 100
    else:
        raise ValueError(f'Unknown task: {args.task}.')

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
args.log_filename = f'dumps/bc-{args.id_string}.log'
args.json_filename = f'dumps/bc-{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))

if not args.debug and args.append_expr:
    with open('./experiments.txt', 'a') as f:
        f.write(args.id_string + '\n')

def build_model(args, domain, **kwargs):
    return Model(domain, **kwargs)

def evaluate(env, model, visualize: bool = False):
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
                
            this_succ = False
            steps = 0
            while steps < MAX_STEPS:
                obs = env.compute_obs()
                state, goal = obs['state'], obs['mission']

                prob = model.forward_state(state, PREDICATES[:2])
                action_index = prob.argmax(dim=0)
                rl_action = ACTIONS[action_index]
                #print(rl_action.name)
                _, _, (done,score), _ = env.step(rl_action)

                steps = steps + 1 
                if done:
                    break
            
            done, score = env.compute_done()
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
    
def validate(model, val_robot_set, val_object_set, val_actions_set):
    outputs = []
    targets = []
    i = 0
    batch_size = 32
    while i < len(val_actions_set):
        robot_tensors, object_tensors = val_robot_set[i:i+batch_size], val_object_set[i:i+batch_size]
        actions = val_actions_set[i:i+batch_size]

        i = i + batch_size

        loss, _, output = model.bc(robot_tensors, object_tensors, actions)
        outputs.append(output)
        targets.append(actions)

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

from hacl.util import accuracy

def main():
    jacinle.reset_global_seed(args.seed, True)
    lib.set_domain_mode(args.action_mode, args.structure_mode)
    domain = lib.get_domain()

    for i in range(len(ACTIONS)):
        ACTIONS[i] = domain.operators[ACTIONS[i]]('r')
    logger.critical('Creating model...')
    from hacl.p.kfac.minibehavior.bc_policy import BCPolicyNetwork
    model = BCPolicyNetwork(domain, action_space=ACTIONS, predicates=PREDICATES)
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
    robot_set = []
    object_set = []
    actions_set = []
    succ_num = 0

    for i in jacinle.tqdm(range(1, 30000)):
        end = time.time()
        states, actions, dones, goal, succ, extra_monitors = worker_inner(args, domain, env)
        if succ:
            robot_tensor, object_tensor, action_tensor = model.encode(states, actions) 
            robot_set.append(robot_tensor)
            object_set.append(object_tensor)
            actions_set.append(action_tensor)
            succ_num += 1
        if succ_num >= args.iterations:
            break
    print(f"Successful demo of all data:{succ_num}")

    if succ_num > 0:
        robot_set = torch.cat(robot_set)
        object_set = torch.cat(object_set)
        actions_set = torch.cat(actions_set)

    # generate val dataset
    val_robot_set = []
    val_object_set = []
    val_actions_set = []

    succ_num = 0
    
    if not args.evaluate:
        for i in jacinle.tqdm(range(2000)):
            states, actions, dones, goal, succ, extra_monitors = worker_inner(args, domain, env)
            if succ:
                robot_tensor, object_tensor, action_tensor = model.encode(states, actions) 
                val_robot_set.append(robot_tensor)
                val_object_set.append(object_tensor)
                val_actions_set.append(action_tensor)
                succ_num += 1

                if succ_num >= args.iterations / 10:
                    break
                
    if succ_num > 0:
        val_robot_set = torch.cat(val_robot_set)
        val_object_set = torch.cat(val_object_set)
        val_actions_set = torch.cat(val_actions_set)

    batch_size = args.batch_size
    model = model.cuda()
    if args.iterations > 0:
        for epoch in range(50):
            outputs = []
            targets = []
            a_loss = []

            i = 0
            while i < len(robot_set):
                robot_tensors, object_tensors = robot_set[i:i+batch_size], object_set[i:i+batch_size]
                actions = actions_set[i:i+batch_size]
                i = i + batch_size

                opt.zero_grad()

                loss, _, output = model.bc(robot_tensors, object_tensors, actions)
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

            validate(model, val_robot_set, val_object_set, val_actions_set)
            
            if (epoch + 1) % 10 == 0:
                ckpt_name = f'dumps/seed{args.seed}/bc-{args.task}-load={load_id}-epoch={epoch}.pth'
                logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
                io.dump(ckpt_name, state_dict(model))

    if args.evaluate:
        if args.env == 'mini_behavior':
            env.set_options(nr_objects=args.evaluate_objects)
            env.env.seed(args.seed+1)
        if not args.visualize:
            succ_rate, avg_expansions = evaluate(env, model, visualize=False)
            if args.append_result:
                with open('./experiments-eval.txt', 'a') as f:
                    print(f'{args.structure_mode}-{args.task}-load={load_id}-objs={args.evaluate_objects},{succ_rate},{avg_expansions}', file=f)
            print(f'succ_rate = {succ_rate}')
            print(f'avg_expansions = {avg_expansions}')
        else:
            evaluate(domain, env, heuristic_model=None, visualize=True)
    else:
        ckpt_name = f'dumps/seed{args.seed}/bc-{args.task}-load={load_id}.pth'
        logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
        io.dump(ckpt_name, state_dict(model))

    # from IPython import embed; embed()

if __name__ == '__main__':
    main()
