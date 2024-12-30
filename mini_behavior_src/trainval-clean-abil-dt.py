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
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--update-interval', type=int, default=50)
parser.add_argument('--print-interval', type=int, default=50)
parser.add_argument('--evaluate-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=500)
parser.add_argument('--iterations', type=int, default=3000)
parser.add_argument('--use-offline', type='bool', default=True)
parser.add_argument('--action-loss-weight', type=float, default=1.0)
parser.add_argument('--goal-loss-weight', type=float, default=1.0)

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
            this_succ = False
            steps = 0
            while steps < MAX_STEPS:
                obs = env.compute_obs()
                state, goal = obs['state'], obs['mission']
                states.append(state)
                pddl_action = model.make_action(domain_gr, states[-2:], his_action[-1:], args.task)
                his_action.append(pddl_action)
                
                # print(pddl_action.name)
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

if args.env == 'mini_behavior':
    lib = mb

def validate(model, val_robot_set, val_object_set, val_actions_set, val_filt_obj_set):
    model.eval()
    outputs = []
    targets = []
    i = 0
    batch_size = 32
    while i < len(val_actions_set):
        robot_tensors, object_tensors = val_robot_set[i:i+batch_size], val_object_set[i:i+batch_size]
        actions, filt_obj_tensors = val_actions_set[i:i+batch_size], val_filt_obj_set[i:i+batch_size]
        i = i + batch_size

        loss, _, output = model.dt(robot_tensors, object_tensors, actions, filt_obj_tensors)
        outputs.append(output)
        targets.append(torch.stack(actions).flatten())


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

from hacl.util import accuracy, goal_test

if args.task == 'CleaningACar':
    ACTIONS_CLEAN = ['forward', 'lturn', 'rturn', 'pickup_2', 'drop_2']
    ACTIONS_PUT = ['forward', 'lturn', 'rturn', 'drop_in' , 'pickup_2', 'drop_2']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_2', 'drop_2', 'drop_in', 'toggle']
    PREDICATES = ['is-clean']
    SEQ_LENGTH = 2
    MAX_STEPS = 90
elif args.task == 'Washing_pots_and_pans':
    ACTIONS_CLEAN = ['forward', 'lturn', 'rturn', 'pickup_1', 'drop_1', 'drop_in', 'toggle']
    ACTIONS_PUT = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_1', 'drop_in', 'open']
    ACTIONS = ['forward', 'lturn', 'rturn', 'pickup_0', 'pickup_1', 'drop_1', 'drop_in', 'toggle']
    PREDICATES = ['is-clean']
    SEQ_LENGTH = 2
    MAX_STEPS = 200

from hacl.p.kfac.minibehavior.abil_dt_policy import ABIL_DTPolicyNetwork

class ABIL_DTModel(nn.Module):
    def __init__(self, task, domain):
        super(ABIL_DTModel, self).__init__()
        self.domain = domain
        self.task = task

        for i in range(len(ACTIONS_CLEAN)):
            ACTIONS_CLEAN[i] = domain.operators[ACTIONS_CLEAN[i]]('r')
        self.clean_abs = ABIL_DTPolicyNetwork(domain, action_space = ACTIONS_CLEAN, predicates=PREDICATES)
        
        for i in range(len(ACTIONS_PUT)):
            ACTIONS_PUT[i] = domain.operators[ACTIONS_PUT[i]]('r')
        self.put_abs = ABIL_DTPolicyNetwork(domain, action_space = ACTIONS_PUT, predicates=PREDICATES)

    def make_action(self, domain_gr, states, his_action, task):
        if task == "CleaningACar":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(exists(?o - item)(and (is-car ?o) (is-dusty ?o) ))")) > 0.9:
                filt_expr = "(foreach (?o - item) (or(is-tool ?o)(is-car ?o)))"
                prob = self.clean_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_CLEAN[action_index]
            else:
                filt_expr = "(foreach (?o - item) (or(is-tool ?o)(is-bucket ?o)))"
                prob = self.put_abs.forward_state(states, his_action, filt_expr, 1)
                action_index = prob.argmax(dim=0)
                return ACTIONS_PUT[action_index]
        
        elif task == "Washing_pots_and_pans":
            if goal_test(domain_gr, states[-1], domain_gr.parse(f"(exists(?o - item)(and(is-pan ?o)(is-dusty ?o) ))")) > 0.9:
                filt_expr = "(foreach (?o - item) (or(is-pan ?o)(is-brush ?o)(is-sink ?o)))"
                prob = self.clean_abs.forward_state(states, his_action, filt_expr, 1)
                if len(states) >= 2:
                    if (states[-1]["robot-feature"].tensor == states[-2]["robot-feature"].tensor).all()\
                    and (states[-1]["robot-holding"].tensor == states[-2]["robot-holding"].tensor).all():
                        last_index = ACTIONS_CLEAN.index(his_action[-1])
                        prob[last_index] -= 100
                action_index = prob.argmax(dim=0)
                return ACTIONS_CLEAN[action_index]
            else:
                # print("put")
                filt_expr = "(foreach (?o - item) (or(is-pan ?o)(is-cabinet ?o)))"
                prob = self.put_abs.forward_state(states, his_action, filt_expr, 1)
                if len(states) >= 2:
                    if (states[-1]["robot-feature"].tensor == states[-2]["robot-feature"].tensor).all()\
                    and (states[-1]["robot-holding"].tensor == states[-2]["robot-holding"].tensor).all():
                        last_index = ACTIONS_PUT.index(his_action[-1])
                        prob[last_index] -= 100
                action_index = prob.argmax(dim=0)
                return ACTIONS_PUT[action_index]
    
    def encod(self, states, actions, filt_expr):
        return self.clean_abs.encode(states, actions, filt_expr)
    
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
    actions_set = [[], []]
    robot_set = [[], []]
    object_set = [[], []]
    filt_obj_set = [[], []]
    succ_num = 0

    for i in jacinle.tqdm(range(1, 30000)):
        if succ_num >= args.iterations:
            break
        end = time.time()
        states, actions, dones, filt_expr, succ, extra_monitors = worker_inner.gen_data(args=args, domain=domain_gr, env=env)
        if succ:
            # clean_set
            for seq in range(len(actions[0])):
                robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.clean_abs.encode(states[0][seq], actions[0][seq], filt_expr[0])
                robot_set[0].append(robot_tensor)
                object_set[0].append(object_tensor)
                actions_set[0].append(action_tensor)
                filt_obj_set[0].append(filt_obj_tensor)
            # put_set
            for seq in range(len(actions[1])):
                robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.put_abs.encode(states[1][seq], actions[1][seq], filt_expr[1])
                robot_set[1].append(robot_tensor)
                object_set[1].append(object_tensor)
                actions_set[1].append(action_tensor)
                filt_obj_set[1].append(filt_obj_tensor)
            succ_num += 1

    print(f"Successful demo of all data:{succ_num}/{args.iterations}")
    
    val_actions_set = [[], []]
    val_robot_set = [[], []]
    val_object_set = [[], []]
    val_filt_obj_set = [[], []]
    
    if not args.evaluate:
        succ_num = 0
        for i in jacinle.tqdm(2000):
            if succ_num >= args.iterations / 10:
                break
            states, actions, dones, filt_expr, succ, extra_monitors = worker_inner.gen_data(args=args, domain=domain_gr, env=env)
            if succ:
                for seq in range(len(actions[0])):
                    robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.clean_abs.encode(states[0][seq], actions[0][seq], filt_expr[0])
                    val_robot_set[0].append(robot_tensor)
                    val_object_set[0].append(object_tensor)
                    val_actions_set[0].append(action_tensor)
                    val_filt_obj_set[0].append(filt_obj_tensor)
                    
                for seq in range(len(actions[1])):
                    robot_tensor, object_tensor, action_tensor, filt_obj_tensor = model.put_abs.encode(states[1][seq], actions[1][seq], filt_expr[1])
                    val_robot_set[1].append(robot_tensor)
                    val_object_set[1].append(object_tensor)
                    val_actions_set[1].append(action_tensor)
                    val_filt_obj_set[1].append(filt_obj_tensor)
                succ_num += 1

    batch_size = args.batch_size
    model = model.cuda()
    
    if succ_num > 0:
        for epoch in range(50):
            # pick
            outputs = []
            targets = []
            a_loss = []
            
            model.clean_abs.train()
            i = 0
            while i < len(robot_set[0]):
                robot_tensors, object_tensors = robot_set[0][i:i+batch_size], object_set[0][i:i+batch_size]
                actions, filt_obj_tensors = actions_set[0][i:i+batch_size], filt_obj_set[0][i:i+batch_size]
                i = i + batch_size
                # print(actions)
                opt.zero_grad()

                loss, _, output = model.clean_abs.dt(robot_tensors, object_tensors, actions, filt_obj_tensors)
                outputs.append(output)
                targets.append(torch.stack(actions).flatten())
                
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
                
            print(f'{end} [Epoch {epoch}] loss: {torch.mean(torch.stack(a_loss)):.4f} clean_abs_ACC: {train_acc:.2f}')
            sys.stdout.flush()
            validate(model.clean_abs, val_robot_set[0], val_object_set[0], val_actions_set[0], val_filt_obj_set[0])
            
            # put
            outputs = []
            targets = []
            a_loss = []
            model.put_abs.train()
            i = 0
            while i < len(robot_set[1]):
                robot_tensors, object_tensors = robot_set[1][i:i+batch_size], object_set[1][i:i+batch_size]
                actions, filt_obj_tensors = actions_set[1][i:i+batch_size], filt_obj_set[1][i:i+batch_size]
                i = i + batch_size
                # print(actions)
                opt.zero_grad()

                loss, _, output = model.put_abs.dt(robot_tensors, object_tensors, actions, filt_obj_tensors)
                outputs.append(output)
                targets.append(torch.stack(actions).flatten())
                
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
                
            print(f'{end} [Epoch {epoch}] loss: {torch.mean(torch.stack(a_loss)):.4f} put_abs_ACC: {train_acc:.2f}')
            sys.stdout.flush()
            validate(model.put_abs, val_robot_set[1], val_object_set[1], val_actions_set[1], val_filt_obj_set[1])
            
            if (epoch + 1) % 5 == 0:
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
