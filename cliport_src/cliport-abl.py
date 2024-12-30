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

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

from cliport_src.data_generator import SUPPORTED_TASKS

parser = jacinle.JacArgumentParser()
parser.add_argument('env', choices=['cliport'])
parser.add_argument('task', choices=SUPPORTED_TASKS)
parser.add_argument('--seed', type=int, default=33)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--update-interval', type=int, default=50)
parser.add_argument('--print-interval', type=int, default=50)
parser.add_argument('--evaluate-interval', type=int, default=250)
parser.add_argument('--save-interval', type=int, default=100)
parser.add_argument('--iterations', type=int, default=1000)
parser.add_argument('--use-offline', type='bool', default=True)
parser.add_argument('--goal-loss-weight', type=float, default=1.0)

parser.add_argument('--structure-mode', type=str, default='abl', choices=["abl", "full"])
parser.add_argument('--load', type='checked_file', default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--visualize-failure', action='store_true', help='visualize the tasks that the planner failed to solve.')
parser.add_argument('--evaluate-objects', type=int, default=4)

# append the results
parser.add_argument('--append-expr', action='store_true')
parser.add_argument('--append-result', action='store_true')

# debug options
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

if args.env == 'cliport':
    import hacl.p.kfac.cliport.ground_models as lib_models
else:
    raise ValueError(f'Unknown environment: {args.env}')

assert args.task in SUPPORTED_TASKS

logger = jacinle.get_logger(__file__)
log_timestr = time.strftime(time.strftime('%Y-%m-%d-%H-%M-%S'))
if args.load is not None:
    load_id = osp.basename(args.load).split('-')
    load_id = load_id[1]
else:
    load_id = 'scratch'

args.id_string = f'{args.task}-lr={args.lr}-load={load_id}-{log_timestr}'
args.log_filename = f'dumps/{args.id_string}.log'
args.json_filename = f'dumps/{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))

if not args.debug and args.append_expr:
    with open('./experiments.txt', 'a') as f:
        f.write(args.id_string + '\n')

def get_data(num):
    data_dir = f"hacl/envs/cliport/data/{args.task}-train"
    import random
    files = random.sample(io.lsdir(data_dir, '*.pkl'), num)
    files = [io.load(f) for f in files]
    return files

def get_goalexpr(predicates, domain):
    if args.task in ["packing-shapes", "packing-5shapes"]:
        shape = predicates[0]
        goal_expr = domain.parse(f"(exists (?o - item) (and(is-{shape} ?o)(exists (?t - container)(is-in ?o ?t))))")
        goal_expr = domain.parse(f"(not(exists (?o - item) (and(is-{shape} ?o)(not(exists (?t - container)(is-in ?o ?t))))))")
    
    elif args.task in ["put-block-in-bowl-seen-colors", "put-block-in-bowl-composed-colors"]:
        block_color = predicates[0]
        bowl_color = predicates[1]
    
        goal_expr = domain.parse(f"(not(exists (?o - item) (and(is-{block_color} ?o)(not(exists (?t - container)(and(c-is-{bowl_color} ?t)(is-in ?o ?t)))))))")
        
    elif args.task == "place-red-in-green":
        block_color = predicates[0]
        bowl_color = predicates[1]
    
        goal_expr = domain.parse(f"(not(exists (?o - item) (and(is-red ?o)(not(exists (?t - container)(and(c-is-green ?t)(is-in ?o ?t)))))))")

    elif args.task in ["separating-piles-seen-colors", "separating-20piles", "separating-10piles"]:
        block_color = predicates[0]
        zone_color = predicates[1]
    
        goal_expr = domain.parse(f"(forall (?o - item)(exists (?t - container)(and(c-is-{zone_color} ?t)(is-in ?o ?t))))")
        goal_expr = None
        
    elif args.task == "assembling-kits":

        goal_expr = domain.parse(f"(forall (?o - item)(exists (?t - container)(and(is-same-shape ?o ?t)(is-in ?o ?t))))")
        goal_expr = None
        
    return goal_expr


def main():
    jacinle.reset_global_seed(args.seed, True)
    # domain = lib.get_domain()
    domain = lib_models.load_domain()
    logger.critical('Creating model...')

    if args.structure_mode == "abl":
        model = lib_models.ModelABL(domain, goal_loss_weight = args.goal_loss_weight)
    elif args.structure_mode == "full":
        model = lib_models.ModelFull(domain, goal_loss_weight = args.goal_loss_weight)

    if args.load is not None:
        try:
            load_state_dict(model, io.load(args.load))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')

    # It will be used for evaluation.
    logger.critical(f'Creating environment (task={args.task})...')
    logger.warning('Using single-threaded mode.')
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    dataset = get_data(args.iterations)
    succ_num = len(dataset)
    print(f"Successful demo of all data:{succ_num}/{args.iterations}")

    data_augmentation = False
    if data_augmentation:
        for data in dataset:
            pass

    train_set = []
    for data in dataset:
        states = []
        actions = []
        goals = []
        dones = []
        for features, action, goal, done in data:
            object_names = list()
            object_types = list()
            item_images = list()
            item_poses = list()
            container_images = list()
            container_poses = list()
            object_type2id = dict()
            for id, feature in features.items():
                obj_type = feature["type"]

                if obj_type.find("container") > -1:
                    object_types.append("container")
                    container_images.append(torch.tensor(feature["image"], dtype=torch.float32).permute((2, 0, 1)).flatten() / 255)
                    container_poses.append(torch.tensor(feature["pose"][:3]))

                else:
                    object_types.append("item")
                    item_images.append(torch.tensor(feature["image"], dtype=torch.float32).permute((2, 0, 1)).flatten() / 255)
                    item_poses.append(torch.tensor(feature["pose"][:3]))
                
                if obj_type not in object_type2id.keys():
                    object_type2id[obj_type] = 0
                obj_name = f'{obj_type}:{object_type2id[obj_type]}'
                object_names.append(obj_name)
                object_type2id[obj_type] += 1

            state = pds.State([domain.types[t] for t in object_types], pds.ValueDict(), object_names)
            ctx = state.define_context(domain)
            ctx.define_feature('item-pose', torch.stack(item_poses).float())
            ctx.define_feature('container-pose', torch.stack(container_poses).float())
            ctx.define_feature('item-image', torch.stack(item_images))
            ctx.define_feature('container-image', torch.stack(container_images))
            states.append(state)
            
            actions.append(action)
            goals.append(get_goalexpr(goal, domain))
            predicates = goal
            dones.append(done)

        dones = torch.tensor(dones)
        train_set.append([states, actions, goals, dones, predicates])

    
    print(f"Length of dataset:{len(train_set)}")

    max_iterations = 10000
    meters = jacinle.GroupMeters()
    for i in jacinle.tqdm(range(1, max_iterations)):
        total_n, total_loss = 0, 0
        while total_n < 8:
            end = time.time()

            index = np.random.randint(0, len(train_set))
            states, actions, goals, dones, predicates = train_set[index]
            extra_monitors = dict()

            extra_monitors['time/data'] = time.time() - end; end = time.time()
            loss, monitors, output_dict = model(feed_dict={'states': states, 'actions': actions, 'dones': dones, 'goal_expr': goals[-1], "predicates": predicates}, task = args.task, forward_augmented=True)

            extra_monitors['time/model'] = time.time() - end; end = time.time()
            extra_monitors['accuracy/succ'] = float(dones[-1])
            monitors.update(extra_monitors)

            n = len(states)
            total_n += n
            total_loss += loss * n

            meters.update(monitors, n=n)
            jacinle.get_current_tqdm().set_description(
                meters.format_simple(values={k: v for k, v in meters.val.items() if k.count('/') <= 1})
            )

        end = time.time()
        if total_loss.requires_grad:
            opt.zero_grad()
            total_loss /= total_n
            total_loss.backward()
            opt.step()
        meters.update({'time/gd': time.time() - end}, n=1)

        if args.print_interval > 0 and i % args.print_interval == 0:
            logger.info(meters.format_simple(f'Iteration {i}', values='avg', compressed=False))
            meters.reset()
        
        if args.save_interval > 0 and i % args.save_interval == 0:
            ckpt_name = f'dumps/seed{args.seed}/{args.structure_mode}-{args.task}-load={load_id}-epoch={i}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))
            
    ckpt_name = f'dumps/seed{args.seed}/{args.structure_mode}-{args.task}-{args.iterations}-{load_id}.pth'
    logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
    io.dump(ckpt_name, state_dict(model))

if __name__ == '__main__':
    main()