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
from transformers import get_cosine_schedule_with_warmup

import hacl.pdsketch as pds
import hacl.envs.cliport.cliport_v2024 as cp

io.set_fs_verbose()
pds.Value.set_print_options(max_tensor_size=100)
# pds.ConditionalAssignOp.set_options(quantize=True)

from cliport_src.data_generator import SUPPORTED_TASKS, get_predicates

parser = jacinle.JacArgumentParser()
parser.add_argument('env', choices=['cliport'])
parser.add_argument('task', choices=SUPPORTED_TASKS)
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
    lib = cp
    import hacl.p.kfac.cliport.ground_models as lib_models
    PREDICATES = get_predicates(args.task)

assert args.task in SUPPORTED_TASKS

logger = jacinle.get_logger(__file__)
log_timestr = time.strftime(time.strftime('%Y-%m-%d-%H-%M-%S'))
if args.load is not None:
    load_id = osp.basename(args.load).split('-')
    load_id = load_id[1]
else:
    load_id = 'scratch'

args.id_string = f'{args.task}-lr={args.lr}-load={load_id}-{log_timestr}'
args.log_filename = f'dumps/dt-{args.id_string}.log'
args.json_filename = f'dumps/dt-{args.id_string}.jsonl'

if not args.debug:
    jacinle.set_logger_output_file(args.log_filename)

logger.critical('Arguments: {}'.format(args))

if not args.debug and args.append_expr:
    with open('./experiments.txt', 'a') as f:
        f.write(args.id_string + '\n')
    
def validate(model, val_set):
    a_loss = []

    i = 1
    while i <= len(val_set):
        end = time.time()
        images, poses, action, goal = val_set[i-1]

        loss, _, output = model(images, poses, action, goal)
        a_loss.append(loss)
        i = i + 1

    end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'{end} Val loss: {torch.mean(torch.stack(a_loss)):.6f}')
    sys.stdout.flush()

def get_data(num):
    data_dir = f"hacl/envs/cliport/data/{args.task}-train"
    import random
    files = random.sample(io.lsdir(data_dir, '*.pkl'), num)
    files = [io.load(f) for f in files]
    return files

def main():
    jacinle.reset_global_seed(args.seed, True)
    logger.critical('Creating model...')
    from hacl.p.kfac.cliport import DTModel
    model = DTModel(args.task, None, predicates=PREDICATES)
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

    len_dataset = 0
    train_set = []
    for data in dataset:
        len_dataset += len(data[:-1])
        images_set = []
        poses_set = []
        goal_set = []
        action_set = []
        for features, action, goal, done in data[:-1]:
            images = []
            poses = []
            for id, feature in features.items():
                
                images.append(torch.tensor(feature["image"], dtype=torch.float32).permute((2, 0, 1)) / 255)
                poses.append(torch.tensor(feature["pose"])[:3])
            
            images_set.append(torch.stack(images))
            poses_set.append(torch.stack(poses))
            action_set.append(torch.from_numpy(action[:,:3]).flatten())
            goal_set.append(model.policy.forward_goal(goal))
        
        images_set = torch.stack(images_set)
        poses_set = torch.stack(poses_set)
        action_set = torch.stack(action_set)
        goal_set = torch.stack(goal_set)
        
        train_set.append([images_set, poses_set, action_set, goal_set])

    print(f"Length of dataset: {len_dataset}")
    
    max_epoch = 300
    batch_size = args.batch_size
    total_steps = (len_dataset // batch_size) * max_epoch if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * max_epoch 
    warm_up_ratio = 0.1 # warmup step
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

    model = model.cuda()

    for epoch in range(max_epoch):
        outputs = []
        targets = []
        a_loss = []
        
        total_n, total_loss = 0, 0
        i = 1
        
        while i <= len(train_set):
            while total_n < batch_size:
                end = time.time()
                images, poses, action, goal = train_set[i-1]

                loss, _, output = model(images, poses, action, goal)

                outputs.append(output)
                targets.append(action)
                a_loss.append(loss)

                total_n += len(goal)
                total_loss += loss * len(goal)

                i = i + 1
                if i > len(train_set):
                    break

            if total_loss.requires_grad:
                opt.zero_grad()
                total_loss /= total_n
                total_loss.backward()
                opt.step()
                scheduler.step()

            total_n, total_loss = 0, 0
                
        end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f'{end} [Epoch {epoch}] loss: {torch.mean(torch.stack(a_loss)):.6f} lr: {scheduler.get_lr()[0]:.6f}')

        # validate(model, val_set)
        
        if (epoch + 1) % 10 == 0:
            ckpt_name = f'dumps/seed{args.seed}/dt-{args.task}-{load_id}-epoch{epoch}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))

    ckpt_name = f'dumps/seed{args.seed}/dt-{args.task}-{load_id}.pth'
    logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
    io.dump(ckpt_name, state_dict(model))

if __name__ == '__main__':
    main()