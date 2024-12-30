import os.path as osp
import time
from copy import deepcopy
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
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
parser.add_argument('--load_domain', type='checked_file', default=None)
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
args.log_filename = f'dumps/abil-dt-{args.id_string}.log'
args.json_filename = f'dumps/abil-dt-{args.id_string}.jsonl'

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
    # files = io.lsdir(data_dir, '*.pkl')[:num]
    import random
    files = random.sample(io.lsdir(data_dir, '*.pkl'), num)
    files = [io.load(f) for f in files]
    return files

from hacl.p.kfac.cliport import ABIL_DTModel

def get_filt_obj(model: ABIL_DTModel, state, predicates, task, action):
    if task in ["packing-shapes", "packing-5shapes"]:
        shape = predicates[0]
        item_filt_expr = f"(foreach (?o - item) (is-{shape} ?o))"
        item_images, item_poses = model.policy.get_obj([state], item_filt_expr)
        container_images, container_poses = model.policy.get_obj([state], None, is_container = True)

        filt_obj_images = torch.cat([item_images, container_images], dim = 0)
        filt_obj_poses = torch.cat([item_poses, container_poses], dim = 0)

        return (filt_obj_images, filt_obj_poses)
    
    elif task in ["place-red-in-green", "put-block-in-bowl-seen-colors", "put-block-in-bowl-composed-colors"]:
        block_color = predicates[0]
        bowl_color = predicates[1]

        item_filt_expr = f"(foreach (?o - item) (is-{block_color} ?o))"
        item_images, item_poses = model.policy.get_obj([state], item_filt_expr)
        container_filt_expr = f"(foreach (?o - container) (c-is-{bowl_color} ?o))"
        container_images, container_poses = model.policy.get_obj([state], container_filt_expr, is_container = True)

        filt_obj_images = torch.cat([item_images, container_images], dim = 0)
        filt_obj_poses = torch.cat([item_poses, container_poses], dim = 0)

        return (filt_obj_images, filt_obj_poses)
    
    elif task in [ "separating-piles-seen-colors", "separating-20piles", "separating-10piles"]:
        block_color = predicates[0]
        zone_color = predicates[1]

        container_filt_expr = f"(foreach (?o - container) (c-is-{zone_color} ?o))"
        container_images, container_poses = model.policy.get_obj([state], None, is_container = True)

        filt_obj_images = container_images
        filt_obj_poses = container_poses

        return (filt_obj_images, filt_obj_poses)
    
    elif task in ["assembling-kits"]:
        items = []

        for obj, typ in state._object_name2index.items():
            if typ[0] == "item":
                items.append(obj)
        item_poses, item_images = state["item-pose"].tensor, state["item-image"].tensor

        pose1 = action[0][:3]

        
        item = items[torch.norm(item_poses-pose1, 1, dim=-1).argmin()]
        
        item_poses = state["item-pose"].tensor
        item_images = state["item-image"].tensor
        filt_item_pose = item_poses[torch.norm(item_poses-pose1, 1, dim=-1).argmin()].unsqueeze(0)
        filt_item_image = item_images[torch.norm(item_poses-pose1, 1, dim=-1).argmin()].unsqueeze(0)
        
        container_filt_expr = f"(foreach(?t - container)(is-same-shape {item} ?t))"
        
        filt_container_image, filt_container_pose = model.policy.get_obj([state], container_filt_expr, is_container = True)

        
        filt_obj_images = torch.cat([filt_item_image, filt_container_image], dim=0)
        filt_obj_poses = torch.cat([filt_item_pose, filt_container_pose], dim=0)

        return (filt_obj_images, filt_obj_poses)

def main():
    jacinle.reset_global_seed(args.seed, True)
    logger.critical('Creating model...')
    domain_gr = lib_models.load_domain()
    model_g = lib_models.ModelABL(domain_gr, goal_loss_weight = 0)
    
    if args.load_domain is not None:
        try:
            load_state_dict(model_g, io.load(args.load_domain))
        except KeyError as e:
            logger.exception(f'Failed to load (part of the) model: {e}.')
    
    model = ABIL_DTModel(args.task, domain_gr, predicates=PREDICATES)
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

    train_set = []
    len_dataset = 0
    for data in dataset:
        len_dataset += len(data[:-1])
        states = []

        for features, action, goal, done in data[:-1]:
            # print(goal)
            images = []
            poses = []

            object_names = list()
            object_types = list()
            item_images = list()
            item_poses = list()
            container_images = list()
            container_poses = list()
            object_type2id = dict()
            for id, feature in features.items():
                images.append(torch.tensor(feature["image"], dtype=torch.float32).permute((2, 0, 1)).flatten() / 255)
                poses.append(torch.tensor(feature["pose"])[:3])
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

            state = pds.State([domain_gr.types[t] for t in object_types], pds.ValueDict(), object_names)
            ctx = state.define_context(domain_gr)
            ctx.define_feature('item-pose', torch.stack(item_poses).float())
            ctx.define_feature('container-pose', torch.stack(container_poses).float())
            ctx.define_feature('item-image', torch.stack(item_images))
            ctx.define_feature('container-image', torch.stack(container_images))
            states.append(state)

            filt_obj_images, filt_obj_poses = get_filt_obj(model, state, goal, args.task, action)

            images = torch.stack(images)
            poses = torch.stack(poses)
            action = torch.from_numpy(action[:,:3]).flatten()
            goal = model.policy.forward_goal(goal)
            train_set.append([images, poses, action, goal, filt_obj_images, filt_obj_poses])

    print(f"Length of dataset:{len(train_set)}")
    
    max_epoch = 250
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
        
        while i < len(train_set):
            while total_n < 32:
                end = time.time()
                images, poses, action, goal, filt_obj_images, filt_obj_poses = train_set[i-1]

                loss, _, output = model(images, poses, action, goal, filt_obj_images, filt_obj_poses)

                outputs.append(output)
                targets.append(action)
                a_loss.append(loss)

                total_n += 1
                total_loss += loss

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
            ckpt_name = f'dumps/seed{args.seed}/abil-dt-{args.task}-load-{load_id}-epoch{epoch}.pth'
            logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
            io.dump(ckpt_name, state_dict(model))

    ckpt_name = f'dumps/seed{args.seed}/abil-dt-{args.task}-load-{load_id}.pth'
    logger.critical('Saving checkpoint to "{}".'.format(ckpt_name))
    io.dump(ckpt_name, state_dict(model))


if __name__ == '__main__':
    main()