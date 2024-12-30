import os.path as osp
from copy import deepcopy

import numpy as np
import torch
import jacinle.io as io
from jactorch.io import load_state_dict, state_dict

import hacl.pdsketch as pds
io.set_fs_verbose()
# pds.ConditionalAssignOp.set_options(quantize=True)

import os
import json
import hydra

from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment
from PIL import Image
import random

from cliport_src.data_generator import SUPPORTED_TASKS, get_predicates


@hydra.main(config_path='../hacl/envs/cliport/cliport/cfg', config_name='data')
def evaluate(cfg):
    # Initialize environment and task.
    def crop_image(env, image, info):
        obj_obs = dict()
        objs_id = env.task.objs_id
       
        for id, type_name in objs_id.items():
            value = info[id]

            value = list(value)
            pos = list(value[0])
            rotation = list(value[1])
            dim = list(value[2])
            if id != 5:
                if dim[0] == 1.25:
                    dim = [0.1, 0.1, 0.01]
                elif dim[0] == 0.003:
                    dim = [0.05, 0.05, 0.01]
                elif dim[0] == 0.006:
                    dim = [0.12, 0.12, 0.01]
                elif dim[0] > 0.2:
                    dim = [dim[0] * 0.2, dim[1] * 0.2, dim[2] * 0.2]
            pos1 = [pos[0] - 0.75*dim[0], pos[1] - 0.75*dim[1]]
            pos2 = [pos[0] + 0.75*dim[0], pos[1] + 0.75*dim[1]]
                
            box = []
            c1 = list(utils.xyz_to_pix(pos1, env.task.bounds, env.task.pix_size))
            box.extend([c1[1], c1[0]])
            c2 = list(utils.xyz_to_pix(pos2, env.task.bounds, env.task.pix_size))
            box.extend([c2[1], c2[0]])

            crop = image.crop(box)
            crop = crop.resize([24,24])

            a = np.array(crop, dtype = np.float32)

            obj_obs[id] = {"type": objs_id[id], "image": a, "pose": np.array(pos + rotation + dim, dtype=np.float32)}

        return obj_obs
    
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize dataset.

    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    ds = dataset.NesyDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")
    
    assert cfg['task'] in SUPPORTED_TASKS
    PREDICATES = get_predicates(cfg['task'])

    import hacl.p.kfac.cliport.ground_models as lib_models
    domain_gr = lib_models.load_domain()
    model_g = lib_models.ModelABL(domain_gr, goal_loss_weight = 0)
    
    try:
        load_state_dict(model_g, io.load("dumps/seed33/abl-put-block-in-bowl-seen-colors-load=scratch.pth"))
        
    except KeyError as e:
        print(f'Failed to load (part of the) model: {e}.')

    import hacl.p.kfac.cliport as act_models
    Model = act_models.get_model(cfg["model"])
    model = Model(cfg['task'], domain_gr, predicates=PREDICATES)

    try:
        load_state_dict(model, io.load(cfg["load"]))
    except KeyError as e:
        print(f'Failed to load (part of the) model: {e}.')

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = ds.max_seed
    if seed < 0:
        
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    rewards = []

    n = 0
    #while ds.n_episodes < cfg['n']:
    while n < cfg['n']:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Test: {}/{} | Seed: {}'.format(n, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{ds.n_episodes+1:06d}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            img = ds.get_image(obs)
            image = Image.fromarray(np.uint8(img[:,:,:3]))
            features = crop_image(env, image, info)

            images = []
            poses = []

            object_names = list()
            object_types = list()
            item_images = list()
            item_poses = list()
            container_images = list()
            container_poses = list()
            for id, feature in features.items():
                images.append(torch.tensor(feature["image"], dtype=torch.float32).permute((2, 0, 1)).flatten() / 255)
                poses.append(torch.tensor(feature["pose"]))

                obj_type = feature["type"]
                if obj_type.find("container") > -1 or obj_type.find("rod") > -1:
                    object_types.append("container")
                    container_images.append(torch.tensor(feature["image"], dtype=torch.float32).permute((2, 0, 1)).flatten() / 255)
                    container_poses.append(torch.tensor(feature["pose"][:3]))

                else:
                    object_types.append("item")
                    item_images.append(torch.tensor(feature["image"], dtype=torch.float32).permute((2, 0, 1)).flatten() / 255)
                    item_poses.append(torch.tensor(feature["pose"][:3]))
                    
                object_names.append(obj_type)

            state = pds.State([domain_gr.types[t] for t in object_types], pds.ValueDict(), object_names)
            ctx = state.define_context(domain_gr)
            ctx.define_feature('item-pose', torch.stack(item_poses).float())
            ctx.define_feature('container-pose', torch.stack(container_poses).float())
            ctx.define_feature('item-image', torch.stack(item_images))
            ctx.define_feature('container-image', torch.stack(container_images))

            images = torch.stack(images)
            poses = torch.stack(poses)
            goal = env.task.goal_predicates

            act = model.make_action(images, poses, goal, domain_gr, state, cfg["task"], info)
            print(act)
            episode.append((features, act, goal))
            
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {goal}')
            
            if done:
                break

        # ds.add(seed, episode)
        n += 1
        rewards.append(total_reward)
        
        # End video recording
        if record:
            env.end_rec()
    
    print(f"Average succ rate: {np.mean(rewards)}")


if __name__ == '__main__':
    evaluate()
