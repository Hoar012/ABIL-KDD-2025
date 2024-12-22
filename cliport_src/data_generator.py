"""Data collection script."""

import os
import hydra
import numpy as np
import random
import sys

sys.path.insert(0, "./")
from hacl.envs.cliport.cliport import tasks
from hacl.envs.cliport.cliport.dataset import NesyDataset
from hacl.envs.cliport.cliport.environments.environment import Environment
from PIL import Image
from hacl.envs.cliport.cliport.utils import utils

import pickle

def dump(data, data_dir, fname):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, fname), 'wb') as f:
        pickle.dump(data, f)

SUPPORTED_TASKS = ["packing-shapes", "packing-5shapes",
                   "assembling-kits", "put-block-in-bowl-seen-colors", "place-red-in-green", "put-block-in-bowl-composed-colors",
                   "separating-piles-seen-colors", "separating-20piles", "separating-10piles"
                    ]

def get_predicates(task):
    if task in ['packing-shapes', 'packing-5shapes']:
        return ['letter_R', 'letter_A', 'triangle', 'square', 'plus', 'letter_T', 'diamond',
                  'pentagon', 'rectangle', 'flower', 'star', 'circle', 'letter_G', 'letter_V',
                  'letter_E', 'letter_L', 'ring', 'hexagon', 'heart', 'letter_M']
    elif task == 'assembling-kits':
        return []
    elif task in ["place-red-in-green", 'put-block-in-bowl-seen-colors', "put-block-in-bowl-composed-colors"]:
        return ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']
    elif task in ['separating-piles-seen-colors', 'separating-20piles', 'separating-10piles']:
        return ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']

@hydra.main(config_path='../hacl/envs/cliport/cliport/cfg', config_name='data')
def main(cfg):
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
    assert cfg['task'] in SUPPORTED_TASKS
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = NesyDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
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

    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        done = False
        # Rollout expert policy
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            if act is None:
                break

            lang_goal = info['lang_goal']
            img = dataset.get_image(obs)
            image = Image.fromarray(np.uint8(img[:,:,:3]))

            obj_obs = crop_image(env, image, info)
            action = np.array([list(act["pose0"][0]) + list(act["pose0"][1]), list(act["pose1"][0]) + list(act["pose1"][1])])
            goal = env.task.goal_predicates
            # print(goal)
            episode.append((obj_obs, action, goal, done))
            
            obs, reward, done, info = env.step(act)
            total_reward += reward
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            
            if done:
                break
            
        img = dataset.get_image(obs)
        image = Image.fromarray(np.uint8(img[:,:,:3]))
        obj_obs = crop_image(env, image, info)

        episode.append((obj_obs, None, goal, done))

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)

        # End video recording
        if record:
            env.end_rec()

if __name__ == '__main__':
    main()