"""Sorting Task."""

import numpy as np
from hacl.envs.cliport.cliport.tasks.task import Task
from hacl.envs.cliport.cliport.utils import utils

import pybullet as p


class PlaceRedInGreen(Task):
    """Sorting Task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the red blocks in a green bowl"
        self.task_completed_desc = "done placing blocks in bowls."

        self.goal_predicates = []
        self.objs_id = {}
        
    def reset(self, env):
        super().reset(env)
        n_blocks = n_bowls = np.random.randint(1, 4)
        # n_blocks = np.random.randint(1, n_bowls + 1)
        self.objs_id = {}
        
        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            self.objs_id[bowl_id] = "greencontainer"
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            self.objs_id[block_id] = "redblock"
            blocks.append((block_id, (0, None)))

        # Goal: each red block is in a different green bowl.
        self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                           bowl_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)
        self.goal_predicates = ["red", "green"]
        
        self.max_steps = len(blocks) + 1
        
        # Colors of distractor objects.
        bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'green']
        block_colors = [utils.COLORS[c] for c in utils.COLORS if c != 'red']
        bowl_colors_names = [c for c in utils.COLORS if c != 'green']
        block_colors_names = [c for c in utils.COLORS if c != 'red']
        
        # Add distractors.
        n_distractors = 0
        while n_distractors < 6:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = block_colors if is_block else bowl_colors
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = colors[n_distractors % len(colors)]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            if is_block:
                self.objs_id[obj_id] = block_colors_names[n_distractors % len(colors)] + "block"
            else:
                self.objs_id[obj_id] = bowl_colors_names[n_distractors % len(colors)] + "container"
            n_distractors += 1
