import os.path as osp
import numpy as np
from typing import Optional
import time
import torch
import jactorch
import hacl.pdsketch as pds
import hacl.pdsketch.interface.v2.rl as pdsrl
from hacl.envs.mini_behavior.mini_behavior import minibehavior
from hacl.envs.mini_behavior.mini_behavior.envs.installing_a_printer import InstallingAPrinterEnv
import random
import gym

import argparse
from gym_minigrid.wrappers import *
from hacl.envs.mini_behavior.mini_behavior.window import Window
from hacl.envs.mini_behavior.mini_behavior.utils.save import get_step, save_demo
from hacl.envs.mini_behavior.mini_behavior.grid import GridDimension

__all__  = [
    'get_domain', 'set_domain_mode',
    'SUPPORTED_ACTION_MODES', 'SUPPORTED_STRUCTURE_MODES', 'SUPPORTED_TASKS',
    'MiniBehaviorEnvV2023', 'MiniBehaviorStateV2023', 'make',
    'visualize_planner', 'visualize_plan'
]


def _map_int(x):
    if isinstance(x, tuple):
        return map(int, x)
    if isinstance(x, np.ndarray):
        return map(int, x)
    if isinstance(x, torch.Tensor):
        return map(int, jactorch.as_numpy(x))


g_domain_action_mode = 'absaction'
g_domain_structure_mode = 'full'
g_domain: Optional[pds.Domain] = None

SUPPORTED_ACTION_MODES = ['absaction', 'envaction']
SUPPORTED_STRUCTURE_MODES = ['basic', 'full', 'abl']
SUPPORTED_TASKS = ['install-a-printer', 'install-a-printer-multi', 'opening_packages', 'opening_packages1', 'opening_packages3', 'MovingBoxesToStorage', 
                   'SortingBooks', 'SortingBooks-multi', 'CollectMisplacedItems-multi',
                   'Throwing_away_leftovers', 'Throwing_away_leftovers1', 'Throwing_away_leftovers2', 'PuttingAwayDishesAfterCleaning', 'BoxingBooksUpForStorage',
                   'CleaningACar', 'CleaningShoes', 'CollectMisplacedItems',
                   'LayingWoodFloors', 'MakingTea', 'OrganizingFileCabinet', 'Washing_pots_and_pans',
                   'WateringHouseplants']


def set_domain_mode(action_mode, structure_mode):
    global g_domain
    assert g_domain is None, 'Domain has been loaded.'
    assert action_mode in SUPPORTED_ACTION_MODES, f'Unsupported action mode: {action_mode}.'
    assert structure_mode in SUPPORTED_STRUCTURE_MODES, f'Unsupported structure mode: {structure_mode}.'

    global g_domain_action_mode
    global g_domain_structure_mode

    g_domain_action_mode = action_mode
    g_domain_structure_mode = structure_mode


def get_domain(force_reload=False):
    global g_domain
    if g_domain is None or force_reload:
        g_domain = pds.load_domain_file(osp.join(
            osp.dirname(__file__),
            'pds_files',
            f'minibehavior-v2023-{g_domain_action_mode}-{g_domain_structure_mode}.pdsketch'
        ))
    return g_domain


class MiniBehaviorEnvV2023(minibehavior.MiniBehaviorEnv):
    def __init__(self, task='', Generalize = False):
        assert task in SUPPORTED_TASKS, f'Unknown task: {task}.'
        self.task = task
        self.options = dict()
        self._gen_grid(Generalize, 6, 6)


    def set_options(self, **kwargs):
        self.options.update(kwargs)

    def get_option(self, name, default=None):
        return self.options.get(name, default)

    def _gen_grid(self, Generalize, width, height):
        if self.task in ['install-a-printer', 'install-a-printer-multi']:
            self._gen_grid_install(Generalize, width, height)
        elif self.task == 'opening_packages':
            self._gen_grid_open(Generalize, width, height)
        elif self.task == 'opening_packages1':
            self._gen_grid_open1(Generalize, width, height)
        elif self.task == 'opening_packages3':
            self._gen_grid_open3(Generalize, width, height)
        elif self.task == 'MovingBoxesToStorage':
            self._gen_grid_moving(Generalize, width, height)
        elif self.task in ['SortingBooks', 'SortingBooks-multi']:
            self._gen_grid_sorting(Generalize, width, height)
        elif self.task == 'Throwing_away_leftovers':
            self._gen_grid_throw(Generalize, width, height)
        elif self.task == 'Throwing_away_leftovers1':
            self._gen_grid_throw1(Generalize, width, height)
        elif self.task == 'Throwing_away_leftovers2':
            self._gen_grid_throw2(Generalize, width, height)
        elif self.task == 'PuttingAwayDishesAfterCleaning':
            self._gen_grid_putting(Generalize, width, height)
        elif self.task == 'BoxingBooksUpForStorage':
            self._gen_grid_boxing(Generalize, width, height)
        elif self.task == 'CleaningACar':
            self._gen_grid_CleaningACar(Generalize, width, height)
        elif self.task == 'CleaningShoes':
            self._gen_grid_CleaningShoes(Generalize, width, height)
        elif self.task in ['CollectMisplacedItems', 'CollectMisplacedItems-multi']:
            self._gen_grid_collect(Generalize, width, height)
        elif self.task == 'LayingWoodFloors':
            self._gen_grid_laying(Generalize, width, height)
        elif self.task == 'MakingTea':
            self._gen_grid_MakingTea(Generalize, width, height)
        elif self.task == 'OrganizingFileCabinet':
            self._gen_grid_Organizing(Generalize, width, height)
        elif self.task == 'Washing_pots_and_pans':
            self._gen_grid_Washing(Generalize, width, height)
        elif self.task == 'WateringHouseplants':
            self._gen_grid_Watering(Generalize, width, height)
        else:
            raise ValueError(f'Unknown task: {self.task}.')

    def _gen_grid_install(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-InstallingAPrinter-Gen-8x8-N2-v0')
        else:
            self.env = gym.make('MiniGrid-InstallingAPrinter-Basic-8x8-N2-v0')
        
        self.printer = self.env.objs["printer"][0]
        self.table = self.env.objs["table"][0]

        self.mission = get_domain().parse(f'(and (toggleon printer_0)(exists (?t - item)(and (is-table ?t)(ontop printer_0 ?t))))')
        
    def _gen_grid_open(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-OpeningPackages-Gen-16x16-N2-v0')
        else:
            self.env = gym.make('MiniGrid-OpeningPackages-Basic-16x16-N2-v0')
        
        self.mission = get_domain().parse("(not(exists (?o - item) (and (is-package ?o)(not (is-open ?o)))))")
    
    def _gen_grid_open1(self, Generalize, width, height):
        self.env = gym.make('MiniGrid-OpeningPackages1-16x16-N2-v0')
        
        self.mission = get_domain().parse("(not(exists (?o - item) (and (is-package ?o)(not (is-open ?o)))))")
        
    def _gen_grid_open3(self, Generalize, width, height):
        self.env = gym.make('MiniGrid-OpeningPackages3-16x16-N2-v0')
        
        self.mission = get_domain().parse("(not(exists (?o - item) (and (is-package ?o)(not (is-open ?o)))))")

    def _gen_grid_moving(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-MovingBoxesToStorage-Gen-16x16-N2-v0')
        else:
            self.env = gym.make('MiniGrid-MovingBoxesToStorage-Basic-16x16-N2-v0')
        
        self.box0 = self.env.objs["carton"][0]
        self.box1 = self.env.objs["carton"][1]
        
        self.mission = get_domain().parse(f'(or (ontop carton_0 carton_1) (ontop carton_1 carton_0))')
    
    def _gen_grid_sorting(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-SortingBooks-Gen-10x10-N2-v0')
        else:
            self.env = gym.make('MiniGrid-SortingBooks-Basic-10x10-N2-v0')
            
        self.book = self.env.objs['book']
        self.hardback = self.env.objs['hardback']
        self.table = self.env.objs['table'][0]
        self.shelf = self.env.objs['shelf'][0]

        self.mission = get_domain().parse(f'(not (exists (?o - item) (and (is-book ?o) (not (exists (?t - item) (and (is-shelf ?t) (ontop ?o ?t)))) )))')
        
    def _gen_grid_throw(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-ThrowingAwayLeftovers-Gen-10x10-N2-v0')
        else:
            self.env = gym.make('MiniGrid-ThrowingAwayLeftovers-Basic-10x10-N2-v0')

        self.hamburgers = self.env.objs['hamburger']
        self.ashcan = self.env.objs['ashcan'][0]
        
        self.mission = get_domain().parse("(not(exists(?o - item) (and (is-hamburger ?o) (not (exists (?t - item)(and (is-ashcan ?t) (inside ?o ?t))))) ))")
    
    def _gen_grid_throw1(self, Generalize, width, height):
        self.env = gym.make('MiniGrid-ThrowingAwayLeftovers1-10x10-N2-v0')

        self.hamburgers = self.env.objs['hamburger']
        self.ashcan = self.env.objs['ashcan'][0]
        
        self.mission = get_domain().parse("(not(exists(?o - item) (and (is-hamburger ?o) (not (exists (?t - item)(and (is-ashcan ?t) (atSameLocation ?o ?t))))) ))")
    
    def _gen_grid_throw2(self, Generalize, width, height):
        self.env = gym.make('MiniGrid-ThrowingAwayLeftovers2-10x10-N2-v0')

        self.hamburgers = self.env.objs['hamburger']
        self.ashcan = self.env.objs['ashcan'][0]
        
        self.mission = get_domain().parse("(not(exists(?o - item) (and (is-hamburger ?o) (not (exists (?t - item)(and (is-ashcan ?t) (inside ?o ?t))))) ))")
    

    def _gen_grid_putting(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-PuttingAwayDishesAfterCleaning-Gen-10x10-N2-v0')
        else:
            self.env = gym.make('MiniGrid-PuttingAwayDishesAfterCleaning-Basic-10x10-N2-v0')
        
        self.plates = self.env.objs['plate']
        self.cabinet = self.env.objs['cabinet'][0]
        
        self.mission = get_domain().parse("(not(exists(?o - item) (and (is-plate ?o) (not (exists (?t - item)(and (is-cabinet ?t) (inside ?o ?t))))) ))")
        
    def _gen_grid_boxing(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-BoxingBooksUpForStorage-Gen-10x10-N2-v0')
        else:
            self.env = gym.make('MiniGrid-BoxingBooksUpForStorage-Basic-10x10-N2-v0')
        
        self.book = self.env.objs['book']
        self.box = self.env.objs['box'][0]
        
        self.mission = get_domain().parse("(not (exists(?o - item)(and (is-book ?o)(not (exists(?t - item)(and (is-box ?t)(inside ?o ?t)))))))")

    def _gen_grid_CleaningACar(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-CleaningACar-Gen-12x12-N2-v0')
        else:
            self.env = gym.make('MiniGrid-CleaningACar-Basic-12x12-N2-v0')
        
        self.car = self.env.objs['car'][0]
        self.rag = self.env.objs['rag'][0]
        self.soap = self.env.objs['soap'][0]
        self.bucket = self.env.objs['bucket'][0]
        # self.sink = self.env.objs['sink'][0]

        self.mission = get_domain().parse(f'(and (exists(?o - item)(and (is-bucket ?o)(inside soap_0 ?o)) ) (exists(?o - item)(and (is-bucket ?o)(inside rag_0 ?o)) )\
                                          (not(is-dusty car_0)))')
    
    def _gen_grid_CleaningShoes(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-CleaningShoes-Gen-10x10-N2-v0')
        else:
            self.env = gym.make('MiniGrid-CleaningShoes-Basic-10x10-N2-v0')
        
        self.mission = get_domain().parse(f'(and (toggleon sink_0))')
    
    def _gen_grid_collect(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-CollectMisplacedItems-Gen-12x12-N2-v0')
        else:
            self.env = gym.make('MiniGrid-CollectMisplacedItems-Basic-12x12-N2-v0')
        
        self.mission = get_domain().parse(f'(forall (?o - item) (or (not(is-collect ?o)) (exists (?t - item) (and (is-table ?t) (ontop ?o ?t))) ))')
    
    def _gen_grid_laying(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-LayingWoodFloors-Gen-16x16-N2-v0')
        else:
            self.env = gym.make('MiniGrid-LayingWoodFloors-Basic-16x16-N2-v0')
            
        self.mission = get_domain().parse(f'(and (or(nextto (item-pose plywood_0)(item-pose plywood_1))(nextto (item-pose plywood_0)(item-pose plywood_2))(nextto (item-pose plywood_0)(item-pose plywood_3)))\
                                          (or(nextto (item-pose plywood_1)(item-pose plywood_0))(nextto (item-pose plywood_1)(item-pose plywood_2))(nextto (item-pose plywood_1)(item-pose plywood_3)))\
                                          (or(nextto (item-pose plywood_2)(item-pose plywood_0))(nextto (item-pose plywood_2)(item-pose plywood_1))(nextto (item-pose plywood_2)(item-pose plywood_3)))\
                                          (or(nextto (item-pose plywood_3)(item-pose plywood_0))(nextto (item-pose plywood_3)(item-pose plywood_1))(nextto (item-pose plywood_3)(item-pose plywood_2))))')
    
    def _gen_grid_MakingTea(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-MakingTea-Gen-12x12-N2-v0')
        else:
            self.env = gym.make('MiniGrid-MakingTea-Basic-12x12-N2-v0')
        
        self.mission = get_domain().parse(f'(exists (?o - item)(and (is-stove ?o)(toggleon ?o) (is-open ?o)(inside tea_bag_0 ?o) (ontop teapot_0 ?o)) )')
        
    def _gen_grid_Organizing(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-OrganizingFileCabinet-Gen-10x10-N2-v0')
        else:
            self.env = gym.make('MiniGrid-OrganizingFileCabinet-Basic-10x10-N2-v0')
        
        self.mission = get_domain().parse(f'(and (exists (?o - item)(and (is-cabinet ?o)(inside folder_0 ?o) ) )\
                                                 (exists (?o - item)(and (is-cabinet ?o)(inside folder_1 ?o) ) )\
                                                 (exists (?o - item)(and (is-cabinet ?o)(inside document_0 ?o) ) )\
                                                 (exists (?o - item)(and (is-cabinet ?o)(inside document_1 ?o) ) )\
                                                 (exists (?o - item)(and (is-cabinet ?o)(inside document_2 ?o) ) )\
                                                 (exists (?o - item)(and (is-cabinet ?o)(inside document_3 ?o) ) )\
                                                 (exists (?o - item)(and (is-table ?o)(ontop marker_0 ?o) ) ) )')
        
    def _gen_grid_Washing(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-WashingPotsAndPans-Gen-12x12-N2-v0')
        else:
            self.env = gym.make('MiniGrid-WashingPotsAndPans-Basic-12x12-N2-v0')
        
        self.mission = get_domain().parse(f'(and(not(exists (?o - item)(and (is-pan ?o)(is-dusty ?o)) ))\
                                            (not(exists (?o - item)(and (is-pan ?o)(not (exists(?t - item)(and (is-cabinet ?t)(inside ?o ?t)))))) ))')
        
    def _gen_grid_Watering(self, Generalize, width, height):
        if Generalize:
            self.env = gym.make('MiniGrid-WateringHouseplants-Gen-10x10-N2-v0')
        else:
            self.env = gym.make('MiniGrid-WateringHouseplants-Basic-10x10-N2-v0')
        
        self.mission = get_domain().parse(f'(not(exists (?o - item)(and (is-plant ?o)(not (exists(?t - item)(and (is-sink ?t)(inside ?o ?t)))))) )')

    def compute_obs(self):
        return {'state': MiniBehaviorStateV2023.from_env(self), 'mission': self.mission}

    def compute_done(self):
        return self.env.compute_done()
        

    def reset(self):
        self.env.reset()
        return self.compute_obs()

    def step_inner(self, action):
        super().step(action)

    def step(self, action: pdsrl.RLEnvAction):
        if action.name == 'forward':
            self.env.step(self.Actions.forward)
        elif action.name == 'lturn':
            self.env.step(self.Actions.left)
        elif action.name == 'rturn':
            self.env.step(self.Actions.right)
        elif action.name == 'toggle':
            self.env.step(self.Actions.toggle)
        elif action.name == 'open':
            self.env.step(self.Actions.open)
        elif action.name == 'close':
            self.env.step(self.Actions.close)
        elif action.name == 'slice':
            self.env.step(self.Actions.slice)
        elif action.name == 'cook':
            self.env.step(self.Actions.cook)
        elif action.name == 'drop_in':
            self.env.step(self.Actions.drop_in)
        elif action.name == 'pickup_0':
            self.env.step(self.Actions.pickup_0)
        elif action.name == 'pickup_1':
            self.env.step(self.Actions.pickup_1)
        elif action.name == 'pickup_2':
            self.env.step(self.Actions.pickup_2)
        elif action.name == 'drop_0':
            self.env.step(self.Actions.drop_0)
        elif action.name == 'drop_1':
            self.env.step(self.Actions.drop_1)
        elif action.name == 'drop_2':
            self.env.step(self.Actions.drop_2)
        else:
            raise ValueError(f'Unknown action: {action}.')

        obs = self.compute_obs()
        done = self.compute_done()
        return obs, -1, done, {}
    
    
    def manual_control(self, tile_size = 32, seed = -1, agent_view = False, save = False, mode = "primitive"):    
        env = self.env

        # Size in pixels of a tile in the full-scale human view
        TILE_PIXELS = 32
        show_furniture = False


        def redraw(img):
            if not agent_view:
                img = env.render('rgb_array', tile_size=tile_size)

            window.no_closeup()
            window.set_inventory(env)
            window.show_img(img)


        def render_furniture():
            global show_furniture
            show_furniture = not show_furniture

            if show_furniture:
                img = np.copy(env.furniture_view)

                # i, j = env.agent.cur_pos
                i, j = env.agent_pos
                ymin = j * TILE_PIXELS
                ymax = (j + 1) * TILE_PIXELS
                xmin = i * TILE_PIXELS
                xmax = (i + 1) * TILE_PIXELS

                img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
                    img[ymin:ymax, xmin:xmax, :], env.agent_dir)
                img = env.render_furniture_states(img)

                window.show_img(img)
            else:
                obs = env.gen_obs()
                redraw(obs)


        def show_states():
            imgs = env.render_states()
            window.show_closeup(imgs)


        def reset():
            if seed != -1:
                env.seed(seed)

            obs = env.reset()

            if hasattr(env, 'mission'):
                print('Mission: %s' % env.mission)
                window.set_caption(env.mission)

            redraw(obs)


        def step(action):
            prev_obs = env.gen_obs()
            obs, reward, done, info = env.step(action)

            print('step=%s, reward=%.2f' % (env.step_count, reward))

            if save:
                all_steps[env.step_count] = (prev_obs, action)

            if done:
                print('done!')
                if save:
                    save_demo(all_steps, a_env, env.episode)
                reset()
            else:
                redraw(obs)


        def switch_dim(dim):
            env.switch_dim(dim)
            print(f'switching to dim: {env.render_dim}')
            obs = env.gen_obs()
            redraw(obs)


        def key_handler_cartesian(event):
            print('pressed', event.key)
            if event.key == 'escape':
                window.close()
                return
            if event.key == 'backspace':
                reset()
                return
            if event.key == 'left':
                step(env.actions.left)
                return
            if event.key == 'right':
                step(env.actions.right)
                return
            if event.key == 'up':
                step(env.actions.forward)
                return
            # Spacebar
            if event.key == ' ':
                render_furniture()
                return
            if event.key == 'pageup':
                step('choose')
                return
            if event.key == 'enter':
                env.save_state()
                return
            if event.key == 'pagedown':
                show_states()
                return
            if event.key == '0':
                switch_dim(None)
                return
            if event.key == '1':
                switch_dim(0)
                return
            if event.key == '2':
                switch_dim(1)
                return
            if event.key == '3':
                switch_dim(2)
                return

        def key_handler_primitive(event):
            print('pressed', event.key)
            if event.key == 'escape':
                window.close()
                return
            if event.key == 'left':
                step(env.actions.left)
                return
            if event.key == 'right':
                step(env.actions.right)
                return
            if event.key == 'up':
                step(env.actions.forward)
                return
            if event.key == '0':
                step(env.actions.pickup_0)
                return
            if event.key == '1':
                step(env.actions.pickup_1)
                return
            if event.key == '2':
                step(env.actions.pickup_2)
                return
            if event.key == '3':
                step(env.actions.drop_0)
                return
            if event.key == '4':
                step(env.actions.drop_1)
                return
            if event.key == '5':
                step(env.actions.drop_2)
                return
            if event.key == 't':
                step(env.actions.toggle)
                return
            if event.key == 'o':
                step(env.actions.open)
                return
            if event.key == 'c':
                step(env.actions.close)
                return
            if event.key == 'k':
                step(env.actions.cook)
                return
            if event.key == 'h':
                step(env.actions.slice)
                return
            if event.key == 'i':
                step(env.actions.drop_in)
                return
            if event.key == 'pagedown':
                show_states()
                return
            if event.key == ' ':
                render_furniture()
                return


        a_env = 'MiniGrid-MakingTea-16x16-N2-v0'
            
        env.teleop_mode()

        all_steps = {}

        if agent_view:
            env = RGBImgPartialObsWrapper(env)
            env = ImgObsWrapper(env)

        window = Window('mini_behavior - ' + a_env)
        
        
        if mode == "cartesian":
            window.reg_key_handler(key_handler_cartesian)
        elif mode == "primitive":
            window.reg_key_handler(key_handler_primitive)

        reset()

        # Blocking event loop
        window.show(block=True)


class MiniBehaviorStateV2023(pds.State):
    @classmethod
    def from_env(cls, env: MiniBehaviorEnvV2023, ignore_walls: bool = True):
        if env.task in ["MovingBoxesToStorage"]:
            ignore_walls = True
        if g_domain_structure_mode == 'full' or g_domain_structure_mode == 'abl':
            return cls.from_env_full(env.env, ignore_walls=ignore_walls)
        else:
            raise ValueError('Unknown domain structure mode: {}.'.format(g_domain_structure_mode))

    @classmethod
    def from_env_full(cls, env: minibehavior.MiniBehaviorEnv, ignore_walls: bool = False):
        object_names = ['r']
        object_types = ['robot']
        object_type2id = dict()
        
        for k in minibehavior.OBJECT_TO_IDX:
            object_type2id[k] = 0
            
        robot_features = list()
        robot_features.append(
            list(env.agent_pos) + [env.agent_dir,]
        )
        
        object_features = list()
        object_images = list()
        object_poses = list()
        object_heights = list()
        objects = list()
        
        for x, y, z, obj in env.iter_objects(ignore_walls = ignore_walls):
            
            obj_name = f'{obj.type}_{object_type2id[obj.type]}'
            object_names.append(obj_name)
            object_types.append('item')
            image = list(obj.encode())
            if obj.type in ["printer", "sink"]:
                toggleon = obj.states['toggleable'].value
                if toggleon:
                    image[-1] = 12
                
            elif obj.type in ["package", "cabinet", "electric_refrigerator"]:
                openable = obj.states['openable'].value
                if openable:
                    image[-1] = 11
            
            elif obj.type in ["car"]:
                dustyable = obj.states['dustyable'].value
                if dustyable:
                    image[-1] = 10
            
            elif obj.type in ["stove"]:
                toggleon = obj.states['toggleable'].value
                openable = obj.states['openable'].value
                if toggleon and openable:
                    image[-1] = 9
                elif toggleon:
                    image[-1] = 12
                elif openable:
                    image[-1] = 11
            elif obj.type in ["teapot", "kettle", "pan"]:
                stainable = obj.states['stainable'].value
                if stainable:
                    image[-1] = 8
            elif obj.type in ["pot_plant"]:
                soakable = obj.states['soakable'].value
                if soakable:
                    image[-1] = 7

            object_images.append(image)
            object_poses.append((x, y))
            object_heights.append([z])
            object_type2id[obj.type] += 1
            objects.append(obj)
            object_features.append(image + [x, y, z])
        
        domain = get_domain()
        state = cls([domain.types[t] for t in object_types], pds.ValueDict(), object_names)
        ctx = pds.TensorDictDefHelper(domain, state)

        predicates = list()
        for obj, obj_name in zip(objects, object_names[1:]):
            if obj.type in ['wall', 'table', 'sofa', 'cabinet']:
                pass
            else:
                predicates.append(ctx.pickable(obj_name))
            if obj.type == 'door' or obj.type == 'printer':
                predicates.append(ctx.toggleable(obj_name))
            
            if obj.type == 'package':
                predicates.append(ctx.openable(obj_name))
                
        if env.carrying is not None and len(env.carrying) > 0:
            obj = list(env.carrying)[0]
            predicates.append(ctx.robot_holding('r', obj.name))

        ctx.define_predicates(predicates)
        ctx.define_feature('robot-pose', torch.tensor(np.array([env.agent_pos]), dtype=torch.float32))
        ctx.define_feature('robot-direction', torch.tensor([[env.agent_dir]], dtype=torch.int64))
        ctx.define_feature('item-pose', torch.tensor(object_poses, dtype=torch.float32))
        ctx.define_feature('item-height', torch.tensor(object_heights, dtype=torch.float32))
        ctx.define_feature('item-image', torch.tensor(object_images, dtype=torch.float32))
        ctx.define_feature('robot-feature', torch.tensor(robot_features, dtype=torch.float32))
        ctx.define_feature('item-feature', torch.tensor(object_features, dtype=torch.float32))

        if env.carrying is not None and len(env.carrying) > 0:
            obj = list(env.carrying)[0]
            image = list(obj.encode())
            if obj.type in ["teapot", "kettle", "pan"]:
                stainable = obj.states['stainable'].value
                if stainable:
                    image[-1] = 8
            ctx.define_feature('holding', torch.tensor([image], dtype=torch.float32))
        else:
            ctx.define_feature('holding', torch.tensor([[0,0,0]], dtype=torch.float32))
        return state

def make(*args, **kwargs):
    return MiniBehaviorEnvV2023(*args, **kwargs)


if __name__ == '__main__':
    env = make()
