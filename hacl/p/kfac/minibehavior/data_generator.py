import time
import torch
import random
import hacl.pdsketch as pds
import numpy as np

from copy import deepcopy
from hacl.envs.mini_behavior.mini_behavior.path_finding import find_path_to_obj

__all__ = ['MGState', 'OfflineDataGenerator', 'worker_offline', 'worker_bc', 'worker_eval']

class MGState(object):
    def __init__(self, agent_pos, agent_dir, grid, carrying):
        self.agent_pos = agent_pos
        self.agent_dir = agent_dir
        self.grid = grid
        self.carrying = carrying

    @classmethod
    def save(cls, env):
        return cls(deepcopy(env.agent_pos), deepcopy(env.agent_dir), deepcopy(env.grid), deepcopy(env.carrying))

    def restore(self, env):
        env.agent_pos = self.agent_pos
        env.agent_dir = self.agent_dir
        env.agent_grid = self.grid
        env.carrying = self.carrying

def execute_plan(env, plan):
    for action in plan:
        rl_action = pds.rl.RLEnvAction(action_to_operator[action.name])
        obs, reward, done, _ = env.step(rl_action)
    
    return env

action_to_operator = {'left': 'lturn', 'right': 'rturn', 'forward': 'forward', 'pickup_0': 'pickup_0', 'pickup_1': 'pickup_1', 'pickup_2': 'pickup_2', 'drop_0': 'drop_0', 'drop_1': 'drop_1', 'drop_2': 'drop_2', 'toggle': 'toggle', 'open': 'open', 'close':'close', 'slice':'slice', 'cook': 'cook', 'drop_in': 'drop_in'}

class OfflineDataGenerator(object):
    def __init__(self, succ_prob):
        self.succ_prob = 1

    def plan(self, env):
        succ_flag = random.random() < self.succ_prob

        if env.task == 'install-a-printer':
            return self.plan_install(env, succ_flag)
        elif env.task == 'install-a-printer-multi':
            return self.plan_install_multi(env, succ_flag)
        elif env.task == 'opening_packages':
            return self.plan_open(env, succ_flag)
        elif env.task == 'opening_packages1':
            return self.plan_open1(env, succ_flag)
        elif env.task == 'opening_packages3':
            return self.plan_open3(env, succ_flag)
        elif env.task == 'MovingBoxesToStorage':
            return self.plan_moving(env, succ_flag)
        elif env.task == 'SortingBooks':
            return self.plan_sort(env, succ_flag)
        elif env.task == 'SortingBooks-multi':
            return self.plan_sort_multi(env, succ_flag)
        elif env.task == 'Throwing_away_leftovers':
            return self.plan_throwing(env, succ_flag)
        elif env.task == 'Throwing_away_leftovers1':
            return self.plan_throwing1(env, succ_flag)
        elif env.task == 'Throwing_away_leftovers2':
            return self.plan_throwing2(env, succ_flag)
        elif env.task == 'PuttingAwayDishesAfterCleaning':
            return self.plan_putting(env, succ_flag)
        elif env.task == 'BoxingBooksUpForStorage':
            return self.plan_boxing(env, succ_flag)
        elif env.task == 'Setting_up_candles':
            return self.plan_setting(env, succ_flag)
        elif env.task == 'CleaningACar':
            return self.plan_CleaningACar(env, succ_flag)
        elif env.task == 'CleaningShoes':
            return self.plan_CleaningShoes(env, succ_flag)
        elif env.task == 'CollectMisplacedItems':
            return self.plan_collect(env, succ_flag)
        elif env.task == 'CollectMisplacedItems-multi':
            return self.plan_collect_multi(env, succ_flag)
        elif env.task == 'LayingWoodFloors':
            return self.plan_laying(env, succ_flag)
        elif env.task == 'MakingTea':
            return self.plan_MakingTea(env, succ_flag)
        elif env.task == 'OrganizingFileCabinet':
            return self.plan_Organizing(env, succ_flag)
        elif env.task == 'Washing_pots_and_pans':
            return self.plan_Washing(env, succ_flag)
        elif env.task == 'WateringHouseplants':
            return self.plan_Watering(env, succ_flag)
        else:
            raise NotImplementedError('Unknown task: {}.'.format(self.task))

    def plan_install(self, env, succ_flag=True):
        c_env = deepcopy(env)
        printer_pos = env.printer.cur_pos
        table_pos = env.table.all_pos

        plan1 = find_path_to_obj(env.env, printer_pos)
        if plan1 is None:
            return None

        plan1.append(env.Actions.pickup_0)  # 捡到物体1
        c_env = execute_plan(c_env, plan1)

        plan2 = None

        agent_pos = c_env.env.agent_pos
        goal_pos = table_pos[np.argmin(np.sum(abs(agent_pos-table_pos),1))] #距离最近的桌子

        plan2 = find_path_to_obj(c_env.env, goal_pos)
        if plan2 is None:
            return plan1

        plan2.append(env.Actions.drop_2)
        if succ_flag:
            plan2.append(env.Actions.toggle)

        return plan1 + plan2

    def plan_install_multi(self, env, succ_flag=True):
        prob = random.random()
        
        c_env = deepcopy(env)
        printer_pos = env.printer.cur_pos
        table_pos = env.table.all_pos
        
        plan1 = find_path_to_obj(env.env, printer_pos)
        if plan1 is None:
            return None
        
        if prob < 0.5:
            plan1.append(env.Actions.toggle)
        plan1.append(env.Actions.pickup_0)  # 捡到物体1
        c_env = execute_plan(c_env, plan1)

        plan2 = None

        agent_pos = c_env.env.agent_pos
        goal_pos = table_pos[np.argmin(np.sum(abs(agent_pos-table_pos),1))] #距离最近的桌子

        plan2 = find_path_to_obj(c_env.env, goal_pos)
        if plan2 is None:
            return plan1

        plan2.append(env.Actions.drop_2)

        if prob >= 0.5:
            plan2.append(env.Actions.toggle)

        return plan1 + plan2

    def plan_open(self, env, succ_flag = True):
        c_env = deepcopy(env)

        package0_pos = env.env.objs["package"][0].cur_pos
        package1_pos = env.env.objs["package"][1].cur_pos
        agent_pos = env.env.agent_pos
        dist0 = sum(abs(package0_pos-agent_pos))
        dist1 = sum(abs(package1_pos-agent_pos))
        if dist0 <= dist1:
            goal0 = package0_pos
            goal1 = package1_pos
        else:
            goal0 = package1_pos
            goal1 = package0_pos
        plan1 = find_path_to_obj(env.env, goal0)
        if plan1 is None:
            return None

        plan1.append(env.Actions.open)  # open1
        c_env = execute_plan(c_env, plan1)

        plan2 = find_path_to_obj(c_env.env, goal1)
        if plan2 is None:
            return plan1

        plan2.append(env.Actions.open) # open2
        return plan1 + plan2

    def plan_open1(self, env, succ_flag = True):
        package = env.env.objs["package"][0]
        package_pos = package.cur_pos
        
        plan1 = find_path_to_obj(env.env, package_pos)
        if plan1 is None:
            return []

        plan1.append(env.Actions.open)  # open1
        return plan1

    def plan_open3(self, env, succ_flag = True):
        c_env = deepcopy(env)
        packages = env.env.objs["package"]

        plan = []
        for package in packages:
            plan1 = find_path_to_obj(c_env.env, package.cur_pos)
            if plan1 is None:
                return []

            plan1.append(env.Actions.open)
            c_env = execute_plan(c_env, plan1)
            
            plan.append(plan1)
        
        return plan[0] + plan[1] + plan[2]
        
    def plan_moving(self, env, succ_flag = True):
        c_env = deepcopy(env)
        box0_pos = env.box0.cur_pos
        box1_pos = env.box1.cur_pos
        agent_pos = env.env.agent_pos
        dist0 = sum(abs(box0_pos-agent_pos))
        dist1 = sum(abs(box1_pos-agent_pos))
        if dist0 <= dist1:
            goal0 = box0_pos
            goal1 = box1_pos
        else:
            goal0 = box1_pos
            goal1 = box0_pos

        plan1 = find_path_to_obj(env.env, goal0)
        if plan1 is None:
            return None

        plan1.append(env.Actions.pickup_0)  # 捡到一个箱子
        c_env = execute_plan(c_env, plan1)

        door = env.env.doors[0].cur_pos
        plan_to_door = find_path_to_obj(c_env.env, door)
        if plan_to_door is None:
            return plan1
        
        c_env = execute_plan(c_env, plan_to_door)

        plan2 = find_path_to_obj(c_env.env, goal1)
        if plan2 is None:
            return plan1 + plan_to_door

        plan2.append(env.Actions.drop_1) # drop

        return plan1 + plan_to_door + plan2

    def plan_sort(self, env, succ_flag=True):
        shelf_index = 0
        c_env = deepcopy(env)
        book = env.book
        hardback = env.hardback
        shelf = env.shelf

        shelf_pos = shelf.all_pos

        all_books = [book[0], hardback[0], book[1], hardback[1]]
        pick_action = [env.Actions.pickup_0, env.Actions.pickup_0, env.Actions.pickup_2, env.Actions.pickup_2]
        pick_order = [0, 1, 2, 3]

        plan = []
        
        for i in pick_order:
            sub_plan = find_path_to_obj(c_env.env, all_books[i].cur_pos)
            if sub_plan is None:
                return []
            
            sub_plan.append(pick_action[i])
            c_env = execute_plan(sub_plan)
                
            while shelf_index < 6:
                path_plan = find_path_to_obj(c_env.env, shelf_pos[shelf_index])
                shelf_index += 1
                if path_plan is not None:
                    break
            if shelf_index >= 6:
                return []
            if path_plan is None:
                return []
            path_plan.append(c_env.Actions.drop_2) # drop it on the shelf
            c_env = execute_plan(path_plan)
            
            plan.append(sub_plan + path_plan)

        return plan[0] + plan[1] + plan[2] + plan[3]
    
    def plan_sort_multi(self, env, succ_flag=True):
        shelf_index = 0
        c_env = deepcopy(env)
        book = env.book
        hardback = env.hardback
        shelf = env.shelf

        shelf_pos = shelf.all_pos

        all_books = [book[0], book[1], hardback[0], hardback[1]]
        pick_action = [env.Actions.pickup_0, env.Actions.pickup_2, env.Actions.pickup_0, env.Actions.pickup_2]
        pick_order = [0, 1, 2, 3]
        random.shuffle(pick_order)
        plan = []
        
        for i in pick_order:
            sub_plan = find_path_to_obj(c_env.env, all_books[i].cur_pos)
            if sub_plan is None:
                return []
            
            sub_plan.append(pick_action[i])
            c_env = execute_plan(sub_plan)
                
            while shelf_index < 6:
                path_plan = find_path_to_obj(c_env.env, shelf_pos[shelf_index])
                shelf_index += 1
                if path_plan is not None:
                    break
            if shelf_index >= 6:
                return []
            if path_plan is None:
                return []
            path_plan.append(c_env.Actions.drop_2)#放在书架上
            c_env = execute_plan(path_plan)
            
            plan.append(sub_plan + path_plan)

        return plan[0] + plan[1] + plan[2] + plan[3]
    
    def plan_throwing(self, env, succ_flag=True):
        c_env = deepcopy(env)
        hamburgers = env.hamburgers
        ashcan = env.ashcan

        ashcan_pos = ashcan.cur_pos
        agent_pos = c_env.env.agent_pos
        ham_pos = np.array([hamburgers[0].cur_pos,hamburgers[1].cur_pos,hamburgers[2].cur_pos])
        dist = np.sum(abs(ham_pos - agent_pos),axis=-1)
        ham_pos = ham_pos[dist.argsort()]

        plan = []
        for pos in ham_pos:
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_2)  #pickup the hamburger
            c_env = execute_plan(plan1)

            plan2 = find_path_to_obj(c_env.env, ashcan_pos)
            if plan2 is None:
                return []
            plan2.append(c_env.Actions.drop_in)  # drop it into the ashcan
            c_env = execute_plan(plan2)
            
            plan.append(plan1+plan2)

        # return plan
        return plan[0] + plan[1] + plan[2]

    def plan_throwing1(self, env, succ_flag=True):
        c_env = deepcopy(env)
        hamburgers = env.hamburgers
        ashcan = env.ashcan

        ashcan_pos = ashcan.cur_pos
        ham_pos = hamburgers[0].cur_pos

        plan1 = find_path_to_obj(c_env.env, ham_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_2)  #捡起桌子上的hamburger
        
        for action in plan1:
            rl_action = pds.rl.RLEnvAction(action_to_operator[action.name])
            obs, reward, done, _ = c_env.step(rl_action)

        plan2 = find_path_to_obj(c_env.env, ashcan_pos)
        if plan2 is None:
            return []
        plan2.append(c_env.Actions.drop_in)  #丢进垃圾桶
        
        return plan1 + plan2
    
    def plan_throwing2(self, env, succ_flag=True):
        c_env = deepcopy(env)
        hamburgers = env.hamburgers
        ashcan = env.ashcan

        ashcan_pos = ashcan.cur_pos

        agent_pos = c_env.env.agent_pos
        ham_pos = np.array([hamburgers[0].cur_pos,hamburgers[1].cur_pos])

        dist = np.sum(abs(ham_pos - agent_pos),axis=-1)
        ham_pos = ham_pos[dist.argsort()]
        plan = []

        for pos in ham_pos:
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_2)  # pickup the hamburger
            c_env = execute_plan(plan1)
            
            plan2 = find_path_to_obj(c_env.env, ashcan_pos)
            if plan2 is None:
                return []
            plan2.append(c_env.Actions.drop_in)  # drop it into the ashcan
            
            c_env = execute_plan(plan2)
            
            plan.append(plan1+plan2)

        # return plan
        return plan[0] + plan[1]
    
    def plan_putting(self, env, succ_flag=True):
        c_env = deepcopy(env)
        plates = env.plates
        cabinet = env.cabinet

        plate_pos = np.array([p.cur_pos for p in plates])
        cabinet_pos = np.array(cabinet.all_pos)
        agent_pos = c_env.env.agent_pos

        dist = np.sum(abs(cabinet_pos - agent_pos),axis=-1)
        cabinet_pos = cabinet_pos[dist.argsort()]
        cabinet_index = 0
        cabinet_vol = [0, 0, 0, 0, 0, 0]
        
        dist = np.sum(abs(plate_pos - agent_pos),axis=-1)
        plate_pos = plate_pos[dist.argsort()]

        is_open = False
        plan = []
        for pos in plate_pos:
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_1)  #捡起countertop上的plate
            
            c_env = execute_plan(plan1)

            if cabinet_vol[cabinet_index] >= 3:
                cabinet_index += 1
            plan2 = None
            while cabinet_index < 6:
                plan2 = find_path_to_obj(c_env.env, cabinet_pos[cabinet_index])
                if plan2 is not None:
                    break
                cabinet_index += 1
            if cabinet_index >= 6:
                return []
            cabinet_vol[cabinet_index] += 1
            
            if not is_open:
                plan2.append(c_env.Actions.open)  #open the cabinet
                is_open = True
            plan2.append(c_env.Actions.drop_in)
            
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)

        # return plan
        return plan[0] + plan[1] + plan[2] + plan[3]
    
    def plan_boxing(self, env, succ_flag=True):
        c_env = deepcopy(env)
        books = env.book[:3]
        box = env.box

        book_pos = np.array([b.cur_pos for b in books])
        box_pos = np.array(box.all_pos)
        agent_pos = c_env.env.agent_pos

        dist = np.sum(abs(box_pos - agent_pos),axis=-1)
        box_pos = box_pos[dist.argsort()]
        box_index = 0
        box_vol = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        dist = np.sum(abs(book_pos - agent_pos),axis=-1)
        book_pos = book_pos[dist.argsort()]

        is_open = False
        plan = []
        for pos in book_pos:
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_0)  #捡起地上的书
            c_env = execute_plan(plan1)

            if box_vol[box_index] >= 1:
                box_index += 1
            plan2 = None
            while box_index < 9:
                plan2 = find_path_to_obj(c_env.env, box_pos[box_index])
                if plan2 is not None:
                    break
                box_index += 1
            if box_index >= 9:
                return []
            box_vol[box_index] += 1
            
            plan2.append(c_env.Actions.drop_in)  #put it in
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)
        
        books = env.book[3:]
        book_pos = np.array([b.cur_pos for b in books])
        
        agent_pos = c_env.env.agent_pos
        
        dist = np.sum(abs(book_pos - agent_pos),axis=-1)
        book_pos = book_pos[dist.argsort()]
        
        for pos in book_pos:
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_2)  #捡shelf上的书
            c_env = execute_plan(plan1)

            if box_vol[box_index] >= 1:
                box_index += 1
            plan2 = None
            while box_index < 9:
                plan2 = find_path_to_obj(c_env.env, box_pos[box_index])
                if plan2 is not None:
                    break
                box_index += 1
            if box_index >= 9:
                return []
            box_vol[box_index] += 1
            
            plan2.append(c_env.Actions.drop_in)  #put it in
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)

        # return plan
        return plan[0] + plan[1] + plan[2] + plan[3] +plan[4]
    
    def plan_setting(self, env, succ_flag=True):
        c_env = deepcopy(env)
        candles = env.candle
        table = env.table

        candle_pos = np.array([c.cur_pos for c in candles])
        agent_pos = c_env.env.agent_pos
    
        dist = np.sum(abs(candle_pos - agent_pos),axis=-1)
        candle_pos = candle_pos[dist.argsort()]
        
        
        table_pos = np.array(table[0].all_pos)
        dist = np.sum(abs(table_pos - agent_pos),axis=-1)
        table_pos = table_pos[dist.argsort()]
        table_index = 0
        table_vol = [0, 0, 0, 0]
        
        plan = []
        for pos in candle_pos[:2]:
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_0)  #捡起箱子里的candle
            c_env = execute_plan(plan1)

            if table_vol[table_index] >= 1:
                table_index += 1
            plan2 = None
            while table_index < 4:
                plan2 = find_path_to_obj(c_env.env, table_pos[table_index])
                if plan2 is not None:
                    break
                table_index += 1
            if table_index >= 4:
                return []
            table_vol[table_index] += 1
            
            plan2.append(c_env.Actions.drop_2)  #放在桌子上
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)
            
        table_pos = np.array(table[1].all_pos)
        agent_pos = c_env.env.agent_pos
        dist = np.sum(abs(table_pos - agent_pos),axis=-1)
        table_pos = table_pos[dist.argsort()]
        table_index = 0
        table_vol = [0, 0, 0, 0]
        
        for pos in candle_pos[2:]:
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_0)  #捡起箱子里的candle
            c_env = execute_plan(plan1)

            if table_vol[table_index] >= 1:
                table_index += 1
            plan2 = None
            while table_index < 4:
                plan2 = find_path_to_obj(c_env.env, table_pos[table_index])
                if plan2 is not None:
                    break
                table_index += 1
            if table_index >= 4:
                return []
            table_vol[table_index] += 1
            
            plan2.append(c_env.Actions.drop_2)  #放在桌子上
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)

        # return plan
        return plan[0] + plan[1] + plan[2] + plan[3]

    def plan_CleaningACar(self, env, succ_flag=True):
        c_env = deepcopy(env)
        car = c_env.car
        rag = c_env.rag
        soap = c_env.soap
        bucket = c_env.bucket
        # sink = c_env.sink
        
        car_pos = np.array(car.all_pos)
        rag_pos = rag.cur_pos
        soap_pos = soap.cur_pos
        bucket_pos = np.array(bucket.all_pos)
        
        agent_pos = c_env.env.agent_pos
        
        plan = []
        
        plan1 = find_path_to_obj(c_env.env, rag_pos)  # pickup a rag
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_2)
        c_env = execute_plan(plan1)
            
        # clean the car
        agent_pos = c_env.env.agent_pos
        dist = np.sum(abs(car_pos - agent_pos),axis=-1)
        car_pos = car_pos[dist.argsort()]
        
        for pos in car_pos:
            plan2 = find_path_to_obj(c_env.env, pos)
            if plan2 is None:
                continue
            plan2.append(c_env.Actions.drop_2)
            break
        if plan2 is None:
            return []
        c_env = execute_plan(plan2)
        plan.append(plan1 + plan2)
        
        # put the tools into the bucket
        plan1 = [c_env.Actions.pickup_2]
        plan2 = find_path_to_obj(c_env.env, bucket_pos[0])
        if plan2 is None:
            return []
        plan2.append(c_env.Actions.drop_in)
        c_env = execute_plan(plan1 + plan2)
        
        plan3 = find_path_to_obj(c_env.env, soap.cur_pos)
        if plan3 is None:
            return []
        plan3.append(c_env.Actions.pickup_2)
        c_env = execute_plan(plan3)

        plan4 = find_path_to_obj(c_env.env, bucket_pos[1])
        if plan4 is None:
            return []
        plan4.append(c_env.Actions.drop_in)
        
        plan.append(plan1 + plan2 + plan3 + plan4)
        
        assert len(plan) == 2
        return plan[0] + plan[1]
    
    def plan_CleaningShoes(self, env, succ_flag=True):
        c_env = deepcopy(env)
        towel = c_env.env.objs['towel'][0]
        rag = c_env.env.objs['rag'][0]
        soap = c_env.env.objs['soap'][0]
        sink = c_env.env.objs['sink'][0]
        shoes = c_env.env.objs['shoe']
        
        shoes_pos = np.array([s.cur_pos for s in shoes])
        towel_pos = towel.cur_pos
        rag_pos = rag.cur_pos
        soap_pos = soap.cur_pos
        sink_pos = np.array(sink.all_pos)
        
        agent_pos = c_env.env.agent_pos
        
        dist = np.sum(abs(shoes_pos - agent_pos),axis=-1)
        shoes_pos = shoes_pos[dist.argsort()]
        
        plan = []
        
        # pick towel
        plan1 = find_path_to_obj(c_env.env, towel_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_0)
        c_env = execute_plan(plan1)
            
        plan2 = find_path_to_obj(c_env.env, sink_pos[0])
        if plan2 is None:
            return []
        plan2.append(c_env.Actions.drop_in)
        c_env = execute_plan(plan2)

        plan.append(plan1 + plan2)
        
        # pick rag
        plan1 = find_path_to_obj(c_env.env, rag_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_1)
        c_env = execute_plan(plan1)
            
        plan2 = find_path_to_obj(c_env.env, sink_pos[1])
        if plan2 is None:
            return []
        plan2.append(c_env.Actions.drop_in)
        c_env = execute_plan(plan2)
        plan.append(plan1 + plan2)
        
        # pick shoes
        i = 0
        for pos in shoes_pos:
            plan1 = find_path_to_obj(c_env.env, pos)
            
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_1)  #捡起床上的shoe
            c_env = execute_plan(plan1)

            plan2 = find_path_to_obj(c_env.env, sink_pos[i])
            if plan2 is None:
                return []
            
            plan2.append(c_env.Actions.drop_1)  #放在towel上
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)
            i += 1
        
        # pick soap
        plan1 = find_path_to_obj(c_env.env, soap_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_1)
        c_env = execute_plan(plan1)
            
        plan2 = find_path_to_obj(c_env.env, sink_pos[2])
        if plan2 is None:
            return []
        plan2.append(c_env.Actions.drop_in)
        plan2.append(c_env.Actions.toggle)
        
        plan.append(plan1 + plan2)
            
        assert len(plan) == 5
        return plan[0] + plan[1] + plan[2] + plan[3] + plan[4]
    
    def plan_collect(self, env, succ_flag=True):
        c_env = deepcopy(env)
        shoe = c_env.env.objs['gym_shoe'][0]
        necklace = c_env.env.objs['necklace'][0]
        notebook = c_env.env.objs['notebook'][0]
        socks = c_env.env.objs['sock']
        table = c_env.env.objs['table'][0]
        
        table_pos = np.array(table.all_pos)
        notebook_pos = notebook.cur_pos
        
        table_p = []
        for i in range(6):
            if (table_pos[i] != notebook_pos).any():
                table_p.append(table_pos[i])
        
        table_index = 0

        plan = []
        height = [0, 0, 1, 2]  # necklace  sock1  sock0  shoe
        pick_action = [c_env.Actions.pickup_0, c_env.Actions.pickup_0, c_env.Actions.pickup_1, c_env.Actions.pickup_2]
        for pos in [necklace.cur_pos, socks[1].cur_pos, socks[0].cur_pos, shoe.cur_pos]:
            
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(pick_action[table_index])  #pick
            c_env = execute_plan(plan1)

            plan2 = None
            plan2 = find_path_to_obj(c_env.env, table_p[table_index])
            if plan2 is None:
                return []

            plan2.append(c_env.Actions.drop_2)  #放在桌子上
            c_env = execute_plan(plan2)
                
            table_index += 1
        
            plan.append(plan1 + plan2)
        
        plan1 = find_path_to_obj(c_env.env, notebook_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_0)
        plan1.append(c_env.Actions.drop_2)
    
        plan.append(plan1)
        return plan[0] + plan[1] + plan[2] + plan[3] +plan[4]
    
    def plan_collect_multi(self, env, succ_flag=True):
        c_env = deepcopy(env)
        shoe = c_env.env.objs['gym_shoe'][0]
        necklace = c_env.env.objs['necklace'][0]
        notebook = c_env.env.objs['notebook'][0]
        socks = c_env.env.objs['sock']
        table = c_env.env.objs['table'][0]
        
        table_pos = np.array(table.all_pos)
        notebook_pos = notebook.cur_pos
        
        table_p = []
        for i in range(6):
            if (table_pos[i] != notebook_pos).any():
                table_p.append(table_pos[i])
        
        table_index = 0
        
        pick_action = [c_env.Actions.pickup_0, c_env.Actions.pickup_0, c_env.Actions.pickup_1, c_env.Actions.pickup_2]
        all_pos = [necklace.cur_pos, socks[1].cur_pos, socks[0].cur_pos, shoe.cur_pos]
        pick_order = [0, 1, 2, 3]
        random.shuffle(pick_order)
        
        plan = []

        for i in pick_order:
            pos = all_pos[i]            
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(pick_action[i]) 
            c_env = execute_plan(plan1)

            plan2 = None
            plan2 = find_path_to_obj(c_env.env, table_p[table_index])
            if plan2 is None:
                return []

            plan2.append(c_env.Actions.drop_2)
            c_env = execute_plan(plan2)
                
            table_index += 1
        
            plan.append(plan1 + plan2)
        
        plan1 = find_path_to_obj(c_env.env, notebook_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_0)
        plan1.append(c_env.Actions.drop_2)
    
        plan.append(plan1)
        return plan[0] + plan[1] + plan[2] + plan[3] +plan[4]

    def plan_laying(self, env, succ_flag=True):
        c_env = deepcopy(env)
        plywoods = c_env.env.objs['plywood']
        hammer = c_env.env.objs['hammer'][0]
        saw = c_env.env.objs['saw'][0]
        
        comp_pos = [[6,5], [6,6], [6,7], [6,8], [6,9], [6,10]]
        target_pos = [(7,7), (7,8), (8,7), (8,8)]
        plan = []
        if hammer.cur_pos in target_pos:
            plan1 = find_path_to_obj(c_env.env, hammer.cur_pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_0)
            c_env = execute_plan(plan1)

            for x,y in comp_pos:
                if c_env.env.grid.get(x,y)[0][1] is None:
                    plan2 = find_path_to_obj(c_env.env, (x,y))
                    break
            if plan2 is None:
                return []
            plan2.append(c_env.Actions.drop_0)
            c_env = execute_plan(plan2)
                
            plan.append(plan1+plan2)
        else:
            plan.append([])
            
        if saw.cur_pos in target_pos:
            plan1 = find_path_to_obj(c_env.env, saw.cur_pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_0)  #捡起地上的saw
            c_env = execute_plan(plan1)

            for x,y in comp_pos:
                if c_env.env.grid.get(x,y)[0][1] is None:
                    plan2 = find_path_to_obj(c_env.env, (x,y))
                    break
            if plan2 is None:
                return []
            plan2.append(c_env.Actions.drop_0)
            c_env = execute_plan(plan2)
                
            plan.append(plan1+plan2)
        else:
            plan.append([])

        agent_pos = c_env.env.agent_pos
        
        plywood_pos = np.array([plywoods[0].cur_pos, plywoods[1].cur_pos, plywoods[2].cur_pos, plywoods[3].cur_pos])

        dist = np.sum(abs(plywood_pos - agent_pos),axis=-1)
        plywood_pos = plywood_pos[dist.argsort()]
        
        i = 0
        for pos in plywood_pos:
            
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_0)
            c_env = execute_plan(plan1)

            plan2 = find_path_to_obj(c_env.env, target_pos[i])
            if plan2 is None:
                return []
            plan2.append(c_env.Actions.drop_0)
            c_env = execute_plan(plan2)
                
            plan.append(plan1+plan2)
            i += 1
 
        assert len(plan) == 6
        # return plan
        return plan[0] + plan[1] + plan[2] + plan[3] + plan[4] + plan[5]
    
    def plan_MakingTea(self, env, succ_flag=True):
        c_env = deepcopy(env)
        teapot = c_env.env.objs['teapot'][0]
        tea_bag = c_env.env.objs['tea_bag'][0]
        lemon = c_env.env.objs['lemon'][0]
        stove = c_env.env.objs['stove'][0]
        
        plan = []
        plan1 = find_path_to_obj(c_env.env, teapot.cur_pos)
            
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.open)
        plan1.append(c_env.Actions.pickup_1)  #捡起cabinet里的teapot
        c_env = execute_plan(plan1)

        stove_pos = np.array(stove.all_pos)
        agent_pos = c_env.env.agent_pos
    
        dist = np.sum(abs(stove_pos - agent_pos),axis=-1)
        stove_pos = stove_pos[dist.argsort()]
        
        plan2 = None
        for pos in stove_pos:
            plan2 = find_path_to_obj(c_env.env, pos)
            if plan2 is not None:
                break
        if plan2 is None:
            return []
        
        plan2.append(c_env.Actions.open)
        plan2.append(c_env.Actions.drop_2)
        c_env = execute_plan(plan2)
        
        plan.append(plan1 + plan2)
        
        # teabag
        plan1 = find_path_to_obj(c_env.env, tea_bag.cur_pos)
            
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_0)
        c_env = execute_plan(plan1)
        
        plan2 = find_path_to_obj(c_env.env, teapot.cur_pos)

        if plan2 is None:
            return []
        
        plan2.append(c_env.Actions.drop_in)
        plan2.append(c_env.Actions.toggle)
        
        plan.append(plan1 + plan2)
        
        assert len(plan) == 2
        return plan[0] + plan[1]
    
    def plan_Organizing(self, env, succ_flag=True):
        c_env = deepcopy(env)
        table = c_env.env.objs['table'][0]
        cabinet = c_env.env.objs['cabinet'][0]
        marker = c_env.env.objs['marker'][0]
        folders = c_env.env.objs['folder']
        documents = c_env.env.objs['document']
        
        pos1 = documents[1].cur_pos
        pos2 = documents[3].cur_pos
        plan = []
        
        plan1 = find_path_to_obj(c_env.env, pos1)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.open)  #open the shelf
        plan1.append(c_env.Actions.pickup_0)  #pick
        plan1.append(c_env.Actions.drop_in)
        
        c_env = execute_plan(plan1)

        plan.append(plan1)
        
        plan2 = find_path_to_obj(c_env.env, pos2)
        if plan2 is None:
            return []
        plan2.append(c_env.Actions.pickup_1)  #pick
        plan2.append(c_env.Actions.drop_in)
        c_env = execute_plan(plan2)

        plan.append(plan2)
        
        pick_action = [c_env.Actions.pickup_2, c_env.Actions.pickup_2, c_env.Actions.pickup_2, c_env.Actions.pickup_0]
        pick_index = 0
        cabinet_p = [pos1, pos1, pos2, pos2]
        for pos in [documents[0].cur_pos, documents[2].cur_pos, folders[0].cur_pos, folders[1].cur_pos]:
            
            plan1 = find_path_to_obj(c_env.env, pos)
            if plan1 is None:
                return []
            plan1.append(pick_action[pick_index])  #pick
            c_env = execute_plan(plan1)

            plan2 = None
            plan2 = find_path_to_obj(c_env.env, cabinet_p[pick_index])
            if plan2 is None:
                return []

            plan2.append(c_env.Actions.drop_in)  #放在cabinet中
            c_env = execute_plan(plan1)

            pick_index += 1        
            plan.append(plan1 + plan2)

        plan1 = find_path_to_obj(c_env.env, marker.cur_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_1)  #pick 
        c_env = execute_plan(plan1)
        
        for pos in table.all_pos:
            plan2 = find_path_to_obj(c_env.env, pos)
            if plan2 is not None:
                break
        if plan2 is None:
            return []
        plan2.append(c_env.Actions.drop_2)  #放在桌子上
    
        plan.append(plan1 + plan2)
        assert len(plan) == 7
        return plan[0] + plan[1] + plan[2] + plan[3] + plan[4] + plan[5] + plan[6]

    def plan_Washing(self, env, succ_flag=True):
        c_env = deepcopy(env)
        teapot = c_env.env.objs['teapot'][0]
        kettle = c_env.env.objs['kettle'][0]
        pan = c_env.env.objs['pan'][0]
        sink = c_env.env.objs['sink'][0]
        scrub_brush = c_env.env.objs['scrub_brush'][0]
        cabinet = c_env.env.objs['cabinet'][0]
        
        plan = []
        sink_pos = sink.all_pos
        sink_index = 1
        for obj in [pan, kettle, teapot]:
            plan1 = find_path_to_obj(c_env.env, obj.cur_pos)
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_1)
            c_env = execute_plan(plan1)
                
            plan2 = find_path_to_obj(c_env.env, sink_pos[sink_index])
            if plan2 is None:
                return []
            plan2.append(c_env.Actions.drop_in)
            c_env = execute_plan(plan2)

            plan.append(plan1 + plan2)
            sink_index += 1
        
        # wash
        plan0 = [c_env.Actions.toggle]
        plan1 = find_path_to_obj(c_env.env, scrub_brush.cur_pos)
        if plan1 is None:
            return []
        plan1.append(c_env.Actions.pickup_1)
        c_env = execute_plan(plan0 + plan1)
            
        plan0 = plan0 + plan1
        
        for obj in [pan, kettle, teapot]:
            plan2 = find_path_to_obj(c_env.env, obj.cur_pos)
            if plan2 is None:
                return []
            plan2.append(c_env.Actions.drop_1)
            if obj is not teapot:
                plan2.append(c_env.Actions.pickup_1)
            
            c_env = execute_plan(plan2)

            plan0 = plan0 + plan2
            
        plan.append(plan0)
        
        # pick-place in the cabinet
        
        for obj in [teapot, kettle, pan]:
            if obj is teapot:
                plan1 = []
            else:
                plan1 = find_path_to_obj(c_env.env, obj.cur_pos)
                if plan1 is None:
                    return []
            plan1.append(c_env.Actions.pickup_0)
            c_env = execute_plan(plan1)
                
            cabinet_pos = np.array(cabinet.all_pos)
            agent_pos = c_env.env.agent_pos
            dist = np.sum(abs(cabinet_pos - agent_pos),axis=-1)
            cabinet_pos = cabinet_pos[dist.argsort()]
            plan2 = None
            for pos in cabinet_pos:
                plan2 = find_path_to_obj(c_env.env, pos)
                if plan2 is not None:
                    break
            if plan2 is None:
                return []
            if obj is teapot:
                plan2.append(c_env.Actions.open)
            plan2.append(c_env.Actions.drop_in)
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)
        assert len(plan) == 7
        return plan[0] + plan[1] + plan[2] + plan[3] + plan[4] + plan[5] + plan[6]

    def plan_Watering(self, env, succ_flag=True):
        c_env = deepcopy(env)
        pot_plants = c_env.env.objs['pot_plant']
        sink = c_env.env.objs['sink'][0]
        
        plants_pos = np.array([p.cur_pos for p in pot_plants])
        sink_pos = np.array(sink.all_pos)
        agent_pos = c_env.env.agent_pos
        
        dist = np.sum(abs(plants_pos - agent_pos),axis=-1)
        plants_pos = plants_pos[dist.argsort()]
        
        plan = []
        
        i = 0
        toggleon = False
        for pos in plants_pos:
            plan1 = find_path_to_obj(c_env.env, pos)
            
            if plan1 is None:
                return []
            plan1.append(c_env.Actions.pickup_0)  #捡起地上的plant
            c_env = execute_plan(plan1)

            plan2 = find_path_to_obj(c_env.env, sink_pos[i])
            if plan2 is None:
                return []
            if not toggleon:
                plan2.append(c_env.Actions.toggle)
                toggleon = True
            plan2.append(c_env.Actions.drop_in)
            c_env = execute_plan(plan2)
            
            plan.append(plan1 + plan2)
            i += 1

        assert len(plan) == 3
        return plan[0] + plan[1] + plan[2]


from hacl.envs.mini_behavior.mini_behavior.grid import is_obj
from hacl.envs.mini_behavior.mini_bddl.actions import ACTION_FUNC_MAPPING

def worker_offline(args, domain, env, action_filter):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, goal = obs['state'], obs['mission']
    data_gen = OfflineDataGenerator(1)
    plan = data_gen.plan(env)

    if plan is None:
        plan = list()

    states = [state]
    actions = []
    dones = [False]
    succ = False

    structure_mode = getattr(args, 'structure_mode', 'basic')
    for action in plan:
        if structure_mode in ('basic', 'robokin', 'abskin'):
            pddl_action = domain.operators[action_to_operator[action.name]]('r')
        elif structure_mode == 'full' or structure_mode == 'abl':
            if action.name == 'pickup_0':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                target = fwd_cell[0][1]
                assert target is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', target.name)
            elif action.name == 'pickup_1':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                target = fwd_cell[1][1]
                assert target is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', target.name)
            elif action.name == 'pickup_2':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                target = fwd_cell[2][1]
                assert target is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', target.name)
            elif action.name == 'drop_0':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                target = fwd_cell[0][1]
                # assert target is None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', list(env.env.carrying)[0].name)
            elif action.name == 'drop_1':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                target = fwd_cell[1][1]
                assert target is None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', list(env.env.carrying)[0].name)
            elif action.name == 'drop_2':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                target = fwd_cell[2][1]
                assert target is None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', list(env.env.carrying)[0].name)
            elif action.name == 'drop_in':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                for j in range(3):
                    if fwd_cell[j][0] is not None:
                        target = fwd_cell[j][0]
                assert target is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', target.name)
            elif action.name == 'toggle':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                target = None
                action_class = ACTION_FUNC_MAPPING[action.name]
                action_done = False

                for dim in fwd_cell:
                    for obj in dim:
                        if is_obj(obj) and action_class(env.env).can(obj):
                            target = obj
                            action_done = True
                            break
                    if action_done:
                        break
                assert target is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', target.name)
            elif action.name == 'open':
                fwd_pos = env.env.front_pos
                fwd_cell = env.env.grid.get(*fwd_pos)
                action_class = ACTION_FUNC_MAPPING[action.name]
                action_done = False

                for dim in fwd_cell:
                    for obj in dim:
                        if is_obj(obj) and action_class(env.env).can(obj):
                            target = obj
                            action_done = True
                            break
                    if action_done:
                        break
                assert target is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', target.name)
            else:
                pddl_action = domain.operators[action_to_operator[action.name]]('r')

        else:
            raise ValueError('Unknown structure mode: {}.'.format(structure_mode))

        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        obs, reward, (done, score), _ = env.step(rl_action)

        states.append(obs['state'])
        actions.append(pddl_action)
        dones.append(done)

        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    extra_monitors['time/generate'] = time.time() - end
    data = (states, actions, dones, goal, succ, extra_monitors)
    return data

def worker_bc(args, domain, env):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, goal = obs['state'], obs['mission']
    data_gen = OfflineDataGenerator(1)
    plan = data_gen.plan(env)

    if plan is None:
        plan = list()

    states = []
    actions = []
    dones = [False]
    succ = False

    for action in plan:
        pddl_action = domain.operators[action_to_operator[action.name]]('r')
        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        states.append(obs['state'])
        actions.append(pddl_action)
        
        obs, reward, (done, score), _ = env.step(rl_action)

        dones.append(done)

        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    extra_monitors['time/generate'] = time.time() - end
    data = (states, actions, dones, goal, succ, extra_monitors)
    return data

def worker_eval(args, domain, env):
    data_gen = OfflineDataGenerator(1)
    plan = data_gen.plan(env)

    if plan is None:
        return False
        
    succ = False

    for action in plan:
        pddl_action = domain.operators[action_to_operator[action.name]]('r')

        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        obs, reward, (done, score), _ = env.step(rl_action)

        if done:
            succ = True
            break
    return succ