import time
import torch
import random
import hacl.pdsketch as pds
import numpy as np


__all__ = ['OfflineDataGenerator', 'worker_offline']

action_to_operator = {'left': 'lturn', 'right': 'rturn', 'forward': 'forward', 'pickup_0': 'pickup_0', 'pickup_1': 'pickup_1', 'pickup_2': 'pickup_2', 'drop_0': 'drop_0', 'drop_1': 'drop_1', 'drop_2': 'drop_2', 'toggle': 'toggle', 'open': 'open', 'close':'close', 'slice':'slice', 'cook': 'cook', 'drop_in': 'drop_in'}

from hacl.p.kfac.minibehavior.data_generator import OfflineDataGenerator, worker_bc
from hacl.pdsketch.interface.v2.state import State, BatchState

class worker_abil(object):
    def __init__(self, succ_prob):
        self.succ_prob = 1
        
    def get_seq_list(self, seq_length):
        L = []
        for i in range(seq_length):
            L.append([[], []])
        return L
    
    def gen_data(self, args, domain, env):
        if args.task == 'CleaningACar':
            return self.worker_CleaningACar(args, domain, env)
        elif args.task == 'CleaningShoes':
            return self.worker_CleaningShoes(args, domain, env)
        elif args.task in ['CollectMisplacedItems', 'CollectMisplacedItems-multi']:
            return self.worker_collect(args, domain, env)
        elif args.task in ['install-a-printer', 'install-a-printer-multi']:
            return self.worker_install(args, domain, env)
        elif args.task == 'LayingWoodFloors':
            return self.worker_laying(args, domain, env)
        elif args.task == 'MakingTea':
            return self.worker_MakingTea(args, domain, env)
        elif args.task == "MovingBoxesToStorage":
            return self.worker_move(args, domain, env)
        elif args.task in ["opening_packages"]:
            return self.worker_open(args, domain, env)
        elif args.task in ["opening_packages1"]:
            return self.worker_open1(args, domain, env)
        elif args.task == "OrganizingFileCabinet":
            return self.worker_organize(args, domain, env)
        elif args.task == 'PuttingAwayDishesAfterCleaning':
            return self.worker_put(args, domain, env)
        elif args.task in ['SortingBooks', 'SortingBooks-multi']:
            return self.worker_sort(args, domain, env)
        elif args.task == 'Throwing_away_leftovers':
            return self.worker_throw(args, domain, env)
        elif args.task == 'Throwing_away_leftovers1':
            return self.worker_throw1(args, domain, env)
        elif args.task == 'Washing_pots_and_pans':
            return self.worker_washing(args, domain, env)
        elif args.task == 'WateringHouseplants':
            return self.worker_watering(args, domain, env)

    def worker_install(self, args, domain, env):
        extra_monitors = dict()
        end = time.time()

        obs = env.reset()
        state, goal = obs['state'], obs['mission']
        
        filt_expr = ["(foreach (?o - item) (or(is-printer ?o)))",
        "(foreach (?o - item) (or(is-table ?o)))"]
        
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = [[], []]
        actions = [[], []]
        dones = [False]
        succ = False

        for action in plan:
            pddl_action = domain.operators[action_to_operator[action.name]]('r')
            rl_action = pds.rl.RLEnvAction(pddl_action.name)
            state = obs['state']
            if pddl_action.name in ["forward", "lturn", "rturn"]:

                if domain.forward_expr(state, [], domain.parse("(hands-free r)")).tensor:
                    # print("go to printer")
                    states[0].append(obs['state'])
                    actions[0].append(pddl_action)
                else:
                    # print("go to table")
                    states[1].append(obs['state'])
                    actions[1].append(pddl_action)

            obs, reward, (done, score), _ = env.step(rl_action)

            dones.append(done)

            if done:
                succ = True
                break
        dones = torch.tensor(dones, dtype=torch.int64)

        extra_monitors['time/generate'] = time.time() - end
        data = (states, actions, dones, filt_expr, succ, extra_monitors)

        return data

    def worker_open(self, args, domain, env):
        extra_monitors = dict()
        end = time.time()

        obs = env.reset()
        state, goal = obs['state'], obs['mission']
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        goal = ["(foreach (?o - item) (and(is-package ?o)))"]
        states = [[]]
        actions = [[]]
        dones = [False]
        succ = False

        for action in plan:
            pddl_action = domain.operators[action_to_operator[action.name]]('r')

            rl_action = pds.rl.RLEnvAction(pddl_action.name)
            if pddl_action.name in ["forward", "lturn", "rturn"]:
                states[0].append(obs['state'])
                actions[0].append(pddl_action)

            obs, reward, (done, score), _ = env.step(rl_action)

            dones.append(done)

            if done:
                succ = True
                break
        dones = torch.tensor(dones, dtype=torch.int64)

        extra_monitors['time/generate'] = time.time() - end
        data = (states, actions, dones, goal, succ, extra_monitors)
        return data

    def worker_open1(self, args, domain, env):
        extra_monitors = dict()
        end = time.time()

        obs = env.reset()
        state, goal = obs['state'], obs['mission']
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        goal = "(foreach (?o - item) (and(is-package ?o)(not(is-open ?o))))"
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
    
    def worker_move(self, args, domain, env):
        extra_monitors = dict()
        end = time.time()

        obs = env.reset()
        state, goal = obs['state'], obs['mission']

        filt_expr = ["(foreach (?o - item)(or (is-carton ?o) (is-door ?o)))"]
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        states = [[]]
        actions = [[]]
        dones = [False]
        succ = False

        if plan is not None and len(plan) > 0:
            for action in plan:
                pddl_action = domain.operators[action_to_operator[action.name]]('r')
                rl_action = pds.rl.RLEnvAction(pddl_action.name)
                state = obs['state']
                
                if pddl_action.name in ["forward", "lturn", "rturn"]:

                    states[0].append(obs['state'])
                    actions[0].append(pddl_action)

                obs, reward, (done, score), _ = env.step(rl_action)
                dones.append(done)

                if done:
                    succ = True
                    break
                    
        dones = torch.tensor(dones, dtype=torch.int64)
        extra_monitors['time/generate'] = time.time() - end
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
        
    def worker_sort(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        filt_expr = ["(foreach (?o - item)(and(is-book ?o) (not(exists (?t - item)(and(is-shelf ?t)(ontop ?o ?t)))) ))",
                "(foreach (?o - item)(or(is-book ?o)(is-shelf ?o)) )"]

        states = self.get_seq_list(4)
        actions = self.get_seq_list(4)

        pick_seq = 0
        for i, action in enumerate(all_actions):
            if i > 0:
                if pred[i - 1] == 0 and pred[i] == 1:
                    pick_seq += 1

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data

    def worker_throw(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        filt_expr = ["(foreach (?o - item) (or(is-countertop ?o)(and(is-hamburger ?o) (not(exists (?t - item)(and(is-ashcan ?t )(inside ?o ?t)))))))", 
                "(foreach (?o - item) (or(is-hamburger ?o)(is-ashcan ?o)))"]
        
        states = self.get_seq_list(3)
        actions = self.get_seq_list(3)
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            if i > 0:
                if pred[i - 1] == 0 and pred[i] == 1:
                    pick_seq += 1

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (all_states, all_actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_throw1(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        filt_expr = "(foreach(?o - item) (or(not (is-hamburger ?o))(and (is-hamburger ?o) (not (exists(?t - item)(and (is-ashcan ?t)(inside ?o ?t)))))))"
        filt_expr = "(foreach(?o - item) (or(is-hamburger ?o)))"
        
        data = (all_states, all_actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_laying(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(1)
        actions = self.get_seq_list(1)
        filt_expr = ["(foreach (?o - item) (and(is-plywood ?o) ) )",
                "(foreach (?o - item) (and(is-plywood ?o) ) )"]
        
        pick_seq = 0
        for i, action in enumerate(all_actions):

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_collect(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor

        filt_expr = ["(foreach (?o - item) (or(is-furniture ?o)(and(is-collect ?o) (not (exists(?t - item) (and (is-table ?t) (ontop ?o ?t)) )) )))",
                "(foreach (?o - item)(or(is-collect ?o) (is-table ?o) ))"]
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = self.get_seq_list(5)
        actions = self.get_seq_list(5)
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            if i > 0:
                if pred[i - 1] == 0 and pred[i] == 1:
                    pick_seq += 1

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_put(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor

        filt_expr = ["(foreach (?o - item)(or(is-countertop ?o) (and(is-plate ?o) (not(exists(?t -item) (and (is-cabinet ?t)(inside ?o ?t))) )) ))",
                "(foreach (?t - item) (or(is-cabinet ?t)(is-plate ?t) ) )"]
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = self.get_seq_list(4)
        actions = self.get_seq_list(4)
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            if i > 0:
                if pred[i - 1] == 0 and pred[i] == 1:
                    pick_seq += 1

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data

    def worker_organize(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        filt_expr = ["(foreach (?o - item) (or(is-collect ?o) (is-furniture ?o)))",
                "(foreach (?o - item) (or(is-collect ?o) (is-furniture ?o)))"]

        states = self.get_seq_list(1)
        actions = self.get_seq_list(1)
        
        pick_seq = 0
        for i, action in enumerate(all_actions):

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_CleaningACar(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(exists(?o - item)(and (is-car ?o) (is-dusty ?o) ))")).tensor
        # print(pred)
        filt_expr = ["(foreach (?o - item) (or(is-tool ?o)(is-car ?o)))",
                "(foreach (?o - item) (or(is-tool ?o)(is-bucket ?o)))"]

        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = self.get_seq_list(1)
        actions = self.get_seq_list(1)
        
        for i, action in enumerate(all_actions):
            
            if pred[i] >= 0.9:
                states[0][0].append(all_states[i])
                actions[0][0].append(action)
            else:
                states[0][1].append(all_states[i])
                actions[0][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_CleaningShoes(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        # print(pred)

        filt_expr = ["(foreach (?o - item)(or(and(is-collect ?o) (not(exists(?t -item) (and (is-sink ?t)(atSameLocation ?o ?t)) )) ) ) )",
                "(foreach (?o - item)(or(is-collect ?o) (is-sink ?o) ))"]
        
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = self.get_seq_list(5)
        actions = self.get_seq_list(5)
        
        pick_seq = 0
        for i, action in enumerate(all_actions[:-1]):
            if i > 0:
                if pred[i - 1] == 0 and pred[i] == 1:
                    pick_seq += 1

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_watering(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor

        filt_expr =  ["(foreach (?o - item)(or (and(is-plant ?o) (not(exists(?t - item) (and (is-sink ?t)(inside ?o ?t)) )) )  ))",
                "(foreach (?o - item)(or (is-plant ?o) (is-sink ?o)))"]
        
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = self.get_seq_list(3)
        actions = self.get_seq_list(3)
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            if i > 0:
                if pred[i - 1] == 0 and pred[i] == 1:
                    pick_seq += 1

            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data

    
    def worker_MakingTea(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor

        filt_expr =  ["(foreach (?o - item)(or(is-teapot ?o) (is-teabag ?o) (is-cabinet ?o) ) )",
                      "(foreach (?o - item)(or(is-teapot ?o) (is-teabag ?o) (is-stove ?o) ) )"]
        
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = self.get_seq_list(2)
        actions = self.get_seq_list(2)
        
        pick_seq = 0
        for i, action in enumerate(all_actions):

            if action.name == "toggle":
                continue
            
            if pred[i] == 1:
                states[pick_seq][0].append(all_states[i])
                actions[pick_seq][0].append(action)
            else:
                states[pick_seq][1].append(all_states[i])
                actions[pick_seq][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_washing(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(exists(?o - item)(and(is-pan ?o)(is-dusty ?o) ))")).tensor

        filt_expr = ["(foreach (?o - item) (or(is-pan ?o)(is-brush ?o)(is-sink ?o)))",
                "(foreach (?o - item) (or(is-pan ?o)(is-cabinet ?o)))"]

        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = self.get_seq_list(1)
        actions = self.get_seq_list(1)
        
        for i, action in enumerate(all_actions):
            
            if pred[i] >= 0.9:
                states[0][0].append(all_states[i])
                actions[0][0].append(action)
            else:
                states[0][1].append(all_states[i])
                actions[0][1].append(action)
            
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data