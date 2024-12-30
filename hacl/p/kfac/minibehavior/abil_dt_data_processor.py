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
    def __init__(self, succ_prob, traj_len = 2):
        self.succ_prob = 1
        self.traj_len = traj_len
        
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
        elif args.task == 'CollectMisplacedItems':
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
        elif args.task == 'SortingBooks':
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

        filt_expr = ["(foreach (?o - item) (is-printer ?o))",
                     "(foreach (?o - item) (is-table ?o))"]

        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = [[], []]
        actions = [[], []]
        all_states = []
        all_actions = []
        dones = [False]
        succ = False

        for action in plan:
            pddl_action = domain.operators[action_to_operator[action.name]]('r')
            rl_action = pds.rl.RLEnvAction(pddl_action.name)
            state = obs['state']
            all_actions.append(pddl_action)
            all_states.append(state)

            if len(all_actions) >= self.traj_len:
                if domain.forward_expr(state, [], domain.parse("(hands-free r)")).tensor:
                    states[0].append(all_states[-self.traj_len:])
                    actions[0].append(all_actions[-self.traj_len:])
                else:
                    states[1].append(all_states[-self.traj_len:])
                    actions[1].append(all_actions[-self.traj_len:])

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

        filt_expr = ["(foreach (?o - item) (and(is-package ?o)))"]
        states = [[]]
        actions = [[]]
        all_states = []
        all_actions = []
        dones = [False]
        succ = False
        
        for i, action in enumerate(plan):
            pddl_action = domain.operators[action_to_operator[action.name]]('r')
            rl_action = pds.rl.RLEnvAction(pddl_action.name)
            state = obs['state']

            all_actions.append(pddl_action)
            all_states.append(state)

            if len(all_actions) >= self.traj_len:
                states[0].append(all_states[-self.traj_len:])
                actions[0].append(all_actions[-self.traj_len:])

            obs, reward, (done, score), _ = env.step(rl_action)
            dones.append(done)
            if done:
                succ = True
                break
                
        dones = torch.tensor(dones, dtype=torch.int64)

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
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

        his_states = []
        his_actions = []
        for action in plan:
            pddl_action = domain.operators[action_to_operator[action.name]]('r')

            rl_action = pds.rl.RLEnvAction(pddl_action.name)
            state = obs['state']
            his_states.append(state)
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                states.append(his_states[-self.traj_len:])
                actions.append(his_actions[-self.traj_len:])
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
        all_states = []
        all_actions = []
        dones = [False]
        succ = False

        if plan is not None and len(plan) > 0:
            for action in plan:
                pddl_action = domain.operators[action_to_operator[action.name]]('r')
                rl_action = pds.rl.RLEnvAction(pddl_action.name)
                state = obs['state']
                all_actions.append(pddl_action)
                all_states.append(state)

                if len(all_actions) >= self.traj_len:

                    states[0].append(all_states[-self.traj_len:])
                    actions[0].append(all_actions[-self.traj_len:])

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
        
        states = self.get_seq_list(4)
        actions = self.get_seq_list(4)
        filt_expr = ["(foreach (?o - item)(and(is-book ?o) (not(exists (?t - item)(and(is-shelf ?t)(ontop ?o ?t)))) ))",
                "(foreach (?o - item)(or(is-book ?o)(is-shelf ?o)) )"]
        
        filt_expr = ["(foreach (?o - item)(or(is-book ?o)))",
                "(foreach (?o - item)(or(is-book ?o)(is-shelf ?o)) )"]
        
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):            
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
        
    def worker_throw(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(3)
        actions = self.get_seq_list(3)
        filt_expr = ["(foreach (?o - item) (or(is-countertop ?o)(and(is-hamburger ?o) (not(exists (?t - item)(and(is-ashcan ?t )(inside ?o ?t)))))))", 
                "(foreach (?o - item) (or(is-hamburger ?o)(is-ashcan ?o)))"]
        filt_expr = ["(foreach (?o - item) (or(is-countertop ?o)(is-hamburger ?o)))", 
                "(foreach (?o - item) (or(is-hamburger ?o)(is-ashcan ?o)))"]
        
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_throw1(self, args, domain, env):
        extra_monitors = dict()
        end = time.time()

        obs = env.reset()
        state, goal = obs['state'], obs['mission']

        filt_expr = "(foreach(?o - item) (is-hamburger ?o))"
        data_gen = OfflineDataGenerator(self.succ_prob)
        plan = data_gen.plan(env)

        if plan is None:
            plan = list()

        states = []
        actions = []
        dones = [False]
        succ = False
        his_states = []
        his_actions = []

        for action in plan:
            pddl_action = domain.operators[action_to_operator[action.name]]('r')
            rl_action = pds.rl.RLEnvAction(pddl_action.name)                
            state = obs['state']
            his_states.append(state)
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                states.append(his_states[-self.traj_len:])
                actions.append(his_actions[-self.traj_len:])

            obs, reward, (done, score), _ = env.step(rl_action)
            dones.append(done)

            if done:
                succ = True
                break
        dones = torch.tensor(dones, dtype=torch.int64)

        extra_monitors['time/generate'] = time.time() - end
        data = (states, actions, dones, filt_expr, succ, extra_monitors)
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
        
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data

    def worker_collect(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(1)
        actions = self.get_seq_list(1)
        filt_expr = ["(foreach (?o - item)(or(is-collect ?o) (is-furniture ?o) ))", 
                "(foreach (?o - item)(or(is-collect ?o) (is-table ?o) ))"]
        
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
        
    def worker_put(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(4)
        actions = self.get_seq_list(4)

        filt_expr = ["(foreach (?o - item)(or(is-countertop ?o) (is-plate ?o)))",
                "(foreach (?t - item) (or(is-cabinet ?t)(is-plate ?t) ) )"]
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_watering(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(3)
        actions = self.get_seq_list(3)

        filt_expr = ["(foreach (?o - item)(or (is-plant ?o) ))",
                     "(foreach (?o - item)(or (is-plant ?o)(is-sink ?o) ))"]
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
        
    def worker_MakingTea(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(3)
        actions = self.get_seq_list(3)

        filt_expr = ["(foreach (?o - item)(or(is-teapot ?o) (is-teabag ?o) (is-cabinet ?o) ) )",
                      "(foreach (?o - item)(or(is-teapot ?o) (is-teabag ?o) (is-stove ?o) ) )"]
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            if action.name == "toggle":
                continue
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_organize(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(3)
        actions = self.get_seq_list(3)

        filt_expr = ["(foreach (?o - item) (or(is-collect ?o) (is-furniture ?o)))",
                "(foreach (?o - item) (or(is-collect ?o) (is-furniture ?o)))"]
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):

            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[pick_seq][1].append(his_states[-self.traj_len:])
                    actions[pick_seq][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
            
    def worker_CleaningACar(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(exists(?o - item)(and (is-car ?o) (is-dusty ?o) ))")).tensor
        
        states = [[], []]
        actions = [[], []]
        filt_expr = ["(foreach (?o - item) (or(is-tool ?o)(is-car ?o)))",
                "(foreach (?o - item) (or(is-tool ?o)(is-bucket ?o)))"]
        
        his_states = []
        his_actions = []
        
        for i, action in enumerate(all_actions):
            
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] >= 0.9:
                    states[0].append(his_states[-self.traj_len:])
                    actions[0].append(his_actions[-self.traj_len:])
                else:
                    states[1].append(his_states[-self.traj_len:])
                    actions[1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_CleaningShoes(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(hands-free r)")).tensor
        
        states = self.get_seq_list(5)
        actions = self.get_seq_list(5)
        filt_expr = ["(foreach (?o - item)(or(and(is-collect ?o) (not(exists(?t -item) (and (is-sink ?t)(atSameLocation ?o ?t)) )) ) ) )",
                "(foreach (?o - item)(or(is-collect ?o) (is-sink ?o) ))"]
        filt_expr = ["(foreach (?o - item)(is-collect ?o ))",
                "(foreach (?o - item)(or(is-collect ?o) (is-sink ?o) ))"]
        his_states = []
        his_actions = []
        
        pick_seq = 0
        for i, action in enumerate(all_actions):
            
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] == 1:
                    states[pick_seq][0].append(his_states[-self.traj_len:])
                    actions[pick_seq][0].append(his_actions[-self.traj_len:])
                else:
                    states[0][1].append(his_states[-self.traj_len:])
                    actions[0][1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data
    
    def worker_washing(self, args, domain, env):
        all_states, all_actions, dones, goal, succ, extra_monitors = worker_bc(args, domain, env)
        if not succ:
            return all_states, all_actions, dones, goal, succ, extra_monitors
        
        batched_states = BatchState.from_states(domain, all_states)
        pred = domain.forward_expr(batched_states, {}, domain.parse("(exists(?o - item)(and (is-pan ?o) (is-dusty ?o) ))")).tensor
        
        states = [[], []]
        actions = [[], []]
        filt_expr = ["(foreach (?o - item) (or(is-pan ?o)(is-brush ?o)(is-sink ?o)))",
                "(foreach (?o - item) (or(is-pan ?o)(is-cabinet ?o)))"]
        
        his_states = []
        his_actions = []
        
        for i, action in enumerate(all_actions):
            
            his_states.append(all_states[i])
            his_actions.append(action)
            if len(his_actions) >= self.traj_len:
                if pred[i] >= 0.9:
                    states[0].append(his_states[-self.traj_len:])
                    actions[0].append(his_actions[-self.traj_len:])
                else:
                    states[1].append(his_states[-self.traj_len:])
                    actions[1].append(his_actions[-self.traj_len:])

        data = (states, actions, dones, filt_expr, succ, extra_monitors)
        return data