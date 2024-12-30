import time
import torch
import random
import hacl.pdsketch as pds

from copy import deepcopy
from hacl.envs.gridworld.minigrid.gym_minigrid.path_finding import find_path_to_obj

__all__ = ['worker_inst', 'worker_put', 'worker_unlock', 'worker_dt_gen']

from hacl.p.kfac.minigrid.data_generator import OfflineDataGenerator

action_to_operator = {'left': 'lturn', 'right': 'rturn', 'forward': 'forward', 'pickup': 'pickup','drop': 'place', 'toggle': 'toggle', 'toggle_tool': 'toggle-tool'}

def worker_inst(args, domain, env):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, goal = obs['state'], obs['mission']
    goal_obj = env.goal_obj
    typ = goal_obj.type
    color = goal_obj.color
    goal = [goal]
    filt_expr = [f"(foreach (?o - item)(and(is-{typ} ?o)(is-{color} ?o)))"]
    data_gen = OfflineDataGenerator(1)
    plan = data_gen.plan(env)

    if plan is None:
        plan = list()

    states = [[]]
    actions = [[]]
    dones = [False]
    succ = False
    
    all_states = []
    all_actions = []
    for action in plan:
        pddl_action = domain.operators[action_to_operator[action.name]]('r')
        
        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        state = obs['state']
        all_states.append(state)
        all_actions.append(pddl_action)
        if len(all_actions) >= 2:
            if pddl_action.name in ["forward", "lturn", "rturn"]:
                states[0].append(all_states[-2:])
                actions[0].append(all_actions[-2:])
        
        obs, reward, done, _ = env.step(rl_action)
        dones.append(done)
        
        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    extra_monitors['time/generate'] = time.time() - end
    data = (states, actions, dones, goal, succ, filt_expr)
    return data


def worker_put(args, domain, env):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, _ = obs['state'], obs['mission']
    
    goal = ['(exists (?o - item) (and (is-ball ?o)))',
            '(exists (?o - item) (and (is-box ?o)))']
    filt_expr = ['(foreach (?o - item) (and (is-ball ?o)))',
            '(foreach (?o - item) (and (is-box ?o)))']
    data_gen = OfflineDataGenerator(1)
    plan = data_gen.plan(env)

    if plan is None:
        plan = list()

    states = [[], []]
    actions = [[], []]
    dones = [False]
    succ = False
    
    all_states = []
    all_actions = []
    for action in plan:
        pddl_action = domain.operators[action_to_operator[action.name]]('r')
        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        state = obs['state']
        
        all_states.append(state)
        all_actions.append(pddl_action)
        if len(all_actions) >= 2:
            if domain.forward_expr(state, [], domain.parse("(hands-free r)")).tensor:
                states[0].append(all_states[-2:])
                actions[0].append(all_actions[-2:])
            else:
                states[1].append(all_states[-2:])
                actions[1].append(all_actions[-2:])

        obs, reward, done, _ = env.step(rl_action)
        dones.append(done)
        
        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    extra_monitors['time/generate'] = time.time() - end
    data = (states, actions, dones, goal, succ, filt_expr)
    return data

def worker_unlock(args, domain, env):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, _ = obs['state'], obs['mission']
    color = env.goal_obj.color
    goal = [f'(exists (?o - item) (and (is-key ?o) (is-{color} ?o)))',
            f'(exists (?o - item) (and (is-door ?o) (is-{color} ?o)))']
    filt_expr = [f'(foreach (?o - item) (and (is-key ?o) (is-{color} ?o)))',
                 f'(foreach (?o - item) (and (is-door ?o) (is-{color} ?o)))']
    
    data_gen = OfflineDataGenerator(1)
    plan = data_gen.plan(env)

    if plan is None:
        plan = list()

    states = [[], []]
    actions = [[], []]
    dones = [False]
    succ = False
    all_states = []
    all_actions = []
    for action in plan:
        pddl_action = domain.operators[action_to_operator[action.name]]('r')
        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        
        state = obs['state']
        all_states.append(state)
        all_actions.append(pddl_action)
        if len(all_actions) >= 2:
            if domain.forward_expr(state, [], domain.parse("(hands-free r)")).tensor:
                states[0].append(all_states[-2:])
                actions[0].append(all_actions[-2:])
            else:
                states[1].append(all_states[-2:])
                actions[1].append(all_actions[-2:])

        obs, reward, done, _ = env.step(rl_action)
        dones.append(done)
        
        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    extra_monitors['time/generate'] = time.time() - end
    data = (states, actions, dones, goal, succ, filt_expr)
    return data

def worker_dt_gen(args, domain, env):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, _ = obs['state'], obs['mission']
    color = env.goal_obj.color
    typ = env.goal_obj.type
    if env.sub_task == "pickup":
        goal = f'(exists (?o - item) (and (is-{typ} ?o) (is-{color} ?o)))'
        filt_expr = f'(foreach (?o - item) (and (is-{typ} ?o) (is-{color} ?o)))'
    elif env.sub_task == "open":
        goal = f'(exists (?o - item) (and (is-door ?o) (is-{color} ?o)))'
        filt_expr = f'(foreach (?o - item) (and (is-door ?o) (is-{color} ?o)))'
    
    data_gen = OfflineDataGenerator(1)
    plan = data_gen.plan(env)

    if plan is None:
        plan = list()

    states = []
    actions = []
    dones = [False]
    succ = False
    
    all_states = []
    all_actions = []
    for action in plan:
        pddl_action = domain.operators[action_to_operator[action.name]]('r')
        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        state = obs['state']
        all_states.append(state)
        all_actions.append(pddl_action)
        if len(all_actions) >= 2:
            states.append(all_states[-2:])
            actions.append(all_actions[-2:])

        obs, reward, done, _ = env.step(rl_action)
        dones.append(done)
        
        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    extra_monitors['time/generate'] = time.time() - end
    data = (states, actions, dones, goal, succ, filt_expr)
    return data
