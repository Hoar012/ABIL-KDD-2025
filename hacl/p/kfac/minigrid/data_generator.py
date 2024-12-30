import time
import torch
import random
import hacl.pdsketch as pds

from copy import deepcopy
from hacl.envs.gridworld.minigrid.gym_minigrid.path_finding import find_path_to_obj

__all__ = ['MGState', 'OfflineDataGenerator', 'worker_offline', 'worker_search']


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
    
action_to_operator = {'left': 'lturn', 'right': 'rturn', 'forward': 'forward', 'pickup': 'pickup','drop': 'place', 'toggle': 'toggle', 'toggle_tool': 'toggle-tool'}

class OfflineDataGenerator(object):
    def __init__(self, succ_prob):
        self.succ_prob = succ_prob

    def plan(self, env):
        succ_flag = random.random() < self.succ_prob

        if env.task in ('goto', 'goto2', 'gotosingle', 'gotodoor'):
            return self.plan_goto(env, succ_flag)
        elif env.task == 'pickup':
            return self.plan_pickup(env, succ_flag)
        elif env.task == "open":
            return self.plan_open(env, succ_flag)
        elif env.task == "unlock":
            return self.plan_unlock(env, succ_flag)
        elif env.task == 'put':
            return self.plan_put(env, succ_flag)
        elif env.task == "generalization":
            return self.plan_gen(env, succ_flag)
        else:
            raise NotImplementedError('Unknown task: {}.'.format(self.task))

    def plan_goto(self, env, succ_flag=True):
        goal_pos = (-1, -1)

        if succ_flag:
            goal_obj = env.goal_obj
            for x, y, obj in env.iter_objects():
                if obj.color == goal_obj.color and obj.type == goal_obj.type:
                    goal_pos = (x, y)
                    break
        else:
            objects = list()
            for x, y, obj in env.iter_objects():
                if obj.type != 'wall':
                    objects.append(((x, y), obj))
            goal_pos, goal_obj = random.choice(objects)


        plan = find_path_to_obj(env, goal_pos)
        if plan is None:
            return None

        if env.task == 'goto2':
            plan.extend([env.Actions.forward, env.Actions.left, env.Actions.forward, env.Actions.left, env.Actions.forward])

        # if not succ_flag:
        #     for i in range(5):
        #         plan.extend([random.choice([env.Actions.forward, env.Actions.left, env.Actions.right])])

        return plan

    def plan_pickup(self, env, succ_flag=True):
        plan = self.plan_goto(env,succ_flag)
        if plan is None:
            return None
        
        plan.append(env.Actions.pickup)
        return plan

    def plan_open(self, env, succ_flag=True):
        plan = self.plan_goto(env, succ_flag)
        if plan is None:
            return None

        plan.append(env.Actions.toggle)
        return plan

    def plan_unlock(self, env, succ_flag=True):
        c_env = deepcopy(env)
        tool_pos = (-1, -1)

        if succ_flag:
            goal_tool = env.goal_tool
            for x, y, obj in env.iter_objects():
                if obj.color == goal_tool.color and obj.type == goal_tool.type:
                    tool_pos = (x, y)
                    break
        else:
            objects = list()
            for x, y, obj in env.iter_objects():
                if obj.type != 'wall' and obj.type != 'door':
                    objects.append(((x, y), obj))
            tool_pos, goal_obj = random.choice(objects)

        plan1 = find_path_to_obj(env, tool_pos)
        if plan1 is None:
            return None
        
        plan1.append(env.Actions.pickup)
        c_env = execute_plan(c_env, plan1)
        
        goal_obj = env.goal_obj
        for x, y, obj in env.iter_objects():
            if obj.color == goal_obj.color and obj.type == goal_obj.type:
                goal_pos = (x, y)
                break

        plan2 = find_path_to_obj(c_env, goal_pos)
        plan2.append(env.Actions.toggle_tool)

        return plan1 + plan2
    
    def plan_put(self, env, succ_flag=True):
        c_env = deepcopy(env)
        
        if random.random() < 0.5:
            goal1_pos = env.goal1_pos
            goal2_pos = env.goal2_pos
        else:
            goal1_pos = env.goal2_pos
            goal2_pos = env.goal1_pos
        
        plan1 = find_path_to_obj(env, goal1_pos)
        if plan1 is None:
            return None
        
        plan1.append(env.Actions.pickup)  # pickup obj1
        c_env = execute_plan(c_env, plan1)

        all_pos = list()
        pos = (goal2_pos[0],goal2_pos[1]+1)
        if env.grid.get(*pos) is None:
            all_pos.append(pos)
        pos = (goal2_pos[0],goal2_pos[1]-1)
        if env.grid.get(*pos) is None:
            all_pos.append(pos)
        pos = (goal2_pos[0]+1,goal2_pos[1])
        if env.grid.get(*pos) is None:
            all_pos.append(pos)
        pos = (goal2_pos[0]-1,goal2_pos[1])
        if env.grid.get(*pos) is None:
            all_pos.append(pos)
        if succ_flag:
            goal_pos = random.choice(all_pos)
        else: 
            goal_pos = goal1_pos
        plan2 = find_path_to_obj(c_env, goal_pos)
        
        if plan2 is not None:
            plan2.append(env.Actions.drop)
            return plan1 + plan2
        
        return plan1
    
    def plan_gen(self, env, succ_flag=True):
        if env.sub_task == "pickup":
            plan = self.plan_goto(env,succ_flag)
            if plan is None:
                return None
            if succ_flag:
                plan.append(env.Actions.pickup)
            else:
                for i in range(5):
                    plan.extend([random.choice([env.Actions.forward, env.Actions.left, env.Actions.right])])
            return plan
        
        elif env.sub_task == "open":
            plan = self.plan_goto(env, succ_flag)
            if plan is None:
                return None

            if succ_flag:
                plan.append(env.Actions.toggle_tool)
            else:
                for i in range(5):
                    plan.extend([random.choice([env.Actions.forward, env.Actions.left, env.Actions.right])])

            return plan
        

def worker_offline(args, domain, env, action_filter):
    extra_monitors = dict()
    end = time.time()

    obs = env.reset()
    state, goal = obs['state'], obs['mission']
    data_gen = OfflineDataGenerator(0.6)
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
        elif structure_mode in ['full', 'abl']:
            if action.name == 'pickup':
                fwd_pos = env.front_pos
                fwd_cell = env.grid.get(*fwd_pos)
                assert fwd_cell is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', fwd_cell.name)
            elif action.name == 'drop':
                fwd_pos = env.front_pos
                fwd_cell = env.grid.get(*fwd_pos)
                assert fwd_cell is None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', env.carrying.name)
            elif action.name == 'toggle':
                fwd_pos = env.front_pos
                fwd_cell = env.grid.get(*fwd_pos)
                assert fwd_cell is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', fwd_cell.name)
            elif action.name == 'toggle_tool':
                fwd_pos = env.front_pos
                fwd_cell = env.grid.get(*fwd_pos)
                assert fwd_cell is not None
                pddl_action = domain.operators[action_to_operator[action.name]]('r', env.carrying.name, fwd_cell.name)
            else:
                pddl_action = domain.operators[action_to_operator[action.name]]('r')
        
        else:
            raise ValueError('Unknown structure mode: {}.'.format(structure_mode))

        rl_action = pds.rl.RLEnvAction(pddl_action.name)
        obs, reward, done, _ = env.step(rl_action)

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

def worker_il(args, domain, env):
    filt_expr = None
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
        obs, reward, done, _ = env.step(rl_action)

        dones.append(done)
        
        if done:
            succ = True
            break
    dones = torch.tensor(dones, dtype=torch.int64)

    data = (states, actions, dones, goal, succ, filt_expr)
    return data

def worker_search(args, domain, env, action_filter):
    extra_monitors = dict()
    end = time.time()
    states, actions, dones, goal, succ = pds.rl.customize_follow_policy(
        domain, env,
        max_episode_length=10, max_expansions=200,
        action_filter=action_filter,
        search_algo=pds.heuristic_search_strips,
        extra_monitors=extra_monitors,
        # search-algo-arguments
        use_tuple_desc=False,
        use_quantized_state=False,
        prob_goal_threshold=0.5,
        strips_heuristic=args.heuristic,
        strips_backward_relevance_analysis=args.relevance_analysis,
    )
    extra_monitors['time/search'] = time.time() - end
    data = (states, actions, dones, goal, succ, extra_monitors)
    return data

