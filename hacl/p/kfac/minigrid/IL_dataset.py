import jacinle
import torch

from torch.utils.data import Dataset

__all__ = ['init_dataset', 'preprocess']

class ILDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        if "filt_obj_set" in self.data:
            filt_obj_tensors = self.data["filt_obj_set"][index]
            return self.data["robot_set"][index], self.data["object_set"][index], self.data["goal_set"][index], self.data["action_set"][index], filt_obj_tensors
        else:
            return self.data["robot_set"][index], self.data["object_set"][index], self.data["goal_set"][index], self.data["action_set"][index]
        

    def __len__(self):
        return len(self.data["robot_set"])


def init_dataset(args, worker, env, domain, model):
    succ_num = 0
    dataset = []
    
    for i in jacinle.tqdm(range(1, args.iterations + 1)):
        states, actions, dones, goal, succ, filt_expr = worker(args, domain, env)
        if succ:
            dataset.append([states, actions, goal, succ, filt_expr])
        
        if succ:
            succ_num += 1
    
    return dataset, succ_num

def preprocess_bc(dataset, task, domain, model, seq_len = 1):
    robot_set = []
    object_set = []
    goal_set = []
    action_set = []
    
    for states, actions, goal, succ, _ in dataset:
        robot_tensor, object_tensor, goal_tensor, action_tensor = model.process_data(states, actions, goal, succ)
        robot_set.append(robot_tensor)
        object_set.append(object_tensor)
        goal_set.append(goal_tensor)
        action_set.append(action_tensor)
        
    if len(dataset) > 0:
        robot_set = torch.cat(robot_set)
        object_set = torch.cat(object_set)
        goal_set = torch.cat(goal_set)
        action_set = torch.cat(action_set)
        
    processed_dataset = {"robot_set": robot_set, 
                         "object_set": object_set,
                         "goal_set": goal_set,
                         "action_set": action_set
                         }
    print("length of dataset:", len(robot_set))
    
    return ILDataset(processed_dataset)

def preprocess_dt(dataset, task, domain, model, seq_len = 2):
    robot_set = []
    object_set = []
    goal_set = []
    action_set = []
    
    for states, actions, goal, succ, _ in dataset:
        history_states = []
        history_actions = []
        
        for i, action in enumerate(actions):
            history_states.append(states[i])
            history_actions.append(actions[i])
        
        if len(history_states) >= seq_len:
            
            robot_tensor, object_tensor, goal_tensor, action_tensor = model.process_data(history_states[-seq_len:], history_actions[-seq_len:], goal, succ)
            robot_set.append(robot_tensor)
            object_set.append(object_tensor)
            goal_set.append(goal_tensor)
            action_set.append(action_tensor)
        
    processed_dataset = {"robot_set": robot_set, 
                         "object_set": object_set,
                         "goal_set": goal_set,
                         "action_set": action_set
                         }
    print("length of dataset:", len(robot_set))
    
    return ILDataset(processed_dataset)

def preprocess(args, dataset, domain, model):
    if args.model == "bc":
        return preprocess_bc(dataset, args.task, domain, model)
    elif args.model == "dt":
        return preprocess_dt(dataset, args.task, domain, model)
