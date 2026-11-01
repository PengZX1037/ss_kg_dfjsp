import json
import random
from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy
from params import configs

def is_pareto_dominant(sol1_mk, sol1_td, sol2_mk, sol2_td):
    return sol1_mk <= sol2_mk and sol1_td < sol2_td or (sol1_mk < sol2_mk and sol1_td <= sol2_td)

def sample_action(p):
    policy_distribution = Categorical(p)
    selected_action = policy_distribution.sample()
    return (selected_action, policy_distribution.log_prob(selected_action))

def eval_actions(p, actions):
    policy_distribution = Categorical(p.squeeze())
    action_log_probs = policy_distribution.log_prob(actions).reshape(-1)
    entropy = policy_distribution.entropy().mean()
    return (action_log_probs, entropy)

def greedy_select_action(p):
    _, index = torch.max(p, dim=1)
    return index

def min_element_index(array):
    min_element = np.min(array)
    matching_indices = np.where(array == min_element)
    return matching_indices

def max_element_index(array):
    max_element = np.max(array)
    matching_indices = np.where(array == max_element)
    return matching_indices

def available_mch_list_for_job(chosen_job, env):
    machine_compatibility = ~env.candidate_process_relation[0, chosen_job]
    available_machines = np.where(machine_compatibility == True)[0]
    mch_free_time = env.mch_free_time[0][available_machines]
    job_free_time = env.candidate_free_time[0][chosen_job]
    if (job_free_time < mch_free_time).all():
        selected_machines = available_machines[min_element_index(mch_free_time)]
    else:
        selected_machines = available_machines[np.where(mch_free_time <= job_free_time)]
    return selected_machines

def heuristic_select_action(method, env):
    chosen_job = -1
    chosen_mch = -1
    job_state = env.mask[0] == 0
    ready_job_mask = env.candidate_free_time[0] <= env.next_schedule_time[0]
    job_state = ready_job_mask & job_state
    eligible_jobs = np.where(job_state == True)[0]
    eligible_operations = env.candidate[0][eligible_jobs]
    if method == 'FIFO':
        candidate_free_time = env.candidate_free_time[0][eligible_jobs]
        job_options = eligible_jobs[min_element_index(candidate_free_time)]
        chosen_job = np.random.choice(job_options)
        mch_state = ~env.candidate_process_relation[0, chosen_job]
        available_mchs = np.where(mch_state == True)[0]
        mch_free_time = env.mch_free_time[0][available_mchs]
        machine_options = available_mchs[min_element_index(mch_free_time)]
        chosen_mch = np.random.choice(machine_options)
    elif method == 'MOR':
        remain_ops = env.op_match_job_left_op_nums[0][eligible_operations]
        job_options = eligible_jobs[max_element_index(remain_ops)]
        chosen_job = np.random.choice(job_options)
        machine_options = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(machine_options)
    elif method == 'SPT':
        masked_processing_times = copy.deepcopy(env.candidate_pt[0])
        masked_processing_times[env.dynamic_pair_mask[0]] = float('inf')
        flattened_processing_times = masked_processing_times.reshape(-1)
        action_options = np.where(flattened_processing_times == np.min(flattened_processing_times))[0]
        action = np.random.choice(action_options)
        return action
    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][eligible_operations]
        chosen_job = eligible_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]
        machine_options = available_mch_list_for_job(chosen_job, env)
        chosen_mch = np.random.choice(machine_options)
    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()
    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()
    action = chosen_job * env.number_of_machines + chosen_mch
    return action

def save_default_params(config):
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)

def nonzero_averaging(x):
    feature_sum = x.sum(dim=-2)
    nonzero_indicators = torch.count_nonzero(x, dim=-1)
    valid_node_counts = (nonzero_indicators != 0).sum(dim=-1, keepdim=True)
    normalization_weights = 1 / valid_node_counts
    normalization_weights[valid_node_counts == 0] = 0
    return torch.mul(normalization_weights, feature_sum)

def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def find_min_not_zero(lst):
    nonzero_items = [num for num in lst if num != 0]
    return min(nonzero_items) if nonzero_items else None

def _normalize_pt(op_pt_matrix):
    pt_lower = 0
    pt_upper = np.max(configs.high)
    pt_range = pt_upper - pt_lower
    if pt_range < 1e-08:
        op_pt_norm = np.zeros_like(op_pt_matrix)
    else:
        op_pt_norm = (op_pt_matrix - pt_lower) / (pt_range + 1e-08)
        op_pt_norm[op_pt_matrix <= 0] = 0
    return (op_pt_norm, pt_lower, pt_range)
