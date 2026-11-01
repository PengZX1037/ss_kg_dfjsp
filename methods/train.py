import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import heapq
import pandas as pd
import torch
import sys
import time
import random
import csv
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from MetaheuristicAlgorithm.GA.ga_main import get_instance_info
from MetaheuristicAlgorithm.VNS_main import run_vns
from draw_util import plot_gantt_chart
from base_tools import is_pareto_dominant
from params import configs
from data_utils import generate_dynamic_scenario, load_or_create_dynamic_validation, count_dynamic_jobs
from base_tools import setup_seed, sample_action, greedy_select_action, strToSuffix, heuristic_select_action, _normalize_pt
from model.PPO import PPO_initialize, Memory
import numpy as np
import warnings
str_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
device = torch.device(configs.device)

def simulate_with_heuristic(instance_job_length, instance_op_pt, op_pt_norm_matrix, t_event, t_event_norm=0.0):
    n_j = instance_job_length.shape[0]
    n_op, n_m = instance_op_pt.shape
    mch_free_time = np.zeros(n_m, dtype=float)
    mch_free_time_norm = np.zeros(n_m, dtype=float)
    job_next_op_ready_time = np.zeros(n_j, dtype=float)
    job_next_op_ready_time_norm = np.zeros(n_j, dtype=float)
    op_status = np.zeros(n_op, dtype=int)
    op_completion_time = np.full(n_op, float('inf'))
    op_completion_time_norm = np.full(n_op, float('inf'))
    op_machine_assignment = np.full(n_op, -1, dtype=int)
    job_first_op_id = np.zeros(n_j, dtype=int)
    job_last_op_id = np.zeros(n_j, dtype=int)
    operation_job_lookup = np.zeros(n_op, dtype=int)
    current_op = 0
    for j in range(n_j):
        if instance_job_length[j] <= 0:
            job_first_op_id[j] = current_op
            job_last_op_id[j] = current_op - 1
            continue
        job_first_op_id[j] = current_op
        job_last_op_id[j] = current_op + instance_job_length[j] - 1
        map_start = current_op
        map_end = current_op + instance_job_length[j]
        if map_start < n_op and map_end <= n_op:
            operation_job_lookup[map_start:map_end] = j
        elif map_start < n_op:
            warnings.warn(f'Job {j} length {instance_job_length[j]} exceeds op_pt rows {n_op - map_start}. Truncating mapping.')
            operation_job_lookup[map_start:n_op] = j
        current_op += instance_job_length[j]
    next_operation_by_job = np.copy(job_first_op_id)
    next_operation_by_job[instance_job_length <= 0] = -1
    num_scheduled_ops = 0
    machine_compatibility_mask = instance_op_pt > 0
    while num_scheduled_ops < n_op:
        feasible_dispatches = []
        normalized_dispatches = []
        for j in range(n_j):
            next_operation_id = next_operation_by_job[j]
            if next_operation_id < 0 or next_operation_id >= n_op or instance_job_length[j] <= 0:
                continue
            job_ready = job_next_op_ready_time[j]
            job_ready_norm = job_next_op_ready_time_norm[j]
            compatible_machines = np.where(machine_compatibility_mask[next_operation_id, :])[0]
            for m in compatible_machines:
                mch_ready = mch_free_time[m]
                mch_ready_norm = mch_free_time_norm[m]
                start_time = max(job_ready, mch_ready)
                start_time_norm = max(job_ready_norm, mch_ready_norm)
                proc_time = instance_op_pt[next_operation_id, m]
                proc_time_norm = op_pt_norm_matrix[next_operation_id, m]
                feasible_dispatches.append((proc_time, start_time, next_operation_id, m, j))
                normalized_dispatches.append((proc_time_norm, start_time_norm, next_operation_id, m, j))
        if not feasible_dispatches:
            remaining_ops = np.sum(op_status == 0)
        feasible_dispatches.sort()
        normalized_dispatches.sort()
        chosen_dispatch = feasible_dispatches[0]
        chosen_normalized_dispatch = normalized_dispatches[0]
        chosen_pt, chosen_start, chosen_op, chosen_mch, chosen_job = chosen_dispatch
        chosen_pt_norm, chosen_start_norm, chosen_op_norm, chosen_mch_norm, chosen_job_norm = chosen_normalized_dispatch
        if chosen_start >= t_event:
            break
        completion_time = chosen_start + chosen_pt
        completion_time_norm = chosen_start_norm + chosen_pt_norm
        op_status[chosen_op] = 1
        op_completion_time[chosen_op] = completion_time
        op_completion_time_norm[chosen_op] = completion_time_norm
        mch_free_time[chosen_mch] = completion_time
        mch_free_time_norm[chosen_mch] = completion_time_norm
        job_next_op_ready_time[chosen_job] = completion_time
        job_next_op_ready_time_norm[chosen_job] = completion_time_norm
        op_machine_assignment[chosen_op] = chosen_mch
        next_op_for_job = chosen_op + 1
        next_operation_by_job[chosen_job] = -1 if next_op_for_job > job_last_op_id[chosen_job] else next_op_for_job
        num_scheduled_ops += 1
    mch_free_time = [max(a, t_event) for a in mch_free_time]
    mch_free_time_norm = [max(a, t_event_norm) for a in mch_free_time_norm]
    simulation_snapshot = {'current_time': t_event, 'current_time_norm': t_event_norm, 'mch_free_time': mch_free_time, 'mch_free_time_norm': mch_free_time_norm, 'job_next_op_ready_time': job_next_op_ready_time, 'job_next_op_ready_time_norm': job_next_op_ready_time_norm, 'op_status': op_status, 'op_completion_time': op_completion_time, 'op_completion_time_norm': op_completion_time_norm, 'job_length': instance_job_length, 'op_machine_assignment_old': op_machine_assignment}
    return simulation_snapshot

def generate_new_job_data(n_m, current_time, low=1, high=99, seed=None):
    operation_count = random.randint(int(0.5 * n_m), int(1.2 * n_m))
    if operation_count <= 0:
        operation_count = 1
    job_length = np.array([operation_count])
    op_pt = np.random.randint(low=low, high=high + 1, size=(operation_count, n_m))
    op_per_mch_min = 1
    op_per_mch_max = n_m
    machines_per_operation = np.random.randint(low=op_per_mch_min, high=op_per_mch_max + 1, size=operation_count)
    for r in range(operation_count):
        mch_num = int(machines_per_operation[r])
        if mch_num == 0:
            mch_num = 1
        if mch_num < n_m:
            machines_to_disable = max(0, min(n_m - mch_num, n_m))
            if machines_to_disable > 0:
                disabled_machine_indices = np.random.choice(np.arange(n_m), machines_to_disable, replace=False)
                op_pt[r][disabled_machine_indices] = 0
        if not np.any(op_pt[r] > 0):
            op_pt[r, np.random.randint(0, n_m)] = random.randint(low, high)
    avg_pt = np.mean(op_pt[op_pt > 0]) if np.any(op_pt > 0) else low
    due_date_factor = random.uniform(1, 6.0)
    due_date = np.array([current_time + due_date_factor * operation_count / n_m * avg_pt / 2])
    new_job_info = {'job_length': job_length, 'op_pt': op_pt, 'due_date': due_date}
    return new_job_info

def determine_heuristic_window(shop_state, new_job_op_indices, existing_op_pt, time_horizon=50):
    window_operation_ids = list(new_job_op_indices)
    op_status_old = shop_state['op_status']
    job_length_old = shop_state['job_length']
    n_j_old = len(job_length_old)
    n_op_old = len(op_status_old)
    job_first_op_id_old = np.zeros(n_j_old, dtype=int)
    job_last_op_id_old = np.zeros(n_j_old, dtype=int)
    current_op_old = 0
    for j in range(n_j_old):
        if job_length_old[j] <= 0:
            job_first_op_id_old[j] = current_op_old
            job_last_op_id_old[j] = current_op_old - 1
            continue
        job_first_op_id_old[j] = current_op_old
        job_last_op_id_old[j] = current_op_old + job_length_old[j] - 1
        current_op_old += job_length_old[j]
    pending_operation_by_job = np.full(n_j_old, -1, dtype=int)
    for j in range(n_j_old):
        if job_length_old[j] <= 0:
            continue
        first_op = job_first_op_id_old[j]
        last_op = job_last_op_id_old[j]
        start_check = min(first_op, n_op_old)
        end_check = min(last_op, n_op_old)
        if start_check > end_check:
            continue
        job_ops_status_old = op_status_old[start_check:end_check + 1]
        unstarted_indices = np.where(job_ops_status_old == 0)[0]
        if len(unstarted_indices) > 0:
            first_unstarted_idx_rel = unstarted_indices[0]
            pending_operation_by_job[j] = start_check + first_unstarted_idx_rel
    pending_job_ids = []
    for j in range(n_j_old):
        if pending_operation_by_job[j] != -1:
            pending_job_ids.append(j)
    for j_idx in pending_job_ids:
        first_unstarted_op = pending_operation_by_job[j_idx]
        last_op_of_job = job_last_op_id_old[j_idx]
        if 0 <= first_unstarted_op <= last_op_of_job < n_op_old:
            remaining_ops_for_job = np.arange(first_unstarted_op, last_op_of_job + 1)
            window_operation_ids.extend(remaining_ops_for_job.tolist())
        elif first_unstarted_op != -1:
            warnings.warn(f'Window generation: Invalid range for job {j_idx} (first_unstarted={first_unstarted_op}, last={last_op_of_job})')
    unique_window_ops = np.unique(np.array(window_operation_ids, dtype=int))
    return unique_window_ops

class Trainer:
    """Coordinate dynamic scenarios, policy updates, and validation."""

    def __init__(self, config, train_env_instance, validation_env_instance):
        self.n_j = int(config.n_j)
        self.n_m = int(config.n_m)
        self.n_new = count_dynamic_jobs(self.n_j, config.dynamic_job_fraction)
        self.low = config.low
        self.high = config.high
        self.config = config
        self.max_updates = config.max_updates
        self.reset_scenario_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs
        self.waiting_penalty = config.waiting_penalty
        model_name_base = f'{self.n_j}x{self.n_m}'
        mode = 'KG' if config.use_kgrl else 'noKG'
        wp_tag = format(float(config.waiting_penalty), '.3g').replace('.', 'p')
        self.model_name = f'{model_name_base}{strToSuffix(config.model_suffix)}_{mode}_WP{wp_tag}_seed{config.seed_train}'
        ppo1_base_path = f'./model'
        self.trained_network_path = f'{ppo1_base_path}/network'
        self.train_log_path = f'{ppo1_base_path}/log'
        os.makedirs(self.trained_network_path, exist_ok=True)
        os.makedirs(self.train_log_path, exist_ok=True)
        self.model_file = os.path.join(self.trained_network_path, f'{self.model_name}.pth')
        self.run_config_file = os.path.join(self.train_log_path, f'run_config_{self.model_name}.json')
        if (os.path.exists(self.model_file) or os.path.exists(self.run_config_file)) and (not config.cover_train_flag):
            raise FileExistsError(f'Run already exists: {self.model_name}. Use a different suffix, or --cover_train_flag true intentionally.')
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)
        self.scenario_rng = np.random.default_rng(int(self.seed_train) + 1000003)
        self.env = train_env_instance
        self.vali_env = validation_env_instance
        self.use_kgrl = config.use_kgrl
        self.kgrl_p_sel = config.kgrl_p_sel
        self.kgrl_knowledge_source = config.kgrl_knowledge_source
        self.vns_max_iter = getattr(config, 'VNS_MAX_ITER', 1)
        self.vns_k_max = getattr(config, 'VNS_K_MAX', 3)
        self.vns_ls_iter = getattr(config, 'VNS_LS_ITER', 5)
        self.vali_scenarios, self.vali_data_path, self.vali_sha256 = load_or_create_dynamic_validation(config)
        print(f'Loaded {len(self.vali_scenarios)} FIXED dynamic validation scenes from {self.vali_data_path}; SHA256={self.vali_sha256[:12]}')
        print(f'Jobs: {self.n_j} at t=0 + {self.n_new} at t_event>0; actual arriving fraction: {self.n_new / (self.n_j + self.n_new):.2%}')
        ms_refs, tt_refs = ([], [])
        for scene in self.vali_scenarios:
            pt = np.vstack((np.asarray(scene['initial_op_pt']), np.asarray(scene['new_op_pt'])))
            valid_pt = pt[pt > 0]
            avg_pt = float(valid_pt.mean()) if valid_pt.size else 1.0
            ref_ms = max(1.0, avg_pt * len(pt) / self.n_m)
            dds = np.concatenate((scene['initial_due_dates'], scene['new_due_dates']))
            ms_refs.append(ref_ms)
            tt_refs.append(max(1.0, sum((max(0.0, ref_ms - d) for d in dds))))
        self.dataset_ref_makespan = float(np.mean(ms_refs))
        self.dataset_ref_tardiness = float(np.mean(tt_refs))
        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        with open(self.run_config_file, 'w', encoding='utf-8') as fh:
            json.dump({'parameters': vars(config), 'validation_file': self.vali_data_path, 'validation_sha256':
                self.vali_sha256, 'n_initial_jobs': self.n_j, 'n_dynamic_jobs': self.n_new}, fh, indent=2, ensure_ascii=False)

    def _prepare_dynamic_scenario(self, scenario):
        initial_job_length = np.asarray(scenario['initial_job_length'], dtype=np.int32)
        initial_op_pt = np.asarray(scenario['initial_op_pt'], dtype=np.int32)
        initial_due_dates = np.asarray(scenario['initial_due_dates'], dtype=float)
        new_job_length = np.asarray(scenario['new_job_length'], dtype=np.int32)
        new_op_pt = np.asarray(scenario['new_op_pt'], dtype=np.int32)
        new_due_dates = np.asarray(scenario['new_due_dates'], dtype=float)
        t_event = float(scenario['t_event'])
        if t_event <= 0.0 or not np.all(initial_op_pt.sum(axis=1) > 0) or (not np.all(new_op_pt.sum(axis=1) > 0)):
            raise ValueError('Scenario must have positive dynamic event time and feasible operations')
        if (len(initial_job_length) != self.n_j or len(new_job_length) != self.n_new or initial_op_pt.shape[1] != self.n_m or
                (new_op_pt.shape[1] != self.n_m) or (int(initial_job_length.sum()) != len(initial_op_pt)) or
                (int(new_job_length.sum()) != len(new_op_pt)) or (len(initial_due_dates) != self.n_j) or
                (len(new_due_dates) != self.n_new)):
            raise ValueError('Inconsistent initial/arriving job dimensions in dynamic scenario')
        op_pt_norm, pt_low, pt_range = _normalize_pt(initial_op_pt)
        t_event_norm = (t_event - pt_low) / (pt_range + 1e-08)
        initial_shop_snapshot = simulate_with_heuristic(initial_job_length, initial_op_pt, op_pt_norm, t_event, t_event_norm)
        combined_job_lengths = np.concatenate((initial_job_length, new_job_length))
        combined_operation_times = np.vstack((initial_op_pt, new_op_pt))
        combined_due_dates = np.concatenate((initial_due_dates, new_due_dates))
        arriving_operation_ids = np.arange(len(initial_op_pt), len(combined_operation_times), dtype=np.int32)
        rescheduling_window = determine_heuristic_window(initial_shop_snapshot, arriving_operation_ids, initial_op_pt)
        rescheduling_window = rescheduling_window[(rescheduling_window >= 0) & (rescheduling_window < len(combined_operation_times))]
        initial_shop_snapshot.pop('job_length', None)
        initial_shop_snapshot.pop('op_pt', None)
        return (initial_shop_snapshot, rescheduling_window, combined_job_lengths, combined_operation_times, combined_due_dates)

    def validate_reschedule_envs(self, validation_env_instance):
        self.ppo.policy.eval()
        num_vali_instances = len(self.vali_scenarios)
        if num_vali_instances != int(self.config.validation_size):
            raise ValueError('The fixed validation set is incomplete')
        makespan_records = []
        tardiness_records = []
        num_validation_waits = 0
        num_validation_decisions = 0
        validation_batch_size = self.num_envs
        plotted_gantt_this_validation = False
        for i in range(0, num_vali_instances, validation_batch_size):
            scenario_batch_indices = list(range(i, min(i + validation_batch_size, num_vali_instances)))
            current_batch_size = len(scenario_batch_indices)
            scenario_inputs = [self._prepare_dynamic_scenario(self.vali_scenarios[k]) for k in scenario_batch_indices]
            vali_initial_states = [x[0] for x in scenario_inputs]
            vali_windows_W = [x[1] for x in scenario_inputs]
            vali_job_lengths_combined = [x[2] for x in scenario_inputs]
            vali_op_pts_combined = [x[3] for x in scenario_inputs]
            vali_due_dates_combined = [x[4] for x in scenario_inputs]
            try:
                state = validation_env_instance.reset_for_reschedule(vali_initial_states, vali_windows_W, vali_job_lengths_combined, vali_op_pts_combined, vali_due_dates_combined)
            except Exception as e:
                raise RuntimeError('Validation environment reset failed') from e
            current_done_flags = np.zeros(current_batch_size, dtype=bool)
            validation_step = 0
            max_validation_steps = 5 * int(validation_env_instance.max_number_of_ops) + 100
            while not np.all(current_done_flags):
                validation_step += 1
                if validation_step > max_validation_steps:
                    stuck = np.flatnonzero(~current_done_flags)
                    global_stuck = [scenario_batch_indices[e] for e in stuck]
                    raise RuntimeError(f'Validation stuck: {global_stuck}')
                active_envs_mask = ~current_done_flags
                active_environment_ids = np.where(active_envs_mask)[0]
                if len(active_environment_ids) == 0:
                    break
                try:
                    active_policy_inputs = {'fea_j': state.fea_j_tensor[active_envs_mask], 'op_mask': state.op_mask_tensor[active_envs_mask], 'candidate': state.candidate_tensor[active_envs_mask], 'fea_m': state.fea_m_tensor[active_envs_mask], 'mch_mask': state.mch_mask_tensor[active_envs_mask], 'comp_idx': state.comp_idx_tensor[active_envs_mask], 'dynamic_pair_mask': state.dynamic_pair_mask_tensor[active_envs_mask], 'fea_pairs': state.fea_pairs_tensor[active_envs_mask], 'wait_action_valid': state.wait_action_valid_tensor[active_envs_mask]}
                except Exception as e:
                    warnings.warn(f'\nError preparing validation state step {validation_step}: {e}')
                    break
                with torch.no_grad():
                    try:
                        pi_active, _ = self.ppo.policy(**active_policy_inputs)
                    except Exception as e:
                        warnings.warn(f'\nError during validation policy forward step {validation_step}: {e}')
                        break
                selected_active_actions = greedy_select_action(pi_active)
                num_validation_waits += int((selected_active_actions == validation_env_instance.number_of_jobs * self.n_m).sum().item())
                num_validation_decisions += int(len(active_environment_ids))
                batch_actions = np.full(current_batch_size, -1, dtype=int)
                batch_actions[active_environment_ids] = selected_active_actions.cpu().numpy()
                try:
                    state, _, done = validation_env_instance.step(actions=batch_actions)
                    if state is None or state.fea_j_tensor.shape[0] != current_batch_size:
                        raise ValueError('Validation step failed.')
                    current_done_flags = done
                except Exception as e:
                    warnings.warn(f'\nError during validation env step {validation_step}: {e}')
                    break
            if not np.all(current_done_flags):
                raise RuntimeError(f'Validation incomplete for indices {scenario_batch_indices}')
            if validation_env_instance.current_makespan.shape == (current_batch_size,):
                batch_makespan_values = validation_env_instance.current_makespan
            else:
                batch_makespan_values = np.full(current_batch_size, float('inf'))
            batch_tardiness_values = np.zeros(current_batch_size)
            if validation_env_instance.job_due_date is not None and validation_env_instance.job_due_date.shape[0] == current_batch_size and (validation_env_instance.true_op_ct is not None) and (validation_env_instance.true_op_ct.shape[0] == current_batch_size) and (validation_env_instance.op_scheduled_flag is not None) and (validation_env_instance.op_scheduled_flag.shape[0] == current_batch_size):
                try:
                    for e in range(current_batch_size):
                        for j in range(validation_env_instance.env_number_of_jobs[e]):
                            last_op = validation_env_instance.job_last_op_id[e, j]
                            if 0 <= last_op < validation_env_instance.max_number_of_ops:
                                if validation_env_instance.op_scheduled_flag[e, last_op] > 0:
                                    comp_time = validation_env_instance.true_op_ct[e, last_op]
                                    due_date = validation_env_instance.job_due_date[e, j]
                                    if np.isfinite(comp_time) and np.isfinite(due_date):
                                        batch_tardiness_values[e] += max(0, comp_time - due_date)
                except Exception as e:
                    warnings.warn(f'Error calculating validation tardiness: {e}')
                    batch_tardiness_values.fill(float('inf'))
            else:
                batch_tardiness_values.fill(float('inf'))
            if i == 0 and (not plotted_gantt_this_validation):
                try:
                    env_to_plot_idx = 0
                    op_status_plot = validation_env_instance.op_scheduled_flag[env_to_plot_idx]
                    true_op_ct_plot = validation_env_instance.true_op_ct[env_to_plot_idx]
                    true_op_pt_plot = validation_env_instance.true_op_pt[env_to_plot_idx]
                    op_to_job_map_plot = validation_env_instance.op_to_job_map[env_to_plot_idx]
                    op_machine_assignment_plot = validation_env_instance.op_machine_assignment[env_to_plot_idx]
                    env_num_ops_plot = validation_env_instance.env_number_of_ops[env_to_plot_idx]
                    num_machines_plot = validation_env_instance.number_of_machines
                    gantt_filename = os.path.join(self.train_log_path, f"gantt_validation_update{(self.ppo.update_step_count if hasattr(self.ppo, 'update_step_count') else '_latest')}_inst{scenario_batch_indices[env_to_plot_idx]}.png")
                    gantt_title = f"Gantt Chart - Validation After Update {(self.ppo.update_step_count if hasattr(self.ppo, 'update_step_count') else 'N/A')} (Instance {scenario_batch_indices[env_to_plot_idx]})"
                    plot_gantt_chart(env_idx=scenario_batch_indices[env_to_plot_idx], op_status=op_status_plot, true_op_ct=true_op_ct_plot, true_op_pt=true_op_pt_plot, op_to_job_map=op_to_job_map_plot, op_machine_assignment=op_machine_assignment_plot, env_num_ops=env_num_ops_plot, num_machines=num_machines_plot, filename=gantt_filename, title=gantt_title)
                    plotted_gantt_this_validation = True
                except Exception as e:
                    warnings.warn(f'\nError plotting Gantt chart for validation instance {scenario_batch_indices[env_to_plot_idx]}: {e}')
                    import traceback
                    traceback.print_exc()
            makespan_records.extend(batch_makespan_values)
            tardiness_records.extend(batch_tardiness_values)
        self.ppo.policy.train()
        if not makespan_records:
            return (float('inf'), float('inf'))
        valid_makespans = np.array(makespan_records)
        valid_tardiness = np.array(tardiness_records)
        finite_mask = np.isfinite(valid_makespans) & np.isfinite(valid_tardiness)
        if not np.all(finite_mask) or len(valid_makespans) != num_vali_instances:
            raise RuntimeError('Missing/invalid fixed validation results; refusing partial mean')
        avg_makespan = np.mean(valid_makespans[finite_mask])
        avg_tardiness = np.mean(valid_tardiness[finite_mask])
        self.last_validation_wait = (num_validation_waits, num_validation_decisions)
        return (avg_makespan, avg_tardiness)

    def generate_reschedule_scenario(self, num_scenarios):
        if num_scenarios <= 0:
            raise ValueError('num_scenarios must be positive')
        batch = [self._prepare_dynamic_scenario(generate_dynamic_scenario(self.config, self.scenario_rng)) for _ in range(num_scenarios)]
        return ([x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch], [x[4] for x in batch])

    def sample_training_instances(self, num_instances):
        from data_utils import generate_shared_fjsp_instance, generate_shared_due_dates
        jobs, pts, dds = ([], [], [])
        for _ in range(num_instances):
            jl, pt = generate_shared_fjsp_instance(self.config, self.n_j, self.scenario_rng)
            dd = generate_shared_due_dates(jl, pt, self.n_m, 0.0, self.scenario_rng)
            jobs.append(jl)
            pts.append(pt)
            dds.append(dd)
        return (jobs, pts, dds)

    def generate_knowledge_solution(self, ppo_solution, ppo_makespan: float, ppo_tardiness: float, initial_state_info, instance_info_combined, combined_due_dates: np.ndarray):
        job_ready_times = np.asarray(initial_state_info['job_next_op_ready_time'], dtype=float)
        n_arrivals = len(combined_due_dates) - len(job_ready_times)
        if n_arrivals < 1:
            raise ValueError('KG VNS expects at least one arriving job')
        job_ready_times = np.concatenate((job_ready_times, np.full(n_arrivals, float(initial_state_info['current_time']), dtype=float)))
        vns_sol, vns_mk, vns_td, target_schedule_log, target_schedule_gantt_data = run_vns(initial_sol=ppo_solution, initial_sol_makespan=ppo_makespan,
                                                                                           initial_sol_tardiness=ppo_tardiness, instance_info=instance_info_combined,
                                                                                           initial_mch_available_times=initial_state_info['mch_free_time'], job_ready_times=job_ready_times,
                                                                                           job_due_dates=combined_due_dates, max_iter_vns=self.vns_max_iter,
                                                                                           k_max=self.vns_k_max, max_iter_local_search=self.vns_ls_iter)
        return (vns_sol, vns_mk, vns_td, target_schedule_log, target_schedule_gantt_data)

    def simulate_target_actions(self, target_schedule_log, target_schedule_gantt_data, initial_state_info,
                                window_W: np.ndarray, combined_len: np.ndarray, combined_pt: np.ndarray, combined_dd: np.ndarray):
        states_k = []
        target_actions_k = []
        temp_env = deepcopy(self.env)
        try:
            current_state = temp_env.reset_for_reschedule(initial_shop_states=[initial_state_info], reschedule_windows_W=[window_W],
                                                          job_lengths=[combined_len], op_pts=[combined_pt], job_due_dates=[combined_dd])
            target_schedule_log = sorted(target_schedule_log, key=lambda x: (x['start_time'], x['end_time']))
            num_machines = temp_env.number_of_machines
            step = 0
            max_replay_steps = 5 * len(window_W) + 100
            while True:
                if step >= max_replay_steps:
                    warnings.warn('KGRL replay failed to complete within the safety bound')
                    return ([], [])
                candidate_j_id = [i for i, t in enumerate(temp_env.candidate_free_time[0]) if t <= temp_env.next_schedule_time]
                temp_scheduled_op_info_list = []
                for j in candidate_j_id:
                    scheduled_op_info = next((d for d in target_schedule_log if d['job_id'] == j), None)
                    if scheduled_op_info == None:
                        continue
                    if temp_env.mch_free_time[0][scheduled_op_info['mch_id']] <= temp_env.next_schedule_time:
                        temp_scheduled_op_info_list.append(scheduled_op_info)
                scheduled_op_info = None
                if len(temp_scheduled_op_info_list) != 0:
                    scheduled_op_info = min(temp_scheduled_op_info_list, key=lambda x: x['start_time'])
                if scheduled_op_info == None:
                    target_action_index = temp_env.number_of_jobs * num_machines
                elif scheduled_op_info['start_time'] != target_schedule_log[0]['start_time']:
                    target_action_index = temp_env.number_of_jobs * num_machines
                else:
                    op_id = scheduled_op_info['op_id']
                    mch_id = scheduled_op_info['mch_id']
                    job_id = temp_env.op_to_job_map[0, op_id]
                    target_action_index = job_id * num_machines + mch_id
                    current_dynamic_mask = current_state.dynamic_pair_mask_tensor[0]
                    if job_id < current_dynamic_mask.shape[0] and mch_id < current_dynamic_mask.shape[1]:
                        if current_dynamic_mask[job_id, mch_id].item():
                            warnings.warn(f'KGRL Sim: Target action ({job_id},{mch_id}) for Op {op_id} is invalid. Skipping incomplete imitation trajectory.')
                            return ([], [])
                    target_schedule_log.remove(scheduled_op_info)
                if target_action_index == temp_env.number_of_jobs * num_machines and (not bool(current_state.wait_action_valid_tensor[0].item())):
                    warnings.warn('KGRL replay requires an infeasible WAIT; discard this imitation trajectory')
                    return ([], [])
                states_k.append(deepcopy(current_state))
                target_actions_k.append(torch.tensor(target_action_index, dtype=torch.long))
                current_state, _, done = temp_env.step(np.array([target_action_index]))
                if done[0]:
                    break
                step += 1
        except Exception as e:
            warnings.warn(f'Error during target action simulation with VNS solution: {e}')
            import traceback
            traceback.print_exc()
            return ([], [])
        return (states_k, target_actions_k)

    def _validation_compromise_score(self, makespan, tardiness):
        if not (np.isfinite(makespan) and np.isfinite(tardiness)):
            return float('inf')
        if self.val_ref_makespan is None:
            self.val_ref_makespan = max(abs(float(makespan)), 1.0)
        if self.val_ref_tardiness is None:
            self.val_ref_tardiness = max(abs(float(tardiness)), 1.0)
        norm_makespan = float(makespan) / self.val_ref_makespan
        norm_tardiness = float(tardiness) / self.val_ref_tardiness
        score = self.val_weight_makespan * norm_makespan + self.val_weight_tardiness * norm_tardiness
        return score

    def _should_save_validation_model(self, makespan, tardiness):
        if not (np.isfinite(makespan) and np.isfinite(tardiness)):
            return (False, float('inf'), 'invalid validation metrics')
        current_score = self._validation_compromise_score(makespan, tardiness)
        if not (np.isfinite(self.best_val_makespan) and np.isfinite(self.best_val_tardiness)):
            return (True, current_score, 'first valid checkpoint')
        current_dominates = is_pareto_dominant(makespan, tardiness, self.best_val_makespan, self.best_val_tardiness)
        best_dominates = is_pareto_dominant(self.best_val_makespan, self.best_val_tardiness, makespan, tardiness)
        if current_dominates:
            return (True, current_score, 'Pareto improvement')
        if best_dominates:
            return (False, current_score, 'Pareto dominated by saved checkpoint')
        should_save = current_score < self.best_val_score - 1e-08
        if should_save:
            reason = 'better balanced trade-off'
        else:
            reason = 'worse balanced trade-off'
        return (should_save, current_score, reason)

    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.episode_performance_log = []
        self.reward_log = []
        self.validation_log = []
        self.validation_tardiness_log = []
        self.guidance_log = []
        self.training_wait_log = []
        self.validation_wait_log = []
        self.val_weight_makespan = 0.5
        self.val_weight_tardiness = 0.5
        self.val_ref_makespan = self.dataset_ref_makespan
        self.val_ref_tardiness = self.dataset_ref_tardiness
        self.best_val_makespan = float('inf')
        self.best_val_tardiness = float('inf')
        self.best_val_score = float('inf')
        print(f'model name :{self.model_name}')
        self.train_st = time.time()
        for update_index in tqdm(range(self.max_updates), file=sys.stdout, desc='progress', colour='blue'):
            ep_st = time.time()
            initial_states, windows_W, combined_job_lengths, combined_op_pts, combined_due_dates = self.generate_reschedule_scenario(self.num_envs)
            batch_size = len(initial_states)
            baseline_inputs = {'initial_shop_states': deepcopy(initial_states), 'reschedule_windows_W': windows_W, 'job_lengths': combined_job_lengths, 'op_pts': combined_op_pts, 'job_due_dates': combined_due_dates}
            state = self.env.reset_for_reschedule(initial_shop_states=initial_states, reschedule_windows_W=windows_W, job_lengths=combined_job_lengths, op_pts=combined_op_pts, job_due_dates=combined_due_dates)
            current_done_flags = np.zeros(batch_size, dtype=bool)
            self.memory.clear_memory()
            episode_step = 0
            max_episode_steps = 5 * int(self.env.max_number_of_ops) + 100
            num_train_waits = 0
            num_train_decisions = 0
            ep_rewards = 0.0
            while not np.all(current_done_flags):
                episode_step += 1
                if episode_step > max_episode_steps:
                    raise RuntimeError(f'Training episode stuck at iteration {update_index + 1}: {episode_step} steps')
                active_envs_mask = ~current_done_flags
                active_environment_ids = np.where(active_envs_mask)[0]
                if len(active_environment_ids) == 0:
                    break
                try:
                    active_state_tensors = {'fea_j': state.fea_j_tensor[active_envs_mask], 'op_mask': state.op_mask_tensor[active_envs_mask], 'candidate': state.candidate_tensor[active_envs_mask], 'fea_m': state.fea_m_tensor[active_envs_mask], 'mch_mask': state.mch_mask_tensor[active_envs_mask], 'comp_idx': state.comp_idx_tensor[active_envs_mask], 'dynamic_pair_mask': state.dynamic_pair_mask_tensor[active_envs_mask], 'fea_pairs': state.fea_pairs_tensor[active_envs_mask], 'wait_action_valid': state.wait_action_valid_tensor[active_envs_mask]}
                except (IndexError, AttributeError, ValueError) as e:
                    warnings.warn(f'\nError preparing state tensors for policy step {episode_step}: {e}')
                    raise RuntimeError('Training state tensor preparation failed') from e
                with torch.no_grad():
                    try:
                        pi_active, vals_active = self.ppo.policy_old(**active_state_tensors)
                    except Exception as e:
                        raise RuntimeError('Training policy forward failed') from e
                action_active, action_logprob_active = sample_action(pi_active)
                num_train_waits += int((action_active == self.env.number_of_jobs * self.n_m).sum().item())
                num_train_decisions += int(len(active_environment_ids))
                batch_actions = np.full(batch_size, -1, dtype=int)
                batch_actions[active_environment_ids] = action_active.cpu().numpy()
                try:
                    state_to_push = deepcopy(state)
                    self.memory.push(state_to_push)
                except Exception as e:
                    raise RuntimeError('Cannot store training trajectory') from e
                try:
                    state, step_reward, done = self.env.step(actions=batch_actions)
                    if state is None or state.fea_j_tensor.shape[0] != batch_size:
                        raise ValueError(f'env.step returned invalid state')
                except Exception as e:
                    warnings.warn(f'\nError during env.step step {episode_step}: {e}')
                    import traceback
                    raise RuntimeError('Training environment step failed') from e
                ep_rewards += step_reward
                current_done_flags = done
                reward_full = torch.zeros(batch_size, device=device)
                if step_reward.shape == (batch_size,):
                    reward_full[active_environment_ids] = torch.from_numpy(step_reward[active_environment_ids]).float().to(device)
                action_full_tensor = torch.full((batch_size,), -1, dtype=torch.long, device=device)
                action_full_tensor[active_environment_ids] = action_active
                logprob_full = torch.full((batch_size,), -float('inf'), device=device)
                logprob_full[active_environment_ids] = action_logprob_active
                vals_full = torch.full((batch_size, 1), 0.0, device=device)
                if vals_active.shape[0] == len(active_environment_ids):
                    vals_full[active_environment_ids] = vals_active
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward_full)
                self.memory.action_seq.append(action_full_tensor)
                self.memory.log_probs.append(logprob_full)
                self.memory.val_seq.append(vals_full.squeeze(1))
                if np.all(current_done_flags):
                    break
            if not np.all(current_done_flags):
                raise RuntimeError(f'Incomplete training episode at iteration {update_index + 1}')
            final_makespans_actual = np.zeros(batch_size)
            final_tardiness_actual = np.zeros(batch_size)
            calculation_successful = True
            try:
                final_true_op_ct = self.env.true_op_ct
                final_job_due_date = self.env.job_due_date
                final_job_last_op_id = self.env.job_last_op_id
                final_env_num_jobs = self.env.env_number_of_jobs
                final_max_N = self.env.max_number_of_ops
                for e in range(batch_size):
                    makespan_e = 0.0
                    tardiness_e = 0.0
                    args_e = baseline_inputs
                    window_W_e = args_e['reschedule_windows_W'][e]
                    jobs_in_W_e = set()
                    if len(window_W_e) > 0:
                        op_map_e = self.env.op_to_job_map[e]
                        valid_w_ops_e = window_W_e[window_W_e < final_max_N]
                        job_indices_for_w_e = op_map_e[valid_w_ops_e]
                        jobs_in_W_e = set(job_indices_for_w_e[job_indices_for_w_e != -1].tolist())
                    makespan_e = self.env.current_makespan[e]
                    for j in jobs_in_W_e:
                        if j < final_env_num_jobs[e]:
                            last_op_j = final_job_last_op_id[e, j]
                            if 0 <= last_op_j < final_max_N:
                                comp_time = final_true_op_ct[e, last_op_j]
                                due_date = final_job_due_date[e, j]
                                tardiness_e += max(0, comp_time - due_date)
                    final_makespans_actual[e] = makespan_e
                    final_tardiness_actual[e] = tardiness_e
            except Exception as e:
                warnings.warn(f'\nError during final performance/baseline calculation: {e}')
                import traceback
                traceback.print_exc()
                calculation_successful = False
            if calculation_successful:
                mean_final_reward = np.mean(ep_rewards)
                self.reward_log.append([update_index, mean_final_reward])
            ppo_solutions_batch = [None] * batch_size
            ppo_makespans_batch = np.full(batch_size, float('inf'))
            ppo_tardiness_batch = np.full(batch_size, float('inf'))
            for e in range(batch_size):
                ppo_makespans_batch[e] = final_makespans_actual[e]
                ppo_tardiness_batch[e] = final_tardiness_actual[e]
                os_ppo1, ms_ppo1, sorted_window_indices = self.reconstruct_ppo1_schedule_for_window(e)
                ppo_solutions_batch[e] = (os_ppo1, ms_ppo1, sorted_window_indices)
            use_guided_update = False
            vns_improvements = 0
            vns_start = time.perf_counter()
            if self.use_kgrl:
                vns_improvements = 0
                knowledge_solutions_for_imitation = [None] * batch_size
                knowledge_solutions_for_gantt_data = [None] * batch_size
                for e in range(batch_size):
                    args_e = baseline_inputs
                    ppo_sol_e = ppo_solutions_batch[e]
                    ppo_mk_e = ppo_makespans_batch[e]
                    ppo_td_e = ppo_tardiness_batch[e]
                    vns_sol, vns_mk, vns_td, target_schedule_log, target_schedule_gantt_data = self.generate_knowledge_solution(ppo_sol_e, ppo_mk_e, ppo_td_e, args_e['initial_shop_states'][e], get_instance_info(args_e['job_lengths'][e], args_e['op_pts'][e]), args_e['job_due_dates'][e])
                    if is_pareto_dominant(vns_mk, vns_td, ppo_mk_e, ppo_td_e):
                        vns_improvements += 1
                        knowledge_solutions_for_imitation[e] = target_schedule_log
                        knowledge_solutions_for_gantt_data[e] = target_schedule_gantt_data
                    else:
                        knowledge_solutions_for_imitation[e] = None
                        knowledge_solutions_for_gantt_data[e] = None
                if vns_improvements >= batch_size and random.random() < self.kgrl_p_sel:
                    use_guided_update = True
                    print(f'\nUpdate {update_index}: Triggering Knowledge Guidance (VNS dominated PPO, p_sel satisfied)')
            vns_elapsed = time.perf_counter() - vns_start if self.use_kgrl else 0.0
            guidance_applied = False
            loss, v_loss = (0.0, 0.0)
            if use_guided_update:
                all_states_k = []
                all_target_actions_k = []
                simulation_success_batch = True
                for e in range(batch_size):
                    if knowledge_solutions_for_imitation[e] is None:
                        simulation_success_batch = False
                        break
                    args_e = baseline_inputs
                    target_schedule_log = knowledge_solutions_for_imitation[e]
                    target_schedule_gantt_data = knowledge_solutions_for_gantt_data[e]
                    states_e, targets_e = self.simulate_target_actions(target_schedule_log, target_schedule_gantt_data, args_e['initial_shop_states'][e], args_e['reschedule_windows_W'][e], args_e['job_lengths'][e], args_e['op_pts'][e], args_e['job_due_dates'][e])
                    if not states_e:
                        simulation_success_batch = False
                        break
                    all_states_k.extend(states_e)
                    all_target_actions_k.extend(targets_e)
                cloning_loss = None
                if simulation_success_batch and all_states_k:
                    cloning_loss = self.ppo.update_knowledge_guided(all_states_k, all_target_actions_k)
                if cloning_loss is not None and np.isfinite(cloning_loss) and (cloning_loss > 0):
                    print(f'  KG Update executed. Imitation Loss: {cloning_loss:.4f}')
                    self.ppo.policy_old.load_state_dict(self.ppo.policy.state_dict())
                    loss = cloning_loss
                    v_loss = 0.0
                    guidance_applied = True
                elif len(self.memory.action_seq) > 0:
                    loss, v_loss = self.ppo.update(self.memory)
            elif len(self.memory.action_seq) > 0:
                loss, v_loss = self.ppo.update(self.memory)
            self.guidance_log.append([update_index + 1, vns_improvements, batch_size, int(use_guided_update), int(guidance_applied), round(vns_elapsed, 6)])
            mean_final_makespan = np.mean(final_makespans_actual[np.isfinite(final_makespans_actual)]) if np.any(np.isfinite(final_makespans_actual)) else -1.0
            mean_final_tardiness = np.mean(final_tardiness_actual[np.isfinite(final_tardiness_actual)]) if np.any(np.isfinite(final_tardiness_actual)) else -1.0
            self.episode_performance_log.append([update_index, mean_final_makespan, mean_final_tardiness])
            self.training_wait_log.append([update_index + 1, num_train_waits, num_train_decisions])
            ep_et = time.time()
            tqdm.write(f'Update {update_index + 1}\t Makespan: {mean_final_makespan:.2f}\t Tardiness: {mean_final_tardiness:.2f}\tLoss: {loss:.4f}\t V_Loss: {v_loss:.4f}\t Time: {ep_et - ep_st:.2f}s')
            if (update_index + 1) % self.validate_timestep == 0:
                if True:
                    try:
                        vali_perf_makespan, vali_perf_tardiness = self.validate_reschedule_envs(self.vali_env)
                        if np.isfinite(vali_perf_makespan):
                            if vali_perf_makespan < self.record_makespan and vali_perf_tardiness < self.record_tradiness:
                                self.save_model()
                                self.record_makespan = vali_perf_makespan
                                self.record_tradiness = vali_perf_tardiness
                                tqdm.write(f'---> New best validation performance: {self.record_makespan:.4f} <---')
                            self.validation_log.append(vali_perf_makespan)
                            self.validation_tardiness_log.append(vali_perf_tardiness)
                            self.save_validation_log()
                            tqdm.write(f'Validation makespan performance: {vali_perf_makespan:.4f} (Best: {self.record:.4f})')
                            tqdm.write(f'Validation tardiness performance: {vali_perf_tardiness:.4f}')
                        else:
                            tqdm.write('Validation returned non-finite value.')
                    except Exception as e:
                        warnings.warn(f'\nError during validation at update {update_index}: {e}')
                        import traceback
                        traceback.print_exc()
                        tqdm.write('Validation failed.')
                else:
                    tqdm.write('Skipping validation - validation data not available.')
        self.train_et = time.time()
        if not os.path.isfile(self.model_file):
            raise RuntimeError('No validated checkpoint was saved. Ensure that max_updates is at least validate_timestep and validation completed.')
        self.save_training_logs()
        print(f'Training finished. Total time: {self.train_et - self.train_st:.2f}s')
        self.plot_training_results()

    def reconstruct_ppo1_schedule_for_window(self, current_batch_env_idx: int=0):
        op_pt_ppo = self.env.op_pt[current_batch_env_idx]
        op_machine_assignment_ppo = self.env.op_machine_assignment[current_batch_env_idx]
        os = np.array(self.env.os[current_batch_env_idx], dtype=int)
        ms_list = np.array(op_machine_assignment_ppo, dtype=int)
        sch_op_idx = np.array(self.env.sch_op_idx[current_batch_env_idx], dtype=int)
        sort_sch_op_idx = sorted(sch_op_idx)
        ms = []
        for idx in sort_sch_op_idx:
            avail_mac = [i for i, t in enumerate(op_pt_ppo[idx]) if t != 0]
            ms.append(avail_mac.index(ms_list[idx]))
        return (os, ms, sort_sch_op_idx)

    def calculate_baseline_performance(self, initial_shop_state, window_W, combined_job_length, combined_op_pt, combined_due_dates, heuristic='SPT'):
        n_j_total = combined_job_length.shape[0]
        n_op_total, n_m = combined_op_pt.shape
        mch_free_time = np.copy(initial_shop_state['mch_free_time'])
        op_status_sim = np.zeros(n_op_total, dtype=int)
        op_completion_time_sim = np.full(n_op_total, float('inf'))
        job_next_op_ready_time_sim = np.full(n_j_total, float('inf'))
        op_to_job_map_sim = np.full(n_op_total, -1, dtype=int)
        job_first_op_id_sim = np.zeros(n_j_total, dtype=int)
        job_last_op_id_sim = np.full(n_j_total, -1, dtype=int)
        current_op_idx_map = 0
        for j in range(n_j_total):
            job_len = combined_job_length[j]
            if job_len > 0:
                job_first_op_id_sim[j] = current_op_idx_map
                job_last_op_id_sim[j] = current_op_idx_map + job_len - 1
                map_end = min(current_op_idx_map + job_len, n_op_total)
                op_to_job_map_sim[current_op_idx_map:map_end] = j
                current_op_idx_map += job_len
        op_status_old = initial_shop_state['op_status']
        op_comp_time_old = initial_shop_state['op_completion_time']
        job_ready_time_old = initial_shop_state['job_next_op_ready_time']
        n_op_old = len(op_status_old)
        n_j_old = len(job_ready_time_old)
        op_status_sim[:n_op_old] = op_status_old
        op_completion_time_sim[:n_op_old][op_status_old == 1] = op_comp_time_old[op_status_old == 1]
        candidate_op_idx_sim = np.full(n_j_total, -1, dtype=int)
        for j in range(n_j_total):
            first_op = job_first_op_id_sim[j]
            last_op = job_last_op_id_sim[j]
            if first_op > last_op:
                continue
            first_unstarted_op = -1
            for op_idx in range(first_op, last_op + 1):
                if op_idx < n_op_total and op_status_sim[op_idx] == 0:
                    first_unstarted_op = op_idx
                    break
            candidate_op_idx_sim[j] = first_unstarted_op
            if j < n_j_old:
                job_next_op_ready_time_sim[j] = job_ready_time_old[j]
            else:
                job_next_op_ready_time_sim[j] = initial_shop_state['current_time']
        ops_in_W_set = set(window_W.tolist())
        num_W_ops_total = len(ops_in_W_set)
        num_W_ops_scheduled = 0
        process_relation_sim = combined_op_pt > 0
        ready_queue = []

        def check_and_add_to_queue(job_idx):
            op_idx = candidate_op_idx_sim[job_idx]
            if op_idx != -1 and op_idx in ops_in_W_set and (op_status_sim[op_idx] == 0):
                is_ready = True
                if op_idx > job_first_op_id_sim[job_idx]:
                    if op_idx - 1 < n_op_total and op_status_sim[op_idx - 1] == 0:
                        is_ready = False
                if is_ready:
                    compatible_machines = np.where(process_relation_sim[op_idx, :])[0]
                    if len(compatible_machines) > 0:
                        valid_pts = combined_op_pt[op_idx, compatible_machines]
                        min_pt = np.min(valid_pts)
                        heapq.heappush(ready_queue, (min_pt, op_idx, job_idx))
        for j in range(n_j_total):
            check_and_add_to_queue(j)
        current_time_sim = initial_shop_state['current_time']
        while num_W_ops_scheduled < num_W_ops_total and ready_queue:
            spt_pt, spt_op, spt_job = heapq.heappop(ready_queue)
            if op_status_sim[spt_op] != 0:
                continue
            compatible_machines = np.where(process_relation_sim[spt_op, :])[0]
            best_mch = -1
            earliest_finish_time = float('inf')
            op_ready_time = job_next_op_ready_time_sim[spt_job]
            for m in compatible_machines:
                mch_ready_time = mch_free_time[m]
                start_time = max(op_ready_time, mch_ready_time)
                proc_time = combined_op_pt[spt_op, m]
                finish_time = start_time + proc_time
                if finish_time < earliest_finish_time:
                    earliest_finish_time = finish_time
                    best_mch = m
            if best_mch != -1:
                op_status_sim[spt_op] = 1
                op_completion_time_sim[spt_op] = earliest_finish_time
                mch_free_time[best_mch] = earliest_finish_time
                job_next_op_ready_time_sim[spt_job] = earliest_finish_time
                num_W_ops_scheduled += 1
                current_time_sim = max(current_time_sim, earliest_finish_time)
                next_op_for_job = spt_op + 1
                last_op_for_job = job_last_op_id_sim[spt_job]
                if next_op_for_job <= last_op_for_job:
                    candidate_op_idx_sim[spt_job] = next_op_for_job
                    check_and_add_to_queue(spt_job)
                else:
                    candidate_op_idx_sim[spt_job] = -1
            else:
                warnings.warn(f'Baseline Calc: Could not find machine for SPT op {spt_op}')
        baseline_makespan = 0.0
        baseline_tardiness = 0.0
        jobs_in_W = set()
        if len(window_W) > 0:
            valid_w_ops = window_W[window_W < n_op_total]
            job_indices_for_w = op_to_job_map_sim[valid_w_ops]
            jobs_in_W = set(job_indices_for_w[job_indices_for_w != -1].tolist())
        if num_W_ops_scheduled > 0:
            scheduled_W_ops_indices = [op for op in window_W if op < n_op_total and op_status_sim[op] == 1]
            if scheduled_W_ops_indices:
                baseline_makespan = np.max(op_completion_time_sim[scheduled_W_ops_indices])
            else:
                baseline_makespan = initial_shop_state['current_time']
        else:
            baseline_makespan = initial_shop_state['current_time']
        for j in jobs_in_W:
            last_op_j = job_last_op_id_sim[j]
            if 0 <= last_op_j < n_op_total:
                job_completion_time = op_completion_time_sim[last_op_j]
                if np.isfinite(job_completion_time):
                    due_date_j = combined_due_dates[j]
                    if np.isfinite(due_date_j):
                        tardiness_j = max(0, job_completion_time - due_date_j)
                        baseline_tardiness += tardiness_j
        return (baseline_makespan, baseline_tardiness)

    def save_training_logs(self):
        perf_log_path = os.path.join(self.train_log_path, f'episode_perf_{self.model_name}.txt')
        try:
            with open(perf_log_path, 'w') as f:
                f.write('Update,AvgMakespan,AvgTardiness\n')
                for i, ms, td in self.episode_performance_log:
                    f.write(f'{i},{ms:.4f},{td:.4f}\n')
        except Exception as e:
            warnings.warn(f'Error saving episode performance log: {e}')
        self.save_validation_log()
        reward_path = os.path.join(self.train_log_path, f'reward_perf_{self.model_name}.txt')
        with open(reward_path, 'w') as fh:
            fh.write('Iteration,AvgReward\n')
            for iteration, reward in self.reward_log:
                fh.write(f'{iteration + 1},{reward:.6f}\n')
        for name, rows in [('train_waiting', getattr(self, 'training_wait_log', [])), ('validation_waiting', getattr(self, 'validation_wait_log', []))]:
            path = os.path.join(self.train_log_path, f'{name}_{self.model_name}.csv')
            with open(path, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(['Iteration', 'WaitingActions', 'TotalDecisions'])
                writer.writerows(rows)
        if hasattr(self, 'guidance_log'):
            knowledge_path = os.path.join(self.train_log_path, f'knowledge_stats_{self.model_name}.csv')
            with open(knowledge_path, 'w', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow(['Iteration', 'VNSDominated', 'BatchSize', 'GuidanceSelected', 'BCExecuted', 'VNSTimeSec'])
                writer.writerows(self.guidance_log)
        time_log_path = os.path.join(self.train_log_path, 'training_time.txt')
        model_full_path = self.model_file
        try:
            with open(time_log_path, 'a') as f:
                f.write(f'model path: {model_full_path}\t\ttraining time: {round(self.train_et - self.train_st, 2)}\t\t local time: {str_time}\n')
        except Exception as e:
            warnings.warn(f'Error saving training time log: {e}')

    def save_validation_log(self):
        val_log_path = os.path.join(self.train_log_path, f'validation_perf_{self.model_name}.txt')
        val_log_tardiness_path = os.path.join(self.train_log_path, f'validation_tardiness_perf_{self.model_name}.txt')
        try:
            with open(val_log_path, 'w') as f:
                f.write('ValidationStep,PerformanceMetric\n')
                for i, perf in enumerate(self.validation_log):
                    f.write(f'{(i + 1) * self.validate_timestep},{perf:.4f}\n')
        except Exception as e:
            warnings.warn(f'Error saving validation log: {e}')
        try:
            with open(val_log_tardiness_path, 'w') as f:
                f.write('ValidationStep,PerformanceMetric\n')
                for i, perf in enumerate(self.validation_tardiness_log):
                    f.write(f'{(i + 1) * self.validate_timestep},{perf:.4f}\n')
        except Exception as e:
            warnings.warn(f'Error saving validation log: {e}')

    def save_model(self):
        model_save_path = self.model_file
        try:
            torch.save(self.ppo.policy.state_dict(), model_save_path)
        except Exception as e:
            raise RuntimeError(f'Failed to save checkpoint {model_save_path}') from e

    def load_model(self):
        model_load_path = self.model_file
        if os.path.exists(model_load_path):
            try:
                state_dict = torch.load(model_load_path, map_location=device)
                self.ppo.policy.load_state_dict(state_dict)
                self.ppo.policy_old.load_state_dict(state_dict)
                print(f'--- Model loaded from {model_load_path} ---')
            except Exception as e:
                warnings.warn(f'Error loading model from {model_load_path}: {e}')
        else:
            print(f'Model file not found at {model_load_path}. Starting from scratch')

    def plot_training_results(self):
        if not self.episode_performance_log:
            return
        if not self.validation_log:
            return
        log_dir = self.train_log_path
        try:
            perf_data = np.array(self.episode_performance_log)
            updates = perf_data[:, 0]
            avg_makespans = perf_data[:, 1]
            avg_tardiness = perf_data[:, 2]
            perf_df = pd.DataFrame({'Update': updates, 'AvgMakespan': avg_makespans, 'AvgTardiness': avg_tardiness})
        except Exception as e:
            warnings.warn(f'Error processing episode performance log: {e}. Skipping performance plot.')
            perf_df = None
        try:
            validation_updates = np.arange(1, len(self.validation_log) + 1) * self.validate_timestep
            validation_scores = np.array(self.validation_log)
            validation_tardiness_updates = np.arange(1, len(self.validation_tardiness_log) + 1) * self.validate_timestep
            validation_tardiness_scores = np.array(self.validation_tardiness_log)
            vali_df = pd.DataFrame({'Update': validation_updates, 'ValidationMetric': validation_scores})
            vali_tardiness_df = pd.DataFrame({'Update': validation_tardiness_updates, 'ValidationMetric': validation_tardiness_scores})
        except Exception as e:
            warnings.warn(f'Error processing validation log: {e}. Skipping validation plot.')
            vali_df = None
            vali_tardiness_df = None
        reward_df = None
        if self.reward_log:
            try:
                reward_data = np.array(self.reward_log)
                reward_df = pd.DataFrame({'Update': reward_data[:, 0], 'AvgFinalReward': reward_data[:, 1]})
            except Exception as e:
                warnings.warn(f'Error processing reward log: {e}. Skipping reward plot.')
        if perf_df is not None:
            try:
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                color1 = 'tab:red'
                ax1.set_xlabel('Training iterations')
                ax1.set_ylabel('Average Makespan', color=color1)
                ax1.plot(perf_df['Update'], perf_df['AvgMakespan'], color=color1, label='Avg Makespan')
                ax1.tick_params(axis='y', labelcolor=color1)
                ax1.legend(loc='upper left')
                ax2 = ax1.twinx()
                color2 = 'tab:blue'
                ax2.set_ylabel('Average Tardiness', color=color2)
                ax2.plot(perf_df['Update'], perf_df['AvgTardiness'], color=color2, label='Avg Tardiness')
                ax2.tick_params(axis='y', labelcolor=color2)
                ax2.legend(loc='upper right')
                plt.title(f'Training Performance ({self.model_name})')
                fig1.tight_layout()
                plot_save_path = os.path.join(log_dir, f'training_performance_{self.model_name}.png')
                plt.savefig(plot_save_path)
                plt.close(fig1)
            except Exception as e:
                warnings.warn(f'Error plotting training performance: {e}')
        if vali_df is not None:
            try:
                fig2, ax3 = plt.subplots(figsize=(10, 5))
                ax3.plot(vali_df['Update'], vali_df['ValidationMetric'], marker='o', linestyle='-', label='Validation Metric')
                ax3.set_xlabel('Training iterations')
                ax3.set_ylabel('Validation Makespan Performance Metric')
                ax3.set_title(f'Validation Makespan Performance ({self.model_name})')
                ax3.legend()
                plot_save_makespan_path = os.path.join(log_dir, f'validation_makespan_performance_{self.model_name}.png')
                plt.savefig(plot_save_makespan_path)
                plt.close(fig2)
            except Exception as e:
                warnings.warn(f'Error plotting validation performance: {e}')
        if reward_df is not None:
            try:
                fig3, ax4 = plt.subplots(figsize=(10, 5))
                ax4.plot(reward_df['Update'], reward_df['AvgFinalReward'], marker='.', linestyle='-', label='Avg Final Reward')
                ax4.set_xlabel('Training iterations')
                ax4.set_ylabel('Average Final Reward (Standardized Improvement)')
                ax4.set_title(f'Average Final Reward per Update ({self.model_name})')
                ax4.legend()
                ax4.axhline(0, color='grey', linestyle='--', linewidth=0.8)
                plot_save_path = os.path.join(log_dir, f'average_reward_{self.model_name}.png')
                plt.savefig(plot_save_path)
                plt.close(fig3)
            except Exception as e:
                warnings.warn(f'Error plotting average reward: {e}')
        if vali_tardiness_df is not None:
            try:
                fig4, ax5 = plt.subplots(figsize=(10, 5))
                ax5.plot(vali_tardiness_df['Update'], vali_tardiness_df['ValidationMetric'], marker='o', linestyle='-', label='Validation Metric')
                ax5.set_xlabel('Training iterations')
                ax5.set_ylabel('Validation Tardiness Performance Metric')
                ax5.set_title(f'Validation Tardiness Performance ({self.model_name})')
                ax5.legend()
                plot_save_tardiness_path = os.path.join(log_dir, f'validation_tardiness_performance_{self.model_name}.png')
                plt.savefig(plot_save_tardiness_path)
                plt.close(fig4)
            except Exception as e:
                warnings.warn(f'Error plotting validation performance: {e}')

def main():
    try:
        from DFJSPEnv import DFJSPEnv
        env_instance = DFJSPEnv(configs.n_j, configs.n_m)
        vali_env_instance = DFJSPEnv(configs.n_j, configs.n_m)
    except ImportError:
        print('Error: Could not import DFJSPEnv')
        sys.exit(1)
    except Exception as e:
        print(f'Error initializing environment instances: {e}')
        sys.exit(1)
    trainer = Trainer(configs, env_instance, vali_env_instance)
    trainer.train()
if __name__ == '__main__':
    try:
        from DFJSPEnv import DFJSPEnv
    except ImportError:
        print('Error: Ensure DFJSPEnv.py exists and is importable.')
        sys.exit(1)
    main()
