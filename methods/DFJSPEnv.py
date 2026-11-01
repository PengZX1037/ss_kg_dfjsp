import numpy as np
import numpy.ma as ma
import copy
import sys
import torch
import warnings
from common_utils import find_min_not_zero
from params import configs
from fjsp_env_same_op_nums import EnvState

class DFJSPEnv:

    def __init__(self, n_j_initial, n_m):
        self.use_step_reward = configs.use_step_reward
        self.waiting_penalty = float(configs.waiting_penalty)
        self.number_of_machines = n_m
        self.old_state = EnvState()
        self.op_fea_dim = configs.fea_j_input_dim
        self.mch_fea_dim = configs.fea_m_input_dim
        self.pair_fea_dim = configs.pair_input_dim
        self.reschedule_mode = configs.reschedule_mode
        self.reward_makespan_weight = configs.reward_makespan_weight
        self.reward_tardiness_weight = configs.reward_tardiness_weight
        self.number_of_envs = 0
        self.number_of_jobs = 0
        self.number_of_ops = 0
        self.max_number_of_ops = 0
        self.job_length = None
        self.op_pt = None
        self.true_op_pt = None
        self.op_pt_norm = None
        self.job_due_date = None
        self.env_number_of_ops = None
        self.env_number_of_jobs = None
        self.reschedule_ops_idx = None
        self.reschedule_ops_mask = None
        self.reschedule_jobs_mask = None
        self.initial_shop_state = None
        self.reschedule_start_time = None
        self.ops_in_W_scheduled_count = None
        self.total_ops_in_W = None
        self.mask_dummy_node = None
        self.op_machine_assignment = None
        self.true_initial_mch_available_times = None

    def get_next_wait_time(self, env_idx, eps=1e-08):
        """Return the next feasible scheduling event strictly after the current decision time."""
        if self.done()[env_idx]:
            return None
        decision_time = float(self.next_schedule_time[env_idx])
        candidate_start_times = np.maximum(self.candidate_free_time[env_idx, :, None], self.mch_free_time[env_idx, None, :])
        eligible_pairs = (~self.candidate_process_relation[env_idx]).copy()
        eligible_pairs &= self.candidate[env_idx, :, None] >= 0
        future_events = eligible_pairs & np.isfinite(candidate_start_times) & (candidate_start_times > decision_time + eps)
        if not np.any(future_events):
            return None
        return float(candidate_start_times[future_events].min())

    def get_wait_action_mask(self):
        return np.array([self.get_next_wait_time(e) is not None for e in range(self.number_of_envs)], dtype=np.bool_)

    def _update_wait_mask_on_state(self):
        wait_mask = self.get_wait_action_mask()
        self.state.wait_action_valid_tensor = torch.as_tensor(wait_mask, dtype=torch.bool, device=self.state.candidate_tensor.device).clone()

    def set_static_properties(self):
        self.multi_env_mch_diag = np.tile(np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0), (self.number_of_envs, 1, 1))
        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(self.number_of_envs, self.number_of_jobs)
        self.op_idx_padded = np.arange(self.max_number_of_ops)[np.newaxis, :]
        self.mask_dummy_node = np.full(shape=[self.number_of_envs, self.max_number_of_ops], fill_value=False, dtype=bool)
        operation_ids = np.arange(self.max_number_of_ops)
        self.mask_dummy_node[self.env_idxs[:, np.newaxis], operation_ids] = operation_ids >= self.env_number_of_ops[:, np.newaxis]
        self.dummy_mask_fea_j = np.tile(self.mask_dummy_node[:, :, np.newaxis], (1, 1, self.op_fea_dim))
        self.flag_exist_dummy_node = np.any(self.mask_dummy_node)
        self.op_to_job_map = np.full((self.number_of_envs, self.max_number_of_ops), -1, dtype=int)
        for e in range(self.number_of_envs):
            operation_offset = 0
            for j in range(self.env_number_of_jobs[e]):
                operations_in_job = self.job_length[e, j]
                if operations_in_job > 0:
                    next_operation_offset = operation_offset + operations_in_job
                    bounded_start = min(operation_offset, self.env_number_of_ops[e])
                    bounded_end = min(next_operation_offset, self.env_number_of_ops[e])
                    if bounded_start < bounded_end:
                        self.op_to_job_map[e, bounded_start:bounded_end] = j
                    operation_offset = next_operation_offset

    def reset_for_reschedule(self, initial_shop_states, reschedule_windows_W, job_lengths, op_pts, job_due_dates):
        """Initialize padded batches for dynamic rescheduling."""
        self.reschedule_mode = True
        self.number_of_envs = len(initial_shop_states)
        self.env_number_of_ops = np.array([arr.shape[0] for arr in op_pts])
        self.env_number_of_jobs = np.array([arr.shape[0] for arr in job_lengths])
        self.max_number_of_ops = np.max(self.env_number_of_ops) if self.env_number_of_ops.size > 0 else 0
        self.number_of_jobs = np.max(self.env_number_of_jobs) if self.env_number_of_jobs.size > 0 else 0
        self.number_of_machines = op_pts[0].shape[1] if len(op_pts) > 0 and len(op_pts[0].shape) > 1 else 0
        self.job_length = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=int)
        self.op_pt = np.zeros((self.number_of_envs, self.max_number_of_ops, self.number_of_machines), dtype=op_pts[0].dtype)
        self.job_due_date = np.full((self.number_of_envs, self.number_of_jobs), float('inf'))
        self.op_machine_assignment = np.full((self.number_of_envs, self.max_number_of_ops), -1, dtype=int)
        self.os = [[] for _ in range(self.number_of_envs)]
        self.sch_op_idx = [[] for _ in range(self.number_of_envs)]
        self.true_initial_mch_available_times = np.array([s['mch_free_time'] for s in initial_shop_states])
        for e in range(self.number_of_envs):
            env_operation_count = self.env_number_of_ops[e]
            env_job_count = self.env_number_of_jobs[e]
            self.job_length[e, :env_job_count] = job_lengths[e]
            self.op_pt[e, :env_operation_count, :] = op_pts[e]
            self.job_due_date[e, :env_job_count] = job_due_dates[e]
            if 'op_machine_assignment_old' in initial_shop_states[e]:
                op_assign_old = initial_shop_states[e]['op_machine_assignment_old']
                n_op_old_e = len(op_assign_old)
                if n_op_old_e > 0:
                    self.op_machine_assignment[e, :n_op_old_e] = op_assign_old
        self.set_static_properties()
        self.pt_lower_bound = 0.0
        self.pt_upper_bound = configs.high
        self.true_op_pt = np.copy(self.op_pt)
        processing_time_range = self.pt_upper_bound - self.pt_lower_bound
        self.time_normalization_scale = float(processing_time_range + 1e-08)
        self.op_pt_norm = (self.op_pt - self.pt_lower_bound) / (processing_time_range + 1e-08)
        self.process_relation = self.true_op_pt != 0
        self.reverse_process_relation = ~self.process_relation
        self.compatible_op = np.sum(self.process_relation, 2)
        self.compatible_mch = np.sum(self.process_relation, 1)
        self.unmasked_op_pt_norm = np.copy(self.op_pt_norm)
        initial_operation_offsets = np.zeros((self.number_of_envs, 1))
        self.job_first_op_id = np.concatenate([initial_operation_offsets, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype('int')
        self.job_last_op_id = np.full((self.number_of_envs, self.number_of_jobs), -1, dtype=int)
        for e in range(self.number_of_envs):
            for j in range(self.env_number_of_jobs[e]):
                last_op_for_job = self.job_first_op_id[e, j] + self.job_length[e, j] - 1
                self.job_last_op_id[e, j] = min(last_op_for_job, self.env_number_of_ops[e] - 1)
        self.initial_vars()
        self.reschedule_start_time = np.array([s['current_time'] for s in initial_shop_states])
        self.true_mch_free_time = np.copy(np.array([s['mch_free_time'] for s in initial_shop_states]))
        self.mch_free_time = np.array([s['mch_free_time_norm'] for s in initial_shop_states])
        self.reschedule_start_time_norm = np.asarray([s['current_time_norm'] for s in initial_shop_states], dtype=float)
        self.next_schedule_time = np.maximum(np.min(self.mch_free_time, axis=1), self.reschedule_start_time_norm)
        self.candidate_free_time = np.full((self.number_of_envs, self.number_of_jobs), float('inf'))
        self.true_candidate_free_time = np.full((self.number_of_envs, self.number_of_jobs), float('inf'))
        self.candidate = np.full((self.number_of_envs, self.number_of_jobs), -1, dtype=int)
        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        for e in range(self.number_of_envs):
            shop_snapshot = initial_shop_states[e]
            previous_op_status = shop_snapshot['op_status']
            previous_actual_completion = shop_snapshot['op_completion_time']
            previous_normalized_completion = shop_snapshot['op_completion_time_norm']
            previous_job_ready_times = shop_snapshot['job_next_op_ready_time_norm']
            previous_job_ready_times = np.maximum(0, previous_job_ready_times)
            n_op_old_e = len(previous_op_status)
            n_j_old_e = len(previous_job_ready_times)
            if n_op_old_e > 0:
                self.true_op_ct[e, :n_op_old_e][previous_op_status == 1] = previous_actual_completion[previous_op_status == 1]
                self.op_ct[e, :n_op_old_e][previous_op_status == 1] = previous_normalized_completion[previous_op_status == 1]
                self.op_scheduled_flag[e, :n_op_old_e][previous_op_status == 1] = 2
            for j in range(n_j_old_e):
                first_op = self.job_first_op_id[e, j]
                last_op = self.job_last_op_id[e, j]
                if first_op >= n_op_old_e:
                    continue
                start_check = first_op
                end_check = min(last_op + 1, n_op_old_e)
                if start_check >= end_check:
                    self.mask[e, j] = True
                    continue
                job_ops_status_old = previous_op_status[start_check:end_check]
                first_unfin_idx_rel = np.argmax(job_ops_status_old == 0)
                if job_ops_status_old[first_unfin_idx_rel] == 0:
                    candidate_op_abs = start_check + first_unfin_idx_rel
                    self.candidate[e, j] = candidate_op_abs
                    self.candidate_free_time[e, j] = previous_job_ready_times[j]
                    self.true_candidate_free_time[e, j] = shop_snapshot['job_next_op_ready_time'][j]
                    if candidate_op_abs > first_op:
                        self.op_scheduled_flag[e, first_op:candidate_op_abs] = 2
                else:
                    self.mask[e, j] = True
            for j_total in range(n_j_old_e, self.env_number_of_jobs[e]):
                first_op_new = self.job_first_op_id[e, j_total]
                if 0 <= first_op_new < self.env_number_of_ops[e]:
                    self.candidate[e, j_total] = first_op_new
                    self.candidate_free_time[e, j_total] = shop_snapshot['current_time_norm']
                    self.true_candidate_free_time[e, j_total] = self.reschedule_start_time[e]
            self.current_makespan[e] = np.max(self.true_initial_mch_available_times[e])
        self.reschedule_ops_idx = reschedule_windows_W
        self.reschedule_ops_mask = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=bool)
        self.ops_in_W_scheduled_count = np.zeros(self.number_of_envs, dtype=int)
        self.total_ops_in_W = np.zeros(self.number_of_envs, dtype=int)
        self.reschedule_jobs_mask = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=bool)
        for e in range(self.number_of_envs):
            window_operation_ids = self.reschedule_ops_idx[e]
            if len(window_operation_ids) > 0:
                valid_window_indices = (window_operation_ids >= 0) & (window_operation_ids < self.max_number_of_ops)
                valid_window_ops = window_operation_ids[valid_window_indices]
                if len(valid_window_ops) > 0:
                    self.reschedule_ops_mask[e, valid_window_ops] = True
                    self.total_ops_in_W[e] = len(valid_window_ops)
                    affected_job_ids = self.op_to_job_map[e, valid_window_ops]
                    valid_jobs_mask = (affected_job_ids >= 0) & (affected_job_ids < self.number_of_jobs)
                    affected_valid_jobs = np.unique(affected_job_ids[valid_jobs_mask])
                    if len(affected_valid_jobs) > 0:
                        self.reschedule_jobs_mask[e, affected_valid_jobs] = True
                else:
                    self.total_ops_in_W[e] = 0
            else:
                self.total_ops_in_W[e] = 0
        self.op_pt_masked = ma.array(self.op_pt_norm, mask=self.reverse_process_relation)
        self.true_op_pt_masked = ma.array(self.true_op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.op_pt_masked, axis=2).data
        self.op_min_pt = np.min(self.op_pt_masked, axis=-1).data
        self.true_op_min_pt = np.min(self.true_op_pt_masked, axis=-1).data
        self.op_max_pt = np.max(self.op_pt_masked, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt
        self.mch_min_pt = np.max(self.op_pt_masked, axis=1).data
        self.mch_mean_pt = np.mean(self.op_pt_masked, axis=1).filled(0)
        self.op_ct_lb = np.zeros_like(self.op_ct)
        self.true_op_ct_lb = np.zeros_like(self.op_ct)
        candidate_mask = self.candidate >= 0
        env_indices, job_indices = np.where(candidate_mask)
        if env_indices.size > 0:
            valid_candidates = self.candidate[env_indices, job_indices]
            valid_op_mask_lb = (valid_candidates >= 0) & (valid_candidates < self.max_number_of_ops)
            env_indices = env_indices[valid_op_mask_lb]
            job_indices = job_indices[valid_op_mask_lb]
            valid_candidates = valid_candidates[valid_op_mask_lb]
            if valid_candidates.size > 0:
                earliest_start_time = self.candidate_free_time[env_indices, job_indices]
                min_pt_candidate = self.op_min_pt[env_indices, valid_candidates]
                min_pt_candidate[min_pt_candidate <= 0] = 0
                initial_lb = earliest_start_time + min_pt_candidate
                self.op_ct_lb[env_indices, valid_candidates] = initial_lb
                self.true_op_ct_lb[env_indices, valid_candidates] = self.true_candidate_free_time[env_indices, job_indices] + self.true_op_min_pt[env_indices, valid_candidates]
        for e in range(self.number_of_envs):
            for j in range(self.env_number_of_jobs[e]):
                if self.candidate[e, j] != -1:
                    current_op_idx = self.candidate[e, j]
                    job_last_op_actual = self.job_last_op_id[e, j]
                    while current_op_idx <= job_last_op_actual:
                        current_op_lb = self.op_ct_lb[e, current_op_idx]
                        true_current_op_lb = self.true_op_ct_lb[e, current_op_idx]
                        if current_op_idx < job_last_op_actual:
                            next_op_idx = current_op_idx + 1
                            min_pt_current = self.op_min_pt[e, next_op_idx] if np.any(self.process_relation[e, current_op_idx]) else 0
                            min_true_pt_current = self.true_op_min_pt[e, next_op_idx] if np.any(self.process_relation[e, current_op_idx]) else 0
                            next_op_lb_candidate = current_op_lb + min_pt_current
                            true_next_op_lb_candidate = true_current_op_lb + min_true_pt_current
                            if self.op_scheduled_flag[e, next_op_idx] != 2:
                                self.op_ct_lb[e, next_op_idx] = next_op_lb_candidate
                                self.true_op_ct_lb[e, next_op_idx] = true_next_op_lb_candidate
                        current_op_idx += 1
        self.op_match_job_left_op_nums = np.zeros_like(self.op_to_job_map, dtype=int)
        self.op_match_job_remain_work = np.zeros_like(self.op_to_job_map, dtype=float)
        for e in range(self.number_of_envs):
            for j in range(self.env_number_of_jobs[e]):
                if self.candidate[e, j] != -1:
                    start_op = self.candidate[e, j]
                    end_op = self.job_last_op_id[e, j]
                    ops_in_job_range = np.arange(start_op, end_op + 1)
                    if ops_in_job_range.size > 0:
                        valid_range_mask = ops_in_job_range < self.max_number_of_ops
                        ops_in_job_range = ops_in_job_range[valid_range_mask]
                        if ops_in_job_range.size > 0:
                            ops_in_job_and_W_mask = self.reschedule_ops_mask[e, ops_in_job_range]
                            num_left_in_W = np.sum(ops_in_job_and_W_mask)
                            remain_work_in_W = np.sum(self.op_mean_pt[e, ops_in_job_range][ops_in_job_and_W_mask])
                            self.op_match_job_left_op_nums[e, ops_in_job_range] = num_left_in_W
                            self.op_match_job_remain_work[e, ops_in_job_range] = remain_work_in_W
        self.init_op_mask()
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        finished_mask = self.op_scheduled_flag == 2
        self.mch_current_available_op_nums -= np.sum(self.process_relation & finished_mask[:, :, np.newaxis], axis=1)
        self.mch_current_available_op_nums = np.maximum(0, self.mch_current_available_op_nums)
        self.candidate_pt_norm = np.zeros((self.number_of_envs, self.number_of_jobs, self.number_of_machines))
        valid_candidate_mask = self.candidate >= 0
        env_idx_vc, job_idx_vc = np.where(valid_candidate_mask)
        if env_idx_vc.size > 0:
            valid_candidate_ops = self.candidate[env_idx_vc, job_idx_vc]
            valid_op_indices_mask = (valid_candidate_ops >= 0) & (valid_candidate_ops < self.max_number_of_ops)
            env_idx_vc = env_idx_vc[valid_op_indices_mask]
            job_idx_vc = job_idx_vc[valid_op_indices_mask]
            valid_candidate_ops = valid_candidate_ops[valid_op_indices_mask]
            if valid_candidate_ops.size > 0:
                self.candidate_pt_norm[env_idx_vc, job_idx_vc, :] = self.unmasked_op_pt_norm[env_idx_vc, valid_candidate_ops, :]
        self.dynamic_pair_mask = self.candidate_pt_norm == 0
        finished_job_mask = ~valid_candidate_mask
        self.dynamic_pair_mask[finished_job_mask, :] = True
        if self.reschedule_mode and self.reschedule_ops_mask is not None:
            candidate_outside_W_mask = np.ones_like(self.candidate, dtype=bool)
            if env_idx_vc.size > 0:
                if valid_candidates.size > 0:
                    is_in_W = self.reschedule_ops_mask[env_idx_vc, valid_candidates]
                    candidate_outside_W_mask[env_idx_vc, job_idx_vc] = ~is_in_W
            candidate_outside_W_mask_expanded = np.expand_dims(candidate_outside_W_mask, axis=2)
            self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, candidate_outside_W_mask_expanded)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.pair_free_time = np.maximum(self.candidate_free_time[:, :, None], self.mch_free_time[:, None, :])
        self.dynamic_pair_mask = self.candidate_process_relation | (self.pair_free_time > self.next_schedule_time[:, None, None] + 1e-08)
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.init_mch_mask()
        self.construct_op_features()
        self.construct_mch_features()
        self.construct_pair_features()
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.init_makespan = np.max(self.true_op_ct_lb, axis=1)
        self.initial_shop_state = {'op_ct_lb': np.copy(self.op_ct_lb), 'true_op_ct': np.copy(self.true_op_ct), 'reschedule_ops_mask': np.copy(self.reschedule_ops_mask), 'reschedule_jobs_mask': np.copy(self.reschedule_jobs_mask), 'job_due_date': np.copy(self.job_due_date)}
        self.state = EnvState()
        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask, self.dynamic_pair_mask, self.comp_idx, self.candidate, self.fea_pairs, reschedule_op_idx=self.reschedule_ops_idx, job_due_date=self.job_due_date, current_time=self.next_schedule_time)
        self._update_wait_mask_on_state()
        self.old_state = copy.deepcopy(self.state)
        return self.state

    def initial_vars(self):
        self.step_count = 0
        self.current_makespan = np.full(self.number_of_envs, 0.0, dtype=float)
        self.op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=float)
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines), dtype=float)
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines), dtype=float)
        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines), dtype=float)
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines), dtype=float)
        self.next_schedule_time = np.zeros(self.number_of_envs, dtype=float)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=float)
        self.true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=float)
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=float)
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines), dtype=float)
        self.candidate = np.full((self.number_of_envs, self.number_of_jobs), -1, dtype=int)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=False, dtype=bool)
        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=int)
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=float)
        self.op_remain_work = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=float)
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs, self.number_of_machines), dtype=float)
        self.delete_mask_fea_j = np.full(shape=(self.number_of_envs, self.max_number_of_ops, self.op_fea_dim), fill_value=False, dtype=bool)
        self.deleted_op_nodes = np.full(shape=(self.number_of_envs, self.max_number_of_ops), fill_value=False, dtype=bool)
        self.op_machine_assignment = np.full((self.number_of_envs, self.max_number_of_ops), -1, dtype=int)
        if self.reschedule_mode:
            self.ops_in_W_scheduled_count = np.zeros(self.number_of_envs, dtype=int)

    def step(self, actions):
        previous_makespan_estimate, old_est_tardiness = self._calculate_estimated_performance()
        previous_done_flags = self.done()
        active_envs_mask = ~previous_done_flags
        active_environment_indices = np.where(active_envs_mask)[0]
        if len(active_environment_indices) == 0:
            reward_batch = np.zeros(self.number_of_envs)
            return (self.state, reward_batch, previous_done_flags)
        selected_actions = actions[active_environment_indices]
        dispatch_count = self.number_of_jobs * self.number_of_machines
        wait_action_id = dispatch_count
        reward_batch = np.zeros(self.number_of_envs)
        reward_signal = np.full(self.number_of_envs, -self.waiting_penalty, dtype=float)
        env_idx_took_schedule_step = []
        env_idx_empty_step = []
        for i, env_idx in enumerate(active_environment_indices):
            action = selected_actions[i]
            if action == wait_action_id:
                next_event_time = self.get_next_wait_time(env_idx)
                if next_event_time is None:
                    raise RuntimeError(f'Invalid waiting action in env {env_idx}: no future feasible scheduling event.')
                env_idx_empty_step.append([env_idx, next_event_time])
                continue
            env_idx_took_schedule_step.append(env_idx)
            action_job_idx_total = action // self.number_of_machines
            action_mch_idx = action % self.number_of_machines
            candidate_ops_active = self.candidate[env_idx, action_job_idx_total]
            if not (0 <= action_job_idx_total < self.number_of_jobs and 0 <= candidate_ops_active < self.max_number_of_ops and (not self.dynamic_pair_mask[env_idx, action_job_idx_total, action_mch_idx])):
                raise RuntimeError(f'Invalid dispatch in env {env_idx}: action={action}, candidate={candidate_ops_active}, decision_time={self.next_schedule_time[env_idx]}')
            env_idx_update = env_idx
            job_idx_update = action_job_idx_total
            op_idx_update = candidate_ops_active
            mch_idx_update = action_mch_idx
            self.step_count += 1
            job_idx = job_idx_update
            op_idx = op_idx_update
            self.os[env_idx_update].append(job_idx)
            self.sch_op_idx[env_idx_update].append(op_idx)
            if self.reschedule_mode:
                is_in_W = self.reschedule_ops_mask[env_idx_update, op_idx_update]
                self.ops_in_W_scheduled_count[env_idx_update[is_in_W]] += 1
            next_op_indices = op_idx_update + 1
            is_last_op_mask = op_idx_update >= self.job_last_op_id[env_idx_update, job_idx_update]
            new_candidate_vals = np.where(is_last_op_mask, -1, next_op_indices)
            if new_candidate_vals != -1 and (not self.reschedule_ops_mask[env_idx_update, new_candidate_vals]):
                new_candidate_vals = -1
            self.candidate[env_idx_update, job_idx_update] = new_candidate_vals
            self.mch_current_available_op_nums[env_idx_update] -= self.process_relation[env_idx_update, op_idx_update]
            self.mch_current_available_op_nums = np.maximum(0, self.mch_current_available_op_nums)
            self.mask[env_idx_update, job_idx_update] = np.array([new_candidate_vals == -1])
            op_st_norm = max(float(self.candidate_free_time[env_idx_update, job_idx_update]), float(self.mch_free_time[env_idx_update, mch_idx_update]), float(self.next_schedule_time[env_idx_update]))
            op_pt_norm_chosen = self.op_pt_norm[env_idx_update, op_idx_update, mch_idx_update]
            op_ct_norm = op_st_norm + op_pt_norm_chosen
            self.op_ct[env_idx_update, op_idx_update] = op_ct_norm
            true_op_st = max(float(self.true_candidate_free_time[env_idx_update, job_idx_update]), float(self.true_mch_free_time[env_idx_update, mch_idx_update]), float(self.next_schedule_time[env_idx_update]) * self.time_normalization_scale)
            true_op_pt_chosen = self.true_op_pt[env_idx_update, op_idx_update, mch_idx_update]
            true_op_ct_val = true_op_st + true_op_pt_chosen
            self.true_op_ct[env_idx_update, op_idx_update] = true_op_ct_val
            self.candidate_free_time[env_idx_update, job_idx_update] = op_ct_norm
            self.true_candidate_free_time[env_idx_update, job_idx_update] = true_op_ct_val
            self.mch_free_time[env_idx_update, mch_idx_update] = op_ct_norm
            self.true_mch_free_time[env_idx_update, mch_idx_update] = true_op_ct_val
            self.current_makespan[env_idx_update] = np.maximum(self.current_makespan[env_idx_update], true_op_ct_val)
            self.op_machine_assignment[env_idx_update, op_idx_update] = mch_idx_update
            finished_job_mask_update = new_candidate_vals == -1
            valid_next_candidate_mask = ~finished_job_mask_update
            next_op_idx_valid = valid_next_candidate_mask
            in_bounds_mask = next_op_idx_valid < self.max_number_of_ops
            is_last_e = candidate_ops_active >= self.job_last_op_id[env_idx, action_job_idx_total]
            if new_candidate_vals != -1 and 0 <= next_op_indices < self.max_number_of_ops:
                self.candidate_pt_norm[env_idx, action_job_idx_total, :] = self.unmasked_op_pt_norm[env_idx, next_op_indices, :]
                self.candidate_process_relation[env_idx, action_job_idx_total, :] = self.reverse_process_relation[env_idx, next_op_indices, :]
            else:
                self.candidate_process_relation[env_idx, action_job_idx_total, :] = True
            self.op_scheduled_flag[env_idx_update, op_idx_update] = 1
            diff_norm = self.op_ct[env_idx_update, op_idx_update] - self.op_ct_lb[env_idx_update, op_idx_update]
            true_diff_norm = self.true_op_ct[env_idx_update, op_idx_update] - self.true_op_ct_lb[env_idx_update, op_idx_update]
            job_idx = job_idx_update
            op_idx = op_idx_update
            last_op_actual = self.job_last_op_id[env_idx, job_idx]
            if op_idx < last_op_actual:
                start_update_idx = op_idx + 1
                end_update_idx = last_op_actual + 1
                start_update_idx = max(0, start_update_idx)
                end_update_idx = min(self.max_number_of_ops, end_update_idx)
                if start_update_idx < end_update_idx:
                    self.op_ct_lb[env_idx, start_update_idx:end_update_idx] += diff_norm
                    self.true_op_ct_lb[env_idx, start_update_idx:end_update_idx] += true_diff_norm
            elif op_idx == last_op_actual:
                self.op_ct_lb[env_idx, op_idx] += diff_norm
                self.true_op_ct_lb[env_idx, op_idx] += true_diff_norm
            current_candidate_op = self.candidate[env_idx, job_idx]
            if current_candidate_op != -1:
                start_op_remain = current_candidate_op
                end_op_remain = last_op_actual
                ops_in_job_range = np.arange(start_op_remain, end_op_remain + 1)
                if ops_in_job_range.size > 0:
                    valid_range_mask = ops_in_job_range < self.max_number_of_ops
                    ops_in_job_range = ops_in_job_range[valid_range_mask]
                    if ops_in_job_range.size > 0:
                        ops_in_job_and_W_mask = self.reschedule_ops_mask[env_idx, ops_in_job_range]
                        num_left_in_W = np.sum(ops_in_job_and_W_mask)
                        remain_work_in_W = np.sum(self.op_mean_pt[env_idx, ops_in_job_range][ops_in_job_and_W_mask])
                        self.op_match_job_left_op_nums[env_idx, ops_in_job_range] = num_left_in_W
                        self.op_match_job_remain_work[env_idx, ops_in_job_range] = remain_work_in_W
        previous_decision_time = np.asarray(self.next_schedule_time, dtype=float).copy()
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)
        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)
        min_times = np.min(schedule_matrix.reshape(self.number_of_envs, -1), axis=1)
        current_max_op_ct = np.max(self.op_ct, axis=1)
        next_time_candidate = np.where(min_times.mask, current_max_op_ct, min_times.data)
        self.next_schedule_time = np.maximum(previous_decision_time, next_time_candidate)
        for sub_list in env_idx_empty_step:
            self.next_schedule_time[sub_list[0]] = sub_list[1]
        self.deleted_op_nodes = np.logical_and(self.op_ct <= self.next_schedule_time[:, np.newaxis], self.op_scheduled_flag == 1)
        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis], (1, 1, self.op_fea_dim))
        self.update_op_mask()
        self.op_waiting_time.fill(0)
        candidate_mask = self.candidate >= 0
        env_indices_cw, job_indices_cw = np.where(candidate_mask)
        if env_indices_cw.size > 0:
            valid_candidates_ops = self.candidate[env_indices_cw, job_indices_cw]
            valid_op_mask_wait = (valid_candidates_ops >= 0) & (valid_candidates_ops < self.max_number_of_ops)
            env_indices_cw = env_indices_cw[valid_op_mask_wait]
            job_indices_cw = job_indices_cw[valid_op_mask_wait]
            valid_candidates_ops = valid_candidates_ops[valid_op_mask_wait]
            if valid_candidates_ops.size > 0:
                wait_time = np.maximum(self.next_schedule_time[env_indices_cw] - self.candidate_free_time[env_indices_cw, job_indices_cw], 0)
                self.op_waiting_time[env_indices_cw, valid_candidates_ops] = wait_time
        self.op_remain_work = np.maximum(self.op_ct - self.next_schedule_time[:, np.newaxis], 0)
        self.construct_op_features()
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)
        self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]
        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.update_mch_mask()
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        mch_free_duration = self.next_schedule_time[:, np.newaxis] - self.mch_free_time
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag.astype(float)
        self.mch_waiting_time = np.maximum(0, (1 - self.mch_working_flag) * mch_free_duration)
        self.mch_remain_work = np.maximum(0, -mch_free_duration)
        self.construct_mch_features()
        self.construct_pair_features()
        new_est_max_lb, new_est_tardiness = self._calculate_estimated_performance()
        for e in env_idx_took_schedule_step:
            tardiness_improved = new_est_tardiness[e] < old_est_tardiness[e]
            tardiness_worsened = new_est_tardiness[e] > old_est_tardiness[e]
            makespan_improved = new_est_max_lb[e] < previous_makespan_estimate[e]
            if tardiness_improved:
                reward_signal[e] = 1.0
            elif tardiness_worsened:
                reward_signal[e] = -1.0
            elif makespan_improved:
                reward_signal[e] = 1.0
            else:
                reward_signal[e] = 0.0
        reward_batch = reward_signal
        completion_flags = self.done()
        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask, self.dynamic_pair_mask, self.comp_idx, self.candidate, self.fea_pairs, job_due_date=self.job_due_date, current_time=self.next_schedule_time)
        self._update_wait_mask_on_state()
        return (self.state, reward_batch, completion_flags)

    def done(self):
        if not self.reschedule_mode:
            num_scheduled = np.sum(self.op_scheduled_flag > 0, axis=1)
            return num_scheduled >= self.env_number_of_ops
        if self.ops_in_W_scheduled_count is None or self.total_ops_in_W is None:
            warnings.warn('Reschedule counters not initialized in done().')
            return np.zeros(self.number_of_envs, dtype=bool)
        done_mask = np.full(self.number_of_envs, False, dtype=bool)
        done_mask[self.total_ops_in_W == 0] = True
        valid_w_mask = self.total_ops_in_W > 0
        if np.any(valid_w_mask):
            done_mask[valid_w_mask] = self.ops_in_W_scheduled_count[valid_w_mask] >= self.total_ops_in_W[valid_w_mask]
        return done_mask

    def construct_op_features(self):
        op_due_date = np.full((self.number_of_envs, self.max_number_of_ops), float('inf'))
        op_slack_time = np.full((self.number_of_envs, self.max_number_of_ops), float('inf'))
        if self.job_due_date is not None:
            valid_job_map_mask = self.op_to_job_map != -1
            env_idx_vm, op_idx_vm = np.where(valid_job_map_mask)
            if env_idx_vm.size > 0:
                job_idx_vm = self.op_to_job_map[env_idx_vm, op_idx_vm]
                valid_job_idx_mask = job_idx_vm < self.job_due_date.shape[1]
                env_idx_vm = env_idx_vm[valid_job_idx_mask]
                op_idx_vm = op_idx_vm[valid_job_idx_mask]
                job_idx_vm = job_idx_vm[valid_job_idx_mask]
                if env_idx_vm.size > 0:
                    op_due_date[env_idx_vm, op_idx_vm] = self.job_due_date[env_idx_vm, job_idx_vm]
                    est_completion = np.where(self.op_scheduled_flag == 2, self.true_op_ct, self.op_ct_lb)
                    finite_due_date_mask_vm = np.isfinite(op_due_date[env_idx_vm, op_idx_vm])
                    env_idx_fin = env_idx_vm[finite_due_date_mask_vm]
                    op_idx_fin = op_idx_vm[finite_due_date_mask_vm]
                    if env_idx_fin.size > 0:
                        op_slack_time[env_idx_fin, op_idx_fin] = op_due_date[env_idx_fin, op_idx_fin] - est_completion[env_idx_fin, op_idx_fin]
        norm_op_due_date = np.zeros_like(op_due_date)
        if self.job_due_date is not None and env_idx_vm.size > 0:
            max_due_date_per_env = np.maximum(1.0, np.max(self.job_due_date, axis=1))
            max_dd_for_ops = max_due_date_per_env[env_idx_vm]
            due_dates_for_ops = op_due_date[env_idx_vm, op_idx_vm]
            normalized_values = due_dates_for_ops / max_dd_for_ops
            norm_op_due_date[env_idx_vm, op_idx_vm] = normalized_values
        norm_op_slack = np.tanh(op_slack_time / (configs.high * self.max_number_of_ops + 1e-06))
        infinite_slack_mask = ~np.isfinite(op_slack_time)
        norm_op_slack[infinite_slack_mask] = 1.0
        norm_op_match_job_left_op_nums = np.zeros_like(self.op_match_job_left_op_nums, dtype=float)
        if env_idx_vm.size > 0:
            max_ops_per_job_val = np.maximum(1, np.max(self.job_length, axis=1))
            max_ops_for_envs = max_ops_per_job_val[env_idx_vm]
            remain_ops_for_ops = self.op_match_job_left_op_nums[env_idx_vm, op_idx_vm]
            normalized_remain_ops = remain_ops_for_ops / max_ops_for_envs
            norm_op_match_job_left_op_nums[env_idx_vm, op_idx_vm] = normalized_remain_ops
        self.fea_j = np.stack((self.op_scheduled_flag / 2.0, self.op_ct_lb, self.op_min_pt, self.pt_span, self.op_mean_pt, self.op_waiting_time, self.op_remain_work, norm_op_match_job_left_op_nums, self.op_match_job_remain_work, self.compatible_op / self.number_of_machines if self.number_of_machines > 0 else 0, norm_op_due_date, norm_op_slack), axis=2)
        self.fea_j[self.mask_dummy_node] = 0
        is_done = self.done()
        if not np.all(is_done):
            finished_before_mask = self.op_scheduled_flag == 2
            exclude_mask_j = np.logical_or(self.mask_dummy_node, finished_before_mask)
            exclude_mask_fea_j = np.tile(exclude_mask_j[:, :, np.newaxis], (1, 1, self.op_fea_dim))
            self.norm_op_features(mask=exclude_mask_fea_j)
        elif self.fea_j.size > 0:
            self.fea_j.fill(0)

    def norm_op_features(self, mask):
        if self.fea_j.size == 0:
            return
        num_nodes = self.max_number_of_ops
        num_masked_nodes = np.sum(mask[:, :, 0], axis=1, keepdims=True)
        num_active_nodes = np.maximum(1e-08, num_nodes - num_masked_nodes)
        self.fea_j[mask] = 0
        sum_mask = ~mask
        masked_fea_j_for_sum = np.where(sum_mask, self.fea_j, 0)
        sum_fea_j = np.sum(masked_fea_j_for_sum, axis=1)
        mean_fea_j = sum_fea_j / num_active_nodes
        sq_fea_j = np.where(sum_mask, self.fea_j ** 2, 0)
        sum_sq_fea_j = np.sum(sq_fea_j, axis=1)
        mean_sq_fea_j = sum_sq_fea_j / num_active_nodes
        var_fea_j = np.maximum(0, mean_sq_fea_j - mean_fea_j ** 2)
        std_fea_j = np.sqrt(var_fea_j)
        non_masked_features = (self.fea_j - mean_fea_j[:, np.newaxis, :]) / (std_fea_j[:, np.newaxis, :] + 1e-08)
        self.fea_j = np.where(mask, 0, non_masked_features)

    def construct_mch_features(self):
        norm_mch_jc_nums = self.mch_current_available_jc_nums / (self.number_of_jobs + 1e-08) if self.number_of_jobs > 0 else 0
        norm_mch_op_nums = self.mch_current_available_op_nums / (self.max_number_of_ops + 1e-08) if self.max_number_of_ops > 0 else 0
        self.fea_m = np.stack((norm_mch_jc_nums, norm_mch_op_nums, self.mch_min_pt, self.mch_mean_pt, self.mch_waiting_time, self.mch_remain_work, self.mch_free_time, self.mch_working_flag), axis=2)
        is_done = self.done()
        if not np.all(is_done) and self.fea_m.size > 0:
            if self.delete_mask_fea_m.shape == self.fea_m.shape:
                self.norm_machine_features(mask=self.delete_mask_fea_m)
            else:
                warnings.warn('Shape mismatch between fea_m and delete_mask_fea_m. Skipping machine normalization.')
        elif self.fea_m.size > 0:
            self.fea_m.fill(0)

    def norm_machine_features(self, mask):
        if self.fea_m.size == 0:
            return
        num_machines = self.number_of_machines
        num_masked_mchs = np.sum(mask[:, :, 0], axis=1, keepdims=True)
        num_active_mchs = np.maximum(1e-08, num_machines - num_masked_mchs)
        self.fea_m[mask] = 0
        sum_mask = ~mask
        masked_fea_m_for_sum = np.where(sum_mask, self.fea_m, 0)
        sum_fea_m = np.sum(masked_fea_m_for_sum, axis=1)
        mean_fea_m = sum_fea_m / num_active_mchs
        sq_fea_m = np.where(sum_mask, self.fea_m ** 2, 0)
        sum_sq_fea_m = np.sum(sq_fea_m, axis=1)
        mean_sq_fea_m = sum_sq_fea_m / num_active_mchs
        var_fea_m = np.maximum(0, mean_sq_fea_m - mean_fea_m ** 2)
        std_fea_m = np.sqrt(var_fea_m)
        non_masked_features = (self.fea_m - mean_fea_m[:, np.newaxis, :]) / (std_fea_m[:, np.newaxis, :] + 1e-08)
        self.fea_m = np.where(mask, 0, non_masked_features)

    def construct_pair_features(self):
        self.fea_pairs = np.zeros((self.number_of_envs, self.number_of_jobs, self.number_of_machines, self.pair_fea_dim))
        chosen_op_max_pt = np.zeros((self.number_of_envs, self.number_of_jobs, 1))
        candidate_mask = self.candidate >= 0
        env_idx_vc, job_idx_vc = np.where(candidate_mask)
        if env_idx_vc.size > 0:
            valid_candidates = self.candidate[env_idx_vc, job_idx_vc]
            valid_op_mask_maxpt = (valid_candidates >= 0) & (valid_candidates < self.max_number_of_ops)
            env_idx_vc = env_idx_vc[valid_op_mask_maxpt]
            job_idx_vc = job_idx_vc[valid_op_mask_maxpt]
            valid_candidates = valid_candidates[valid_op_mask_maxpt]
            if valid_candidates.size > 0:
                max_pts_for_candidates = self.op_max_pt[env_idx_vc, valid_candidates]
                chosen_op_max_pt[env_idx_vc, job_idx_vc, 0] = max_pts_for_candidates
        chosen_op_max_pt = np.maximum(1e-08, chosen_op_max_pt)
        current_remain_mask = self.process_relation & (self.op_scheduled_flag[:, :, np.newaxis] == 0)
        remain_op_pt_norm_current = ma.array(self.op_pt_norm, mask=~current_remain_mask)
        max_remain_op_pt = np.max(remain_op_pt_norm_current, axis=(1, 2), keepdims=True)
        max_remain_op_pt = np.maximum(1e-08, max_remain_op_pt)
        mch_max_remain_op_pt = np.max(remain_op_pt_norm_current, axis=1, keepdims=True)
        mch_max_remain_op_pt = np.maximum(1e-08, mch_max_remain_op_pt)
        pair_max_pt = np.max(self.candidate_pt_norm, axis=(1, 2), keepdims=True)
        pair_max_pt = np.maximum(1e-08, pair_max_pt)
        mch_max_candidate_pt = np.max(self.candidate_pt_norm, axis=1, keepdims=True)
        mch_max_candidate_pt = np.maximum(1e-08, mch_max_candidate_pt)
        op_wait_time_candidate = np.zeros((self.number_of_envs, self.number_of_jobs, 1))
        if env_idx_vc.size > 0:
            op_wait_time_candidate[env_idx_vc, job_idx_vc, 0] = self.op_waiting_time[env_idx_vc, valid_candidates]
        pair_wait_time = op_wait_time_candidate + self.mch_waiting_time[:, np.newaxis, :]
        chosen_job_remain_work = np.zeros((self.number_of_envs, self.number_of_jobs, 1))
        if env_idx_vc.size > 0:
            chosen_job_remain_work[env_idx_vc, job_idx_vc, 0] = self.op_match_job_remain_work[env_idx_vc, valid_candidates]
        chosen_job_remain_work = np.maximum(1e-08, chosen_job_remain_work)
        self.fea_pairs[..., 0] = self.candidate_pt_norm
        self.fea_pairs[..., 1] = self.candidate_pt_norm / chosen_op_max_pt
        self.fea_pairs[..., 2] = self.candidate_pt_norm / mch_max_candidate_pt
        self.fea_pairs[..., 3] = self.candidate_pt_norm / max_remain_op_pt
        self.fea_pairs[..., 4] = self.candidate_pt_norm / mch_max_remain_op_pt
        self.fea_pairs[..., 5] = self.candidate_pt_norm / pair_max_pt
        self.fea_pairs[..., 6] = self.candidate_pt_norm / chosen_job_remain_work
        self.fea_pairs[..., 7] = pair_wait_time
        invalid_pair_mask_expanded = np.expand_dims(self.dynamic_pair_mask, axis=-1)
        self.fea_pairs = np.where(invalid_pair_mask_expanded, 0, self.fea_pairs)

    def update_mch_mask(self):
        active_unscheduled_op_mask = (self.op_scheduled_flag == 0) & ~self.mask_dummy_node
        relevant_process_relation = self.process_relation & active_unscheduled_op_mask[:, :, np.newaxis]
        self.mch_mask = self.logic_operator(relevant_process_relation).sum(axis=-1) > 0
        self.mch_mask[self.multi_env_mch_diag] = True
        can_process_remaining_active = np.sum(relevant_process_relation, axis=1) > 0
        self.delete_mask_fea_m = np.tile(~can_process_remaining_active[:, :, np.newaxis], (1, 1, self.mch_fea_dim))

    def init_mch_mask(self):
        self.mch_mask = self.logic_operator(self.process_relation).sum(axis=-1) > 0
        self.mch_mask[self.multi_env_mch_diag] = True
        can_process_any = np.sum(self.process_relation, axis=1) > 0
        self.delete_mask_fea_m = np.tile(~can_process_any[:, :, np.newaxis], (1, 1, self.mch_fea_dim))

    def init_op_mask(self):
        self.op_mask = np.full(shape=(self.number_of_envs, self.max_number_of_ops, 3), fill_value=False, dtype=bool)
        for e in range(self.number_of_envs):
            for j in range(self.env_number_of_jobs[e]):
                first_op = self.job_first_op_id[e, j]
                last_op = self.job_last_op_id[e, j]
                if 0 <= first_op < self.max_number_of_ops:
                    self.op_mask[e, first_op, 0] = True
                if 0 <= last_op < self.max_number_of_ops:
                    self.op_mask[e, last_op, 2] = True
        self.op_mask[self.mask_dummy_node, :] = True
        self.op_mask = self.op_mask.astype(np.float32)

    def update_op_mask(self):
        mask_prev_succ = np.zeros_like(self.op_mask[:, :, 2], dtype=bool)
        mask_next_pred = np.zeros_like(self.op_mask[:, :, 0], dtype=bool)
        deleted_rolled_fwd = np.roll(self.deleted_op_nodes, 1, axis=1)
        deleted_rolled_bwd = np.roll(self.deleted_op_nodes, -1, axis=1)
        deleted_rolled_fwd[:, 0] = False
        deleted_rolled_bwd[:, -1] = False
        deleted_rolled_fwd[self.mask_dummy_node] = False
        deleted_rolled_bwd[self.mask_dummy_node] = False
        mask_prev_succ = deleted_rolled_fwd
        mask_next_pred = deleted_rolled_bwd
        current_mask_bool = self.op_mask.astype(bool)
        new_pred_mask = np.logical_or(current_mask_bool[:, :, 0], mask_next_pred)
        new_succ_mask = np.logical_or(current_mask_bool[:, :, 2], mask_prev_succ)
        self.op_mask[:, :, 0] = new_pred_mask.astype(np.float32)
        self.op_mask[:, :, 2] = new_succ_mask.astype(np.float32)
        deleted_expanded = np.expand_dims(self.deleted_op_nodes, axis=-1)
        self.op_mask = np.where(deleted_expanded, 1.0, self.op_mask)
        self.op_mask[self.mask_dummy_node, :] = 1.0

    def logic_operator(self, x, flagT=True):
        if flagT:
            x = x.transpose(0, 2, 1)
            d1_exp = np.expand_dims(x, 2)
            d2_exp = np.expand_dims(x, 1)
            return np.logical_and(d1_exp, d2_exp).astype(np.float32)
        else:
            d1_exp = np.expand_dims(x, 2)
            d2_exp = np.expand_dims(x, 1)
            return np.logical_and(d1_exp, d2_exp).astype(np.float32)

    def _calculate_estimated_performance(self):
        est_max_lb = np.zeros(self.number_of_envs)
        est_tardiness = np.zeros(self.number_of_envs)
        for e in range(self.number_of_envs):
            ops_in_w_mask_e = self.reschedule_ops_mask[e, :]
            if np.any(ops_in_w_mask_e):
                lb_values_in_w = self.op_ct_lb[e, ops_in_w_mask_e]
                valid_lb_mask = np.isfinite(lb_values_in_w) & (lb_values_in_w > 0)
                if np.any(valid_lb_mask):
                    est_max_lb[e] = np.max(lb_values_in_w[valid_lb_mask])
                else:
                    est_max_lb[e] = self.reschedule_start_time[e] if hasattr(self, 'reschedule_start_time') else 0
            else:
                est_max_lb[e] = self.reschedule_start_time[e] if hasattr(self, 'reschedule_start_time') else 0
            jobs_in_w_mask_e = self.reschedule_jobs_mask[e, :]
            if np.any(jobs_in_w_mask_e):
                job_indices_in_w = np.where(jobs_in_w_mask_e)[0]
                for j in job_indices_in_w:
                    if j < self.env_number_of_jobs[e]:
                        last_op_j = self.job_last_op_id[e, j]
                        if 0 <= last_op_j < self.max_number_of_ops:
                            est_comp_time = self.true_op_ct_lb[e, last_op_j]
                            due_date_j = self.job_due_date[e, j]
                            if np.isfinite(est_comp_time) and est_comp_time > 0 and np.isfinite(due_date_j):
                                tardiness_j = max(0, est_comp_time - due_date_j)
                                est_tardiness[e] += tardiness_j
        return (est_max_lb, est_tardiness)

    def _calculate_avg_utilization(self):
        avg_utilization = np.zeros(self.number_of_envs)
        eps = 1e-06
        for e in range(self.number_of_envs):
            machine_utilizations = []
            last_mch_time_e = self.true_mch_free_time[e]
            for m in range(self.number_of_machines):
                initial_release_m = self.true_initial_mch_available_times[e, m]
                last_mch_time_e_m = last_mch_time_e[m]
                total_time_span = max(0, last_mch_time_e_m - initial_release_m)
                total_busy_time = 0.0
                ops_on_machine_mask = (self.op_machine_assignment[e] == m) & (self.op_scheduled_flag[e] > 0)
                ops_on_machine_indices = np.where(ops_on_machine_mask)[0]
                if len(ops_on_machine_indices) > 0:
                    valid_pt_mask = self.true_op_pt[e, ops_on_machine_indices, m] > 0
                    total_busy_time = np.sum(self.true_op_pt[e, ops_on_machine_indices[valid_pt_mask], m])
                if total_time_span < eps:
                    util_m = 1.0 if total_busy_time > eps else 0.0
                else:
                    util_m = total_busy_time / total_time_span
                    util_m = np.clip(util_m, 0.0, 1.0)
                machine_utilizations.append(util_m)
            if machine_utilizations:
                avg_utilization[e] = np.mean(machine_utilizations)
        return avg_utilization

    def temp_reward_method(self):
        est_r2 = np.zeros(self.number_of_envs)
        for e in range(self.number_of_envs):
            jobs_in_w_mask_e = self.reschedule_jobs_mask[e, :]
            if np.any(jobs_in_w_mask_e):
                job_indices_in_w = np.where(jobs_in_w_mask_e)[0]
                for j in job_indices_in_w:
                    if j < self.env_number_of_jobs[e]:
                        last_op_j = self.job_last_op_id[e, j]
                        if 0 <= last_op_j < self.max_number_of_ops:
                            est_comp_time = self.true_op_ct_lb[e, last_op_j]
                            due_date_j = self.job_due_date[e, j]
                            if np.isfinite(est_comp_time) and est_comp_time > 0 and np.isfinite(due_date_j):
                                r2 = max(0, due_date_j - est_comp_time)
                                est_r2[e] += r2
        return est_r2
