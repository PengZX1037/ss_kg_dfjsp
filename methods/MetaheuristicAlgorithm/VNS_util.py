import numpy as np
import random
import copy
import warnings
from typing import Tuple, List, Dict, Any
from collections import deque
from MetaheuristicAlgorithm.GA.Decoding import decode_chromosome_and_evaluate
from MetaheuristicAlgorithm.N5_util import neighborhood_N5_critical_block_swap

def swap_mutation_os(os_chromosome):
    updated_sequence = np.copy(os_chromosome)
    if len(updated_sequence) > 1:
        first_position, second_position = random.sample(range(len(updated_sequence)), 2)
    updated_sequence[first_position], updated_sequence[second_position] = (updated_sequence[second_position], updated_sequence[first_position])
    return updated_sequence

def N5_os(os_chromosome, ms_chromosome, decoded_solution, instance_info, job_due_dates, initial_mch_available_times,
          job_ready_times, f_local_best_mk, f_local_best_td):
    local_operation_sequence = os_chromosome
    local_machine_choices = ms_chromosome
    has_improved = False
    critical_operation_ids, _ = get_critical_and_tardy_info(decoded_solution, instance_info, job_due_dates)
    neighbor_sequences = neighborhood_N5_critical_block_swap(os_chromosome, decoded_solution, instance_info, critical_operation_ids)
    for candidate_sequence in neighbor_sequences:
        candidate_evaluation = decode_chromosome_and_evaluate(candidate_sequence, ms_chromosome, instance_info, initial_mch_available_times, job_due_dates, job_ready_times)
        candidate_makespan, candidate_tardiness = (candidate_evaluation['makespan'], candidate_evaluation['total_tardiness'])
        if candidate_makespan <= f_local_best_mk and candidate_tardiness < f_local_best_td or (candidate_makespan < f_local_best_mk and candidate_tardiness <= f_local_best_td):
            has_improved = True
            local_operation_sequence, local_machine_choices = (candidate_sequence, ms_chromosome)
            f_local_best_mk, f_local_best_td = (candidate_makespan, candidate_tardiness)
    return (local_operation_sequence, local_machine_choices, f_local_best_mk, f_local_best_td, has_improved)

def change_critcalOpe_machine(os_chromosome, ms_chromosome, decoded_solution, instance_info, job_due_dates,
                              initial_mch_available_times, job_ready_times, f_local_best_mk, f_local_best_td):
    local_operation_sequence = os_chromosome
    local_machine_choices = ms_chromosome
    machine_choices_by_operation = instance_info['op_machine_options']
    critical_operation_ids, _ = get_critical_and_tardy_info(decoded_solution, instance_info, job_due_dates)
    has_improved = False
    for op_idx in critical_operation_ids:
        if len(machine_choices_by_operation[op_idx]) == 1:
            continue
        for i in range(len(machine_choices_by_operation[op_idx])):
            if i == ms_chromosome[op_idx]:
                continue
            candidate_machine_choices = np.copy(ms_chromosome)
            candidate_machine_choices[op_idx] = i
            candidate_evaluation = decode_chromosome_and_evaluate(os_chromosome, candidate_machine_choices, instance_info, initial_mch_available_times, job_due_dates, job_ready_times)
            candidate_makespan, candidate_tardiness = (candidate_evaluation['makespan'], candidate_evaluation['total_tardiness'])
            if candidate_makespan <= f_local_best_mk and candidate_tardiness < f_local_best_td or (candidate_makespan < f_local_best_mk and candidate_tardiness <= f_local_best_td):
                has_improved = True
                local_operation_sequence, local_machine_choices = (os_chromosome, candidate_machine_choices)
                f_local_best_mk, f_local_best_td = (candidate_makespan, candidate_tardiness)
    return (local_operation_sequence, local_machine_choices, f_local_best_mk, f_local_best_td, has_improved)

def single_point_mutation_ms(os_chromosome, ms_chromosome: np.ndarray, instance_info: dict, decoded_solution, initial_mch_available_times,
                             job_ready_times, job_due_dates, f_local_best_mk, f_local_best_td):
    _, tardy_operation_ids = get_critical_and_tardy_info(decoded_solution, instance_info, job_due_dates)
    machine_choices_by_operation = instance_info['op_machine_options']
    local_operation_sequence = os_chromosome
    local_machine_choices = ms_chromosome
    has_improved = False
    for op_idx in tardy_operation_ids:
        if len(machine_choices_by_operation[op_idx]) == 1:
            continue
        for i in range(len(machine_choices_by_operation[op_idx])):
            if i == ms_chromosome[op_idx]:
                continue
            candidate_machine_choices = np.copy(ms_chromosome)
            candidate_machine_choices[op_idx] = i
            candidate_evaluation = decode_chromosome_and_evaluate(os_chromosome, candidate_machine_choices, instance_info,
                                                                  initial_mch_available_times, job_due_dates, job_ready_times)
            candidate_makespan, candidate_tardiness = (candidate_evaluation['makespan'], candidate_evaluation['total_tardiness'])
            if candidate_makespan <= f_local_best_mk and candidate_tardiness < f_local_best_td or (candidate_makespan < f_local_best_mk and candidate_tardiness <= f_local_best_td):
                has_improved = True
                local_operation_sequence, local_machine_choices = (os_chromosome, candidate_machine_choices)
                f_local_best_mk, f_local_best_td = (candidate_makespan, candidate_tardiness)
    return (local_operation_sequence, local_machine_choices, f_local_best_mk, f_local_best_td, has_improved)

def get_critical_and_tardy_info(decoded_solution: Dict, instance_info: Dict, job_due_dates: np.ndarray) -> Tuple[List[int], List[int]]:
    """Identify one critical path and operations belonging to tardy jobs."""
    makespan = decoded_solution['makespan']
    scheduled_operations = decoded_solution.get('schedule', [])
    job_count = instance_info['num_jobs']
    operation_details = instance_info['op_details']
    schedule_by_operation = {}
    operations_by_machine: Dict[int, List[int]] = {m: [] for m in range(instance_info['num_machines'])}
    operation_details_by_id = {idx: op for idx, op in enumerate(operation_details)}
    for item in scheduled_operations:
        op_id = item['op_id']
        if op_id in operation_details_by_id:
            schedule_by_operation[op_id] = {'job_id': item['job_id'], 'mch_id': item['mch_id'], 'start': item['start_time'],
                                            'end': item['end_time'], 'pt': item.get('pt', item['end_time'] - item['start_time']),
                                            'job_pred': None, 'job_succ': None, 'mch_pred': None, 'mch_succ': None}
            operations_by_machine[item['mch_id']].append(op_id)
    for mch_id in operations_by_machine:
        operations_by_machine[mch_id].sort(key=lambda op_id: schedule_by_operation[op_id]['start'])
    for op_id, info in schedule_by_operation.items():
        job_id = info['job_id']
        op_id_in_job = instance_info['job_op_indices'][job_id].index(op_id)
        if op_id_in_job > 0:
            info['job_pred'] = op_id - 1
        if op_id_in_job < len(instance_info['job_op_indices'][job_id]) - 1:
            info['job_succ'] = op_id + 1
        machine_operation_sequence = operations_by_machine[info['mch_id']]
        try:
            machine_schedule_position = machine_operation_sequence.index(op_id)
            if machine_schedule_position > 0:
                info['mch_pred'] = machine_operation_sequence[machine_schedule_position - 1]
            if machine_schedule_position < len(machine_operation_sequence) - 1:
                info['mch_succ'] = machine_operation_sequence[machine_schedule_position + 1]
        except ValueError:
            warnings.warn(f'Op {op_id} not found in its own machine schedule. This should not happen.')
    critical_operation_path = []
    pending_critical_operations = deque()
    time_tolerance = 1e-05
    for op_id, info in schedule_by_operation.items():
        if abs(info['end'] - makespan) < time_tolerance:
            critical_operation_path.append(op_id)
            pending_critical_operations.append(op_id)
    while pending_critical_operations:
        current_operation_id = pending_critical_operations.popleft()
        if current_operation_id not in schedule_by_operation:
            continue
        current_operation_info = schedule_by_operation[current_operation_id]
        preceding_job_operation = current_operation_info['job_pred']
        if preceding_job_operation is not None:
            preceding_job_details = schedule_by_operation[preceding_job_operation]
            if abs(preceding_job_details['end'] - current_operation_info['start']) < time_tolerance:
                if preceding_job_operation not in critical_operation_path:
                    critical_operation_path.insert(0, preceding_job_operation)
                    pending_critical_operations.append(preceding_job_operation)
                    continue
        preceding_machine_operation = current_operation_info['mch_pred']
        if preceding_machine_operation is not None:
            preceding_machine_details = schedule_by_operation[preceding_machine_operation]
            if abs(preceding_machine_details['end'] - current_operation_info['start']) < time_tolerance:
                if preceding_machine_operation not in critical_operation_path:
                    critical_operation_path.insert(0, preceding_machine_operation)
                    pending_critical_operations.append(preceding_machine_operation)
    late_job_operation_ids = []
    actual_job_completion_times = np.full(job_count, -1.0)
    for j_idx in range(job_count):
        job_operation_indices = instance_info['job_op_indices'][j_idx]
        if job_operation_indices:
            last_job_operation = job_operation_indices[-1]
            if last_job_operation in schedule_by_operation:
                actual_job_completion_times[j_idx] = schedule_by_operation[last_job_operation]['end']
    if job_due_dates is not None:
        for j_idx in range(job_count):
            if actual_job_completion_times[j_idx] > -1 and j_idx < len(job_due_dates) and (actual_job_completion_times[j_idx] > job_due_dates[j_idx]):
                late_job_operation_ids.extend(instance_info['job_op_indices'][j_idx])
    return (critical_operation_path, list(set(late_job_operation_ids)))
