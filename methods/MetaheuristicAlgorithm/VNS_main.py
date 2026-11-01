import os
import numpy as np
import random
import copy
import warnings
from typing import Tuple, List, Dict, Any
import draw_util
from MetaheuristicAlgorithm.GA.Decoding import decode_chromosome_and_evaluate
from MetaheuristicAlgorithm.VNS_util import swap_mutation_os, single_point_mutation_ms, change_critcalOpe_machine, N5_os

def neighborhood_tardy_op_move(current_sol: Tuple[np.ndarray, np.ndarray], instance_info: Dict, tardy_job_ops: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    candidate_operation_sequence, candidate_machine_choices = copy.deepcopy(current_sol)
    if not tardy_job_ops or len(candidate_operation_sequence) < 2:
        return (candidate_operation_sequence, candidate_machine_choices)
    for _ in range(min(3, len(tardy_job_ops))):
        candidate_operation_sequence = swap_mutation_os(candidate_operation_sequence)
    return (candidate_operation_sequence, candidate_machine_choices)

def run_vns(initial_sol: Tuple[np.ndarray, np.ndarray], initial_sol_makespan: float, initial_sol_tardiness: float, instance_info: Dict,
            initial_mch_available_times: np.ndarray, job_ready_times: np.ndarray, job_due_dates: np.ndarray, max_iter_vns: int=1,
            k_max: int=3, max_iter_local_search: int=10) -> Tuple[Tuple[np.ndarray, np.ndarray], float, float]:
    """Explore scheduling neighborhoods and restore original operation identifiers."""
    current_operation_sequence, current_machine_choices, scheduled_operation_ids = copy.deepcopy(initial_sol)
    window_machine_options = [v for i, v in enumerate(instance_info['op_machine_options']) if i in scheduled_operation_ids]
    window_operation_details = [v for i, v in enumerate(instance_info['op_details']) if i in scheduled_operation_ids]
    next_window_operation_index = 0
    window_job_index = 0
    original_job_by_window_job = {}
    original_operation_by_window_operation = {}
    window_operations_by_job = {}
    window_due_dates = []
    for source_job_index, job_operation_list in enumerate(instance_info['job_op_indices']):
        selected_job_operations = [item for item in job_operation_list if item in scheduled_operation_ids]
        if selected_job_operations:
            mapped_operation_ids = list(range(next_window_operation_index, next_window_operation_index + len(selected_job_operations)))
            window_operations_by_job[window_job_index] = mapped_operation_ids
            for mapped_operation_id in mapped_operation_ids:
                window_operation_details[mapped_operation_id]['new_job_id'] = window_job_index
            original_job_by_window_job[window_job_index] = source_job_index
            for i in range(len(selected_job_operations)):
                original_operation_by_window_operation[mapped_operation_ids[i]] = selected_job_operations[i]
            if source_job_index != window_job_index:
                current_operation_sequence = [window_job_index if x == source_job_index else x for x in current_operation_sequence]
            next_window_operation_index += len(selected_job_operations)
            window_job_index += 1
            window_due_dates.append(job_due_dates[source_job_index])
    window_job_ready_times = [0] * (window_job_index + 1)
    for window_job_index, source_job_index in original_job_by_window_job.items():
        window_job_ready_times[window_job_index] = job_ready_times[source_job_index]
    job_ready_times = np.array(window_job_ready_times)
    instance_info['op_machine_options'] = window_machine_options
    instance_info['op_details'] = window_operation_details
    instance_info['job_op_indices'] = window_operations_by_job
    instance_info['num_jobs'] = len(window_due_dates)
    instance_info['num_operations'] = next_window_operation_index
    job_due_dates = window_due_dates
    window_processing_times = [[] for _ in range(len(instance_info['op_machine_options']))]
    for op_idx, machine_options_for_operation in enumerate(instance_info['op_machine_options']):
        processing_time_row = [0] * instance_info['num_machines']
        for item in machine_options_for_operation:
            processing_time_row[item[0]] = item[1]
        window_processing_times[op_idx].extend(processing_time_row)
    current_makespan = initial_sol_makespan
    current_tardiness = initial_sol_tardiness
    best_operation_sequence, best_machine_choices = (current_operation_sequence, current_machine_choices)
    best_makespan, best_tardiness = (current_makespan, current_tardiness)
    vns_iteration = 0
    while vns_iteration < max_iter_vns:
        vns_iteration += 1
        k = 0
        while k < k_max:
            current_evaluation = decode_chromosome_and_evaluate(best_operation_sequence, best_machine_choices, instance_info, initial_mch_available_times, job_due_dates, job_ready_times)
            local_best_makespan, local_best_tardiness = (current_evaluation['makespan'], current_evaluation['total_tardiness'])
            has_improved = False
            if k == 0:
                local_best_operation_sequence, local_best_machine_choices, local_best_makespan, local_best_tardiness, has_improved = (
                    change_critcalOpe_machine(best_operation_sequence, best_machine_choices, current_evaluation, instance_info, job_due_dates, initial_mch_available_times, job_ready_times, best_makespan, best_tardiness))
            elif k == 1:
                local_best_operation_sequence, local_best_machine_choices, local_best_makespan, local_best_tardiness, has_improved =\
                    N5_os(best_operation_sequence, best_machine_choices, current_evaluation, instance_info, job_due_dates, initial_mch_available_times, job_ready_times, best_makespan, best_tardiness)
            elif k == 2:
                local_best_operation_sequence, local_best_machine_choices, local_best_makespan, local_best_tardiness, has_improved =\
                    single_point_mutation_ms(best_operation_sequence, best_machine_choices, instance_info, current_evaluation, initial_mch_available_times,
                                             job_ready_times, job_due_dates, best_makespan, best_tardiness)
            if has_improved:
                best_operation_sequence, best_machine_choices = (local_best_operation_sequence, local_best_machine_choices)
                best_makespan, best_tardiness = (local_best_makespan, local_best_tardiness)
                k = 0
            else:
                k += 1
    final_evaluation = decode_chromosome_and_evaluate(best_operation_sequence, best_machine_choices, instance_info,
                                                      initial_mch_available_times, job_due_dates, job_ready_times)
    final_schedule = final_evaluation.get('schedule', [])
    final_gantt_payload = final_evaluation.get('gantt_data', [])
    final_gantt_payload['true_op_pt'] = np.array(window_processing_times)
    restored_operation_sequence = []
    restored_machine_choices = []
    for i in range(len(best_operation_sequence)):
        window_job_identifier = best_operation_sequence[i]
        restored_operation_sequence.append(original_job_by_window_job[window_job_identifier])
        restored_machine_choices.append(window_machine_options[best_machine_choices[i]][0])
        final_schedule[i]['job_id'] = original_job_by_window_job[final_schedule[i]['job_id']]
        final_schedule[i]['op_id'] = original_operation_by_window_operation[final_schedule[i]['op_id']]
    return ((restored_operation_sequence, restored_machine_choices), best_makespan, best_tardiness, final_schedule, final_gantt_payload)
