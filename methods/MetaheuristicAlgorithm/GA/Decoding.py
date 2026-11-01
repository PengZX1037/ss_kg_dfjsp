import copy
import numpy as np
import warnings

def decode_chromosome_and_evaluate(os_chromosome: np.ndarray, ms_chromosome: np.ndarray, instance_info: dict,
                                   initial_mch_available_times: np.ndarray, job_due_dates: np.ndarray=None, job_ready_times: np.ndarray=None):
    """Decode OS/MS chromosomes using active scheduling and calculate the objective metrics."""
    job_count = instance_info['num_jobs']
    machine_count = instance_info['num_machines']
    operation_count = instance_info['num_operations']
    operation_metadata = instance_info['op_details']
    machine_options_by_operation = instance_info['op_machine_options']
    operation_indices_by_job = instance_info['job_op_indices']
    next_ready_time_by_job = np.zeros(job_count)
    if job_ready_times is not None:
        next_ready_time_by_job = copy.deepcopy(job_ready_times)
    operation_finish_times = np.full(operation_count, -1.0)
    machine_assignments = np.full(operation_count, -1, dtype=int)
    machine_available_slots = [[(initial_mch_available_times[m], float('inf'))] for m in range(machine_count)]
    next_operation_by_job = np.zeros(job_count, dtype=int)
    scheduled_operations = []
    for job_id in os_chromosome:
        position_in_job = next_operation_by_job[job_id]
        if position_in_job >= len(operation_indices_by_job[job_id]):
            warnings.warn(f'Decoding Error: Job {job_id} has more occurrences in OS than its actual number of ops.')
            continue
        operation_index = operation_indices_by_job[job_id][position_in_job]
        machine_choice_index = ms_chromosome[operation_index]
        operation_machine_options = machine_options_by_operation[operation_index]
        if not operation_machine_options:
            warnings.warn(f'Decoding Error: Op {operation_index} has no machine options. Cannot schedule.')
            next_operation_by_job[job_id] += 1
            continue
        if not 0 <= machine_choice_index < len(operation_machine_options):
            warnings.warn(f'Decoding Error: Invalid MS index {machine_choice_index} for Op {operation_index} (only {len(operation_machine_options)} options). Using index 0.')
            machine_choice_index = 0
        assigned_machine, processing_time = operation_machine_options[machine_choice_index]
        precedence_ready_time = next_ready_time_by_job[job_id]
        earliest_machine_start = float('inf')
        selected_slot_index = -1
        available_slots = machine_available_slots[assigned_machine]
        for i, (slot_start, slot_end) in enumerate(available_slots):
            feasible_start = max(precedence_ready_time, slot_start)
            if feasible_start + processing_time <= slot_end:
                earliest_machine_start = feasible_start
                selected_slot_index = i
                break
        if selected_slot_index == -1:
            earliest_machine_start = max(precedence_ready_time, available_slots[-1][0] if available_slots else 0)
        scheduled_start = earliest_machine_start
        scheduled_end = scheduled_start + processing_time
        operation_finish_times[operation_index] = scheduled_end
        machine_assignments[operation_index] = assigned_machine
        next_ready_time_by_job[job_id] = scheduled_end
        next_operation_by_job[job_id] += 1
        if selected_slot_index != -1:
            previous_slot_start, previous_slot_end = available_slots[selected_slot_index]
            del available_slots[selected_slot_index]
            if scheduled_start > previous_slot_start:
                available_slots.insert(selected_slot_index, (previous_slot_start, scheduled_start))
                selected_slot_index += 1
            if scheduled_end < previous_slot_end:
                available_slots.insert(selected_slot_index, (scheduled_end, previous_slot_end))
        else:
            machine_available_slots[assigned_machine].append((scheduled_start, scheduled_end))
        scheduled_operations.append({'job_id': job_id, 'op_id': operation_index, 'mch_id': assigned_machine, 'start_time': scheduled_start, 'end_time': scheduled_end})
    schedule_makespan = 0.0
    if np.any(operation_finish_times > -1):
        schedule_makespan = np.max(operation_finish_times[operation_finish_times > -1])
    accumulated_tardiness = 0.0
    if job_due_dates is not None:
        for j in range(job_count):
            last_job_operation = operation_indices_by_job[j][-1] if operation_indices_by_job[j] else -1
            if last_job_operation != -1 and operation_finish_times[last_job_operation] > -1:
                job_completion_time = operation_finish_times[last_job_operation]
                job_deadline = job_due_dates[j]
                accumulated_tardiness += max(0, job_completion_time - job_deadline)
    makespan_weight = 0.5
    tardiness_weight = 0.5
    weighted_objective = makespan_weight * schedule_makespan + tardiness_weight * accumulated_tardiness
    fitness_score = 1.0 / (weighted_objective + 1e-06)
    scheduled_operation_mask = np.where(operation_finish_times > -1, 1, 0)
    gantt_payload = {'op_status': scheduled_operation_mask, 'true_op_ct': operation_finish_times, 'true_op_pt': instance_info.get('op_pt_matrix_original', None),
                     'op_to_job_map': np.array([d['job_id'] for d in operation_metadata]), 'op_machine_assignment': machine_assignments, 'env_num_ops': operation_count, 'num_machines': machine_count}
    return {'schedule': scheduled_operations, 'makespan': schedule_makespan, 'total_tardiness': accumulated_tardiness, 'fitness': fitness_score, 'gantt_data': gantt_payload}
