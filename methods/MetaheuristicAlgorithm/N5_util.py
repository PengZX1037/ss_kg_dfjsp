import numpy as np
import random
import copy
import warnings
from typing import Tuple, List, Dict, Any
from collections import deque

def neighborhood_N5_critical_block_swap(current_sol_os: np.ndarray, decoded_solution_current: Dict, instance_info: Dict, critical_ops_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sequence neighbors by swapping operations in critical blocks."""
    scheduled_operations = decoded_solution_current.get('schedule', [])
    neighbor_sequences = []
    critical_operation_records = {}
    for item in scheduled_operations:
        op_id = item['op_id']
        if op_id in critical_ops_indices:
            critical_operation_records[op_id] = item
    critical_machine_blocks: List[Tuple[int, List[int]]] = []
    previous_machine_id = critical_operation_records[critical_ops_indices[0]]['mch_id']
    current_critical_block = []
    for i in range(len(critical_operation_records)):
        if critical_operation_records[critical_ops_indices[i]]['mch_id'] == previous_machine_id:
            current_critical_block.append(critical_operation_records[critical_ops_indices[i]]['op_id'])
        elif current_critical_block:
            critical_machine_blocks.append((previous_machine_id, list(current_critical_block)))
            previous_machine_id = critical_operation_records[critical_ops_indices[i]]['mch_id']
            current_critical_block = []
            current_critical_block.append(critical_operation_records[critical_ops_indices[i]]['op_id'])
    if current_critical_block:
        critical_machine_blocks.append((previous_machine_id, list(current_critical_block)))
    if not critical_machine_blocks:
        return neighbor_sequences
    block_index = 0
    for selected_machine, block_operation_ids in critical_machine_blocks:
        block_size = len(block_operation_ids)
        if block_size < 2:
            block_index += 1
            continue
        is_first_block = block_index == 0
        is_last_block = block_index == len(critical_machine_blocks) - 1
        exchange_pairs: List[Tuple[int, int]] = []
        if is_first_block and (not is_last_block):
            exchange_pairs.append((block_operation_ids[-2], block_operation_ids[-1]))
        elif is_last_block and (not is_first_block):
            exchange_pairs.append((block_operation_ids[0], block_operation_ids[1]))
        elif not is_first_block and (not is_last_block):
            exchange_pairs.append((block_operation_ids[0], block_operation_ids[1]))
            if block_size >= 4:
                exchange_pairs.append((block_operation_ids[-2], block_operation_ids[-1]))
            elif block_size == 3:
                pass
            elif block_size == 2:
                pass
        elif is_first_block and is_last_block:
            if block_size >= 2:
                exchange_pairs.append((block_operation_ids[0], block_operation_ids[1]))
        trial_operation_sequence = np.copy(current_sol_os)
        for first_operation_id, second_operation_id in exchange_pairs:
            first_sequence_position = find_os_positions(trial_operation_sequence, first_operation_id, instance_info)
            second_sequence_position = find_os_positions(trial_operation_sequence, second_operation_id, instance_info)
            if first_sequence_position != -1 and second_sequence_position != -1 and (first_sequence_position != second_sequence_position):
                trial_operation_sequence[first_sequence_position], trial_operation_sequence[second_sequence_position] = (trial_operation_sequence[second_sequence_position], trial_operation_sequence[first_sequence_position])
        neighbor_sequences.append(trial_operation_sequence)
    return neighbor_sequences

def find_os_positions(os_chromosome: np.ndarray, op_global_idx: int, instance_info: Dict) -> int:
    operation_details = instance_info['op_details'][op_global_idx]
    target_job = operation_details['new_job_id']
    target_job_operation_position = instance_info['job_op_indices'][target_job].index(op_global_idx)
    job_occurrence_count = 0
    for i, sequence_job_id in enumerate(os_chromosome):
        if sequence_job_id == target_job:
            if job_occurrence_count == target_job_operation_position:
                return i
            job_occurrence_count += 1
    return -1
