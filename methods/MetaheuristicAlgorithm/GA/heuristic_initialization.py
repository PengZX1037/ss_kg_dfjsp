import numpy as np
import random
import warnings

def generate_ms_spt(instance_info: dict):
    operation_count = instance_info['num_operations']
    machine_choices_by_operation = instance_info['op_machine_options']
    ms_chromosome = np.zeros(operation_count, dtype=int)
    for operation_index in range(operation_count):
        machine_options = machine_choices_by_operation[operation_index]
        if not machine_options:
            ms_chromosome[operation_index] = 0
            continue
        minimum_processing_time = float('inf')
        fastest_machine_choice = 0
        for choice_index, (machine_id, processing_duration) in enumerate(machine_options):
            if processing_duration < minimum_processing_time:
                minimum_processing_time = processing_duration
                fastest_machine_choice = choice_index
        ms_chromosome[operation_index] = fastest_machine_choice
    return ms_chromosome

def generate_os_based_on_job_priority(instance_info: dict, job_priorities: list):
    job_count = instance_info['num_jobs']
    operation_count = instance_info['num_operations']
    operation_indices_by_job = instance_info['job_op_indices']
    os_chromosome = np.zeros(operation_count, dtype=int)
    next_operation_offsets = np.zeros(job_count, dtype=int)
    sequence_cursor = 0
    while sequence_cursor < operation_count:
        found_eligible_operation = False
        for job_id in job_priorities:
            job_operation_cursor = next_operation_offsets[job_id]
            if job_operation_cursor < len(operation_indices_by_job[job_id]):
                os_chromosome[sequence_cursor] = job_id
                next_operation_offsets[job_id] += 1
                sequence_cursor += 1
                found_eligible_operation = True
                break
        if not found_eligible_operation and sequence_cursor < operation_count:
            warnings.warn('OS generation error: Could not find next operation.')
            jobs_with_pending_ops = [j for j in range(job_count) if next_operation_offsets[j] < len(operation_indices_by_job[j])]
            if not jobs_with_pending_ops:
                break
            job_id = random.choice(jobs_with_pending_ops)
            os_chromosome[sequence_cursor] = job_id
            next_operation_offsets[job_id] += 1
            sequence_cursor += 1
    return os_chromosome

def generate_heuristic_chromosome(instance_info: dict, rule: str):
    """Initialize chromosomes with SPT, LPT, or MOR job priorities."""
    job_count = instance_info['num_jobs']
    operation_count = instance_info['num_operations']
    machine_choices_by_operation = instance_info['op_machine_options']
    operation_indices_by_job = instance_info['job_op_indices']
    ms_chromosome = generate_ms_spt(instance_info)
    priority_order = list(range(job_count))
    if rule == 'SPT':
        shortest_job_processing_load = np.zeros(job_count)
        for j in range(job_count):
            for op_global_idx in operation_indices_by_job[j]:
                machine_options = machine_choices_by_operation[op_global_idx]
                if machine_options:
                    shortest_job_processing_load[j] += min((pt for m_id, pt in machine_options))
        priority_order = np.argsort(shortest_job_processing_load).tolist()
    elif rule == 'LPT':
        longest_job_processing_load = np.zeros(job_count)
        for j in range(job_count):
            for op_global_idx in operation_indices_by_job[j]:
                machine_options = machine_choices_by_operation[op_global_idx]
                if machine_options:
                    longest_job_processing_load[j] += max((pt for m_id, pt in machine_options))
        priority_order = np.argsort(longest_job_processing_load)[::-1].tolist()
    elif rule == 'MOR':
        operations_per_job = np.array([len(ops) for ops in operation_indices_by_job])
        priority_order = np.argsort(operations_per_job)[::-1].tolist()
    else:
        warnings.warn(f"Unknown heuristic rule '{rule}' for OS generation. Using default job order.")
    os_chromosome = generate_os_based_on_job_priority(instance_info, priority_order)
    return (os_chromosome, ms_chromosome)

def harbid_generate_initial_population(pop_size: int, instance_info: dict, num_heuristic_seeds: int=6):
    """Combine heuristic seeds with random chromosomes."""
    initial_population = []
    job_count = instance_info['num_jobs']
    operation_count = instance_info['num_operations']
    operation_indices_by_job = instance_info['job_op_indices']
    machine_choices_by_operation = instance_info['op_machine_options']
    if operation_count == 0:
        return []
    seeding_rules = ['SPT', 'LPT', 'MOR']
    rule_count = len(seeding_rules)
    seeds_per_heuristic = max(1, num_heuristic_seeds // rule_count)
    heuristic_seed_count = 0
    for rule in seeding_rules:
        for _ in range(seeds_per_heuristic):
            if heuristic_seed_count < num_heuristic_seeds and heuristic_seed_count < pop_size:
                try:
                    seed_operation_sequence, seed_machine_selection = generate_heuristic_chromosome(instance_info, rule)
                    initial_population.append((seed_operation_sequence, seed_machine_selection))
                    heuristic_seed_count += 1
                except Exception as e:
                    warnings.warn(f'Error generating heuristic seed with rule {rule}: {e}')
    initial_population = initial_population[:min(pop_size, len(initial_population))]
    random_population_count = pop_size - len(initial_population)
    for _ in range(random_population_count):
        os_chromosome = []
        shuffled_job_pool = []
        for j in range(job_count):
            operation_count_for_job = len(operation_indices_by_job[j])
            if operation_count_for_job > 0:
                shuffled_job_pool.extend([j] * operation_count_for_job)
        random.shuffle(shuffled_job_pool)
        os_chromosome = np.array(shuffled_job_pool, dtype=int)
        ms_chromosome = np.zeros(operation_count, dtype=int)
        for operation_index in range(operation_count):
            machine_options = machine_choices_by_operation[operation_index]
            if not machine_options:
                ms_chromosome[operation_index] = 0
            else:
                ms_chromosome[operation_index] = random.randint(0, len(machine_options) - 1)
        initial_population.append((os_chromosome, ms_chromosome))
    return initial_population
