import warnings
import numpy as np
import random
import copy

def generate_initial_population(pop_size: int, instance_info: dict):
    """Create random operation-order and machine-choice chromosomes."""
    initial_population = []
    job_count = instance_info['num_jobs']
    operation_count = instance_info['num_operations']
    operation_indices_by_job = instance_info['job_op_indices']
    machine_choices_by_operation = instance_info['op_machine_options']
    if operation_count == 0:
        return []
    for _ in range(pop_size):
        os_chromosome = []
        op_counters = [0] * job_count
        shuffled_job_pool = []
        for j in range(job_count):
            for position_in_job in range(len(operation_indices_by_job[j])):
                shuffled_job_pool.append(j)
        random.shuffle(shuffled_job_pool)
        os_chromosome = np.array(shuffled_job_pool, dtype=int)
        ms_chromosome = np.zeros(operation_count, dtype=int)
        for operation_index in range(operation_count):
            machine_options = machine_choices_by_operation[operation_index]
            if not machine_options:
                warnings.warn(f'Initialization: Op {operation_index} has no machine options. Assigning index 0 (may fail later).')
                ms_chromosome[operation_index] = 0
            else:
                ms_chromosome[operation_index] = random.randint(0, len(machine_options) - 1)
        initial_population.append((os_chromosome, ms_chromosome))
    return initial_population

def precedence_preserving_order_crossover(parent1_os: np.ndarray, parent2_os: np.ndarray, num_jobs: int):
    """Combine operation sequences while respecting job occurrence counts."""
    chromosome_length = len(parent1_os)
    child1_os = np.full(chromosome_length, -1, dtype=int)
    child2_os = np.full(chromosome_length, -1, dtype=int)
    shuffled_job_ids = list(range(num_jobs))
    random.shuffle(shuffled_job_ids)
    job_group_boundary = random.randint(1, num_jobs - 1) if num_jobs > 1 else 1
    first_job_group = set(shuffled_job_ids[:job_group_boundary])
    second_job_group = set(shuffled_job_ids[job_group_boundary:])
    second_parent_offset = 0
    for i in range(chromosome_length):
        job_id = parent1_os[i]
        if job_id in first_job_group:
            child1_os[i] = job_id
        else:
            while parent2_os[second_parent_offset] in first_job_group:
                second_parent_offset += 1
            child1_os[i] = parent2_os[second_parent_offset]
            second_parent_offset += 1
    first_parent_offset = 0
    for i in range(chromosome_length):
        job_id = parent2_os[i]
        if job_id in second_job_group:
            child2_os[i] = job_id
        else:
            while parent1_os[first_parent_offset] in second_job_group:
                first_parent_offset += 1
            child2_os[i] = parent1_os[first_parent_offset]
            first_parent_offset += 1
    return (child1_os, child2_os)

def two_point_crossover(parent1_ms: np.ndarray, parent2_ms: np.ndarray):
    chromosome_length = len(parent1_ms)
    first_offspring = np.copy(parent1_ms)
    second_offspring = np.copy(parent2_ms)
    if chromosome_length <= 1:
        return (first_offspring, second_offspring)
    first_cut_position = random.randint(0, chromosome_length - 1)
    second_cut_position = random.randint(0, chromosome_length - 1)
    if first_cut_position == second_cut_position:
        second_cut_position = (first_cut_position + 1) % chromosome_length
    if first_cut_position > second_cut_position:
        first_cut_position, second_cut_position = (second_cut_position, first_cut_position)
    first_offspring[first_cut_position:second_cut_position] = parent2_ms[first_cut_position:second_cut_position]
    second_offspring[first_cut_position:second_cut_position] = parent1_ms[first_cut_position:second_cut_position]
    return (first_offspring, second_offspring)

def swap_mutation_os(os_chromosome: np.ndarray):
    mutated_sequence = np.copy(os_chromosome)
    chromosome_length = len(mutated_sequence)
    if chromosome_length < 2:
        return mutated_sequence
    attempt_count = 0
    attempt_limit = chromosome_length
    while attempt_count < attempt_limit:
        attempt_count += 1
        first_position = random.randint(0, chromosome_length - 2)
        second_position = first_position + 1
        if mutated_sequence[first_position] != mutated_sequence[second_position]:
            mutated_sequence[first_position], mutated_sequence[second_position] = (mutated_sequence[second_position], mutated_sequence[first_position])
            return mutated_sequence
    return mutated_sequence

def single_point_mutation_ms(ms_chromosome: np.ndarray, instance_info: dict):
    mutated_machine_choices = np.copy(ms_chromosome)
    operation_count = instance_info['num_operations']
    machine_choices_by_operation = instance_info['op_machine_options']
    if operation_count == 0:
        return mutated_machine_choices
    chosen_operation = random.randint(0, operation_count - 1)
    machine_options = machine_choices_by_operation[chosen_operation]
    available_choice_count = len(machine_options)
    if available_choice_count <= 1:
        return mutated_machine_choices
    previous_machine_choice = mutated_machine_choices[chosen_operation]
    replacement_machine_choice = previous_machine_choice
    while replacement_machine_choice == previous_machine_choice:
        replacement_machine_choice = random.randint(0, available_choice_count - 1)
    mutated_machine_choices[chosen_operation] = replacement_machine_choice
    return mutated_machine_choices

def tournament_selection(population: list, fitnesses: list, tournament_size: int):
    population_count = len(population)
    if population_count == 0:
        return None
    tournament_contenders = random.sample(range(population_count), k=min(tournament_size, population_count))
    winning_index = -1
    winning_fitness = -float('inf')
    for idx in tournament_contenders:
        if fitnesses[idx] > winning_fitness:
            winning_fitness = fitnesses[idx]
            winning_index = idx
    return population[winning_index]

def selection_with_elitism(population: list, fitnesses: list, pop_size: int, elite_size: int, tournament_size: int):
    """Retain elite solutions and fill the population by tournament selection."""
    selected_population = []
    available_individuals = len(population)
    elite_size = min(elite_size, available_individuals)
    if elite_size > 0:
        ranked_population_indices = np.argsort(fitnesses)[::-1]
        for i in range(elite_size):
            elite_individual_index = ranked_population_indices[i]
            selected_population.append(population[elite_individual_index])
    remaining_slots = pop_size - elite_size
    for _ in range(remaining_slots):
        selected_parent = tournament_selection(population, fitnesses, tournament_size)
        if selected_parent is not None:
            selected_population.append(selected_parent)
        elif population:
            selected_population.append(random.choice(population))
    while len(selected_population) < pop_size and population:
        warnings.warn('Filling remaining population randomly due to selection issues.')
        selected_population.append(random.choice(population))
    return selected_population
