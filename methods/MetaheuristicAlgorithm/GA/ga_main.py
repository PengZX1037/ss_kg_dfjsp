import numpy as np
import random
import time
import os
import warnings
from MetaheuristicAlgorithm.GA.Decoding import decode_chromosome_and_evaluate
from MetaheuristicAlgorithm.GA.Genetic_Operators import (single_point_mutation_ms, swap_mutation_os, two_point_crossover,
                                                         precedence_preserving_order_crossover, selection_with_elitism, generate_initial_population)
from MetaheuristicAlgorithm.GA.heuristic_initialization import harbid_generate_initial_population
from data_utils import SD2_instance_generator
from draw_util import plot_gantt_chart

def get_instance_info(job_lengths: np.ndarray, op_pt_matrix: np.ndarray):
    """Build operation and machine-compatibility metadata from an FJSP instance."""
    job_count = len(job_lengths)
    if job_count == 0:
        return {'num_jobs': 0, 'num_machines': 0, 'num_operations': 0, 'op_details': [], 'op_machine_options': [], 'job_op_indices': []}
    operation_count, machine_count = op_pt_matrix.shape
    if np.sum(job_lengths) != operation_count:
        raise ValueError('Sum of job_lengths does not match total number of operations.')
    operation_records = []
    machine_choices_by_operation = []
    operation_indices_by_job = [[] for _ in range(job_count)]
    operation_index = 0
    for j in range(job_count):
        num_ops_in_job = job_lengths[j]
        for position_in_job in range(num_ops_in_job):
            operation_records.append({'job_id': j, 'op_id_in_job': position_in_job, 'op_index_global': operation_index})
            eligible_machines = np.where(op_pt_matrix[operation_index] > 0)[0]
            machine_options = []
            if len(eligible_machines) > 0:
                machine_options = [(machine_index, op_pt_matrix[operation_index, machine_index]) for machine_index in eligible_machines]
            if not machine_options:
                warnings.warn(f'Operation {operation_index} (Job {j}, Op {position_in_job}) has no compatible machines!')
            machine_choices_by_operation.append(machine_options)
            operation_indices_by_job[j].append(operation_index)
            operation_index += 1
    return {'num_jobs': job_count, 'num_machines': machine_count, 'num_operations': operation_count, 'op_details': operation_records,
            'op_machine_options': machine_choices_by_operation, 'job_op_indices': operation_indices_by_job}

def run_genetic_algorithm(job_lengths: np.ndarray, op_pt_matrix: np.ndarray, initial_mch_available_times: np.ndarray,
                          job_due_dates: np.ndarray=None, pop_size: int=50, num_generations: int=100, time_limit: float=60.0,
                          elite_size: int=5, tournament_size: int=3, crossover_rate: float=0.8, mutation_rate_os: float=0.1,
                          mutation_rate_ms: float=0.1, plot_gantt: bool=False, output_dir: str='./ga_results', job_ready_times: np.ndarray=None, op_pt_norm: np.ndarray=None):
    """Evolve OS/MS chromosomes and return the best decoded schedule."""
    started_at = time.time()
    instance_info = get_instance_info(job_lengths, op_pt_matrix)
    instance_info['op_pt_matrix_original'] = op_pt_matrix
    job_count = instance_info['num_jobs']
    operation_count = instance_info['num_operations']
    if operation_count == 0:
        warnings.warn('Instance has no operations. Exiting GA.')
        return {'best_chromosome': ([], []), 'best_fitness': -float('inf'), 'best_makespan': 0, 'best_tardiness': 0, 'best_schedule_log': []}
    candidate_population = harbid_generate_initial_population(pop_size, instance_info)
    population_fitness = []
    best_chromosome = None
    best_fitness = -float('inf')
    best_evaluation = {}
    for i, chromosome in enumerate(candidate_population):
        operation_sequence, machine_sequence = chromosome
        evaluation = decode_chromosome_and_evaluate(operation_sequence, machine_sequence, instance_info, initial_mch_available_times, job_due_dates, job_ready_times)
        population_fitness.append(evaluation['fitness'])
        if evaluation['fitness'] > best_fitness:
            best_fitness = evaluation['fitness']
            best_chromosome = chromosome
            best_evaluation = evaluation
    completed_generations = 0
    fitness_trajectory = [best_fitness]
    while time.time() - started_at < time_limit:
        completed_generations += 1
        selected_population = selection_with_elitism(candidate_population, population_fitness, pop_size, elite_size, tournament_size)
        offspring = []
        offspring.extend(selected_population[:elite_size])
        num_offspring_needed = pop_size - elite_size
        parent_pool_indices = list(range(len(selected_population)))
        while len(offspring) < pop_size:
            first_parent_idx = random.choice(parent_pool_indices)
            second_parent_idx = random.choice(parent_pool_indices)
            first_parent = selected_population[first_parent_idx]
            second_parent = selected_population[second_parent_idx]
            first_child_os, second_child_os = (first_parent[0], second_parent[0])
            first_child_ms, second_child_ms = (first_parent[1], second_parent[1])
            if random.random() < crossover_rate:
                first_child_os, second_child_os = precedence_preserving_order_crossover(first_parent[0], second_parent[0], job_count)
                first_child_ms, second_child_ms = two_point_crossover(first_parent[1], second_parent[1])
            if random.random() < mutation_rate_os:
                first_child_os = swap_mutation_os(first_child_os)
            if random.random() < mutation_rate_os:
                second_child_os = swap_mutation_os(second_child_os)
            if random.random() < mutation_rate_ms:
                first_child_ms = single_point_mutation_ms(first_child_ms, instance_info)
            if random.random() < mutation_rate_ms:
                second_child_ms = single_point_mutation_ms(second_child_ms, instance_info)
            offspring.append((first_child_os, first_child_ms))
            if len(offspring) < pop_size:
                offspring.append((second_child_os, second_child_ms))
        candidate_population = offspring[:pop_size]
        population_fitness = []
        generation_best_fitness = -float('inf')
        generation_best_chromosome = None
        generation_best_evaluation = {}
        for chromosome in candidate_population:
            operation_sequence, machine_sequence = chromosome
            evaluation = decode_chromosome_and_evaluate(operation_sequence, machine_sequence, instance_info, initial_mch_available_times, job_due_dates, job_ready_times)
            population_fitness.append(evaluation['fitness'])
            if evaluation['fitness'] > generation_best_fitness:
                generation_best_fitness = evaluation['fitness']
                generation_best_chromosome = chromosome
                generation_best_evaluation = evaluation
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_chromosome = generation_best_chromosome
            best_evaluation = generation_best_evaluation
        fitness_trajectory.append(best_fitness)
    finished_at = time.time()
    elapsed_seconds = finished_at - started_at
    print(f'--- GA Finished in {finished_at - started_at:.2f} seconds ---')
    print(f'Best Solution Found:')
    print(f'  Fitness: {best_fitness:.4f}')
    print(f"  Makespan: {best_evaluation.get('makespan', 'N/A'):.2f}")
    print(f"  Tardiness: {best_evaluation.get('total_tardiness', 'N/A'):.2f}")
    if plot_gantt and best_evaluation and best_evaluation.get('gantt_data'):
        os.makedirs(output_dir, exist_ok=True)
        chart_file_path = os.path.join(output_dir, 'best_schedule_gantt.png')
        gantt_title = f"Best GA Schedule (Fitness: {best_fitness:.4f})\nMakespan: {best_evaluation['makespan']:.2f}, Tardiness: {best_evaluation['total_tardiness']:.2f}"
        gantt_payload = best_evaluation['gantt_data']
        required_chart_fields = ['op_status', 'true_op_ct', 'true_op_pt', 'op_to_job_map', 'op_machine_assignment', 'env_num_ops', 'num_machines']
        if gantt_payload and all((key in gantt_payload for key in required_chart_fields)) and (gantt_payload['true_op_pt'] is not None):
            try:
                plot_gantt_chart(env_idx=0, filename=chart_file_path, title=gantt_title, **gantt_payload)
            except Exception as plot_e:
                warnings.warn(f'Error plotting final Gantt chart: {plot_e}')
        else:
            warnings.warn('Could not plot final Gantt chart due to missing or incomplete gantt_data.')
    return {'best_chromosome': best_chromosome, 'best_fitness': best_fitness, 'best_makespan': best_evaluation.get('makespan', float('inf')),
            'best_tardiness': best_evaluation.get('total_tardiness', float('inf')), 'best_schedule_log': best_evaluation.get('schedule', []),
            'iterations': completed_generations, 'fitness_history': fitness_trajectory, 'runtime': elapsed_seconds}
if __name__ == '__main__':

    class DummyConfig:
        n_j = 10
        n_m = 5
        op_per_job = 5
        low = 1
        high = 99
        data_suffix = 'mix'
        op_per_mch_min = 1
        op_per_mch_max = 5
    config_dummy = DummyConfig()
    example_job_lengths, example_op_pt_matrix, _ = SD2_instance_generator(config_dummy)
    example_due_dates = np.random.randint(50, 200, size=config_dummy.n_j)
    example_initial_mch_times = np.random.randint(0, 10, size=config_dummy.n_m)
    ga_result = run_genetic_algorithm(job_lengths=example_job_lengths, op_pt_matrix=example_op_pt_matrix,
                                      initial_mch_available_times=example_initial_mch_times,job_due_dates=example_due_dates,
                                      pop_size=60, num_generations=150, elite_size=6, tournament_size=3, crossover_rate=0.85,
                                      mutation_rate_os=0.15, mutation_rate_ms=0.15, plot_gantt=True, output_dir='./ga_initial_schedule_results')
