import os
import time
import argparse

from factory import get_problems, get_algorithm, problem_configuration4algorithm
from operators.crossover import PointCrossover
from operators.mutation import BitStringMutation
from operators.sampling.random_sampling import RandomSampling
from operators.selection import RankAndCrowdingSurvival

from zero_cost_methods import get_config_for_zero_cost_predictor, get_zero_cost_predictor

def main(kwargs):
    """ ============================================== Set up directory ============================================ """
    try:
        os.makedirs(f'{kwargs.res_path}/{kwargs.problem}')
    except FileExistsError:
        pass
    RES_PATH = f'{kwargs.res_path}/{kwargs.problem}'

    ''' ============================================== Set up problem ============================================== '''
    database_path = './database'
    problem = get_problems(problem_name=kwargs.problem, database_path=database_path, max_eval=kwargs.max_eval)
    problem.set_up()

    ''' ==================================================================================================== '''
    pop_size = kwargs.pop_size

    n_run = kwargs.n_run
    init_seed = kwargs.init_seed

    # Set up general hyperparameters for all algorithms, i.e., crossover, mutation, sampling methods.
    sampling = RandomSampling()
    crossover = PointCrossover('2X')
    mutation = BitStringMutation()

    algorithm = get_algorithm(algorithm_name=kwargs.algorithm)
    survival = RankAndCrowdingSurvival()

    if '201' in problem.name:
        objectives = problem_configuration4algorithm[kwargs.algorithm]['201']
    elif '101' in problem.name:
        objectives = problem_configuration4algorithm[kwargs.algorithm]['101']
    else:
        raise ValueError()

    algorithm.set_hyperparameters(pop_size=pop_size,
                                  sampling=sampling,
                                  crossover=crossover,
                                  mutation=mutation,
                                  survival=survival,
                                  debug=bool(kwargs.debug),
                                  f0=objectives['f0'],
                                  f1=objectives['f1'])

    ''' ==================================== Set up experimental environment ======================================= '''
    dir_name = f'{kwargs.problem}_{kwargs.algorithm}'

    algo_res_path = RES_PATH + '/' + dir_name
    os.mkdir(algo_res_path)
    print(f'--> Create folder {algo_res_path} - Done\n')

    random_seeds_list = [init_seed + run * 100 for run in range(n_run)]
    executed_time_list = []

    ''' =============================================== Log Information ============================================ '''
    print(f'******* PROBLEM *******')
    print(f'- Benchmark: {problem.name}')
    print(f'- Dataset: {problem.dataset}')
    print(f'- Maximum number of evaluations: {problem.maxEvals}')
    print(f'- Search:')
    print(f'\t+ The first objective (minimize): {objectives["f0"]}')
    print(f'\t+ The second objective (minimize): {objectives["f1"]}')
    print(f'- Evaluate:')
    print(f'\t+ The first objective (minimize): {problem.objective_0}')
    print(f'\t+ The second objective (minimize): {problem.objective_1}\n')

    print(f'******* ALGORITHM *******')
    print(f'- Algorithm name: {algorithm.name}')
    print(f'- Population size: {pop_size}')
    print(f'- Crossover method: {algorithm.crossover.method}')
    print(f'- Mutation method: Bit-string')
    print(f'- Selection method: {algorithm.survival.name}\n')

    print(f'******* ENVIRONMENT *******')
    print(f'- Number of running experiments: {n_run}')
    print(f'- Random seed each run: {random_seeds_list}')
    print(f'- Path for saving results: {algo_res_path}')
    print(f'- Debug: {algorithm.debug}\n')

    with open(f'{algo_res_path}/configurations.txt', 'w') as f:
        f.write(f'******* PROBLEM *******\n')
        f.write(f'- Benchmark: {problem.name}\n')
        f.write(f'- Dataset: {problem.dataset}\n')
        f.write(f'- Maximum number of evaluations: {problem.maxEvals}\n')
        f.write(f'- Search:\n')
        f.write(f'\t+ The first objective (minimize): {objectives["f0"]}\n')
        f.write(f'\t+ The second objective (minimize): {objectives["f1"]}\n')
        f.write(f'- Evaluate:\n')
        f.write(f'\t+ The first objective (minimize): {problem.objective_0}\n')
        f.write(f'\t+ The second objective (minimize): {problem.objective_1}\n\n')

        f.write(f'******* ALGORITHM *******\n')
        f.write(f'- Algorithm name: {algorithm.name}\n')
        f.write(f'- Population size: {pop_size}\n')
        f.write(f'- Crossover method: {algorithm.crossover.method}\n')
        f.write(f'- Mutation method: Bit-string\n')
        f.write(f'- Selection method: {algorithm.survival.name}\n\n')

        f.write(f'******* ENVIRONMENT *******\n')
        f.write(f'- Number of running experiments: {n_run}\n')
        f.write(f'- Random seed each run: {random_seeds_list}\n')
        f.write(f'- Path for saving results: {algo_res_path}\n')
        f.write(f'- Debug: {algorithm.debug}\n\n')

    ''' ==================================================================================================== '''
    for rid in range(n_run):
        algorithm.reset()
        problem.reset()
        print(f'---- Run {rid + 1}/{n_run} ----')
        random_seed = random_seeds_list[rid]

        ## Set up for training-free search
        if kwargs.algorithm in ['E-TF-MOENAS', 'E-TF-MOENAS-C',
                                'synflow', 'jacov', 'snip', 'grasp', 'grad_norm', 'fisher',
                                'MOENAS_TF_PSI', 'ENAS_TFI']:
            config = get_config_for_zero_cost_predictor(problem=problem, seed=random_seed, database_path=database_path)
            if 'E-TF-MOENAS' in kwargs.algorithm:
                ZC_predictor = get_zero_cost_predictor(config=config, method_type='synflow-jacov')
            elif kwargs.algorithm in ['MOENAS_TF_PSI', 'ENAS_TFI']:
                ZC_predictor = get_zero_cost_predictor(config=config, method_type='synflow')
            else:
                ZC_predictor = get_zero_cost_predictor(config=config, method_type=kwargs.algorithm)
            if kwargs.algorithm == 'ENAS_TFI':
                algorithm.tf_calculator = ZC_predictor
            problem.set_zero_cost_predictor(ZC_predictor)

        exp_res_path = algo_res_path + '/' + f'{rid}'
        os.mkdir(exp_res_path)
        s = time.time()

        algorithm.set_hyperparameters(res_path=exp_res_path)
        algorithm.solve(problem, random_seed)

        executed_time = time.time() - s
        executed_time_list.append(executed_time)
        print('This run take', executed_time_list[-1], 'seconds')
    print('==' * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ''' PROBLEM '''
    parser.add_argument('--problem', type=str, default='NAS201-C10', help='the problem name',
                        choices=['NAS101', 'NAS201-C10', 'NAS201-C100', 'NAS201-IN16'])

    ''' EVOLUTIONARY ALGORITHM '''
    parser.add_argument('--pop_size', type=int, default=20, help='the population size')
    parser.add_argument('--algorithm', type=str, default='NSGA-II', help='the algorithm name',
                        choices=['val_error', 'val_loss', 'train_loss',
                                 'synflow', 'jacov', 'snip', 'grad_norm', 'grasp', 'fisher',
                                 'E-TF-MOENAS', 'E-TF-MOENAS-C',
                                 'MOENAS_PSI', 'MOENAS_TF_PSI', 'ENAS_TFI'])

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=31, help='number of experiment runs')
    parser.add_argument('--max_eval', type=int, default=3000, help='the maximum number of evaluation each run')
    parser.add_argument('--init_seed', type=int, default=0, help='random seed')
    parser.add_argument('--res_path', type=str, default='./exp_res', help='path for saving results')
    parser.add_argument('--debug', type=int, default=0, help='debug mode')
    args = parser.parse_args()

    main(args)
