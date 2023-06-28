import pickle as p
from algorithms import NSGAII
import numpy as np
from model import Population
from utils import get_front_0, get_hashKey
from zero_cost_methods import modify_input_for_fitting


def get_population_X(tmp_P, pop_size, problem, tf_calculator):
    tmp_P_F = []

    for i in range(len(tmp_P)):
        X_modified = modify_input_for_fitting(tmp_P[i].X, problem.name)

        score = tf_calculator.query_(arch=X_modified)
        computational_metric = problem.get_computational_metric(tmp_P[i].X)

        tmp_P_F.append([computational_metric, -score])

    tmp_P_F = np.array(tmp_P_F)
    P_X = get_k_best_solutions(tmp_P=tmp_P, tmp_P_F=tmp_P_F, k=pop_size)
    return P_X

def get_k_best_solutions(tmp_P, tmp_P_F, k):
    P_X = []
    n = 0
    idx = np.array(list(range(len(tmp_P_F))))
    while True:
        idx_front_0 = get_front_0(tmp_P_F)
        front_0 = idx[idx_front_0].copy()
        for i in front_0:
            P_X.append(tmp_P[i].X)
            n += 1
            if n == k:
                return np.array(P_X)
        idx = idx[~idx_front_0]
        tmp_P_F = tmp_P_F[~idx_front_0]


class ENAS_TFI(NSGAII):
    """
    Efficiency Enhancement of Evolutionary Neural Architecture Search via Training-Free Initialization
    (Quan Minh Phan, Hoang Ngoc Luong) (NICS 2021)
    """

    def __init__(self, n_sample=500, path_pre_pop=None, **kwargs):
        super().__init__(**kwargs)
        self.n_sample = n_sample
        self.path_pre_pop = path_pre_pop
        self.tf_calculator = None

    def _initialize(self):
        self.sampling.n_sample = self.n_sample
        if self.problem.name == 'NASBench101':
            self.pop_size = 100
        try:
            P_X = p.load(
                open(self.path_pre_pop + '/' + f'{self.problem.name}_{self.problem.dataset}_'
                                       f'population_synflow_'
                                       f'{self.sampling.n_sample}_{self.seed}.p', 'rb')
            )
        except FileNotFoundError:
            tmp_P = self.sampling.do(self.problem)
            P_X = get_population_X(tmp_P=tmp_P, pop_size=self.pop_size,
                                   problem=self.problem, tf_calculator=self.tf_calculator)
            p.dump(P_X, open(self.path_pre_pop + '/' + f'{self.problem.name}_{self.problem.dataset}_'
                                                       f'population_synflow_'
                                                       f'{self.sampling.n_sample}_{self.seed}.p', 'wb'))
        P = Population(self.pop_size)
        for i, X in enumerate(P_X):
            hashKey = get_hashKey(X, problem_name=self.problem.name)
            F = self.evaluate(X)
            P[i].set('X', X)
            P[i].set('hashKey', hashKey)
            P[i].set('F', F)
            self.E_Archive_search.update(P[i], algorithm=self)
        self.pop = P


if __name__ == '__main__':
    pass
