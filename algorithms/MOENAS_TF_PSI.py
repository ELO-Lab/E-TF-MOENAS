import numpy as np
from algorithms import MOENAS_PSI
from utils import get_hashKey
import itertools


def find_all_neighbors(X, distance=1, problem=None):
    """
    This function is used to find all neighboring solutions of the considering solution. The distance from the current
    solution to the neighbors is 1 (or 2).
    """
    all_neighbors = []
    idx = list(itertools.combinations(range(len(X)), distance))
    if problem.name == 'NASBench101':
        for i in idx:
            if i[0] == 0 or i[0] == 21:
                continue
            elif i[0] in problem.IDX_OPS:
                available_ops = problem.OPS.copy()
            else:
                available_ops = problem.EDGES.copy()

            for op in available_ops:
                if op != X[i]:
                    neighbor = X.copy()
                    neighbor[i] = op
                    all_neighbors.append(neighbor)
    else:
        available_ops = problem.available_ops.copy()
        if distance == 1:
            for i in idx:
                for op in available_ops:
                    if op != X[i[0]]:
                        neighbor = X.copy()
                        neighbor[i[0]] = op
                        all_neighbors.append(neighbor)
        elif distance == 2:
            for i, j in idx:
                for op_0 in available_ops:
                    if op_0 != X[i]:
                        neighbor = X.copy()
                        neighbor[i] = op_0
                        for op_1 in available_ops:
                            if op_1 != X[j]:
                                neighbor[j] = op_1
                                all_neighbors.append(neighbor)
        else:
            for i0, i1, i2 in idx:
                for op_0 in available_ops:
                    if op_0 != X[i0]:
                        neighbor = X.copy()
                        neighbor[i0] = op_0
                        for op_1 in available_ops:
                            if op_1 != X[i1]:
                                neighbor[i1] = op_1
                                for op_2 in available_ops:
                                    if op_2 != X[i2]:
                                        neighbor[i2] = op_2
                                        all_neighbors.append(neighbor)
    return all_neighbors

def check_valid(hash_key, **kwargs):
    """
    This function is used to check if the current solution is valid or not.
    """
    return np.all([hash_key not in kwargs[L] for L in kwargs])


def find_the_better(F_x, F_y, position=-1):
    """
    This function is used to find the better solution between two input solutions.\n
    If one of the solutions is an extreme solution, just using only one objective which
    that solution is the best at for comparing.
    """
    if isinstance(F_x, list):
        F_x = np.array(F_x)
    if isinstance(F_y, list):
        F_y = np.array(F_y)
    if position == -1:
        if isinstance(F_x[-1], dict):
            vote_lst = []
            for key in F_x[-1]:
                x_new = np.array([F_x[0], F_x[-1][key]])
                y_new = np.array([F_y[0], F_y[-1][key]])
                sub_ = x_new - y_new
                x_better = np.all(sub_ <= 0)
                y_better = np.all(sub_ >= 0)
                if x_better == y_better:  # True - True
                    vote_lst.append(-1)
                elif y_better:  # False - True
                    vote_lst.append(1)
                else:
                    vote_lst.append(0)  # True - False
            count_vote_lst = [vote_lst.count(-1), vote_lst.count(0), vote_lst.count(1)]
            better_lst = np.array([-1, 0, 1])
            # if count_vote_lst[0] == count_vote_lst[1] == count_vote_lst[2] == 1:
            if count_vote_lst[0] == 1 or count_vote_lst[1] == 1 or count_vote_lst[2] == 1:
                return None
            idx = np.argmax(count_vote_lst)
            return better_lst[idx]
        else:
            sub_ = F_x - F_y
            x_better = np.all(sub_ <= 0)
            y_better = np.all(sub_ >= 0)
            if x_better == y_better:  # True - True
                return -1
            if y_better:  # False - True
                return 1
            return 0  # True - False
    else:
        if position == 0:
            if F_x[position] < F_y[position]:
                return 0
            elif F_x[position] > F_y[position]:
                return 1
            else:
                if F_x[1] < F_y[1]:
                    return 0
                elif F_x[1] > F_y[1]:
                    return 1
                return -1
        else:
            if isinstance(F_x[-1], dict):
                vote_lst = []
                for key in F_x[-1]:
                    x_new = F_x[-1][key]
                    y_new = F_y[-1][key]
                    if x_new < y_new:
                        vote_lst.append(0)
                    elif x_new > y_new:
                        vote_lst.append(1)
                    else:
                        vote_lst.append(-1)
                count_vote_lst = [vote_lst.count(-1), vote_lst.count(0), vote_lst.count(1)]
                better_lst = np.array([-1, 0, 1])
                # if count_vote_lst[0] == count_vote_lst[1] == count_vote_lst[2] == 1:
                if count_vote_lst[0] == 1 or count_vote_lst[1] == 1 or count_vote_lst[2] == 1:
                    return None
                idx = np.argmax(count_vote_lst)
                return better_lst[idx]
            else:
                if F_x[1] < F_y[1]:
                    return 0
                elif F_x[1] > F_y[1]:
                    return 1
                else:
                    return -1

def get_idx_non_dominated_front(F):
    """
    This function is used to get the zero front in the population.
    """
    l = len(F)
    r = np.zeros(l, dtype=np.int8)
    idx = np.array(list(range(l)))
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = find_the_better(F[i], F[j])
                if better_sol == 0:  # the current is better
                    r[j] += 1
                elif better_sol == 1:
                    r[i] += 1
                    break
    idx_non_dominated_front = idx[r == 0]
    return idx_non_dominated_front


class MOENAS_TF_PSI(MOENAS_PSI):
    """
    Paper:
    Enhancing Multi-Objective Evolutionary Neural Architecture Search with Training-Free Pareto Local Search
    (Quan Minh Phan, Ngoc Hoang Luong) (Applied Intelligence 2022)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.neighbors_history = []

    def _reset(self):
        self.neighbors_history = []

    def __evaluate(self, arch):
        comp_metric, perf_metric, _, _ = self.problem.evaluate(arch=arch,
                                                               comp_metric=self.f0, perf_metric='negative_tfi',
                                                               epoch=None, subset=None)
        return [comp_metric, perf_metric]

    def improving(self, P, non_dominated_set, potential_sols, idx_non_dominated_front):
        """
        The second phase in the method: \n Training-free improving potential solutions
        """
        problem_name = self.problem.name

        _non_dominated_set = P.new()
        P_hashKey = P.get('hashKey')

        for i, property_sol in potential_sols:
            if non_dominated_set[i].hashKey not in self.neighbors_history:
                potentialSol_F_proxy = self.__evaluate(non_dominated_set[i].X)

                self.neighbors_history.append(non_dominated_set[i].hashKey)

                betterSol_X_lst = []
                betterSol_F_lst = []
                nonDominatedSol_X_lst = []
                nonDominatedSol_F_lst = []
                all_neighbors = find_all_neighbors(non_dominated_set[i].X, distance=self.nPoints, problem=self.problem)
                for neighbor_X in all_neighbors:
                    if self.problem.isValid(neighbor_X):
                        neighborSol_hashKey = get_hashKey(neighbor_X, problem_name)
                        if check_valid(neighborSol_hashKey, P=P_hashKey):
                            neighborSol_F_proxy = self.__evaluate(neighbor_X)

                            if property_sol == 'best_f0':
                                betterSol = find_the_better(neighborSol_F_proxy, potentialSol_F_proxy, position=0)
                            elif property_sol == 'best_f1':
                                betterSol = find_the_better(neighborSol_F_proxy, potentialSol_F_proxy, position=1)
                            else:
                                betterSol = find_the_better(neighborSol_F_proxy, potentialSol_F_proxy)

                            if betterSol == 0:
                                betterSol_X_lst.append(neighbor_X)
                                betterSol_F_lst.append(neighborSol_F_proxy)
                            elif betterSol == -1:
                                nonDominatedSol_X_lst.append(neighbor_X)
                                nonDominatedSol_F_lst.append(neighborSol_F_proxy)

                idx_bestSol1 = get_idx_non_dominated_front(betterSol_F_lst)
                idx_bestSol2 = get_idx_non_dominated_front(nonDominatedSol_F_lst)

                for idx in idx_bestSol1:
                    neighborSol_hashKey = get_hashKey(betterSol_X_lst[idx], problem_name)
                    tmp_pop = P.new(1)
                    neighborSol_F = self.evaluate(betterSol_X_lst[idx])

                    tmp_pop[0].set('X', betterSol_X_lst[idx])
                    tmp_pop[0].set('hashKey', neighborSol_hashKey)
                    tmp_pop[0].set('F', neighborSol_F)
                    self.E_Archive_search.update(tmp_pop[0], algorithm=self)

                    if property_sol == 'best_f0':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=0)
                    elif property_sol == 'best_f1':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=1)
                    else:
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F)

                    if betterSol == 0:  # --> the neighbor is better
                        tmp = P.new(1)
                        tmp[0].set('X', non_dominated_set[i].X)
                        tmp[0].set('hashKey', non_dominated_set[i].hashKey)
                        tmp[0].set('F', non_dominated_set[i].F)
                        _non_dominated_set = _non_dominated_set.merge(tmp)

                        non_dominated_set[i].set('X', betterSol_X_lst[idx])
                        non_dominated_set[i].set('hashKey', neighborSol_hashKey)
                        non_dominated_set[i].set('F', neighborSol_F)
                    else:
                        tmp_pop = P.new(1)
                        tmp_pop[0].set('X', betterSol_X_lst[idx])
                        tmp_pop[0].set('hashKey', neighborSol_hashKey)
                        tmp_pop[0].set('F', neighborSol_F)
                        _non_dominated_set = _non_dominated_set.merge(tmp_pop)

                for idx in idx_bestSol2:
                    neighborSol_hashKey = get_hashKey(nonDominatedSol_X_lst[idx], problem_name)
                    tmp_pop = P.new(1)
                    neighborSol_F = self.evaluate(nonDominatedSol_X_lst[idx])
                    tmp_pop[0].set('X', nonDominatedSol_X_lst[idx])
                    tmp_pop[0].set('hashKey', neighborSol_hashKey)
                    tmp_pop[0].set('F', neighborSol_F)
                    self.E_Archive_search.update(tmp_pop[0], algorithm=self)

                    if property_sol == 'best_f0':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=0)
                    elif property_sol == 'best_f1':
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=1)
                    else:
                        betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F)

                    if betterSol == 0:  # --> the neighbor is better
                        tmp = P.new(1)
                        tmp[0].set('X', non_dominated_set[i].X)
                        tmp[0].set('hashKey', non_dominated_set[i].hashKey)
                        tmp[0].set('F', non_dominated_set[i].F)
                        _non_dominated_set = _non_dominated_set.merge(tmp)

                        non_dominated_set[i].set('X', nonDominatedSol_X_lst[idx])
                        non_dominated_set[i].set('hashKey', neighborSol_hashKey)
                        non_dominated_set[i].set('F', neighborSol_F)
                    else:
                        tmp_pop = P.new(1)
                        tmp_pop[0].set('X', nonDominatedSol_X_lst[idx])
                        tmp_pop[0].set('hashKey', neighborSol_hashKey)
                        tmp_pop[0].set('F', neighborSol_F)
                        _non_dominated_set = _non_dominated_set.merge(tmp_pop)

        P[idx_non_dominated_front] = non_dominated_set
        pool = P.merge(_non_dominated_set)
        P = self.survival.do(pool, len(P))
        return P


if __name__ == '__main__':
    pass
