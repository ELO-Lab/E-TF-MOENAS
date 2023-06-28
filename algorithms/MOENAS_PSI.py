import numpy as np
from algorithms import NSGAII
from utils import get_hashKey


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


def check_above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
    """
    This function is used to check if the considering point is above or below
    the line connecting two remaining points.\n
    1: above\n
    -1: below
    """
    orthogonal_vector = remaining_pt_2 - remaining_pt_1
    line_connecting_pt1_and_pt2 = -orthogonal_vector[1] * (considering_pt[0] - remaining_pt_1[0]) \
                                  + orthogonal_vector[0] * (considering_pt[1] - remaining_pt_1[1])
    if line_connecting_pt1_and_pt2 > 0:
        return 1
    return -1


def calculate_angle_measure(considering_pt, neighbor_1, neighbor_2):
    """
    This function is used to calculate the angle measure is created by the considering point
    and two its nearest neighbors
    """
    line_1 = neighbor_1 - considering_pt
    line_2 = neighbor_2 - considering_pt
    cosine_angle = (line_1[0] * line_2[0] + line_1[1] * line_2[1]) \
                   / (np.sqrt(np.sum(line_1 ** 2)) * np.sqrt(np.sum(line_2 ** 2)))
    if cosine_angle < -1:
        cosine_angle = -1
    if cosine_angle > 1:
        cosine_angle = 1
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)


class MOENAS_PSI(NSGAII):
    """
    Paper:
    Enhancing Multi-objective Evolutionary Neural Architecture Search
    with Surrogate Models and Potential Point-Guided Local Searches
    (Quan Minh Phan, Ngoc Hoang Luong) (IEA/AIE 2021)
    """
    def __init__(self, gamma=210, nPoints=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.nPoints = nPoints

    def seeking(self, P):
        """
            The first phase in the method: Identifying potential solutions
        """
        P_F = P.get('F')

        idx_non_dominated_front = get_idx_non_dominated_front(P_F)

        non_dominated_set = P[idx_non_dominated_front].copy()

        non_dominated_front = P_F[idx_non_dominated_front].copy()

        new_idx = np.argsort(non_dominated_front[:, 0])

        non_dominated_set = non_dominated_set[new_idx]
        non_dominated_front = non_dominated_front[new_idx]

        non_dominated_front_norm = non_dominated_front.copy()

        min_f0 = np.min(non_dominated_front[:, 0])
        max_f0 = np.max(non_dominated_front[:, 0])

        min_f1 = np.min(non_dominated_front[:, 1])
        max_f1 = np.max(non_dominated_front[:, 1])

        non_dominated_front_norm[:, 0] = (non_dominated_front_norm[:, 0] - min_f0) / (max_f0 - min_f0)
        non_dominated_front_norm[:, 1] = (non_dominated_front_norm[:, 1] - min_f1) / (max_f1 - min_f1)

        idx_non_dominated_front = idx_non_dominated_front[new_idx]

        potential_sols = [
            [0, 'best_f0']  # (idx (in pareto front), property)
        ]

        for i in range(len(non_dominated_front) - 1):
            if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i + 1])) != 0:
                break
            else:
                potential_sols.append([i, 'best_f0'])

        for i in range(len(non_dominated_front) - 1, -1, -1):
            if np.sum(np.abs(non_dominated_front[i] - non_dominated_front[i - 1])) != 0:
                break
            else:
                potential_sols.append([i - 1, 'best_f1'])
        potential_sols.append([len(non_dominated_front) - 1, 'best_f1'])

        # find the knee solutions
        start_idx = potential_sols[0]
        end_idx = potential_sols[-1][0]

        for sols in potential_sols:
            if sols[-1] == 'best_f1':
                break
            else:
                start_idx = sols[0] + 1

        for i in range(len(potential_sols) - 1, -1, -1):
            if potential_sols[i][1] == 'best_f0':
                break
            else:
                end_idx = potential_sols[i][0] - 1

        for i in range(start_idx, end_idx + 1):
            l = None
            h = None
            for m in range(i - 1, -1, -1):
                if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                    l = m
                    break
            for m in range(i + 1, len(non_dominated_front), 1):
                if np.sum(np.abs(non_dominated_front[m] - non_dominated_front[i])) != 0:
                    h = m
                    break

            if (h is not None) and (l is not None):
                position = check_above_or_below(considering_pt=non_dominated_front[i],
                                                remaining_pt_1=non_dominated_front[l],
                                                remaining_pt_2=non_dominated_front[h])
                if position == -1:
                    angle_measure = calculate_angle_measure(considering_pt=non_dominated_front_norm[i],
                                                            neighbor_1=non_dominated_front_norm[l],
                                                            neighbor_2=non_dominated_front_norm[h])
                    if angle_measure > self.gamma:
                        potential_sols.append([i, 'knee'])
        return non_dominated_set, potential_sols, idx_non_dominated_front

    def improving(self, P, non_dominated_set, potential_sols, idx_non_dominated_front):
        """
        The second phase in the method: \n Improving potential solutions
        """
        problem_name = self.problem.name

        _non_dominated_set = P.new()
        P_hashKey = P.get('hashKey')

        l_sol = len(P[0].X)

        for i, property_sol in potential_sols:
            nSearchs, maxSearchs = 0, l_sol
            _nSearchs, _maxSearchs = 0, 100

            found_neighbors_list = [non_dominated_set[i].hashKey]
            while (nSearchs < maxSearchs) and (_nSearchs < _maxSearchs):
                _nSearchs += 1

                """ Find a neighboring solution """
                neighborSol = non_dominated_set[i].copy()

                if problem_name == 'NASBench101':
                    neighborSol_X = neighborSol.X.copy()
                    idxs_lst = []
                    for _ in range(self.nPoints):
                        while True:
                            idx = np.random.randint(22)
                            if idx == 0 or idx == 21:
                                pass
                            else:
                                idxs_lst.append(idx)
                                break
                    for idx in idxs_lst:
                        if idx in self.problem.IDX_OPS:
                            available_ops = self.problem.OPS.copy()
                        else:
                            available_ops = self.problem.EDGES.copy()
                        available_ops.remove(neighborSol_X[idx])
                        neighborSol_X[idx] = np.random.choice(available_ops)
                else:
                    idxs = np.random.choice(range(self.problem.maxLength), size=self.nPoints, replace=False)
                    neighborSol_X = non_dominated_set[i].X.copy()
                    for idx in idxs:
                        allowed_ops = self.problem.available_ops.copy()
                        allowed_ops.remove(non_dominated_set[i].X[idx])
                        new_op = np.random.choice(allowed_ops)
                        neighborSol_X[idx] = new_op

                if self.problem.isValid(neighborSol_X):
                    neighborSol_hashKey = get_hashKey(neighborSol_X, problem_name)

                    if check_valid(neighborSol_hashKey,
                                   neighbors=found_neighbors_list,
                                   P=P_hashKey):
                        """ Improve the neighboring solution """
                        found_neighbors_list.append(neighborSol_hashKey)
                        nSearchs += 1

                        neighborSol_F = self.evaluate(neighborSol_X)
                        neighborSol.set('X', neighborSol_X)
                        neighborSol.set('hashKey', neighborSol_hashKey)
                        neighborSol.set('F', neighborSol_F)

                        self.E_Archive_search.update(neighborSol, algorithm=self)
                        if property_sol == 'best_f0':
                            betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=0)
                        elif property_sol == 'best_f1':
                            betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F, position=1)
                        else:
                            betterSol = find_the_better(neighborSol_F, non_dominated_set[i].F)

                        if betterSol == 0:  # --> the neighbor is better
                            non_dominated_set[i].set('X', neighborSol_X)
                            non_dominated_set[i].set('hashKey', neighborSol_hashKey)
                            non_dominated_set[i].set('F', neighborSol_F)

        P[idx_non_dominated_front] = non_dominated_set
        return P

    def psi(self, pop):
        non_dominated_set, potential_sols, idx_non_dominated_front = self.seeking(pop)
        pop = self.improving(pop, non_dominated_set, potential_sols, idx_non_dominated_front)
        return pop

    def _next(self, pop):
        offsprings = self._mating(pop)
        pool = pop.merge(offsprings)
        pop = self.survival.do(pool, self.pop_size)
        pop = self.psi(pop=pop)
        self.pop = pop


if __name__ == '__main__':
    pass
