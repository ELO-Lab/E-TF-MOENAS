# from .compare import find_the_better
import numpy as np

def find_the_better(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(x[-1], dict):
        vote_lst = []
        for key in x[-1]:
            x_new = np.array([x[0], x[-1][key]])
            y_new = np.array([y[0], y[-1][key]])
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
        sub_ = x - y
        x_better = np.all(sub_ <= 0)
        y_better = np.all(sub_ >= 0)
        if x_better == y_better:  # True - True
            return -1
        if y_better:  # False - True
            return 1
        return 0  # True - False

class ElitistArchive:
    """
        Note: No limited the size
    """
    def __init__(self, log_each_change=True):
        self.X, self.hashKey, self.F = [], [], []
        self.size = np.inf
        self.log_each_change = log_each_change

    def update(self, idv, **kwargs):
        X = idv.X
        hashKey = idv.hashKey
        F = idv.F

        l = len(self.X)
        r = np.zeros(l, dtype=np.int8)
        status = False
        if hashKey not in self.hashKey:
            status = True
            for i, F_ in enumerate(self.F):
                better_idv = find_the_better(F, F_)
                if better_idv == 0:
                    r[i] += 1
                elif better_idv == 1:
                    status = False
                    break
            if status:
                self.X.append(X)
                self.hashKey.append(hashKey)
                self.F.append(F)
                r = np.append(r, 0)
        self.X = np.array(self.X)[r == 0].tolist()
        self.hashKey = np.array(self.hashKey)[r == 0].tolist()
        self.F = np.array(self.F)[r == 0].tolist()

        if status and self.log_each_change:
            kwargs['algorithm'].log_elitist_archive(new_idv=idv)

class ElitistArchive1:
    """
        Note: No limited the size
    """
    def __init__(self, log_each_change=True):
        self.X, self.hashKey, self.F = [], [], []
        self.size = np.inf
        self.log_each_change = log_each_change

        self.certain_objective_value_set = set()  # [FLOPs, #params, latency, ...]

    def update(self, idv, **kwargs):
        new = False
        if idv.F[0] not in self.certain_objective_value_set:
            self.certain_objective_value_set.add(idv.F[0])
            new = True

        X = idv.X
        hashKey = idv.hashKey
        F = idv.F

        l = len(self.X)
        r = np.zeros(l, dtype=np.int8)
        status = False
        if hashKey not in self.hashKey:
            if new:
                status = True
            else:
                status = True
                for i, F_ in enumerate(self.F):
                    if F_[0] == F[0]:
                        better_idv = find_the_better(F, F_)
                        if better_idv == 0:
                            r[i] += 1
                        elif better_idv == 1:
                            status = False
                            break
            if status:
                self.X.append(X)
                self.hashKey.append(hashKey)
                self.F.append(F)
                r = np.append(r, 0)
        self.X = np.array(self.X)[r == 0].tolist()
        self.hashKey = np.array(self.hashKey)[r == 0].tolist()
        self.F = np.array(self.F)[r == 0].tolist()

        if status and self.log_each_change:
            kwargs['algorithm'].log_elitist_archive(new_idv=idv)
