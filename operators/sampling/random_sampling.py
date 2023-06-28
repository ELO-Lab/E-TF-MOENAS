from model.population import Population
from utils import get_hashKey


class RandomSampling:
    def __init__(self, n_sample=0):
        self.n_sample = n_sample

    def do(self, problem, **kwargs):
        problem_name = problem.name
        P = Population(self.n_sample)
        n = 0

        P_hash_key = []
        while n < self.n_sample:
            X = problem.sample_a_compact_architecture()
            if problem.isValid(X):
                hashKey = get_hashKey(X, problem_name)
                if hashKey not in P_hash_key:
                    P[n].set('X', X)
                    P[n].set('hashKey', hashKey)
                    n += 1
        return P

if __name__ == '__main__':
    pass