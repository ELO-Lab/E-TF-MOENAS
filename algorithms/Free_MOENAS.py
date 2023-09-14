import time
from algorithms import NSGAII
import numpy as np

class Free_NSGAII(NSGAII):
    """
    Training-free NSGA-II (only using 1 single training-free performance metrics)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup(self):
        pass

    def _evaluate(self, arch):
        """
        Call function *problem.evaluate* to evaluate the fitness values of solutions.
        """
        self.finish_executed_time_algorithm = time.time()
        self.executed_time_algorithm_history.append(
            self.executed_time_algorithm_history[-1] + (
                        self.finish_executed_time_algorithm - self.start_executed_time_algorithm))

        indicator_time = 0.0

        log_synflow, cost = self.problem.get_free_metrics(arch, 'log_synflow')
        indicator_time += cost
        nwot, cost = self.problem.get_free_metrics(arch, 'nwot')
        indicator_time += cost
        skip = self.problem.get_skip(arch)
        comp_metric = self.problem.get_computational_metric(arch=arch, metric=self.f0)

        self.n_eval += 1
        self.benchmark_time_algorithm_history.append(self.benchmark_time_algorithm_history[-1] + 0.0)
        self.indicator_time_history.append(self.indicator_time_history[-1] + indicator_time)
        self.evaluated_time_history.append(self.evaluated_time_history[-1] + 0.0 + indicator_time)
        self.running_time_history.append(
            self.evaluated_time_history[-1] + self.tmp + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        return [comp_metric, log_synflow, nwot, skip]

    def _initialize(self):
        """
        Workflow in 'Initialization' step:
        + Sampling 'pop_size' architectures.
        + For each architecture, evaluate its fitness.
            _ Update the Elitist Archive (search).
        """
        P = self.sampling.do(self.problem)
        F_pop = np.zeros((self.pop_size, 2), dtype=float)
        all_free_metrics = []
        for i in range(self.pop_size):
            F = self.evaluate(P[i].X)
            F_pop[i][0] = F[0]
            all_free_metrics.append(F[1:])
        all_free_metrics = np.array(all_free_metrics)
        max_values = np.max(np.abs(all_free_metrics), axis=0) + 1e-9
        all_free_metrics /= max_values
        sum_free_metrics = np.sum(all_free_metrics, axis=1)
        F_pop[:, 1] = -sum_free_metrics
        for i in range(self.pop_size):
            P[i].set('F', F_pop[i])
            self.E_Archive_search.update(P[i], algorithm=self)
        self.pop = P

    def _mating(self, P):
        """
         Workflow in 'Mating' step:
        + Create the offspring throughout 'crossover' step.
        + Perform 'mutation' step on the offspring.
        """
        O = self.crossover.do(self.problem, P, algorithm=self)
        O = self.mutation.do(self.problem, P, O, algorithm=self)

        F_off = np.zeros((self.pop_size, 2), dtype=float)
        all_free_metrics = []
        for i in range(self.pop_size):
            F = self.evaluate(O[i].X)
            F_off[i][0] = F[0]
            all_free_metrics.append(F[1:])
        all_free_metrics = np.array(all_free_metrics)
        max_values = np.max(np.abs(all_free_metrics), axis=0) + 1e-9
        all_free_metrics /= max_values
        sum_free_metrics = np.sum(all_free_metrics, axis=1)
        F_off[:, 1] = -sum_free_metrics
        for i in range(self.pop_size):
            O[i].set('F', F_off[i])
            self.E_Archive_search.update(O[i], algorithm=self)

        return O

if __name__ == '__main__':
    pass
