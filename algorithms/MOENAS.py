"""
Inspired: https://github.com/msu-coinlab/pymoo
"""
import time

from algorithms import Algorithm
from model.individual import Individual

INF = 9999999
class NSGAII(Algorithm):
    """
    NSGA-Net
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default: f0 -> efficiency metric; f1 -> performance metric
        self.f0, self.f1 = None, None
        self.individual = Individual(rank=INF, crowding=-1)

    def _setup(self):
        attr_perf_metric = self.f1.split('_')
        self.subset = attr_perf_metric[0]
        self.perf_metric = attr_perf_metric[1]
        self.epoch = int(attr_perf_metric[-1])

    def _evaluate(self, arch):
        """
        Call function *problem.evaluate* to evaluate the fitness values of solutions.
        """
        self.finish_executed_time_algorithm = time.time()
        self.executed_time_algorithm_history.append(
            self.executed_time_algorithm_history[-1] + (self.finish_executed_time_algorithm - self.start_executed_time_algorithm))

        comp_metric, perf_metric, benchmark_time, indicator_time = self.problem.evaluate(arch=arch, comp_metric=self.f0,
                                                                                         perf_metric=self.perf_metric,
                                                                                         epoch=self.epoch,
                                                                                         subset=self.subset)
        self.n_eval += 1
        self.benchmark_time_algorithm_history.append(self.benchmark_time_algorithm_history[-1] + benchmark_time)
        self.indicator_time_history.append(self.indicator_time_history[-1] + indicator_time)
        self.evaluated_time_history.append(self.evaluated_time_history[-1] + benchmark_time + indicator_time)
        self.running_time_history.append(self.evaluated_time_history[-1] + self.tmp + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        return [comp_metric, perf_metric]

    def _initialize(self):
        """
        Workflow in 'Initialization' step:
        + Sampling 'pop_size' architectures.
        + For each architecture, evaluate its fitness.
            _ Update the Elitist Archive (search).
        """
        P = self.sampling.do(self.problem)
        for i in range(self.pop_size):
            F = self.evaluate(P[i].X)
            P[i].set('F', F)
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
        for i in range(len(O)):
            o_F = self.evaluate(O[i].X)
            O[i].set('F', o_F)
            self.E_Archive_search.update(O[i], algorithm=self)

        return O

    def _next(self, pop):
        """
         Workflow in 'Next' step:
        + Create the offspring.
        + Select the new population.
        """
        offsprings = self._mating(pop)
        pool = pop.merge(offsprings)
        pop = self.survival.do(pool, self.pop_size)
        self.pop = pop

    def log_elitist_archive(self, **kwargs):
        self.nEvals_history.append(self.n_eval)

        elitist_archive_search = {
            'X': self.E_Archive_search.X.copy(),
            'hashKey': self.E_Archive_search.hashKey.copy(),
            'F': self.E_Archive_search.F.copy(),
        }
        self.E_Archive_search_history.append(elitist_archive_search)

if __name__ == '__main__':
    pass
