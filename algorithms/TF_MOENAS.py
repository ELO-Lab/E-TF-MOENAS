import time
from algorithms import NSGAII


class TF_NSGAII(NSGAII):
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

        comp_metric, perf_metric, benchmark_time, indicator_time = self.problem.evaluate(arch=arch, comp_metric=self.f0,
                                                                                         perf_metric='negative_tfi',
                                                                                         epoch=None,
                                                                                         subset=None)
        self.n_eval += 1
        self.benchmark_time_algorithm_history.append(self.benchmark_time_algorithm_history[-1] + benchmark_time)
        self.indicator_time_history.append(self.indicator_time_history[-1] + indicator_time)
        self.evaluated_time_history.append(self.evaluated_time_history[-1] + benchmark_time + indicator_time)
        self.running_time_history.append(
            self.evaluated_time_history[-1] + self.tmp + self.executed_time_algorithm_history[-1])
        self.start_executed_time_algorithm = time.time()
        return [comp_metric, perf_metric]


if __name__ == '__main__':
    pass
