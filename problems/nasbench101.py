import time
import numpy as np
import pickle as p
from problems import Problem
from zero_cost_methods import modify_input_for_fitting

from nasbench import wrap_api as api

"""
0: CONV 1x1
1: CONV 3x3
2: MAXPOOL 3x3
"""
class NASBench101(Problem):
    def __init__(self, max_eval, dataset='CIFAR-10', **kwargs):
        """
        # NAS-Benchmark-101 provides us the information (e.g., the testing accuracy, the validation accuracy,
        the number of parameters) of all architectures in the search space. Therefore, if we want to evaluate any
        architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        - database_path -> the path contains NAS-Bench-101 data.
        - data -> NAS-Bench-101 data.
        - pareto_front_testing -> the Pareto-optimal front in the search space (nPrams - validation error)
        - OPS -> the available operators can choose in the search space.
        - IDX_OPS -> the index of operators in compact architecture.
        - EDGES -> 0: doesn't have edge; 1: have edge.
        - IDX_EDGES -> the index of edges in compact architecture.
        - maxLength -> the maximum length of compact architecture.
        """

        super().__init__(max_eval, 'NASBench101', dataset, **kwargs)

        self.objective_0 = 'nParams'
        self.objective_1 = 'test_error'

        self.OPS = [2, 3, 4]
        self.IDX_OPS = [1, 3, 6, 10, 15]

        self.EDGES = [0, 1]
        self.IDX_EDGES = [2, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27]

        self.maxLength = 28
        self.api = api.NASBench_()

        self.database_path = kwargs['database_path'] + '/NASBench101'
        self.data = None
        self.min_max = None

        self.pareto_front_testing = None
        self.zc_predictor = None

    def _set_up(self):
        f_data = open(f'{self.database_path}/data.p', 'rb')
        self.data = p.load(f_data)
        f_data.close()

        f_pareto_front_testing = open(f'{self.database_path}/pareto_front(testing).p', 'rb')
        self.pareto_front_testing = p.load(f_pareto_front_testing)
        f_pareto_front_testing.close()

        print('--> Set Up - Done')

    def X2matrices(self, X):
        edges_matrix = np.zeros((7, 7), dtype=np.int8)
        for row in range(6):
            idx_list = None
            if row == 0:
                idx_list = [2, 4, 7, 11, 16, 22]
            elif row == 1:
                idx_list = [5, 8, 12, 17, 23]
            elif row == 2:
                idx_list = [9, 13, 18, 24]
            elif row == 3:
                idx_list = [14, 19, 25]
            elif row == 4:
                idx_list = [20, 26]
            elif row == 5:
                idx_list = [27]
            for i, edge in enumerate(idx_list):
                if X[edge] - 1 == 0:
                    edges_matrix[row][row + i + 1] = 1

        ops_matrix = ['input']
        for i in self.IDX_OPS:
            if X[i] == 2:
                ops_matrix.append('conv1x1-bn-relu')
            elif X[i] == 3:
                ops_matrix.append('conv3x3-bn-relu')
            else:
                ops_matrix.append('maxpool3x3')
        ops_matrix.append('output')

        return edges_matrix, ops_matrix

    def set_zero_cost_predictor(self, zc_predictor):
        self.zc_predictor = zc_predictor

    def _get_a_compact_architecture(self):
        arch = np.zeros(self.maxLength, dtype=np.int8)
        arch[self.IDX_OPS] = np.random.choice(self.OPS, len(self.IDX_OPS))
        arch[self.IDX_EDGES] = np.random.choice(self.EDGES, len(self.IDX_EDGES))
        arch[0] = 1
        arch[21] = 5
        return arch

    def get_key_in_data(self, X):
        edges_matrix, ops_matrix = self.X2matrices(X)
        model_spec = api.ModelSpec(edges_matrix, ops_matrix)
        key = self.api.get_module_hash(model_spec)
        return key

    """--------------------------------------------- Performance Metrics --------------------------------------------"""
    def get_cost_time(self, arch, epoch):
        key = self.get_key_in_data(arch)
        if key not in self.D.keys():
            cost_time = self.data[f'{epoch}'][key]['train_time']
            self.D[key] = {'cost_time': cost_time}
        else:
            try:
                cost_time = self.D[key]['cost_time']
            except KeyError:
                cost_time = self.data[f'{epoch}'][key]['train_time']
                self.D[key]['cost_time'] = cost_time
        return cost_time

    def get_accuracy(self, arch, epoch, subset):
        indicator_time = 0.0
        key = self.get_key_in_data(arch)
        if key not in self.D.keys():
            if subset == 'test':
                epoch = 108
                accuracy = self.data['108'][key]['test_acc']
                self.D[key] = {'test_acc': accuracy}
            elif subset == 'val':
                accuracy = self.data[f'{epoch}'][key]['val_acc']
                self.D[key] = {'val_acc': accuracy}
            elif subset == 'train':
                accuracy = self.data[f'{epoch}'][key]['train_acc']
                self.D[key] = {'train_acc': accuracy}
            else:
                raise ValueError()
        else:
            try:
                if subset == 'test':
                    epoch = 108
                    accuracy = self.D[key]['test_acc']
                elif subset == 'val':
                    accuracy = self.D[key]['val_acc']
                elif subset == 'train':
                    accuracy = self.D[key]['train_acc']
                else:
                    raise ValueError()
            except KeyError:
                if subset == 'test':
                    epoch = 108
                    accuracy = self.data['108'][key]['test_acc']
                    self.D[key]['test_acc'] = accuracy
                elif subset == 'val':
                    accuracy = self.data[f'{epoch}'][key]['val_acc']
                    self.D[key]['val_acc'] = accuracy
                elif subset == 'train':
                    accuracy = self.data[f'{epoch}'][key]['train_acc']
                    self.D[key] = {'train_acc': accuracy}
                else:
                    raise ValueError()
        benchmark_time = self.get_cost_time(arch, epoch)
        return np.round(accuracy, 4), benchmark_time, indicator_time

    def get_error(self, arch, epoch, subset):
        accuracy, benchmark_time, indicator_time = self.get_accuracy(arch, epoch, subset)
        return np.round(1 - accuracy, 4), benchmark_time, indicator_time

    def get_tfi(self, arch):
        benchmark_time = 0.0
        key = self.get_key_in_data(arch)
        if key not in self.D.keys():
            s = time.time()
            X_modified = modify_input_for_fitting(arch, self.name)
            score = self.zc_predictor.query_(arch=X_modified)
            indicator_time = time.time() - s
            self.D[key] = {'tfi': score, 'indicator_time': indicator_time}
        else:
            try:
                score = self.D[key]['tfi']
                indicator_time = self.D[key]['indicator_time']
            except KeyError:
                s = time.time()
                X_modified = modify_input_for_fitting(arch, self.name)
                score = self.zc_predictor.query_(arch=X_modified)
                indicator_time = time.time() - s
                self.D[key]['tfi'] = score
                self.D[key]['indicator_time'] = indicator_time
        return score, benchmark_time, indicator_time

    def _get_performance_metric(self, arch, epoch, metric='error', subset='val'):
        """
        - Get the performance of architecture. E.g., the testing error, the validation error, the TSE_E\n
        - arch: the architecture we want to get its performance.
        - epoch: the epoch which we want to get the performance at.
        - metric = {'accuracy', 'error'}\n
        - subset = {'train', 'val', 'test'}\n
        """
        if metric == 'accuracy':
            perf_metric, benchmark_time, indicator_time = self.get_accuracy(arch=arch, subset=subset, epoch=epoch)
        elif metric == 'error':
            perf_metric, benchmark_time, indicator_time = self.get_error(arch=arch, subset=subset, epoch=epoch)
        elif metric == 'loss':
            pass
        elif 'tfi' in metric:
            list_perf_metric, benchmark_time, indicator_time = self.get_tfi(arch=arch)
            if 'negative' in metric:  # in case we want to replace the error rate
                if len(list_perf_metric) > 1:  # return a dict
                    perf_metric = {}
                    for metric in list_perf_metric:
                        perf_metric[metric] = -list_perf_metric[metric]
                else:  # return a single value (not a dict)
                    for metric in list_perf_metric.keys():
                        perf_metric = -list_perf_metric[metric]
        else:
            raise ValueError()
        return perf_metric, benchmark_time, indicator_time

    """--------------------------------------------- Computational Metrics ------------------------------------------"""
    def get_FLOPs(self, arch):
        pass

    def get_params(self, arch):
        """
        - Get #params of the architecture.
        """
        key = self.get_key_in_data(arch)
        params = np.round(self.data['108'][key]['n_params']/1e8, 6)
        return params

    def _get_computational_metric(self, arch, metric=None):
        """
        - In NAS-Bench-201 problem, the computational metric can be one of following metrics: {nFLOPs, #params}
        """
        assert metric is not None
        if metric == 'FLOPs':
            return self.get_FLOPs(arch)
        elif metric == 'params':
            return self.get_params(arch)
        else:
            raise ValueError(f'{metric}')

    """--------------------------------------------------------------------------------------------------------------"""
    def _evaluate(self, arch, comp_metric, perf_metric, epoch, subset):
        computational_metric = self.get_computational_metric(arch=arch, metric=comp_metric)
        performance_metric, benchmark_time, indicator_time = self.get_performance_metric(arch=arch, metric=perf_metric,
                                                                                         epoch=epoch, subset=subset)
        return computational_metric, performance_metric, benchmark_time, indicator_time

    def _isValid(self, X):
        edges_matrix, ops_matrix = self.X2matrices(X)
        model_spec = api.ModelSpec(edges_matrix, ops_matrix)
        return self.api.is_valid(model_spec)


if __name__ == '__main__':
    pass
