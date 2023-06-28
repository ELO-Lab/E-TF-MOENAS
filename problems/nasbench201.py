import time
import numpy as np
import pickle as p
from problems import Problem
from zero_cost_methods import modify_input_for_fitting

def get_key_in_data(arch):
    """
    Get the key which is used to represent the architecture in "self.data".
    """
    return ''.join(map(str, arch))


class NASBench201(Problem):
    def __init__(self, dataset, max_eval, **kwargs):
        """
        # NAS-Benchmark-201 provides us the information (e.g., the training loss, the testing accuracy,
        the validation accuracy, the number of FLOPs, etc) of all architectures in the search space. Therefore, if we
        want to evaluate any architectures in the search space, we just need to query its information in the data.\n
        -----------------------------------------------------------------

        # Additional Hyper-parameters:\n
        - database_path -> the path contains NAS-Bench-201 data.
        - data -> NAS-Bench-201 data.
        - pareto_front_testing -> the Pareto-optimal front in the search space (nFLOPs - testing error)
        - available_ops -> the available operators can choose in the search space.
        - maxLength -> the maximum length of compact architecture.
        """

        super().__init__(max_eval, 'NASBench201', dataset, **kwargs)

        self.objective_0 = 'FLOPs'
        self.objective_1 = 'test_error'

        # 'none': 0
        # 'skip_connect': 1
        # 'nor_conv_1x1': 2
        # 'nor_conv_3x3': 3
        # 'avg_pool_3x3': 4
        self.available_ops = [0, 1, 2, 3, 4]
        self.maxLength = 6

        self.database_path = kwargs['database_path'] + '/NASBench201'
        self.data = None
        self.min_max = None

        self.pareto_front_testing = None
        self.zc_predictor = None

    def _set_up(self):
        available_subsets = ['CIFAR-10', 'CIFAR-100', 'ImageNet16-120']
        if self.dataset not in available_subsets:
            raise ValueError(f'Just only supported these subsets: CIFAR-10; CIFAR-100; ImageNet16-120.'
                             f'{self.dataset} subset is not supported at this time.')

        f_data = open(f'{self.database_path}/[{self.dataset}]_data.p', 'rb')
        self.data = p.load(f_data)
        f_data.close()

        f_pareto_front_testing = open(f'{self.database_path}/[{self.dataset}]_pareto_front(testing).p', 'rb')
        self.pareto_front_testing = p.load(f_pareto_front_testing)
        self.pareto_front_testing[:, 0] = np.round(self.pareto_front_testing[:, 0]/1e3, 4)
        f_pareto_front_testing.close()

        print('--> Set Up - Done')

    def set_zero_cost_predictor(self, zc_predictor):
        self.zc_predictor = zc_predictor

    def _get_a_compact_architecture(self):
        return np.random.choice(self.available_ops, self.maxLength)

    """--------------------------------------------- Performance Metrics --------------------------------------------"""
    def get_cost_time(self, arch, epoch):
        key = get_key_in_data(arch)
        if key not in self.D.keys():
            if self.dataset == 'CIFAR-10':
                cost_time = self.data['200'][key]['train_time'] / 2 * epoch
            else:
                cost_time = self.data['200'][key]['train_time'] * epoch
            self.D[key] = {'cost_time': cost_time}
        else:
            try:
                cost_time = self.D[key]['cost_time']
            except KeyError:
                if self.dataset == 'CIFAR-10':
                    cost_time = self.data['200'][key]['train_time'] / 2 * epoch
                else:
                    cost_time = self.data['200'][key]['train_time'] * epoch
                self.D[key]['cost_time'] = cost_time
        return cost_time

    def get_accuracy(self, arch, epoch, subset):
        epoch_ = epoch if epoch == -1 else epoch - 1
        indicator_time = 0.0
        key = get_key_in_data(arch)
        if key not in self.D.keys():
            if subset == 'test':
                accuracy = self.data['200'][key]['test_acc'][epoch_]
                self.D[key] = {'test_acc': accuracy}
            elif subset == 'val':
                accuracy = self.data['200'][key]['val_acc'][epoch_]
                self.D[key] = {'val_acc': accuracy}
            elif subset == 'train':
                accuracy = self.data['200'][key]['train_acc'][epoch_]
                self.D[key] = {'train_acc': accuracy}
            else:
                raise ValueError()
        else:
            try:
                if subset == 'test':
                    accuracy = self.D[key]['test_acc']
                elif subset == 'val':
                    accuracy = self.D[key]['val_acc']
                elif subset == 'train':
                    accuracy = self.D[key]['train_acc']
                else:
                    raise ValueError()
            except KeyError:
                if subset == 'test':
                    accuracy = self.data['200'][key]['test_acc'][epoch_]
                    self.D[key]['test_acc'] = accuracy
                elif subset == 'val':
                    accuracy = self.data['200'][key]['val_acc'][epoch_]
                    self.D[key]['val_acc'] = accuracy
                elif subset == 'train':
                    accuracy = self.data['200'][key]['train_acc'][epoch_]
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
        key = get_key_in_data(arch)
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

    def get_loss(self, arch, epoch, subset):
        # Following the paper:
        # Revisiting the Train Loss: an Efficient Performance Estimator for Neural Architecture Search
        # TSE-E (E = 1)
        key = get_key_in_data(arch)
        indicator_time = 0.0
        if key not in self.D.keys():
            if subset == 'train':
                loss = self.data['200'][key]['train_loss'][epoch - 1]
                self.D[key] = {'train_loss': loss}
            elif subset == 'val':
                loss = self.data['200'][key]['val_loss'][epoch - 1]
                self.D[key] = {'val_loss': loss}
            else:
                raise ValueError()
        else:
            try:
                if subset == 'train':
                    loss = self.D[key]['train_loss']
                elif subset == 'val':
                    loss = self.D[key]['val_loss']
                else:
                    raise ValueError()
            except KeyError:
                if subset == 'train':
                    loss = self.data['200'][key]['train_loss'][epoch - 1]
                    self.D[key] = {'train_loss': loss}
                elif subset == 'val':
                    loss = self.data['200'][key]['val_loss'][epoch - 1]
                    self.D[key] = {'val_loss': loss}
                else:
                    raise ValueError()
        benchmark_time = self.get_cost_time(arch, epoch)
        return np.round(loss, 4), benchmark_time, indicator_time

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
            perf_metric, benchmark_time, indicator_time = self.get_loss(arch=arch, subset=subset, epoch=epoch)
        elif 'tfi' in metric:
            list_perf_metric, benchmark_time, indicator_time = self.get_tfi(arch=arch)
            if 'negative' in metric:  # in case we want to replace the error rate
                if len(list_perf_metric) > 1:
                    perf_metric = {}
                    for metric in list_perf_metric:
                        perf_metric[metric] = -list_perf_metric[metric]
                else:
                    for key in list_perf_metric.keys():
                        perf_metric = -list_perf_metric[key]
        else:
            raise ValueError()
        return perf_metric, benchmark_time, indicator_time

    """--------------------------------------------- Computational Metrics ------------------------------------------"""
    def get_FLOPs(self, arch):
        """
        - Get #FLOPs of the architecture.
        """
        key = get_key_in_data(arch)
        nFLOPs = np.round(self.data['200'][key]['FLOPs'], 4)/1e3  # Convert to Giga (Millions at current) for reducing the IGD value
        return nFLOPs

    def get_params(self, arch):
        """
        - Get #params of the architecture.
        """
        key = get_key_in_data(arch)
        params = self.data['200'][key]['params']
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

    def _isValid(self, arch):
        return True


if __name__ == '__main__':
    pass
