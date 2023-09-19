import time
import numpy as np
import pickle as p
from problems import Problem
from zero_cost_methods import modify_input_for_fitting

OP_NAMES_NB201 = ['skip_connect', 'none', 'nor_conv_3x3', 'nor_conv_1x1', 'avg_pool_3x3']
EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
available_ops = [0, 1, 2, 3, 4]

def encode_int_list_2_ori_input(int_list):
    list_ops = np.array(['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3'])
    list_int_ops = np.array(int_list)
    list_str_ops = list_ops[list_int_ops]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(*list_str_ops)

def convert_str_to_op_indices(str_encoding):
    """
    Converts NB201 string representation to op_indices
    """
    nodes = str_encoding.split('+')

    def get_op(x):
        return x.split('~')[0]

    node_ops = [list(map(get_op, n.strip()[1:-1].split('|'))) for n in nodes]

    enc = []
    for u, v in EDGE_LIST:
        enc.append(OP_NAMES_NB201.index(node_ops[v - 2][u - 1]))

    return tuple(enc)


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

        self.objective_0 = kwargs['f0']
        self.objective_1 = 'test_error'

        assert self.objective_0 in ['params', 'flops', 'latency'], ValueError(f'Wrong objective: {self.objective_0}')

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

        self.zc_predictor = None

    def _set_up(self):
        available_subsets = ['CIFAR-10', 'CIFAR-100', 'ImageNet16-120']
        if self.dataset not in available_subsets:
            raise ValueError(f'Just only supported these subsets: CIFAR-10; CIFAR-100; ImageNet16-120.'
                             f'{self.dataset} subset is not supported at this time.')

        f_data = open(f'{self.database_path}/[{self.dataset}]_data.p', 'rb')
        self.data = p.load(f_data)
        f_data.close()

        import json
        self.zc_benchmark = json.load(open(self.database_path + '/zc_nasbench201.json'))

        self.logsynflow_nwot = p.load(open(self.database_path + '/logsynflow_nwot.p', 'rb'))

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
        # key = get_key_in_data(arch)
        # if key not in self.D.keys():
        #     s = time.time()
        #     X_modified = modify_input_for_fitting(arch, self.name)
        #     score = self.zc_predictor.query_(arch=X_modified)
        #     indicator_time = time.time() - s
        #     self.D[key] = {'tfi': score, 'indicator_time': indicator_time}
        # else:
        #     try:
        #         score = self.D[key]['tfi']
        #         indicator_time = self.D[key]['indicator_time']
        #     except KeyError:
        #         s = time.time()
        #         X_modified = modify_input_for_fitting(arch, self.name)
        #         score = self.zc_predictor.query_(arch=X_modified)
        #         indicator_time = time.time() - s
        #         self.D[key]['tfi'] = score
        #         self.D[key]['indicator_time'] = indicator_time
        str_input = encode_int_list_2_ori_input(arch)
        op_indices = str(convert_str_to_op_indices(str_input))
        metric = self.zc_predictor.method_type
        score = {}
        indicator_time = 0.0
        if isinstance(metric, list):
            for m in metric:
                score[m] = self.zc_benchmark['cifar10'][op_indices][m]['score']
                indicator_time += self.zc_benchmark['cifar10'][op_indices][m]['time']
        else:
            score = {metric: self.zc_benchmark['cifar10'][op_indices][metric]['score']}
            indicator_time = self.zc_benchmark['cifar10'][op_indices][metric]['time']
        # score, indicator_time = {metric: self.zc_benchmark['cifar10'][op_indices][metric]['score']}, self.zc_benchmark['cifar10'][op_indices][metric]['time']
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

    def get_free_metrics(self, arch, metric):
        # Following the paper:
        # FreeREA: Training-Free Evolution-Based Architecture Search
        str_input = encode_int_list_2_ori_input(arch)
        op_indices = str(convert_str_to_op_indices(str_input))

        key = get_key_in_data(arch)
        idx = self.data['200'][key]['idx']
        if self.dataset == 'CIFAR-10':
            dataset_ = 'cifar10'
        elif self.dataset == 'CIFAR-100':
            dataset_ = 'cifar100'
        else:
            dataset_ = self.dataset
        info = self.logsynflow_nwot[dataset_][idx][metric]
        score = info['score']
        if 'synflow' in metric:
            indicator_time = self.zc_benchmark['cifar10'][op_indices]['synflow']['time']
        elif metric == 'nwot':
            indicator_time = self.zc_benchmark['cifar10'][op_indices]['jacov']['time']
        else:
            raise ValueError
        return score, indicator_time

    @staticmethod
    def get_skip(arch):
        genotype = encode_int_list_2_ori_input(arch)
        levels = genotype.split('+')
        max_len = 0
        counter = 0

        for idx, level in enumerate(levels):
            level = level.split('|')[1:-1]
            n_genes = len(level)

            for i in range(n_genes):
                if 'skip' in level[i]:
                    counter += 1
                    min_edge = idx - i
                    max_len += min_edge
        if counter:
            return max_len / counter
        return 0

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
        if metric == 'flops':
            return self.get_FLOPs(arch)
        elif metric == 'params':
            return self.get_params(arch)
        elif metric == 'latency':
            key = get_key_in_data(arch)
            latency = self.data['200'][key]['latency']
            return latency
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
