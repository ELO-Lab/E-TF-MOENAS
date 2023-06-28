class Problem:
    def __init__(self, max_eval, name, dataset, **kwargs):
        """
        # Hyper-parameters:\n
        - *max_eval* -> the maximum number of evaluated architecture.
        - *name* -> the name of used benchmark (or problem's name). E.g., MacroNAS, NAS-Bench-101, NAS-Bench-201, NAS-Bench-301.
        - *dataset* -> the dataset is used to train and evaluate architectures. E.g., CIFAR-10; CIFAR-100; ImageNet16-120
        - *objective_0* -> the first objective which we want to minimize. This objective usually is an efficiency metric.
        - *objective_1* -> the second objective which we want to minimize. This objective usually is the architecture's error.
        """
        self.max_eval = max_eval
        self.name = name
        self.dataset = dataset

        self.objective_0 = None
        self.objective_1 = None

        self.D = {}

    def reset(self):
        self.D = {}

    def set_up(self):
        """
        - Set up necessary things.
        """
        self._set_up()
        if (self.objective_0 is None) or (self.objective_1 is None):
            raise ValueError('The optimization objectives have not been set up.')

    def sample_a_compact_architecture(self):
        """
        Sample a compact architecture in the search space.
        """
        return self._get_a_compact_architecture()

    def get_computational_metric(self, arch, metric):
        """
        Get computational metric which be wanted to minimize, e.g., nFLOPs, nParams, MMACs, etc.
        """
        return self._get_computational_metric(arch, metric)

    def get_performance_metric(self, arch, epoch, metric, subset):
        """
        Get performance metric which be wanted to minimize, e.g., accuracy, error, etc.
        """
        return self._get_performance_metric(arch, epoch, metric, subset)

    def evaluate(self, arch, comp_metric, perf_metric, epoch, subset):
        """
        Calculate the objective values: [computational metric; performance metric]
        """
        return self._evaluate(arch, comp_metric, perf_metric, epoch, subset)

    def isValid(self, arch):
        """
        - Checking if the architecture is valid or not.\n
        - NAS-Bench-101 doesn't provide information of all architecture in the search space. Therefore, when doing experiments on this benchmark, we need to check if the architecture is valid or not.\n
        """
        return self._isValid(arch)

    def _set_up(self):
        pass

    def _get_a_compact_architecture(self):
        raise NotImplementedError

    def _get_computational_metric(self, arch, metric):
        raise NotImplementedError

    def _get_performance_metric(self, arch, epoch, metric, subset):
        raise NotImplementedError

    def _evaluate(self, X, comp_metric, perf_metric, epoch, subset):
        raise NotImplementedError

    def _isValid(self, arch):
        raise NotImplementedError
