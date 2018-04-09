class ModelInfo:
    Sd, Hd, dataset_name, directory, name, DS, t, W = 0, 0, "", "", "", -5, -5, 256

    def __init__(self, Sd=64, Hd=256, dataset_name="", directory="", DS=-5, t=-5, W=256, name="model"):
        """
        :param Sd: signal_dimension
        :param Hd: hidden_dimension
        :param dataset_name: signal_name - name in which the model was saved
        :param directory: where the model was saved
        :param DS: recorded dataset #
        :param t:  recorded epoch # for corresponding dataset
        :param W: default window size
        :param name: model name to be displayed
        """

        self.Sd = Sd
        self.Hd = Hd
        self.dataset_name = dataset_name
        self.directory = directory
        self.name = name
        self.Sd = Sd
        self.t = t
        self.W = W
        self.DS = DS

    def to_signal2model(self):
        return Signal2Model(self.dataset_name,
                            self.directory,
                            signal_dim=self.Sd,
                            hidden_dim=self.Hd,
                            window_size=self.W)


class SignalInfo:
    type, directory, index, size, name, file_name = "", "", 0, 0, "", ""

    def __init__(self, type, directory, index, size, name, file_name=""):
        self.type = type
        self.directory = directory
        self.index = index
        self.size = size
        self.name = name
        self.file_name = file_name


#TODO: Make SignalType Default in all functions
class SignalType:
    EEG, ECG, EMG, GSR, RESP, ACC, BIOMETRIC_ACC, OTHER = range(8)

class ModelType:
    MINI_BATCH, SGD, CROSS_SGD, CROSS_MBSGD = range(4)

class Signal2Model(object):
    signal_dim = 0
    hidden_dim = 0
    model_name = None
    signal_directory = None
    learning_rate_val = 0
    batch_size = 0
    window_size = 0
    number_of_epochs = 0
    save_interval = 0
    number_of_batches = 0
    mini_batch_size = 0
    bptt_truncate = 0
    signal_type, model_type = [0, 0]
    decay = 0
    lower_error, lower_learning_rate, count_to_break_max = 0, 0, 0
    n_grus = 0
    global_optimizer_norm = 0
    n_signals = 0

    def __init__(self, model_name, signal_directory, signal_dim=64, hidden_dim=256, learning_rate_val=0.01,
                 batch_size=256, window_size=256, number_of_epochs=100000, save_interval=1000, number_of_batches=1,
                 mini_batch_size=16, bptt_truncate=-1, signal_type=None, model_type=None, decay=0.9, tolerance=1e-5,
                 lower_error=1e-3, lower_learning_rate=1e-5, count_to_break_max=5, n_grus=2, global_optimizer_norm=1,
                 n_signals=1):
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.signal_directory = signal_directory
        self.learning_rate_val = learning_rate_val
        self.batch_size = batch_size
        self.mini_batch_size = batch_size
        self.window_size = window_size
        self.number_of_epochs = number_of_epochs
        self.save_interval = save_interval
        self.number_of_batches = number_of_batches
        self.mini_batch_size = mini_batch_size
        self.bptt_truncate = bptt_truncate
        self.tolerance = tolerance
        self.n_grus = n_grus
        self.lower_error, self.lower_learning_rate, self.count_to_break_max = \
            lower_error, lower_learning_rate, count_to_break_max
        self.global_optimizer_norm = global_optimizer_norm
        self.n_signals = n_signals

        if signal_type is None:
            signal_type = SignalType.OTHER
        else:
            self.signal_type = signal_type

        if model_type is None:
            self.model_type = ModelType.MINI_BATCH
        else:
            self.signal_type = signal_type

        self.signal_dim = signal_dim
        self.decay = decay


class Signal(object):
    train_signal, test_signal, train_windows, processed_test_windows, processed_train_windows, test_windows, name = \
        [], [], [], [], [], [], ""

    def __init__(self, train_signal, test_signal, train_windows, test_windows, user_name):
        self.train_signal, self.test_signal, self.train_windows, self.test_windows, self.name = \
            train_signal, test_signal, train_windows, test_windows, user_name


class TimeWindow(object):
    signal, start_index, end_index, size = [], 0, 0, 0

    def __init__(self, start_index, end_index, full_signal=None):
        if full_signal is not None:
            self.signal = full_signal[start_index:end_index]

        self.size = end_index - start_index
        self.end_index = end_index
        self.start_index = start_index

    def set_signal(self, full_signal):
            self.signal = full_signal[self.start_index:self.end_index]
