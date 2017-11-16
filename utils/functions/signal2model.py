
from DeepLibphys.utils.functions.database import SignalType


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

    def __init__(self, model_name, signal_directory, signal_dim=64, hidden_dim=256, learning_rate_val=0.01,
                 batch_size=256, window_size=256, number_of_epochs=100000, save_interval=1000, number_of_batches=1,
                 mini_batch_size=16, bptt_truncate=-1, signal_type=None, model_type=None, decay=0.9, tolerance=1e-5,
                 lower_error=1e-3, lower_learning_rate=1e-5, count_to_break_max=5):
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
        self.lower_error, self.lower_learning_rate, self.count_to_break_max = \
            lower_error, lower_learning_rate, count_to_break_max


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
            train_signal, test_signal, test_signal, train_windows, user_name


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
