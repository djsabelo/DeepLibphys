from DeepLibphys.utils.functions.common import ModelType
from DeepLibphys.utils.functions.database import SignalType


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

    def __init__(self, model_name, signal_directory, signal_dim=64, hidden_dim=256, learning_rate_val=0.01,
                 batch_size=256, window_size=256, number_of_epochs=100000, save_interval=1000, number_of_batches=1,
                 mini_batch_size=16, bptt_truncate=-1, signal_type=None, model_type=None, decay=0.9, tolerance=1e-5):
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