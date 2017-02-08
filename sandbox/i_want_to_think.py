from BiosignalsDeepLibphys.utils.functions.common import get_signals
from BiosignalsDeepLibphys.utils.functions.libphys_GRU import LibPhys_GRU

SIGNAL_DIRECTORY = '../utils/data/trained/'

signal_dim = 64
hidden_dim = 128
signal_name = "eeg_2"
signal_directory = 'EEG_Attention[128.256]'
learning_rate_val = 0.05
batch_size = 128
window_size = 256
number_of_epochs = 5000

X, Y = get_signals(signal_dim, 'EEG_Attention', peak_into_data=False)

x = X[1]
y = Y[1]

X = None
Y = None

model = LibPhys_GRU(signal_dim, hidden_dim=hidden_dim, signal_name=signal_name)
model.save(signal_directory, model.get_file_tag(-1, -1))

model.train(x, y, track_loss=False, window_size=window_size, number_of_batches=1, batch_size=batch_size,
            learning_rate_val=learning_rate_val, save_directory=signal_directory, nepoch_val=number_of_epochs)
