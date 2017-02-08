from BiosignalsDeepLibphys.utils.functions.common import get_signals
from BiosignalsDeepLibphys.utils.functions.libphys_GRU import LibPhys_GRU

SIGNAL_DIRECTORY = '../utils/data/trained/'

signal_dim = 32
hidden_dim = 64
signal_name = "driver_gsr"
signal_directory = 'GSR'
learning_rate_val = 0.05
batch_size = 64
window_size = 256
number_of_epochs = 2000

X, Y = get_signals(signal_dim, 'DRIVER_GSR', peak_into_data=False)
x = X[2]
y = Y[2]

X = None
Y = None

model = LibPhys_GRU(signal_dim, hidden_dim=hidden_dim, signal_name=signal_name)
model.save(signal_directory, model.get_file_tag(-1, -1))

model.train(x, y, track_loss=False, window_size=window_size, batch_size=batch_size,
            learning_rate_val=learning_rate_val, save_directory=signal_directory, nepoch_val=number_of_epochs)
