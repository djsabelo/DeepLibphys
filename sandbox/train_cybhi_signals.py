from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import Signal2Model
import DeepLibphys.models.LibphysMBGRU as GRU

signal_dim = 256
hidden_dim = 256
mini_batch_size = 8
batch_size = 112
window_size = 1024
save_interval = 1000

cyb_dir = RAW_SIGNAL_DIRECTORY + 'CYBHi/data/long-term'
processed_data_path = '../data/biometry_cybhi[256].npz'
# train_dates, train_names, train_signals, test_dates, test_names, test_signals = \
#     get_cyb_dataset_files(signal_dim, val='val', row=0, dataset_dir=cyb_dir, peak_into_data=1000)
#
# np.savez(processed_data_path, train_dates=train_dates, train_names=train_names, train_signals=train_signals,
#          test_dates=test_dates, test_names=test_names, test_signals=test_signals)


signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

npzfile = np.load(processed_data_path)
train_dates, train_names, train_signals, test_dates, test_names, test_signals = \
    npzfile['train_dates'], npzfile['train_names'], npzfile['train_signals'], npzfile['test_dates'], \
    npzfile['test_names'], npzfile['test_signals']
z = 3
indexes = sorted(range(len(train_names)), key=lambda k: train_names[k])
train_dates, train_names, train_signals = train_dates[indexes], train_names[indexes], train_signals[indexes]
train_data = [train_dates, train_names, train_signals]

indexes = sorted(range(len(test_signals)), key=lambda k: test_signals[k])
train_dates, train_names, train_signals = train_dates[indexes], train_names[indexes], train_signals[indexes]
train_data = [train_dates, train_names, train_signals]

for i, signal in zip(range(z, len(train_signals)), train_signals[z:]):

    name = 'ecg_cybhi_' + str(i)
    signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                batch_size=batch_size,
                                mini_batch_size=mini_batch_size, window_size=window_size,
                                save_interval=save_interval)

    running_ok = False
    while not running_ok:
        print("Compiling Model {0} for {1} and date {2}".format(name, train_names[i], train_dates[i]))
        model = GRU.LibphysMBGRU(signal2model)
        print("Initiating training... ")
        running_ok = model.train(signal, signal2model, train_ratio=1)
