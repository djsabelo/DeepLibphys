from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import Signal2Model
import DeepLibphys.models.LibphysMBGRU as GRU

signal_dim = 256
hidden_dim = 256
mini_batch_size = 8
batch_size = 128
window_size = 1024
save_interval = 1000

cyb_dir = RAW_SIGNAL_DIRECTORY + 'CYBHi/data/long-term'
processed_data_path = '../data/biometry_segmented_cybhi[256].npz'
# train_dates, train_names, train_signals, test_dates, test_names, test_signals = \
#     get_cyb_dataset_segmented(signal_dim, window_size, peak_into_data=False, confidence=0.01)
#
# np.savez(processed_data_path, train_dates=train_dates, train_names=train_names, train_signals=train_signals,
#                   test_dates=test_dates, test_names=test_names, test_signals=test_signals)
npzfile = np.load(processed_data_path)
train_dates, train_names, train_signals, test_dates, test_names, test_signals = \
    npzfile['train_dates'], npzfile['train_names'], npzfile['train_signals'], npzfile['test_dates'], \
    npzfile['test_names'], npzfile['test_signals']


noisy_data = ['ABD', 'AG', 'AR', 'ARF', 'CB', 'DC', 'JB', 'GF', 'JCA', 'JN', 'MBA', 'MC', 'MMJ', 'MQ', 'PMA', 'RD',
              'RF', 'RL', 'RR', 'SR', 'VM']
print("Training {0} of {1}".format(len(train_dates) - len(noisy_data), len(train_dates)))
signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

z = 4
for i, signal, person_name in zip(range(z, len(train_signals)), train_signals[z:], train_names[z:]):
    if noisy_data.count(person_name) == 0:
        name = 'ecg_cybhi_' + person_name
        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                    batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval)

        running_ok = False
        while not running_ok:
            print("Compiling Model {0} for {1} and date {2}".format(name, person_name, train_dates[i]))
            model = GRU.LibphysMBGRU(signal2model)
            print("Initiating training... ")
            n_for_each = batch_size if batch_size % signal2model.mini_batch_size == 0 \
                else signal2model.mini_batch_size * int(batch_size / signal2model.mini_batch_size)

            indexes = np.random.permutation(n_for_each)
            x_train = signal[indexes, :-1]
            y_train = signal[indexes, 1:]
            running_ok = model.train_model(x_train, y_train, signal2model)
            # running_ok = model.train(signal, signal2model, train_ratio=1)
        model.save(dir_name=signal_directory)
