from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import *
import DeepLibphys.models.LibphysMBGRU as GRU


def prepare_data(windows, signal2model, overlap=0.11, batch_percentage=1):
    x__matrix, y__matrix = [], []
    window_size, max_batch_size, mini_batch_size = \
        signal2model.window_size, signal2model.batch_size, signal2model.mini_batch_size
    reject = 0
    total = 0
    for w, window in enumerate(windows):
        if len(window) > window_size+1:
            for s_w in range(0, len(window) - window_size - 1, int(window_size*overlap)):
                small_window = np.round(window[s_w:s_w + window_size])
                # print(np.max(small_window) - np.min(small_window))
                if (np.max(small_window) - np.min(small_window)) \
                        > 0.8*signal2model.signal_dim:
                    x__matrix.append(window[s_w:s_w + window_size])
                    y__matrix.append(window[s_w + 1:s_w + window_size + 1])
                else:
                    reject += 1

                total += 1

    x__matrix = np.array(x__matrix)
    y__matrix = np.array(y__matrix)

    max_index = int(np.shape(x__matrix)[0]*batch_percentage)
    batch_size = int(max_batch_size) if max_batch_size < max_index else \
        max_index - max_index % mini_batch_size

    indexes = np.random.permutation(int(max_index))[:batch_size]

    print("Windows of {0}: {1}; Rejected: {2} of {3}".format(signal2model.model_name, batch_size, reject, total))
    return x__matrix[indexes], y__matrix[indexes]


if __name__ == "__main__":
    signal_dim = 256
    hidden_dim = 256
    mini_batch_size = 16
    batch_size = 128
    window_size = 512
    save_interval = 1000


    # old_signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(128, 1024)
    signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format('NEW', window_size)
    old_signal_directory = signal_directory
    noise_removed_path = "Data/CYBHi/signals.npz"
    # processed_data_path = '../data/processed/biometry_cybhi[256].npz'
    fileDir = "Data/CYBHi"

    pos_noise_rem_signals = np.load(noise_removed_path)
    signals = np.load(fileDir + "/signals.npz")["signals"]
    z = 0
    z1 = 70
    for s, signal_data in enumerate(signals[z:z1]):
        name = 'ecg_cybhi_' + signal_data.name
        signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                    batch_size=batch_size,
                                    mini_batch_size=mini_batch_size, window_size=window_size,
                                    save_interval=save_interval, tolerance=1e-9, count_to_break_max=10)

        x_train, y_train = prepare_data(signal_data.processed_train_windows, signal2model, batch_percentage=0.5)
        model = GRU.LibphysMBGRU(signal2model)
        try:
            model.load(model.get_file_tag(-5, -5), old_signal_directory)
        except:
            try:
                model.load(model.get_file_tag(-5, -5), signal_directory)
            except:
                model = GRU.LibphysMBGRU(signal2model)

        running_ok = False
        while not running_ok:
            print("Compiling Model {0} for {1}".format(name, signal_data.name))
            print("Initiating training... ")
            running_ok = model.train_model(x_train, y_train, signal2model)
            if not running_ok:
                model = GRU.LibphysMBGRU(signal2model)

        model.save(dir_name=signal_directory)


# print("Training {0} of {1}".format(len(train_dates) - len(noisy_data), len(train_dates)))


# for i, signal, person_name in zip(range(z, len(train_signals)), train_signals[z:], train_names[z:]):
#     if noisy_data.count(person_name) == 0:
#         name = 'ecg_cybhi_' + person_name
#         signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
#                                     batch_size=batch_size,
#                                     mini_batch_size=mini_batch_size, window_size=window_size,
#                                     save_interval=save_interval)
#
#         running_ok = False
#         while not running_ok:
#             print("Compiling Model {0} for {1} and date {2}".format(name, person_name, train_dates[i]))
#             model = GRU.LibphysMBGRU(signal2model)
#             print("Initiating training... ")
#             n_for_each = batch_size if batch_size % signal2model.mini_batch_size == 0 \
#                 else signal2model.mini_batch_size * int(batch_size / signal2model.mini_batch_size)
#
#
#             x_train = signal[indexes, :-1]
#             y_train = signal[indexes, 1:]
#             running_ok = model.train_model(x_train, y_train, signal2model)
#             # running_ok = model.train(signal, signal2model, train_ratio=1)
#         model.save(dir_name=signal_directory)
