from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import Signal2Model
import DeepLibphys.models.LibphysMBGRU as GRU
import time

# mit_dir, processed_data_path, core_name = RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia', \
#                                           '../data/processed/biometry_mit[256].npz', 'ecg_mit_arrythmia_'
# mit_dir, processed_data_path, core_name = RAW_SIGNAL_DIRECTORY + 'MIT-Sinus',\
#                                          '../data/processed/biometry_mit_sinus[256].npz', 'ecg_mit_sinus_'
# mit_dir, processed_data_path, core_name = RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term',\
#                                           '../data/processed/biometry_mit_long_term[256].npz', 'ecg_mit_long_term_'
def get_variables(param):
    if param == "arr":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia', '../data/processed/biometry_mit[256].npz', 'ecg_mit_arrythmia_'
    if param == "sinus":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Sinus', '../data/processed/biometry_mit_sinus[256].npz', 'ecg_mit_sinus_'
    if param == "long":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term', '../data/processed/biometry_mit_long_term[256].npz', \
                'ecg_mit_long_term_'

    return None

signal_dim = 256
hidden_dim = 256
mini_batch_size = 16
batch_size = 128
window_size = 1024
save_interval = 1000
signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

# signal_directory_old = 'ECG_BIOMETRY[{0}.{1}]'.format(128, 512)

# mit_dir, processed_data_path, core_name = get_variables('arr')
# signals = get_mit_dataset_files(signal_dim, type='arr', val='val', row=0, dataset_dir=mit_dir, peak_into_data=False,
#                                 confidence=0.05)
# np.savez(processed_data_path, signals=signals)
# for i in signals:
#     plt.plot(i[34000:44000])
#     plt.ylim([-1, 257])
#     plt.show()

signals = []
core_names = []
for s_index, db_name in enumerate(["arr", "sinus", "long"]):
    mit_dir, processed_data_path, core_name = get_variables(db_name)
    signals_aux = np.load(processed_data_path)['signals']
    signals += extract_test_part(signals_aux)
    core_names += [core_name + str(i) for i in range(0, np.shape(signals_aux)[0])]

signals = np.array(signals)
core_names = np.array(core_names)
# renegade_indexes = [3, 30, 32, 38, 39, 40, 41, 44, 49, 51, 64]
# for i, signal in enumerate(signals[renegade_indexes]):
#     plt.subplot(5, 1, (i % 5) + 1)
#     sep = int(len(signal)*0.33)
#     plt.plot(np.arange(sep), signal[:sep], 'r')
#     plt.plot(np.arange(sep, len(signal)), signal[sep:], 'k')
#     if i % 5 == 4 :
#         plt.show()
#
# plt.show()

# for signal in signals:
#     plt.plot(signal)
#     plt.show()

# print(np.shape(signals)[0])
step = 10
N = np.shape(signals)[0]
z = [np.arange(s, s + step if N > (s + step) else N) for s in range(0, N, step)]
index = 6
indexes = z[index]
# indexes = renegade_indexes
index = "Last"
indexes = [19]

print("Training {1} - {0}".format(indexes, index))

# for i, signal, core_name in zip(indexes, signals[indexes], core_names[indexes]):
#     if i not in renegade_indexes:
#         signal2model = Signal2Model(core_name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
#                                     batch_size=batch_size,
#                                     mini_batch_size=mini_batch_size, window_size=window_size,
#                                     save_interval=save_interval, lower_error=1e-5, count_to_break_max=15)
#
#         running_ok = False
#         while not running_ok:
#             # model.load(model.get_file_tag(), signal_directory)
#             print("Initiating training... ")
#             last_index = int(len(signal) * 0.33)
#             x_train, y_train = prepare_test_data([signal[:last_index]], signal2model, mean_tol=0.9, std_tol=0.5)
#             print("Compiling Model {0}".format(core_name))
#             model = GRU.LibphysMBGRU(signal2model)
#
#             model.start_time = time.time()
#             returned = model.train_model(x_train, y_train, signal2model)
#             # if i == 16:
#             #     model.load(dir_name=signal2model.signal_directory, file_tag=model.get_file_tag(0, 1000))
#             if returned:
#                 model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))
#             running_ok = returned

x_trains, y_trains, signals_2_models = [], [], []
for i, signal, core_name in zip(indexes, signals[indexes], core_names[indexes]):
    signal2model = Signal2Model(core_name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                batch_size=batch_size+5,
                                mini_batch_size=mini_batch_size, window_size=window_size,
                                save_interval=save_interval, lower_error=1e-6, count_to_break_max=15)
    # last_index = int(len(signal) * 0.33)
    # x_train, y_train = prepare_test_data([signal[:last_index]], signal2model, mean_tol=0.9, std_tol=0.2)
    # x_train, y_train = windows_selection(x_train, y_train)
    # x_trains.append(x_train)
    # y_trains.append(y_train)
    signals_2_models.append(signal2model)

# np.savez("x_trains__.npz", x_trains=x_trains, y_trains=y_trains)
file = np.load("x_trains__.npz")
x_trains, y_trains = file["x_trains"], file["y_trains"]

for x_train, y_train, signal2model in zip(x_trains, y_trains, signals_2_models):
    running_ok = False
    while not running_ok:
        print("Initiating training... ")
        print("Compiling Model {0}".format(signal2model.model_name))
        model = GRU.LibphysMBGRU(signal2model)

        model.start_time = time.time()
        indexes = range(len(x_train) - (len(x_train) % mini_batch_size))
        returned = model.train_model(x_train[indexes], y_train[indexes], signal2model)
        # if i == 16:
        model.load(dir_name=signal2model.signal_directory, file_tag=model.get_file_tag(0, 1000))
        if returned:
            model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))
        running_ok = returned