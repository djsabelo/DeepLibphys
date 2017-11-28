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

signal_dim = 256
hidden_dim = 256
mini_batch_size = 16
batch_size = 256
window_size = 512
save_interval = 1000
signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)

# signal_directory_old = 'ECG_BIOMETRY[{0}.{1}]'.format(128, 512)

# signals = get_mit_dataset_files(signal_dim, val='val', row=0, dataset_dir=mit_dir, peak_into_data=False, confidence=0.005)
# np.savez(processed_data_path, signals=signals)
#


def get_variables(param):
    if param == "arr":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia', '../data/processed/biometry_mit[256].npz', 'ecg_mit_arrythmia_'
    if param == "sinus":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Sinus', '../data/processed/biometry_mit_sinus[256].npz', 'ecg_mit_sinus_'
    if param == "long":
        return RAW_SIGNAL_DIRECTORY + 'MIT-Long-Term', '../data/processed/biometry_mit_long_term[256].npz', \
                'ecg_mit_long_term_'

    return None


mit_dir, processed_data_path, core_name = get_variables("sinus")
signals = np.load(processed_data_path)['signals']
# for signal in signals:
#     plt.plot(signal)
#     plt.show()


z = np.arange(16, np.shape(signals)[0])#(35, 43)
# z = np.arange(22,33)
# z = np.arange(33, 45)
# z = np.arange(43,45)
for i, signal in zip(z, signals[z]):
    name = core_name + str(i)
    signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                batch_size=batch_size,
                                mini_batch_size=mini_batch_size, window_size=window_size,
                                save_interval=save_interval, lower_error=1e-5, count_to_break_max=15)

    running_ok = False
    while not running_ok:
        # model.load(model.get_file_tag(), signal_directory)
        print("Initiating training... ")
        last_index = int(len(signal) * 0.33)
        x_train, y_train = prepare_test_data([signal[:last_index]], signal2model, mean_tol=0.9, std_tol=0.5)
        print("Compiling Model {0}".format(name))
        model = GRU.LibphysMBGRU(signal2model)

        model.start_time = time.time()
        returned = model.train_model(x_train, y_train, signal2model)
        # if i == 16:
        #     model.load(dir_name=signal2model.signal_directory, file_tag=model.get_file_tag(0, 1000))
        if returned:
            model.save(signal2model.signal_directory, model.get_file_tag(-5, -5))
        running_ok = returned