from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.signal2model import Signal2Model
import DeepLibphys.models.LibphysMBGRU as GRU

mit_dir = RAW_SIGNAL_DIRECTORY + 'MIT-Arrythmia'
processed_data_path = '../data/biometry_mit[64].npz'
signals = get_dataset_files(64, val='val', row=0, dataset_dir=mit_dir, peak_into_data=False)
np.savez(processed_data_path, signals=signals)

signal_dim = 64
hidden_dim = 256
mini_batch_size = 16
batch_size = 128
window_size = 256
save_interval = 1000
signal_directory = 'ECG_BIOMETRY[{0}.{1}]'.format(batch_size, window_size)


signals = np.load(processed_data_path)['signals']
z = 0
for i, signal in zip(range(z, len(signals)), signals[z:]):
    name = 'ecg_mit_' + str(i)
    signal2model = Signal2Model(name, signal_directory, signal_dim=signal_dim, hidden_dim=hidden_dim,
                                batch_size=batch_size,
                                mini_batch_size=mini_batch_size, window_size=window_size,
                                save_interval=save_interval)

    running_ok = False
    while not running_ok:
        print("Compiling Model {0}".format(name))
        model = GRU.LibphysMBGRU(signal2model)
        print("Initiating training... ")
        running_ok = model.train(signal, signal2model)
