import DeepLibphys.utils.functions.libphys_GRU as GRU
from DeepLibphys.utils.functions.common import get_fantasia_dataset
from DeepLibphys.utils.functions.signal2model import Signal2Model

fantasia_list = [2, 3, 4 ,5, 6, 7, 8, 9, 10]
signals = [Signal2Model("resp_"+str(i), "RESP_FANTASIA[128.256]", save_interval=1000, number_of_batches=1, batch_size=256) for i in fantasia_list]

X_train, Y_train = get_fantasia_dataset(signals[0].signal_dim, fantasia_list, 'Fantasia/RESP/mat/', peak_into_data=False)

signal2model = Signal2Model("resp_[1.2.3.4.5.6.7.8.9.10]_old_fantasia", "RESP_FANTASIA[1000.256]", save_interval=1000, hidden_dim=256, batch_size=500)

model = GRU.LibPhys_GRU(signal2model.signal_dim, hidden_dim=signal2model.hidden_dim, signal_name=signal2model.signal_name)
model.save(signal2model.signal_directory, model.get_file_tag(-1, -1))
model.train_signals(X_train, Y_train, signal2model)

for i in [0, 1, 2, 3, 4, 5, 7]:
    model = GRU.LibPhys_GRU(signals[i].signal_dim, hidden_dim=signals[i].hidden_dim, signal_name=signals[i].signal_name)
    model.save(signals[i].signal_directory, model.get_file_tag(-1, -1))
    model.train_signal(X_train[i], Y_train[i], signals[i], track_loss=False, save_distance=100)