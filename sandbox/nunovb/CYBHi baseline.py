from DeepLibphys.utils.functions.common import *


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
                        > 0.7 * signal2model.signal_dim:
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




#windows_size = [256, 256, 64, 64]
win = 512
fs = 250


# Load Signal and Info
# fileDir = "Data/CYBHi-short"
# cyb_dir = RAW_SIGNAL_DIRECTORY + 'CYBHi/data/short-term'
# full_processed_dir = fileDir + "/cibhi_short_term_signals.npz"
fileDir = "/home/bento/Data/"
RAW_SIGNAL_DIRECTORY = "/home/bento"
cyb_dir = RAW_SIGNAL_DIRECTORY + 'CYBHi/data/long-term'
full_processed_dir = fileDir + "/raw_signals.npz"


file = np.load(fileDir+"/signals.npz")
signals_info = file["signals"]
names = [signal_info.name for signal_info in signals_info]

train_dates, train_names1, train_signals1, test_dates1, test_names1, test_signals = \
    get_cyb_dataset_raw_files(dataset_dir=CYBHi_ECG)

print(train_signals1.shape)

file = np.load(full_processed_dir)
train_dates, train_names, train_signals, test_dates, test_names, test_signals = \
    file["train_dates"], file["train_names"], file["train_signals"], file["test_dates"],file["test_names"], \
    file["test_signals"]


file = np.load(fileDir+"/signals.npz")
signals_info = file["signals"]
names = [signal_info.name for signal_info in signals_info]
for i, name in enumerate(train_names):
    if name not in names:
        plt.figure("Train of "+name)
        plt.plot(train_signals[i])
        plt.figure("Test of "+name)
        plt.plot(test_signals[test_names == "name"])

plt.show()
print(SSS)

train_signalz, test_signalz, names = [], [], []
for train_name, test_name in zip(train_names, test_names):
    train_signal = train_signals[np.where(train_names == train_name)[0]]
    test_signal = train_signals[np.where(test_names == train_name)[0]]
    train_signalz.append(train_signal)
    test_signalz.append(test_signal)
    names.append(train_name)

file = fileDir+"/signalz.npz"

# np.savez(fileDir+"/signalz.npz", test_signals=test_signalz, train_signals=train_signalz, names=names)