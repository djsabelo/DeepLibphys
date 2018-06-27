from DeepLibphys.utils.functions.common import *

DATA = '' #????

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