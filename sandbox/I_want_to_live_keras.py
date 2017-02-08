import numpy as np
import matplotlib.pyplot as plt
import pickle
from DeepLearningTryAndError.load_ecg_data import get_fantasia_dataset, get_dictionary
import os.path
from keras.models import Sequential, Model
from keras.layers import GRU,TimeDistributedDense

def generate_predicted_signal(model, N):
    # We start the sentence with the start token
    new_signal = [0]
    # Repeat until we get an end token
    for i in range(N):
        next_sample_probs = model.predict(new_signal)
        sample = np.random.multinomial(1, next_sample_probs[-1])
        next_sample = np.argmax(sample)
        new_signal.append(next_sample)
        # sampled_word = word_to_index[unknown_token]

    return new_signal


def prepare_ecg_data(filename='fantasia_dataset', dictionary_size=10000,
                     window_size=200, n_examples=20, n_windows=300, Nx=[1]):
    N = n_examples * n_windows * window_size
    M = window_size
    B = n_windows * window_size  # each batch size

    ecgs, ecgs_dict = get_fantasia_dataset(dictionary_size, Nx)
    if ecgs[0] is not list:
        ecgs = [ecgs]
    X_train = np.zeros((N, M), dtype=int)
    y_train = np.zeros(N, dtype=int)

    ecg_example_index = 0

    for ecg in ecgs[0:n_examples]:
        for i in range(M):
            index = ecg_example_index * n_windows + i
            X_train[list(range(index, index + B, window_size)), :] = np.reshape(ecg[i: window_size * n_windows + i],
                                                                                (n_windows, window_size))
            y_train[index:index + B: window_size] = np.asarray(
                ecg[i + window_size + 1: i + (1 + n_windows) * window_size: window_size])

        ecg_example_index += 1

    with open(filename + '.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([X_train, y_train, ecgs_dict], f)

    # Matrix

    return X_train, y_train, ecgs_dict


def get_tranning_dictionary(y_train, dictionary_size):
    y = np.zeros((np.size(y_train),dictionary_size))
    y[y_train-min(y_train),:] = 1
    return y


def prepare_old_ecg_data(filename='fantasia_dataset', dictionary_size=10000):
    ecgs, ecgs_dict = get_fantasia_dataset(dictionary_size)
    X_train = ecgs
    N = len(ecgs)
    y_train = np.zeros(N).tolist()

    for i in range(N):
        y_train[i] = X_train[i][1:] + [0]

    with open(filename + '.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([X_train, y_train, ecgs_dict], f)

    return X_train, y_train, ecgs_dict


# configuration
window_size = 128
n_examples = 1
n_windows = 300
dictionary_size = 1000
batch_size = 32

filename = 'fantasia_dataset_' + str(dictionary_size) + "_" + str(n_examples) + "_keras"
X_train, y_train, ecgs_dict = [[], [], {}]

if os.path.isfile(filename):
    with open(filename + '.pickle', 'rb') as f:
        X_train, y_train, ecgs_dict = pickle.load(f)
else:
    X_train, y_train, ecgs_dict = prepare_ecg_data(filename, dictionary_size, Nx=[1])

plt.plot(X_train[0, :])
plt.show()

np.random.seed(10)

dictionary_size = len(ecgs_dict)

y_train = get_tranning_dictionary(y_train, dictionary_size)

ecg_model = Sequential()
# language_model.add(Embedding(dictionary_size, 256, input_length=128))
ecg_model.add(GRU(256, return_sequences=True, input_shape=window_size))
# ecg_model.add(TimeDistributedDense(256))
ecg_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
ecg_model.fit(X_train, y_train, batch_size=16, nb_epoch=100)
open('ecg_model.json', 'wb').write(ecg_model.to_json())


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# train the model, output generated text after each iteration


N = 1000
plt.plot(generate_predicted_signal(ecg_model, N))
plt.show()
plt.figure()
plt.plot(X_train[:, range(1000)])
plt.show()
