from DeepLibphys.utils.functions.common import get_fantasia_dataset
from DeepLibphys.utils.functions.libphys_GRU import LibPhys_GRU

SIGNAL_DIRECTORY = '../utils/data/trained/'

signal_dim = 32

X, Y = get_fantasia_dataset(signal_dim, [5], 'Fantasia/mat/', peak_into_data=True)
x = X[2]
y = Y[2]

