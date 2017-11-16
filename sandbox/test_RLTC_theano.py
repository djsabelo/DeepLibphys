import DeepLibphys.classification.RLTC as RLTC
import numpy as np

filename = "../data/validation/Sep_FANTASIA_512/NEW_LOSS_FOR_SNR_RAW.npz"
loss_tensor = np.load(filename)["loss_tensor"]

normal = RLTC.normalize_theano_tensor(loss_tensor, 5)
print(normal)