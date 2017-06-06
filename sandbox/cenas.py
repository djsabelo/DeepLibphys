import theano
import theano.tensor as T
from theano.tensor import printing
import numpy as np

import DeepLibphys
import DeepLibphys.models.LibphysMBGRU as GRU
import DeepLibphys.utils.functions.database as db
from DeepLibphys.utils.functions.common import *
from DeepLibphys.utils.functions.database import ModelInfo
from DeepLibphys.utils.functions.signal2model import Signal2Model
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import math
import time
import seaborn
from matplotlib.backends.backend_pdf import PdfPages
from theano.tensor.raw_random import RandomStreamsBase

N = T.iscalar('N')
U, W = T.dtensor3s('Ux', 'Wx')
E, V = T.dmatrices('Ex', 'Vx')
b = T.dmatrix('bx')
c = T.dvector('cx')
XC = T.ivector('XC')
y_hat_ = T.ivector('y_hat_')
model_info = db.ecg_1024_clean_models[0]

signal2Model = Signal2Model(model_info.dataset_name, model_info.directory, signal_dim=model_info.Sd,
                            hidden_dim=model_info.Hd)
model = DeepLibphys.models.LibphysMBGRU.LibphysMBGRU(signal2Model)
model.load(dir_name=model_info.directory)

def forward_prop_predict(x_t, s_prev1, s_prev2, s_prev3, E, V, U, W, b, c):
    trng = T.shared_randomstreams.RandomStreams(1234)
    # Embedding layer
    x_e = E[:, x_t]
    print_x = printing.Print('x')


    def GRU(i, U, W, b, x_0, s_previous):
        z = T.nnet.hard_sigmoid(U[i * 3 + 0].dot(x_0) + W[i * 3 + 0].dot(s_previous) + b[i * 3])
        r = T.nnet.hard_sigmoid(U[i * 3 + 1].dot(x_0) + W[i * 3 + 1].dot(s_previous) + b[i * 3 + 1])
        s_candidate = T.tanh(U[i * 3 + 2].dot(x_0) + W[i * 3 + 2].dot(s_previous * r) + b[i * 3 + 2])

        return (T.ones_like(z) - z) * s_candidate + z * s_previous

    # GRU Layer 1
    s1 = GRU(0, U, W, b, print_x(x_e), s_prev1)

    # GRU Layer 2
    s2 = GRU(1, U, W, b, s1, s_prev2)

    # GRU Layer 3
    s3 = GRU(2, U, W, b, s2, s_prev3)

    # Final output calculation
    o_t = T.nnet.softmax(V.dot(s3) + c)[0]
    print_p = printing.Print('y')
    # o_ = print_ot(print_p(o_t))

    y_hat = T.argmax(o_t)
    # prob = o_t/T.sum(o_t)
    #
    # ran = RandomStreamsBase()
    #
    # y_hat = ran.choice(size=(1), a=64, p=prob)
    # sample = np.random.multinomial(1, next_sample_probs[-1] / np.sum(
    #     next_sample_probs[-1]))
    # next_sample = np.argmax(sample)
    # z = T.lt(T.shape(o_t)[0], 1024)
    # y_hat = T.switch(z, y_hat[-1024:], y_hat)

    return [print_p(y_hat), s1, s2, s3]


outputs, updates = theano.scan(
    fn=forward_prop_predict,
    sequences=XC,
    n_steps=np.shape(XC)[0],
    outputs_info=[None, np.zeros(model.hidden_dim), np.zeros(model.hidden_dim), np.zeros(model.hidden_dim)],
    non_sequences=[E, V, U, W, b, c]

)

#
# outputs, _ = theano.scan(
#     fn=forward_prop_predict,
#     sequences=y_hat_,
#     n_steps=np.shape(y_hat_)[0],
#     outputs_info=[None],
#     non_sequences=[E, V, U, W, b, c]
# )

predict = theano.function(
    inputs=[XC, E, V, U, W, b, c],
    outputs=outputs,
    allow_input_downcast=True
)

def cenas(n, y_hat, E, V, U, W, b, c):
    # y_hat = T.lt(T.shape(y_hat)[0], 512)
    return predict(y_hat[0], E, V, U, W, b, c)


outputs_, _ = theano.scan(
    fn=cenas,
    sequences=np.arange(N, dtype=float32),
    n_steps=N,
    non_sequences=[E, V, U, W, b, c],
    outputs_info=[np.array([np.int64(3)])]
)

cenitas = theano.function(
    inputs=[N, E, V, U, W, b, c],
    outputs=outputs_,
    allow_input_downcast=True
)

[E, V, U, W, b, c] = model.get_parameters()
x = cenitas(2000, E, V, U, W, b, c)
# print(x)
plt.plot(x)
plt.show()
