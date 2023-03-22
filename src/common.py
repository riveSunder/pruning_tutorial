from autograd import numpy as np
from autograd import grad

relu = lambda x: x * (x > 0.0)
fn_type = type(relu)
nparray = type(np.random.rand(1,2))

sm = lambda x: np.exp(x)/(np.sum(np.exp(x), \
        axis=-1, keepdims=True))
mse_loss_fn = lambda y, p: np.mean((y-p)**2)
nll_loss_fn = lambda y, pred: -np.mean(\
        y*np.log(sm(pred)) + (1-y)*np.log(1. - sm(pred)))
