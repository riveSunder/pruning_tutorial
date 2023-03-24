from autograd import numpy as np
from autograd import grad

from src.common import relu,\
    fn_type,\
    nparray,\
    mse_loss_fn

def initialize_layer_weights(in_dim: int, out_dim: int) -> nparray:

    sigma = np.sqrt(2 / (in_dim + out_dim))

    return sigma * np.random.randn(in_dim, out_dim) 

def initialize_weights(dimensions: list) -> list:

    layers = []
    for dims in dimensions:

        layers.append(initialize_layer_weights(dims[0], dims[1]))

    return layers

def initialize_model(my_seed: int, in_dim: int,\
        h_dim: int, out_dim: int, number_hidden: int) -> list:

    dims = [[in_dim, h_dim]]
    for layer_index in range(number_hidden):
        dims.append([h_dim, h_dim])

    dims.append([h_dim, out_dim])

    np.random.seed(my_seed)

    return initialize_weights(dims)

def forward(x: nparray, layers: list,\
        dropout_rate: float=0.25) -> nparray:

    for layer in layers[:-1]:
        
        x = relu(x @ layer) 
        #x = x * (np.random.rand(*x.shape) > dropout_rate)

    x = x @ layers[-1]

    return x
    
def compute_loss(x: nparray,\
        y: nparray,\
        loss_function: fn_type,\
        layers: list) -> np.float64:


    predicted = forward(x, layers)

    loss = loss_function(y, predicted)

    return loss

grad_loss = grad(compute_loss, argnum=3) 

def sgd_update(layers: list, grad_layers: list, lr: float) -> list:

    new_layers = []
    for index, grad_layer in enumerate(grad_layers):

        # multiplying by abs value of layer 
        # freezes weights with value zero 
        update = - lr * grad_layer * (np.abs(layers[index]) > 0.0)
        new_layers.append(layers[index] + update)

    return new_layers


if __name__ == "__main__":

    x = np.random.rand(128,64)
    targets = np.random.rand(128,10)
    loss_fn = mse_loss_fn 
    lr = 1e-3

    dims = [[64,32], [32,32], [32,10]]

    layers = initialize_weights(dims)

    for step in range(1000):
        
        grad_layers = grad_loss(x, targets, loss_fn, layers)

        if step % 100 == 0 or step == 999:
            loss = compute_loss(x, targets, loss_fn, layers) 
            print(f"loss at step {step} = {loss:.3f}")

        layers = sgd_update(layers, grad_layers, lr=lr)
