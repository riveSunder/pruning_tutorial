import copy 

from autograd import numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad

import src
from src.mlp import sgd_update,\
        compute_loss,\
        grad_loss,\
        initialize_weights

from src.common import relu,\
        fn_type,\
        nparray

def forward_skeletonize(x: nparray, nodes: list, layers: list):
    # special forward pass with nodes for dE/da

    for layer, alpha in zip(layers[:-1], nodes):
        
        x = relu((x * alpha) @ layer)

    x = (x * nodes[-1]) @ layers[-1]

    return x
    
def compute_skeleton_loss(x: nparray,\
        y: nparray,\
        loss_function: fn_type,\
        nodes: list,\
        layers: list) -> np.float64:

    predicted = forward_skeletonize(x, nodes, layers)

    loss = loss_function(y, predicted)

    return loss

grad_skeleton_loss = grad(compute_skeleton_loss, argnum=3) 
grad2_loss = egrad(egrad(compute_loss, argnum=3), argnum=3)

def prune_node(layers: list,\
        grad_nodes: list) -> list:

    # prunes only the least important input node 

    new_layers = layers

    prune_indices = []

    lowest_grad = float("Inf") 

    for index, grad_layer in enumerate(grad_nodes[1:]):

        prune_indices.append(np.argmin(grad_layer))

        if grad_layer[prune_indices[-1]] < lowest_grad:
            lowest_grad = grad_layer[prune_indices[-1]]
            prune_index = index

    p = prune_indices[prune_index]

    new_layer = layers[prune_index+1][0:p, :]
    new_layer = np.append(new_layer,\
            layers[prune_index+1][p+1:, :], axis=0) 

    new_layers[prune_index+1] = new_layer

    new_layer_1 = layers[prune_index][:, 0:p]
    new_layer_1 = np.append(new_layer_1,\
            layers[prune_index][:, p+1:], axis=1)

    new_layers[prune_index] = new_layer_1

    return new_layers

def prune_weights_by_grad2(layers: list,\
        grad2_layers: list,\
        prune_per_layer: int=10,\
        initial_threshold: float=1e-5) -> list:

    new_layers = []

    for layer, grad2_layer in zip(layers, grad2_layers):
        threshold = 1.0 * initial_threshold
        done = False

        while not done:
        
            prunable_weights = np.sum(\
                    np.abs(grad2_layer) < threshold)

            if prunable_weights >= prune_per_layer:
                done = True
            else:
                threshold *= 2.

        new_layer = layer * (np.abs(grad2_layer) > threshold)
        new_layers.append(new_layer)

    return new_layers

def prune_weights_by_magnitude(layers: list,\
        prune_per_layer: int=10,\
        initial_threshold: float=1e-3) -> list:

    return prune_weights_by_grad2(layers,\
            layers, prune_per_layer,\
            initial_threshold)

if __name__ == "__main__":

    x = np.random.rand(128,64)
    targets = np.random.rand(128,10)
    loss_fn = lambda y, p: np.mean((y-p)**2)
    lr = 1e-2

    dims = [[64,32], [32,32], [32,32], [32,10]]

    layers = initialize_weights(dims)

    for step in range(1000):
        
        grad_layers = grad_loss(x, targets, loss_fn, layers)

        if step % 100 == 0:
            loss = compute_loss(x, targets, loss_fn, layers) 
            print(f"loss at step {step} = {loss:.3f}")

        layers = sgd_update(layers, grad_layers, lr=lr)

    for layer in layers:
        print(layer.shape)
        print(np.sum(np.abs(layer) > 0))
    
    for step in range(1):

        nodes = [np.ones(elem.shape[0]) for elem in layers]
        
        grad_nodes = grad_skeleton_loss(\
                x, targets, loss_fn, nodes, layers)

        layers = prune_node(layers, grad_nodes)

        loss = compute_loss(x, targets, loss_fn, layers) 
        print(f"after node pruning loss = {loss:.3f}")

        for steps in range(100):
            
            grad_layers = grad_loss(x, targets, loss_fn, layers)
            layers = sgd_update(layers, grad_layers, lr=lr)

        loss = compute_loss(x, targets, loss_fn, layers) 
        print(f"after re-training loss = {loss:.3f}")

        layers = prune_weights_by_magnitude(layers)

        loss = compute_loss(x, targets, loss_fn, layers) 
        print(f"loss after weight magnitude pruning = {loss:.3f}")

        for steps in range(100):
            
            grad_layers = grad_loss(x, targets, loss_fn, layers)
            layers = sgd_update(layers, grad_layers, lr=lr)

        loss = compute_loss(x, targets, loss_fn, layers) 
        print(f"loss after re-training = {loss:.3f}")

        for layer in layers:
            print(layer.shape)
            print(np.sum(np.abs(layer) > 0))

        grad2_layers = grad2_loss(\
                x, targets, loss_fn, layers)

        layers = prune_weights_by_grad2(layers, grad2_layers)

        loss = compute_loss(x, targets, loss_fn, layers) 
        print(f"loss after grad2 weight pruning = {loss:.3f}")

        for steps in range(100):
            
            grad_layers = grad_loss(x, targets, loss_fn, layers)
            layers = sgd_update(layers, grad_layers, lr=lr)

        loss = compute_loss(x, targets, loss_fn, layers) 
        print(f"loss after re-training = {loss:.3f}")

    for layer in layers:
        print(layer.shape)
        print(np.sum(np.abs(layer) > 0))
