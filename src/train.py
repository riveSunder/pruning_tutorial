import os
import copy

import matplotlib.pyplot as plt
from src.draw_nn import draw_nn

from autograd import numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad

import sklearn
import sklearn.datasets

from src.mlp import sgd_update,\
        forward,\
        compute_loss,\
        grad_loss,\
        initialize_weights,\
        initialize_model

from src.prune import forward_skeletonize,\
       compute_skeleton_loss,\
       grad_skeleton_loss,\
       grad2_loss,\
       prune_node,\
       prune_weights_by_grad2,\
       prune_weights_by_magnitude

from src.common import nparray,\
        sm,\
        fn_type,\
        nparray,\
        mse_loss_fn,\
        nll_loss_fn

def indices_to_one_hot(y: nparray, max_index=None) -> nparray: 

    if max_index is None:
        max_index = np.max(y)

    y_target = np.zeros((y.shape[0], max_index+1))

    for ii in range(y.shape[0]):
        y_target[ii, y[ii]] = 1.0

    return y_target

def compute_accuracy(y: nparray, predicted: nparray) -> float:

    return np.mean(y.argmax(-1) == predicted.argmax(-1))


def prune_mode_0(layers: list,\
        train_x: nparray=None, train_y: nparray=None) -> list:
    return layers

def prune_mode_1(layers: list,  train_x: nparray, train_y: nparray) -> list:

    nodes = [np.ones(elem.shape[0]) for elem in layers]
    grad_nodes = grad_skeleton_loss(\
            train_x, train_y, nll_loss_fn, nodes, layers)

    layers = prune_node(layers, grad_nodes)

    return layers

def prune_mode_2(layers: list, train_x: nparray, train_y: nparray) -> list: 

    grad2_layers = grad2_loss(\
            train_x, train_y, nll_loss_fn, layers)
    layers = prune_weights_by_grad2(layers, grad2_layers)

    return layers 

def prune_mode_3(layers: list,\
        train_x: nparray=None, train_y: nparray=None) -> list:

    return prune_weights_by_magnitude(layers)

def retrieve_prune_fn(mode: int=0):

    if mode == 0:
        return prune_mode_0
    elif mode == 1:
        return prune_mode_1
    elif mode == 2:
        return prune_mode_2
    elif mode == 3:
        return prune_mode_3


def split_digits(my_seed: int=13) -> tuple:

    x, y_indices = sklearn.datasets.load_digits(return_X_y = True)
    x = x / np.max(x)
    y = indices_to_one_hot(y_indices)

    np.random.seed(my_seed)
    np.random.shuffle(x)

    np.random.seed(my_seed)
    np.random.shuffle(y)

    split_validation = int(0.1*x.shape[0])

    val_x = x[:split_validation,:]
    val_y = y[:split_validation,:]
    train_x = x[split_validation:,:]
    train_y = y[split_validation:,:]

    return train_x, train_y, val_x, val_y

def print_progress(layers: list=None,\
        batch_x: nparray=None,\
        batch_y: nparray=None,\
        tag: str="train",\
        step: int=0,\
        verbose: bool=True):

    if layers is None:
        # print column labels
        msg = f"split, step, loss, accuracy"
    else:
        loss = compute_loss(batch_x, batch_y, nll_loss_fn, layers) 

        msg = f"{tag}, {step:05}, {loss:.4f}, "

        predicted = forward(batch_x, layers)
        accuracy = compute_accuracy(batch_y, predicted)

        msg += f"{accuracy:.4f}\n"

    if verbose:
        print(msg)

    return msg

def train(my_seed: int=13,\
        number_epochs: int=100,\
        mode: int=0,\
        lr: float=1e-3,\
        verbose: bool=True):
    """
    mode 0 - no pruning
    mode 1 - pruning nodes (Mozer and Smolensky 1989)
    mode 2 - pruning w by 2nd derivative (LeCun et al. 1990)
    mode 3 - by magnitude (e.g. Han et al. 2015 and others)
    """

    my_prune_fn = retrieve_prune_fn(mode)
    batch_size = 1024
    number_prunes = 8
    h_dim = 16
    number_hidden = 3
    display_every = number_epochs // 10

    train_x, train_y, val_x, val_y = split_digits(my_seed)

    in_dim = train_x.shape[-1]
    out_dim = train_y.shape[-1]

    layers = initialize_model(my_seed, in_dim, h_dim, out_dim, number_hidden)

    ticket_layers = copy.deepcopy(layers)

    if verbose:
        fig, ax = plt.subplots(1,1,figsize=(32,16))
        ax = draw_nn(ax, layers)
        plt.savefig(f"mode_{mode}_start.png")
        plt.clf()

    progress = print_progress()
    for step in range(number_epochs):

        if step % display_every == 0:
            progress += print_progress(layers,\
                    train_x, train_y, tag="train", verbose=verbose, step=step)
            progress += print_progress(layers,\
                    val_x, val_y, tag="valid", verbose=verbose, step=step)

        batch_indices = np.random.randint(train_x.shape[0],\
                size=(batch_size,))
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, layers)

        layers = sgd_update(layers, grad_layers, lr=lr)

    for pruning_step in range(number_prunes):
        layers = my_prune_fn(layers,  train_x, train_y)
        ticket_layers = my_prune_fn(ticket_layers, train_x, train_y)

    progress += print_progress(layers,\
            train_x, train_y, tag="train_post_prune", verbose=verbose, step=step)
    progress += print_progress(layers,\
            val_x, val_y, tag="valid_post_prune", verbose=verbose, step=step)

    for steps in range(display_every):
        
        grad_layers = grad_loss(train_x, train_y, nll_loss_fn, layers)
        layers = sgd_update(layers, grad_layers, lr=lr)

    progress += print_progress(layers,\
            train_x, train_y, tag="train_retrained", verbose=verbose, step=step)
    progress += print_progress(layers,\
            val_x, val_y, tag="valid_retrained", verbose=verbose, step=step)

    save_dir = os.path.join("parameters", f"mode_{mode}")

    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)

    print(f"model shape with mode {mode} pruning")

    for ii, layer in enumerate(layers):
        print(layer.shape, np.sum(np.abs(layer) > 0))
        save_filepath = os.path.join(save_dir,f"layer{ii}.npy")
        np.save(save_filepath, layer)

    # examine lottery ticket hypothesis
    pruned_dims = [layer.shape for layer in ticket_layers]
    noticket_layers = initialize_weights(pruned_dims)

    for step in range(number_epochs):
        
        batch_indices = np.random.randint(train_x.shape[0],\
                size=(batch_size,))
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        ticket_grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, ticket_layers)
        noticket_grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, noticket_layers)

        ticket_layers = sgd_update(ticket_layers, ticket_grad_layers, lr=lr)
        noticket_layers = sgd_update(noticket_layers, noticket_grad_layers, lr=lr)

        if step % display_every == 0 or step == (number_epochs-1):
            progress += print_progress(ticket_layers,\
                    train_x, train_y, tag="train_ticket", verbose=verbose, step=step)
            progress +=  print_progress(ticket_layers,\
                    val_x, val_y, tag="valid_ticket", verbose=verbose, step=step)
            progress +=  print_progress(noticket_layers,\
                    train_x, train_y, tag="train_noticket", verbose=verbose, step=step)
            progress +=  print_progress(noticket_layers,\
                    val_x, val_y, tag="valid_noticket", verbose=verbose, step=step)

    if verbose:
        fig, ax = plt.subplots(1,1,figsize=(32,16))
        ax = draw_nn(ax, layers)

        plt.savefig(f"mode_{mode}_pruned.png")
        plt.clf()

    progress_filepath = f"mode_{mode}_log.csv"
    with open(progress_filepath, "w") as f:
        f.write(progress)

if __name__ == "__main__":

    for mode in range(4):
       train(number_epochs=10000,mode=mode, lr=6e-2) 
