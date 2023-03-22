import os
import copy

from autograd import numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad

import sklearn
import sklearn.datasets

from src.mlp import sgd_update,\
        forward,\
        compute_loss,\
        grad_loss,\
        initialize_weights

from src.prune import forward_skeletonize,\
       compute_skeleton_loss,\
       grad_skeleton_loss,\
       grad2_loss,\
       prune_node,\
       prune_weights_by_grad2,\
       prune_weights_by_magnitude

nparray = type(np.random.rand(1,2))

sm = lambda x: np.exp(x)/(np.sum(np.exp(x), \
        axis=-1, keepdims=True))
mse_loss_fn = lambda y, p: np.mean((y-p)**2)
nll_loss_fn = lambda y, pred: -np.mean(\
        y*np.log(sm(pred)) + (1-y)*np.log(1. - sm(pred)))


def indices_to_one_hot(y: nparray, max_index=None) -> nparray: 

    if max_index is None:
        max_index = np.max(y)

    y_target = np.zeros((y.shape[0], max_index+1))

    for ii in range(y.shape[0]):
        y_target[ii, y[ii]] = 1.0

    return y_target

def compute_accuracy(y: nparray, predicted: nparray) -> float:

    return np.mean(y.argmax(-1) == predicted.argmax(-1))

def train(my_seed: int=13,\
        number_epochs: int=100,\
        mode: int=0,\
        lr: float=1e-3):
    """
    mode 0 - no pruning
    mode 1 - pruning nodes (Mozer and Smolensky 1989)
    mode 2 - pruning w by 2nd derivative (LeCun et al. 1990)
    mode 3 - by magnitude (e.g. Han et al. 2015 and others)
    """


    batch_size = 1024
    number_prunes = 10

    display_every = number_epochs // 10
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

    in_dim = train_x.shape[-1]
    out_dim = train_y.shape[-1]
    dims = [[in_dim,512], [512,512], [512, out_dim]]

    np.random.seed(my_seed)
    layers = initialize_weights(dims)
    ticket_layers = copy.deepcopy(layers)

    for step in range(number_epochs):

        batch_indices = np.random.randint(train_x.shape[0],\
                size=(batch_size,))
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, layers)

        if step % display_every == 0:
            loss = compute_loss(train_x, train_y, nll_loss_fn, layers) 
            val_loss = compute_loss(val_x, val_y, nll_loss_fn, layers) 
            print(f"loss at step {step} = {loss:.3f}")
            print(f"val loss at step {step} = {val_loss:.3f}")
            predicted = forward(train_x, layers)
            val_predicted = forward(val_x, layers)
            acc = compute_accuracy(train_y, predicted)
            val_acc = compute_accuracy(val_y, val_predicted)
            print(f"accuracy = {acc:.3f}")
            print(f"val acc  = {val_acc:.3f}")

        layers = sgd_update(layers, grad_layers, lr=lr)

    if mode == 0:
        pass
    elif mode == 1 or mode == 4:
        
        for prune_event in range(number_prunes):
            nodes = [np.ones(elem.shape[0]) for elem in layers]
            grad_nodes = grad_skeleton_loss(\
                    train_x, train_y, nll_loss_fn, nodes, layers)

            layers = prune_node(layers, grad_nodes)
            ticket_layers = prune_node(ticket_layers, grad_nodes)

            for ii in range(100):
                grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, layers)
                layers = sgd_update(layers, grad_layers, lr=lr)

        loss = compute_loss(train_x, train_y, nll_loss_fn, layers) 
        val_loss = compute_loss(val_x, val_y, nll_loss_fn, layers) 
        print(f"after node pruning loss = {loss:.3f}")
        print(f"val loss  = {val_loss:.3f}")

        predicted = forward(train_x, layers)
        val_predicted = forward(val_x, layers)
        acc = compute_accuracy(train_y, predicted)
        val_acc = compute_accuracy(val_y, val_predicted)
        print(f"accuracy = {acc:.3f}")
        print(f"validation accuracy  = {val_acc:.3f}")

    elif mode == 2:

        for prune_event in range(number_prunes):
            grad2_layers = grad2_loss(\
                    train_x, train_y, nll_loss_fn, layers)
            layers = prune_weights_by_grad2(layers, grad2_layers)
            ticket_layers = prune_weights_by_grad2(ticket_layers, grad2_layers)
            for ii in range(100):
                grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, layers)
                layers = sgd_update(layers, grad_layers, lr=lr)

        loss = compute_loss(train_x, train_y, nll_loss_fn, layers) 
        val_loss = compute_loss(val_x, val_y, nll_loss_fn, layers) 
        print(f"loss after grad2 weight pruning = {loss:.3f}")
        print(f"val loss  = {val_loss:.3f}")

        predicted = forward(train_x, layers)
        val_predicted = forward(val_x, layers)
        acc = compute_accuracy(train_y, predicted)
        val_acc = compute_accuracy(val_y, val_predicted)
        print(f"accuracy = {acc:.3f}")
        print(f"validation accuracy  = {val_acc:.3f}")


    elif mode == 3:
        for prune_event in range(number_prunes):
            layers = prune_weights_by_magnitude(layers)
            ticket_layers = prune_weights_by_magnitude(ticket_layers)
            for ii in range(100):
                grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, layers)
                layers = sgd_update(layers, grad_layers, lr=lr)

        loss = compute_loss(train_x, train_y, nll_loss_fn, layers) 
        val_loss = compute_loss(val_x, val_y, nll_loss_fn, layers) 
        print(f"loss after weight magnitude pruning = {loss:.3f}")
        print(f"val loss  = {val_loss:.3f}")

        predicted = forward(train_x, layers)
        val_predicted = forward(val_x, layers)
        acc = compute_accuracy(train_y, predicted)
        val_acc = compute_accuracy(val_y, val_predicted)
        print(f"accuracy = {acc:.3f}")
        print(f"validation accuracy  = {val_acc:.3f}")

    for steps in range(display_every):
        
        grad_layers = grad_loss(train_x, train_y, nll_loss_fn, layers)
        layers = sgd_update(layers, grad_layers, lr=lr)

    loss = compute_loss(train_x, train_y, nll_loss_fn, layers) 
    val_loss = compute_loss(val_x, val_y, nll_loss_fn, layers) 
    print(f"loss after re-training = {loss:.3f}")
    print(f"val loss after re-training = {val_loss:.3f}")

    predicted = forward(train_x, layers)
    val_predicted = forward(val_x, layers)
    acc = compute_accuracy(train_y, predicted)
    val_acc = compute_accuracy(val_y, val_predicted)
    print(f"accuracy = {acc:.3f}")
    print(f"val acc  = {val_acc:.3f}")

    save_dir = os.path.join("parameters", f"mode_{mode}")

    if os.path.exists(save_dir):
        pass
    else:
        os.mkdir(save_dir)

    print(f"model stats with mode {mode} pruning")

    for ii, layer in enumerate(layers):
        print(layer.shape)
        print(np.sum(np.abs(layer) > 0))
        save_filepath = os.path.join(save_dir,f"layer{ii}.npy")
        np.save(save_filepath, layer)

    # examine lottery ticket hypothesis
    pruned_dims = [layer.shape for layer in ticket_layers]
    noticket_layers = initialize_weights(dims)

    for step in range(number_epochs):
        
        batch_indices = np.random.randint(train_x.shape[0],\
                size=(batch_size,))
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        ticket_grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, ticket_layers)
        noticket_grad_layers = grad_loss(batch_x, batch_y, nll_loss_fn, noticket_layers)

        ticket_layers = sgd_update(ticket_layers, ticket_grad_layers, lr=lr)
        noticket_layers = sgd_update(noticket_layers, noticket_grad_layers, lr=lr)

        if step % display_every == 0:
            loss = compute_loss(train_x, train_y, nll_loss_fn, ticket_layers) 
            val_loss = compute_loss(val_x, val_y, nll_loss_fn, ticket_layers) 
            print(f"\n tikcet loss at step {step} = {loss:.3f}")
            print(f"val loss at step {step} = {val_loss:.3f}")
            predicted = forward(train_x, ticket_layers)
            val_predicted = forward(val_x, ticket_layers)
            acc = compute_accuracy(train_y, predicted)
            val_acc = compute_accuracy(val_y, val_predicted)
            print(f"accuracy = {acc:.3f}")
            print(f"val acc  = {val_acc:.3f}")

            loss = compute_loss(train_x, train_y, nll_loss_fn, noticket_layers) 
            val_loss = compute_loss(val_x, val_y, nll_loss_fn, noticket_layers) 
            print(f"noticket loss at step {step} = {loss:.3f}")
            print(f"noticket val loss at step {step} = {val_loss:.3f}")
            predicted = forward(train_x, noticket_layers)
            val_predicted = forward(val_x, noticket_layers)
            acc = compute_accuracy(train_y, predicted)
            val_acc = compute_accuracy(val_y, val_predicted)
            print(f"accuracy = {acc:.3f}")
            print(f"val acc  = {val_acc:.3f}")


if __name__ == "__main__":

    for mode in range(4):
       train(number_epochs=5000,mode=mode, lr=1e-1) 
