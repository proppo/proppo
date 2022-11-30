import numpy as np
import proppo
import torch
import argparse
import sys

from torch.distributions.normal import Normal
from proppo.utils import expand
from proppo.propagators import TotalProp, LRProp, RPProp, PauseProp, LossProp

from examples.rnn_demo import (sigmoid_layer, make_weights, init_state,
                               make_manager)

from functools import partial


def sigmoid_layer(x, W, beta):
    return torch.sigmoid(beta * torch.matmul(x, W))


make_weights = partial(make_weights, init_method='random')
init_state = partial(init_state, init_state_method='random')
make_manager = partial(make_manager, option=4)


def main(conf):

    # parameters
    W = make_weights(conf.state_size)
    beta = torch.Tensor([conf.beta])
    beta.requires_grad = True

    # The current implementation of TotalProp expands the variable into a batch
    # so that the individual gradients can be obtained for inverse variance
    # weighting. Note that the memory for the variables are shared.
    if conf.no_expand:  # Turn expanding off to check the effect on efficiency.
        expanded_beta = beta
    else:
        expanded_beta = expand(beta, conf.batch_size)

    # Configuration code.
    manager = make_manager(expanded_beta, grad_method=conf.grad_method)

    manager.add_propagator(name='pause', propagator=PauseProp(backprop=False))

    # In the current example, the default LossProp for LRProp is not
    # appropriate. The default for LRProp is
    # LossProp(lossgrad=False, backprop=False); however, in the current
    # example, there is a parallel computational path, and the gradients
    # should flow through this path, hence we have to backprop the gradients.
    if conf.grad_method == 'LR' and conf.loss_grads:
        manager.loss_propagator = LossProp()

    # Program code.
    s = init_state(conf.batch_size, conf.state_size)
    s.requires_grad_()

    for step in range(conf.horizon):
        # Here we add a parallel computation path s_2. The meaning of
        # this computation is irrelevant, but the aim of the demo
        # is to show a common problem where the gradients go past the
        # propagation node causing an error.
        s_2 = 2 * s + 0.02

        s = sigmoid_layer(s, W, expanded_beta)

        # Apply manager.forward and record node
        s = manager.forward(s)
        if conf.pause_grad:
            s_2 = manager.forward(s_2, local_propagator='pause')

        s = (s_2 + s) / 2

    batch_loss = (0.5 * (1.0 - s)**2).sum(dim=1, keepdim=True)

    # Total propagation or other backward method via PropagationManager
    try:
        manager.backward(batch_loss)
    except RuntimeError as e:
        print('RuntimeError:')
        print(e)
        if not args.pause_grad:
            print(
                'The above error occurred because the gradients go past'
                ' the propagation nodes.',
                'To fix the error add the argument --pause-grad.',
                'This argument turns on the PauseProp propagator',
                'that pauses the gradients at the parallel path.')
        else:
            print('The above error occurred because the default LossProp',
                  'for LRProp does not propagate gradients from the loss.',
                  'The PauseProp expects gradients to propagate into its',
                  'node, but as there are none, it throws an error.',
                  'To fix the issue turn on the --loss-grads input flag.')
        sys.exit()

    beta_grad = beta.grad.data.cpu().numpy()[0]

    print('Gradient estimate is', beta_grad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--state-size', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--beta', type=float, default=2.4)
    parser.add_argument('--grad-method', type=str, default='TP')
    parser.add_argument('--no-expand', action='store_true')
    parser.add_argument('--pause-grad', action='store_true')
    parser.add_argument('--loss-grads', action='store_true')

    args = parser.parse_args()

    main(conf=args)
