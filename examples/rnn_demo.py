import numpy as np
import proppo
import torch
import argparse

from tqdm import trange
from torch.distributions.normal import Normal
from proppo.utils import expand
from proppo.propagators import TotalProp, LRProp, RPProp


def sigmoid_layer(x, W, beta):
    return torch.sigmoid(beta * torch.matmul(x, W))


def make_weights(state_size, init_method='random'):
    if init_method == 'random':
        weights = torch.randn(state_size, state_size)
    elif init_method == 'chaotic':
        assert state_size == 2
        weights = torch.tensor([[-5.0, -25.0], [5.0, 25.0]])
    else:
        raise NotImplementedError('Initialization method ', init_method,
                                  ' is not implemented.')

    return weights


def init_state(batch_size, state_size, init_state_method='random'):
    if init_state_method == 'random':
        s = torch.randn(batch_size, state_size)
    elif init_state_method == 'chaotic':
        init_pos = [0.35, 0.55]
        s = torch.tensor(init_pos)
        s = s.repeat([batch_size, 1])
    else:
        raise NotImplementedError('State init method ', init_state_method,
                                  ' is not implemented.')
    return s


def make_manager(expanded_beta, option=4, grad_method='TP'):
    # TotalProp, LRProp or RPProp
    if grad_method == 'TP':
        propagator = TotalProp
    elif grad_method == 'RP':
        propagator = RPProp
    elif grad_method == 'LR':
        propagator = LRProp

    if option >= 3:
        zero_mean_param = lambda x: {
            'loc': torch.zeros_like(x),
            'scale': 0.001
        }

    # Proppo
    if option == 1:
        manager = proppo.PropagationManager(default_propagator=propagator())
    if option == 2:
        manager = proppo.PropagationManager(default_propagator=propagator(
            dist_class=Normal))
    if option == 3:
        manager = proppo.PropagationManager(default_propagator=propagator(
            dist_class=Normal, dist_params=zero_mean_param))
    if option == 4:
        manager = proppo.PropagationManager(
            default_propagator=propagator(dist_class=Normal,
                                          dist_params=zero_mean_param,
                                          ivw_target=expanded_beta))
    return manager


def main(conf):

    # parameters
    W = make_weights(conf.state_size, conf.weight_init)
    beta = torch.Tensor([conf.beta])
    beta.requires_grad = True

    # The current implementation of TotalProp expands the variable into a batch
    # so that the individual gradients can be obtained for inverse variance
    # weighting. Note that the memory for the variables are shared.
    if conf.no_expand:  # Turn expanding off to check the effect on efficiency.
        expanded_beta = beta
    else:
        expanded_beta = expand(beta, conf.batch_size)

    option = conf.option

    manager = make_manager(expanded_beta,
                           option=option,
                           grad_method=conf.grad_method)

    # histories for plots
    beta_history = []
    beta_grad_history = []

    for _ in trange(conf.num_iters):
        s = init_state(conf.batch_size, conf.state_size,
                       conf.init_state_method)

        for step in range(conf.horizon):
            s = sigmoid_layer(s, W, expanded_beta)

            # Apply manager.forward and record node
            if option == 1:
                # Correspoding to
                # manager = proppo.PropagationManager(default_propagator=propagator())
                s = manager.forward(s,
                                    dist_class=Normal,
                                    dist_params={
                                        'loc': torch.zeros_like(s),
                                        'scale': 0.001
                                    },
                                    ivw_target=expanded_beta)
            if option == 2:
                # Correspoding to
                # manager = proppo.PropagationManager(default_propagator=propagator(
                #     dist_class=Normal))
                s = manager.forward(s,
                                    dist_params={
                                        'loc': torch.zeros_like(s),
                                        'scale': 0.001
                                    },
                                    ivw_target=expanded_beta)
            if option == 3:
                # Correspoding to
                # manager = proppo.PropagationManager(default_propagator=propagator(
                #     dist_class=Normal, dist_params=zero_mean_param))
                s = manager.forward(s, ivw_target=expanded_beta)
            if option == 4:
                # Correspoding to
                # manager = proppo.PropagationManager(
                #     default_propagator=propagator(dist_class=Normal,
                #                                   dist_params=zero_mean_param,
                #                                   ivw_target=expanded_beta))
                s = manager.forward(s)

        batch_loss = (0.5 * (1.0 - s)**2).sum(dim=1, keepdim=True)

        # Total propagation or other backward method via PropagationManager
        manager.backward(batch_loss)

        # save history
        beta_history.append(beta.data.cpu().numpy()[0])
        beta_grad_history.append(beta.grad.data.cpu().numpy()[0])

        # reset gradient
        beta.grad.zero_()

    print('Gradient estimate is ', np.mean(beta_grad_history), ' +- ',
          np.std(beta_grad_history) / np.sqrt(conf.num_iters))
    print('Variance of gradient estimate is ', np.var(beta_grad_history))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # The 'option' argument demonstrates different ways to configure manager
    # and apply the algorithm in the code.
    parser.add_argument('--option', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--state-size', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--beta', type=float, default=2.4)
    parser.add_argument('--weight-init', type=str, default='chaotic')
    parser.add_argument('--init-state-method', type=str, default='chaotic')
    parser.add_argument('--grad-method', type=str, default='TP')
    parser.add_argument('--no-expand', action='store_true')

    args = parser.parse_args()

    if args.state_size != 2:
        print('Warning: chaotic init is not supported for state_size != 2.',
              'Switched state and weight init methods to random.')
        args.init_state_method = 'random'
        args.weight_init = 'random'

    main(conf=args)
