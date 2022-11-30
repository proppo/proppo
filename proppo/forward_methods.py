import torch
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable


def detached_output(x, requires_grad=True, **kwargs):
    x_detached = x.detach()
    if requires_grad:
        x_detached.requires_grad_()
    node = {'output': x_detached, 'pre_output': x, 'register_node': True}
    return node


def loss_forward(x, lossgrad=False, sum_loss=True, **kwargs):
    """ Method to register a loss node. It will either
    just pass the loss in the local_loss slot, or will
    also backprop the gradient. It also has an option to
    sum the incoming losses to keep a cumulative future loss.

    """
    if x.dim() == 1:
        x = x.reshape([x.numel(), 1])

    node = detached_output(x, requires_grad=False)
    node['lossgrad'] = lossgrad
    node['sum_loss'] = sum_loss
    return node


def rp_noise(x, dist_class, dist_params, detach=True, **kwargs):
    """ Returns noisy node for reparametrization trick
    
    """
    if isinstance(dist_params, (tuple, list)):
        dist = dist_class(*dist_params)
    if isinstance(dist_params, dict):
        dist = dist_class(**dist_params)
    if isinstance(dist_params, Callable):
        dist = dist_class(**dist_params(x))

    x_noisy = x + dist.rsample()

    if detach:
        node = detached_output(x_noisy)
    else:
        node = {'output': x_noisy, 'register_node': False}
    return node


def lr_noise(x, dist_class, dist_params, requires_grad=False, **kwargs):
    """ Returns noisy node for likelihood ratio gradients.
    
    """
    if isinstance(dist_params, (tuple, list)):
        dist = dist_class(*dist_params)
    if isinstance(dist_params, dict):
        dist = dist_class(**dist_params)
    if isinstance(dist_params, Callable):
        dist = dist_class(**dist_params(x))

    x_noisy = x + dist.rsample()

    log_prob = dist.log_prob(x_noisy.detach() - x)

    node = detached_output(x_noisy, requires_grad=requires_grad)
    node['log_prob'] = log_prob
    return node


def totalprop_noise(x,
                    dist_class,
                    dist_params,
                    ivw_target,
                    k_interval=1,
                    **kwargs):
    """ Returns noisy node for total propagation gradients.
    
    """
    node = lr_noise(x,
                    dist_class=dist_class,
                    dist_params=dist_params,
                    requires_grad=True)
    node['ivw_target'] = ivw_target
    node['k_interval'] = k_interval
    return node


def gauss_resample(x, detach=True, **kwargs):
    """ Forward method for ResampleProp.
    P: batchsize
    D: number of dimensions

    """
    # Fit a Gaussian and perform a cholesky decomposition
    dims = x.size()
    x_view = x.view(dims[0], -1, 1)  # size [P, D, 1]
    mu = x_view.mean(axis=0, keepdims=True)  # size [1, D, 1]
    diff = x_view - mu  # size [P, D, 1]
    covar = torch.addbmm(torch.zeros(1, device=diff.device),
                         diff,
                         diff.permute([0, 2, 1]),
                         beta=0)
    covar = covar / (dims[0] - 1)  # Bessel correction, size [D, D]
    chol = torch.cholesky(covar)  # size [D, D]

    # Resample from the fitted Gaussian distribution
    resamp_x = chol @ torch.randn_like(x_view) + mu
    resamp_x = resamp_x.view(dims)

    if detach:
        node = detached_output(resamp_x)
    else:
        node = {'output': resamp_x, 'register_node': False}
    return node


def gauss_shape(x, shaped_grad=False, **kwargs):
    """ Forward method for GSProp.
    P: batchsize
    D: number of dimensions

    TODO: if the input is a dict, we need to extract the dict,
    concatenate everything, and finally deconcatenate the extracted
    variables and put them back into the dict.
    """
    # Unpack a dictionary, and put it in the correct shape
    unpack = isinstance(x, dict)
    if unpack:
        shapes = []
        lengths = []
        tensors = []
        for key in x:
            dims = x[key].size()
            shapes.append(dims)
            lengths.append(sum(dims[1:]))
            tensors.append(x[key].view(dims[0], -1))
        x_c = torch.cat(tensors, axis=1)
    else:
        x_c = x
    # Fit a Gaussian and perform a cholesky decomposition
    dims = x_c.size()
    x_view = x_c.view(dims[0], -1, 1)  # size [P, D, 1]
    mu = x_view.mean(axis=0, keepdims=True)  # size [1, D, 1]
    mu_detached = mu.detach()
    mu_detached.requires_grad_()
    diff = x_view - mu  # size [P, D, 1]
    covar = torch.addbmm(torch.zeros(1, device=diff.device),
                         diff,
                         diff.permute([0, 2, 1]),
                         beta=0)
    covar = covar / (dims[0] - 1)  # Bessel correction, size [D, D]
    covar = (covar + covar.T) / 2  # Symmetrize
    covar = covar + 1e-6 * torch.eye(covar.size()[0],
                                     device=covar.device)  # Add jitter
    covar_detached = covar.detach()
    covar_detached.requires_grad_()
    chol = torch.cholesky(covar_detached)  # size [D, D]

    # Resample from the fitted Gaussian distribution
    resamp_x = chol @ torch.randn_like(x_view) + mu_detached
    resamp_x = resamp_x.view(dims)

    if unpack:
        split_x = torch.split(resamp_x, lengths, dim=1)
        out_x = {}
        for ind, key in enumerate(x):
            out_x[key] = split_x[ind].view(shapes[ind])
    else:
        out_x = resamp_x

    node = {
        'output': out_x
    }  # Currently, the covar and mu are always detached. TODO: make it modular
    node['covar_detached'] = covar_detached
    node['mu_detached'] = mu_detached
    node['covar'] = covar
    node['mu'] = mu
    node['x_view'] = x_view
    node['diff'] = diff
    node['shaped_grad'] = shaped_grad
    return node
