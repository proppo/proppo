import torch
from proppo.utils import inverse_variance_weighting
from proppo.containers import Listed


def backward(node, message, grad_name=None):
    """ Removes outputs and grads from message, calls backward,
    and passes the remaining message backwards.

    """
    if 'outputs' in message:
        outputs = message.pop('outputs')
        grads = message.pop('grads')
    else:
        return message

    torch.autograd.backward(tensors=outputs, grad_tensors=grads)
    return {}


def loss_backward(node, message_in, loss_name):
    """ Backward method for LossBase.
    It has features to sum the incoming future losses,
    and also allows to optionally backprop the gradients
    at the loss node.

    """
    local_loss = node['output']
    if node['sum_loss']:
        if loss_name in message_in:
            local_loss += message_in[loss_name]

    outputs = node['pre_output']
    # Grads is set so that the gradient of the average loss is computed.
    message = {loss_name: local_loss}

    if node['lossgrad']:
        ones_matrix = torch.tensor([1.0], device=outputs.device).expand(
            outputs.size())
        grads = {
            'outputs': Listed(outputs),
            'grads': Listed(ones_matrix / torch.numel(outputs))
        }
        message.update(grads)

    return message


def rp_gradient(node, message_in):
    """ Returns output tensor and its gradient for reparametrization trick.
    The backward method used in the BackPropagator Propagator can 
    initiate backprop at these tensors.

    """
    detached_output = node['output']
    output = node['pre_output']
    message = {
        'outputs': Listed(output),
        'grads': Listed(detached_output.grad)
    }

    return message


def lr_gradient(node, message_in):
    """ Returns output tensor and its gradient for likelihood ratio gradients

    """
    if 'baselined_loss' in message_in:
        local_loss = message_in['baselined_loss']
    else:
        local_loss = message_in['local_loss']

    lr_grad_outputs = local_loss / torch.numel(local_loss)
    log_prob = node['log_prob']

    lr_grad_outputs = lr_grad_outputs.expand(log_prob.shape)

    message = {'outputs': Listed(log_prob), 'grads': Listed(lr_grad_outputs)}
    return message


def totalprop_gradient(node,
                       message_in,
                       var_weighting_func=inverse_variance_weighting):
    """ Returns output tensors and their gradients for total propagation
    gradients.

    The totalpropagation is a combination of the reparametrization trick and
    the likelihood ratio.
    Each gradient will be combined based on inverse variance weighting.
    
    """
    detached_output = node['output']
    output = node['pre_output']
    log_prob = node['log_prob']
    ivw_target = node['ivw_target']
    k_interval = node['k_interval']

    if 'k_counter' in message_in:
        k_counter = message_in.pop('k_counter')
    else:
        k_counter = 0

    if 'baselined_loss' in message_in:
        local_loss = message_in['baselined_loss']
    else:
        local_loss = message_in['local_loss']
    # Make LR gradients compute mean gradient
    local_loss = local_loss / torch.numel(local_loss)
    #lr_grad_outputs = local_loss
    lr_grad_outputs = local_loss.expand(log_prob.shape)

    if k_counter % k_interval == 0:
        rp_grads = torch.autograd.grad(outputs=output,
                                       inputs=ivw_target,
                                       grad_outputs=detached_output.grad,
                                       retain_graph=True)

        lr_grads = torch.autograd.grad(outputs=log_prob,
                                       inputs=ivw_target,
                                       grad_outputs=lr_grad_outputs,
                                       retain_graph=True)

        k_lr, k_rp = var_weighting_func(lr_grads, rp_grads)
    else:
        k_lr, k_rp = message_in['k_lr_k_rp']

    outputs = Listed(log_prob, output)
    grads = Listed(k_lr * lr_grad_outputs, k_rp * detached_output.grad)
    message_grads = {'outputs': outputs, 'grads': grads, 'targets': -1}

    message_k = {
        'k_counter': k_counter + 1,
        'k_lr_k_rp': (k_lr, k_rp),
        'targets': 0
    }

    if k_interval > 1:
        messages = (message_grads, message_k)
    else:
        messages = message_grads
    return messages


def resample_back(node, message_in):
    """ Backward method for ResampleProp.
    Currently, it will just backpropagate the gradient, just like RP

    """
    message = rp_gradient(node, message_in)
    return message


def gs_back(node, message_in):
    """ Performs the Gaussian shaping gradient computations.

    """
    mu = node['mu_detached']  # shape [1, D, 1]
    mut = mu.permute([0, 2, 1])  # shape [1, 1, D]

    m = node['diff']  # shape [P, D, 1]
    m = m.detach()

    xv = node['x_view'].detach()
    xxt = torch.bmm(xv, xv.permute([0, 2, 1]))
    v = xxt - xxt.mean(axis=0, keepdims=True)

    w = m @ mut  # shape [P, D, D]

    dcdmu = mu.grad  # shape [1, D, 1]
    dcds = node['covar_detached'].grad  # shape [D, D]

    g = torch.sum(dcdmu * m, axis=[1, 2], keepdims=True).view(-1, 1) \
        + torch.sum(dcds * (v - 2 * w), axis=[1, 2], keepdims=True).view(-1, 1)

    message = {'shaped_stat': g}
    if node['shaped_grad']:
        message['outputs'] = Listed(node['mu'], node['covar'])
        message['grads'] = Listed(node['mu_detached'].grad,
                                  node['covar_detached'].grad)

    return message
