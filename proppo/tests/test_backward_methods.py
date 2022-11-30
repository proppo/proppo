import torch

import proppo.back_methods as f
import proppo.tests.proppo_test_utils as utils
import pytest

from proppo.utils import inverse_variance_weighting


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("dim", [3])
def test_rp_gradient(batch_size, dim):
    """Tests that the reparameterization gradient method correctly passes
    the gradient at the detached x to the previous x, which was not
    detached.

    """

    x = torch.randn(batch_size, dim)
    x.requires_grad_()

    detached_x = x.detach()
    detached_x.requires_grad_()
    detached_x.grad = torch.randn(batch_size, dim)

    node = {'output': detached_x, 'pre_output': x}

    message = f.rp_gradient(node, None)

    output = message['outputs'].get()[0]
    grad = message['grads'].get()[0]

    torch.autograd.backward(tensors=output, grad_tensors=grad)

    assert output.data_ptr() == x.data_ptr()
    utils.check_identical(detached_x.grad, grad)


@pytest.mark.parametrize("batch_size", [2])
def test_lr_gradient(batch_size):
    """Tests likelihood ratio gradient (LR) gradient for correctnes, by
    checking that the output gradient matches (dlogp\dtheta*L).mean(),
    where logp is the log probability, theta is a parameter of the
    distribution (in the test, log_prob itself will be the parameter),
    L is the local loss used for the LR method, and the mean() is used
    to indicate that the average gradient is computed.

    """

    # log prob is a scalar for each member in batch
    log_prob = torch.randn(batch_size, 1)
    log_prob.requires_grad_()

    # one loss for each member in batch
    local_loss = torch.randn(batch_size, 1)

    message_in = {'local_loss': local_loss}
    node = {'log_prob': log_prob}

    message = f.lr_gradient(node, message_in)

    output = message['outputs'].get()[0]
    grad = message['grads'].get()[0]

    lr_loss = (log_prob * local_loss).mean()
    loss_grad = torch.autograd.grad(outputs=lr_loss,
                                    inputs=log_prob,
                                    retain_graph=True)[0]

    torch.autograd.backward(tensors=output, grad_tensors=grad)

    assert output.data_ptr() == log_prob.data_ptr()
    utils.check_identical(grad, local_loss / batch_size)
    # The gradient for each log_prob should be loss/batch_size
    utils.check_identical(log_prob.grad, local_loss / batch_size)
    utils.check_identical(log_prob.grad, loss_grad)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("dim", [3])
def test_totalprop_gradient(batch_size, dim):
    """Tests that totalprop correctly computes the gradient. Currently
    ivw_target is the node where we will check that the correct
    gradient was computed in the end.

    """
    ivw_target = torch.rand(batch_size, dim)
    ivw_target.requires_grad_()

    x = torch.rand(batch_size, dim) + ivw_target

    detached_x = x.detach()
    detached_x.requires_grad_()
    detached_x.grad = torch.rand(batch_size, dim)

    log_prob = (1.0 - ivw_target).sum(dim=1, keepdim=True)

    local_loss = torch.rand(batch_size, 1)

    node = {
        'pre_output': x,
        'output': detached_x,
        'log_prob': log_prob,
        'ivw_target': ivw_target,
        'k_interval': 1
    }

    message_in = {'local_loss': local_loss}

    rp_grads = torch.autograd.grad(outputs=x,
                                   inputs=ivw_target,
                                   grad_outputs=detached_x.grad,
                                   retain_graph=True)

    lr_loss = (local_loss * log_prob).mean()
    lr_grads = torch.autograd.grad(outputs=lr_loss,
                                   inputs=ivw_target,
                                   retain_graph=True)

    # TODO: this should be modified with multiple ivw targets
    k_lr, k_rp = inverse_variance_weighting(lr_grads[0], rp_grads[0])

    leaf_loss = [(k_lr * lr_loss), x]
    grads_loss = [None, k_rp * detached_x.grad]

    ref_grad = torch.autograd.grad(outputs=leaf_loss,
                                   inputs=ivw_target,
                                   grad_outputs=grads_loss,
                                   retain_graph=True)

    message = f.totalprop_gradient(node, message_in)

    outputs = message['outputs'].get()
    grads = message['grads'].get()

    torch.autograd.backward(tensors=outputs, grad_tensors=grads)

    ref_outputs = [log_prob, x]
    ref_grads = [k_lr * local_loss / batch_size, k_rp * detached_x.grad]

    utils.check_identical(outputs[0], ref_outputs[0])
    utils.check_identical(outputs[1], ref_outputs[1])
    utils.check_identical(grads[0], ref_grads[0])
    utils.check_identical(grads[1], ref_grads[1])
    # Check that gradient is correctly computed when passed to
    # backward
    utils.check_identical(ivw_target.grad, ref_grad[0])
