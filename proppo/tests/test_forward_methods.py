import torch
import pytest

import proppo.forward_methods as f
import proppo.tests.proppo_test_utils as utils

from torch.distributions.normal import Normal


@pytest.mark.parametrize("param_type", ['tuple', 'list', 'dict'])
def test_rp_noise(param_type):
    x = torch.rand(2, 3)
    x.requires_grad_()

    dist_type = Normal
    if param_type == 'tuple':
        dist_params = (0, 0.001)
    elif param_type == 'list':
        dist_params = [0, 0.001]
    elif param_type == 'dict':
        dist_params = {'loc': 0, 'scale': 0.001}

    node = f.rp_noise(x, dist_type, dist_params)

    # check keys
    utils.check_keys(node, ['output'])

    # check shape
    utils.check_shape(node['output'], x)

    # check output is sampled from distribution
    utils.check_not_identical(node['output'], x)

    # check if differntiable
    utils.check_differentiable(node['output'], [node['output']])
    utils.check_differentiable(node['pre_output'], [x])


@pytest.mark.parametrize("param_type", ['tuple', 'list', 'dict'])
def test_lr_noise(param_type):
    x = torch.rand(2, 3)
    x.requires_grad_()

    dist_type = Normal
    if param_type == 'tuple':
        dist_params = (0, 0.001)
    elif param_type == 'list':
        dist_params = [0, 0.001]
    elif param_type == 'dict':
        dist_params = {'loc': 0, 'scale': 0.001}

    dist = Normal(loc=0, scale=0.001)

    node = f.lr_noise(x, dist_type, dist_params)

    utils.check_keys(node, ['output', 'log_prob'])

    utils.check_shape(node['output'], x)

    utils.check_not_identical(node['output'], x)

    utils.check_differentiable(node['pre_output'], [x])
    utils.check_not_differentiable(node['output'], [node['output']])

    # check log probability
    log_prob = dist.log_prob(node['output'] - x)
    utils.check_shape(node['log_prob'], log_prob)
    utils.check_identical(node['log_prob'], log_prob)


@pytest.mark.parametrize("param_type", ['tuple', 'list', 'dict'])
def test_totalprop_noise(param_type):
    x = torch.rand(2, 3)
    x.requires_grad_()

    dist_type = Normal
    if param_type == 'tuple':
        dist_params = (0, 0.001)
    elif param_type == 'list':
        dist_params = [0, 0.001]
    elif param_type == 'dict':
        dist_params = {'loc': 0, 'scale': 0.001}

    dist = Normal(loc=0, scale=0.001)

    node = f.totalprop_noise(x, dist_type, dist_params, x)

    utils.check_keys(node, ['output', 'log_prob', 'ivw_target'])

    utils.check_shape(node['output'], x)

    utils.check_not_identical(node['output'], x)

    utils.check_differentiable(node['pre_output'], [x])
    utils.check_differentiable(node['output'], [node['output']])

    log_prob = dist.log_prob(node['output'] - x)
    utils.check_shape(node['log_prob'], log_prob)
    utils.check_identical(node['log_prob'], log_prob)

    assert node['ivw_target'].data_ptr() == x.data_ptr()
