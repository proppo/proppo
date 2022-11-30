import torch.nn as nn
import torch
import copy
import pytest

import proppo.tests.proppo_test_utils as utils

from torch.distributions.normal import Normal
from proppo.modules import BatchedLinear
from proppo.propagation_manager import PropagationManager
from proppo.propagators import TotalProp
from proppo.forward_methods import totalprop_noise
from proppo.back_methods import totalprop_gradient
from .module_tester import module_tester


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("input_size", [8])
@pytest.mark.parametrize("output_size", [2])
@pytest.mark.parametrize("detach", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_linear(batch_size, input_size, output_size, detach, bias):
    batched_linear = BatchedLinear(input_size,
                                   output_size,
                                   batch_size,
                                   detach=detach,
                                   bias=bias)
    linear = nn.Linear(input_size, output_size, bias=bias)

    param_names = ['weight']
    if bias:
        param_names += ['bias']

    module_tester(batched_linear,
                  linear, (batch_size, input_size),
                  param_names=param_names)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("input_size", [8])
@pytest.mark.parametrize("horizon", [10])
def test_linear_with_propagation_manager(batch_size, input_size, horizon):
    model = BatchedLinear(input_size, input_size, batch_size)

    manager = PropagationManager(default_propagator=TotalProp())

    x = torch.randn(batch_size, input_size)
    for _ in range(horizon):
        x = torch.sigmoid(model(x))

        x = manager.forward(x,
                            dist_class=Normal,
                            dist_params={
                                'loc': torch.zeros_like(x),
                                'scale': 0.01
                            },
                            ivw_target=model.get_batched_parameters())

    batch_loss = ((1.0 - x)**2).sum(dim=1, keepdim=True)

    # backward with PropagationManager
    manager.backward(batch_loss)

    # check if original parameters have gradients
    utils.check_not_zero(model.weight.grad)
    utils.check_not_zero(model.bias.grad)
