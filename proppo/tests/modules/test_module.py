import torch.nn as nn
import torch
import copy
import pytest

import proppo.tests.proppo_test_utils as utils

from proppo.modules import BatchedLinear
from proppo.modules import get_batched_parameters


@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("input_size", [32])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("output_size", [8])
def test_get_batched_parameters(batch_size, input_size, hidden_size,
                                output_size):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = BatchedLinear(input_size, hidden_size, batch_size)
            self.fc2 = BatchedLinear(hidden_size, output_size, batch_size)

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            return self.fc2(h)

    model = Model()

    parameters = get_batched_parameters(model)
    ref_shapes = [(batch_size, hidden_size, input_size),
                  (batch_size, hidden_size),
                  (batch_size, output_size, hidden_size),
                  (batch_size, output_size)]

    for ref_shape, parameter in zip(ref_shapes, parameters):
        assert parameter.shape == ref_shape
