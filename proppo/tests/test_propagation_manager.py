import torch

import proppo.tests.proppo_test_utils as utils

from proppo.propagation_manager import PropagationManager
from proppo.propagators import Propagator


def stub_forward(x, should_dummy):
    assert should_dummy == 'dummy'
    return {'output': x, 'register_node': True}


def stub_backward(grad, target_local_loss, metrics):

    def _f(node, local_loss):
        # track the number of calls
        metrics['counter'] += 1
        utils.check_identical(local_loss, target_local_loss)
        return [node['output']], grad

    return _f


class StubPropagator():

    def __init__(self):
        self.forward = stub_forward
        self.backward = stub_backward

    def loss_propagator(self):
        return StubPropagator()


# class StubPropagator(Propagator):

#     def __init__(self):
#         self.forward_impl = stub_forward
#         self.backward_impl = stub_backward

#     def loss_propagator(self):
#         return StubPropagator()


def test_forward():
    prop = StubPropagator()

    manager = PropagationManager(default_propagator=prop,
                                 terminal_propagator=None)

    x = torch.rand(2, 3)

    y = manager.forward(x, should_dummy='dummy', local_propagator=prop)

    utils.check_identical(x, y)
    assert y.data_ptr() == x.data_ptr()
    assert manager.size() == 1
    assert 'output' in manager.nodes[0]


def test_forward_with_default_propagator():
    manager = PropagationManager(default_propagator=StubPropagator())

    x = torch.rand(2, 3)

    y = manager.forward(x, should_dummy='dummy')

    utils.check_identical(x, y)


def test_backward():
    # This is tested for example in modules/test_linear
    # TODO: But it would be good to write a better test here
    # Note, that I deleted the previous test when I added BackPropagators
    pass
