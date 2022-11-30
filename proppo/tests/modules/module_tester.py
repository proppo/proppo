import copy
import torch
import proppo.tests.proppo_test_utils as utils


def module_tester(batched_module,
                  module,
                  input_shape,
                  param_names=['weight', 'bias']):

    for name in param_names:
        param = copy.deepcopy(getattr(batched_module, name))
        setattr(module, name, param)

    # check if backward does not refresh the graph
    for _ in range(3):
        inputs = []
        if not isinstance(input_shape[0], int):
            for shape in input_shape:
                inputs.append(torch.rand(shape))
        else:
            inputs.append(torch.rand(input_shape))

        # check forward
        batched_y = batched_module(*inputs)
        y = module(*inputs)

        utils.check_identical(batched_y, y, exact_match=False)

        # check backward
        batched_loss = ((1.0 - batched_y)**2).mean()
        batched_loss.backward()

        loss = ((1.0 - y)**2).mean()
        loss.backward()

        if batched_module.is_detached:
            batched_module.apply_gradients()

        for name in param_names:
            grad = getattr(module, name).grad
            batched_grad = getattr(batched_module, name).grad
            utils.check_identical(batched_grad, grad, exact_match=False)

        # check zero_grad
        batched_module.zero_grad()
        module.zero_grad()
        for name in param_names:
            grad = getattr(batched_module, name).grad
            utils.check_zero(grad)
            if batched_module.is_detached:
                batched_grad = getattr(batched_module, 'batched_' + name).grad
                utils.check_zero(batched_grad)
