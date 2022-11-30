import torch


def check_keys(node, keys=[]):
    for k in keys:
        assert k in node


def check_differentiable(output, targets=[]):
    loss = ((output - 1.0)**2).mean()

    loss.backward()

    for t in targets:
        assert t.grad is not None


def check_not_differentiable(output, targets=[]):
    loss = ((output - 1.0)**2).mean()

    if output.requires_grad:
        loss.backward()

        for t in targets:
            assert t.grad is None
    else:
        assert not output.requires_grad


def check_identical(output, target, exact_match=True):
    if exact_match:
        assert torch.all(output == target)
    else:
        assert torch.allclose(output, target, rtol=1e-6, atol=1e-7)


def check_not_identical(output, target, exact_match=True):
    if exact_match:
        assert not torch.all(output == target)
    else:
        assert not torch.allclose(output, target, atol=1e-6)


def check_zero(target):
    check_identical(target, torch.zeros_like(target))


def check_not_zero(target):
    check_not_identical(target, torch.zeros_like(target))


def check_shape(output, target):
    assert output.shape == target.shape
