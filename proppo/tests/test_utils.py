import torch

import proppo.utils as utils

from proppo.tests.proppo_test_utils import check_identical


def test_expand():
    x = torch.rand(2, 3)
    batch_size = 8

    y = utils.expand(x, batch_size)

    assert y.shape == (8, 2, 3)


def test_inverse_variance_weighting():
    x1 = torch.rand(2, 3)
    x2 = torch.rand(2, 3)

    x1_var = torch.var(x1, dim=0).mean()
    x2_var = torch.var(x2, dim=0).mean()

    ref_k_x1 = x2_var / (x1_var + x2_var)
    ref_k_x2 = 1 - ref_k_x1

    k_x1, k_x2 = utils.inverse_variance_weighting(x1, x2)

    check_identical(k_x1, ref_k_x1, exact_match=False)
    check_identical(k_x2, ref_k_x2, exact_match=False)

    assert k_x1.shape == torch.Size([])
    assert k_x2.shape == torch.Size([])


def test_inverse_variance_weighting_with_lists():
    x1 = [torch.rand(2, 3, 4), torch.rand(2, 4, 5)]
    x2 = [torch.rand(2, 3, 4), torch.rand(2, 4, 5)]

    flatten_x1 = torch.cat([x1[0].view(2, 12), x1[1].view(2, 20)], dim=1)
    flatten_x2 = torch.cat([x2[0].view(2, 12), x2[1].view(2, 20)], dim=1)

    x1_var = torch.var(flatten_x1, dim=0).mean()
    x2_var = torch.var(flatten_x2, dim=0).mean()

    ref_k_x1 = x2_var / (x1_var + x2_var)
    ref_k_x2 = 1 - ref_k_x1

    k_x1, k_x2 = utils.inverse_variance_weighting(x1, x2)

    check_identical(k_x1, ref_k_x1, exact_match=False)
    check_identical(k_x2, ref_k_x2, exact_match=False)

    assert k_x1.shape == torch.Size([])
    assert k_x2.shape == torch.Size([])
