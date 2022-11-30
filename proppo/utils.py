import torch


def expand(data, batch_size):
    """ Returns the tensor with batch dimension expanded.

    Arguments:
        data (torch.Tensor): input tensor
        batch_size (int): batch size for expansion

    Returns:
        torch.Tensor: output tensor

    """
    return data.expand((batch_size, ) + data.shape)


def inverse_variance_weighting(x1, x2, scalar_estimate=True):
    """ Returns weights of inverse variance weighting for each input.

    Arguments:
        x1 (torch.Tensor or list): input tensor
        x2 (torch.Tensor or list): input tensor

    Returns:
        torch.Tensor: weight for x1
        torch.Tensor: weight for x2

    """

    if isinstance(x1, (list, tuple)) and isinstance(x2, (list, tuple)):
        x1_vars = []
        x2_vars = []
        c_list = []
        for v1, v2 in zip(x1, x2):
            assert v1.shape == v2.shape
            batch_size = v1.shape[0]

            d1 = v1 - v1.mean(dim=0, keepdims=True)
            d2 = v2 - v2.mean(dim=0, keepdims=True)

            c1 = torch.max(torch.abs(d1))
            c2 = torch.max(torch.abs(d2))
            c = torch.max(c1, c2)
            c_list.append(c)

            if c == 0:
                x1_vars.append(torch.tensor(0.0, device=c.device))
                x2_vars.append(torch.tensor(0.0, device=c.device))
            else:
                x1_vars.append(torch.sum((d1 / c)**2))
                x2_vars.append(torch.sum((d2 / c)**2))
        x1vec = torch.tensor(x1_vars)
        x2vec = torch.tensor(x2_vars)
        cvec = torch.tensor(c_list)
        cmax = torch.max(cvec)
        if cmax == 0:
            x1_var = torch.tensor(1.0, device=cmax.device)
            x2_var = torch.tensor(1.0, device=cmax.device)
        else:
            cvec = (cvec / cmax)**2
            x1_var = torch.sum(x1vec * cvec)
            x2_var = torch.sum(x2vec * cvec)
    else:
        assert x1.shape == x2.shape
        batch_size = x1.shape[0]

        d1 = x1 - x1.mean(dim=0, keepdims=True)
        d2 = x2 - x2.mean(dim=0, keepdims=True)

        c1 = torch.max(torch.abs(d1))
        c2 = torch.max(torch.abs(d2))
        c = torch.max(c1, c2)

        if c == 0:
            x1_var = torch.tensor(1.0, device=c.device)
            x2_var = torch.tensor(1.0, device=c.device)
        else:
            x1_var = torch.sum((d1 / c)**2)
            x2_var = torch.sum((d2 / c)**2)

    k_x1 = x2_var / (x1_var + x2_var)
    k_x1 = torch.clip(k_x1, 0, 1)

    if torch.isnan(k_x1):
        print('Warning: estimated k was nan. Automatically changed to 0.5.')
    k_x1[torch.isnan(k_x1)] = 0.5  # when 0/0 error occurs, take them equally.

    return k_x1, 1.0 - k_x1
