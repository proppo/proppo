from proppo.modules.linear import BatchedLinear


def get_batched_parameters(model):
    """ Retruns list of batched parameters at each module.

    Arguments:
        model (torch.nn.Module): torch.nn.Module model.

    Returns:
        list: list of parameters.

    """
    parameters = []
    for layer in model.modules():
        if isinstance(layer, BatchedLinear):
            parameters += layer.get_batched_parameters()
    return parameters
