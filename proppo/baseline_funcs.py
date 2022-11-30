import torch


def mean_baseline(local_losses, node=None):
    """ Subtracts the mean baseline from the losses for use in the
    likelihood ratio gradient estimator. A Bessel correction is
    added to debiase the magnitude of the gradient.
    
    """
    with torch.no_grad():
        batch_size = local_losses.numel()

        # leave-one-out baselines
        sum_loss = local_losses.sum()
        # The division by (batch_size - 1) instead of (batch_size)
        # is algebraically equivalent to a leave-one-out baseline.
        sum_loss = (sum_loss - local_losses) / (batch_size - 1)
        losses = local_losses - sum_loss
    return losses


def no_baseline(local_losses, node=None):
    """ A method that does not subtract a baseline.
    Set the baseline in LRProp or TotalProp to no_baseline if
    it is desired not to use a baseline.

    """
    return local_losses
