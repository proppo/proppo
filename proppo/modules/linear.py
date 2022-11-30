import torch
import torch.nn as nn
import torch.nn.functional as F

from proppo.utils import expand


class BatchedLinear(nn.Linear):
    """ Linear layer with batched parameters inherited from nn.Linear.

    Attributes:
        batch_size (int); batch size for parameters.
        batched_weight (torch.Tensor): weight parameter with shape of
                                       (batch size, output size, input size).
        batched_bias (torch.Tensor): bias parameter with shape of
                                     (batch size, output size).
        is_detached (bool): flag to detach batched parameters.

    """

    def __init__(self,
                 in_features,
                 out_features,
                 batch_size,
                 bias=True,
                 detach=False):
        """ __init__ method.

        Arguments:
            in_features (int): input size.
            out_features (int): output size.
            batch_size (int): batch size.
            bias (bool): flag to use bias.
            detach (bool): flag to detach batched parameters. This is necessary
                           to hold batched gradients.

        """
        super().__init__(in_features, out_features, bias)
        self.batch_size = batch_size
        self.is_detached = detach
        self.rebuild_batched_parameters()

    def forward(self, input, use_batched_parameters=True):
        """ forward method.

        Arguments:
            input (torch.Tensor): input tensor.
            use_batched_parameters (bool): flag to use batched parameters.

        Returns:
            torch.Tensor: output tensor.

        """
        if not use_batched_parameters:
            return super().forward(input)

        # (batch, input) -> (batch, 1, input)
        reshaped_input = input.view(self.batch_size, 1, -1)
        # (batch, output, input) -> (batch, input, output)
        transposed_weight = self.batched_weight.transpose(1, 2)
        # (batch, 1, input) x (batch, input, output) -> (batch, 1, output)
        output = torch.bmm(reshaped_input, transposed_weight)
        # (batch, 1, output) -> (batch, output)
        output = output.view(self.batch_size, -1)
        if self.batched_bias is not None:
            output += self.batched_bias
        return output

    def get_batched_parameters(self):
        """ Return batched parameters.

        Returns:
            list: list of batched weight and bias.

        """
        batched_parameters = [self.batched_weight]
        if self.batched_bias is not None:
            batched_parameters += [self.batched_bias]
        return batched_parameters

    def apply_gradients(self):
        """ Apply gradients at batched parameters to non-batched ones.

        This method should be called after backward of the PropagationManager.

        """
        assert self.is_detached
        self.weight.grad = self.batched_weight.grad.sum(dim=0).detach().clone()
        if self.batched_bias is not None:
            self.bias.grad = self.batched_bias.grad.sum(dim=0).detach().clone()

    def zero_grad(self):
        super().zero_grad()
        if self.is_detached:
            self.batched_weight.grad.zero_()
            if self.batched_bias is not None:
                self.batched_bias.grad.zero_()

    def rebuild_batched_parameters(self):
        """ Rebuild batched parameters from the latest non-batched parameters.

        This method should be called after the non-batched parameters are
        updated.

        """
        # batched weight parameter
        self.batched_weight = expand(self.weight, self.batch_size)
        if self.is_detached:
            self.batched_weight = self.batched_weight.detach()
            self.batched_weight.requires_grad_()

        # batched bias parameter
        if self.bias is None:
            self.batched_bias = None
        else:
            self.batched_bias = expand(self.bias, self.batch_size)
            if self.is_detached:
                self.batched_bias = self.batched_bias.detach()
                self.batched_bias.requires_grad_()
