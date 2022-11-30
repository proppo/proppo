import proppo.forward_methods as fm
import proppo.back_methods as bm
import proppo.baseline_funcs as baselines
import proppo as pp
from proppo.utils import inverse_variance_weighting
from proppo.containers import Node, Message, Container
from proppo.initializers import (ChainInit, Optional, Init, Empty,
                                 ChainInitTemplate)

import copy
# TODO: move the Monte Carlo gradient estimator propagators into a
# different file and directory.


class Propagator:
    """ Propagators are the basic building blocks of Automatic Propagation
    software, such as Proppo.
    Each Propagtor pairs together the forward and backward methods of 
    the implemeneted algorithm. All Propagators should inherit from this
    Propagator class.

    """

    def __init_subclass__(cls, **kwargs):
        cls.default_init_kwargs = kwargs

    def __init__(self, **kwargs):
        self.default_forward_kwargs = kwargs

    def forward(self, x, **kwargs):
        # Overwrite default arguments, then pass as input
        # TODO: This could be made better by using nested dictionary
        # merges in case the input kwargs also contain dictionaries
        input_kwargs = {**self.default_forward_kwargs, **kwargs}

        # Must create a new node, and pass this to forward, otherwise
        # the propagators at different forward steps will overwrite, the
        # contents of the previous propagation.
        node = {}
        node = self.forward_impl(x, node, **input_kwargs)

        # Flag to store the node in manager.
        if 'register_node' not in node:
            node['register_node'] = True
        # Flag to clear node in manager after backwarding the node.
        if 'clear' not in node:
            node['clear'] = True

        if isinstance(node, dict):  # Backward compatibility for dictionaries.
            node = Container(cont_dict=node)

        return node

    def forward_impl(self, x, node={}, **kwargs):
        return node

    def backward(self, node, message):
        #message_in = copy.copy(message)
        message_in = message
        message_out = self.backward_impl(node, message_in)
        # TODO: Is this problematic if the backward_impl removes gradients,
        # and this is not desired?

        # for backwards compatibility, convert dictionaries
        if not isinstance(message_out, Message):
            if isinstance(message_out, dict):
                if 'targets' in message_out:
                    target = message_out.pop('targets')
                    message_out = Message(cont_dict=message_out, target=target)
                else:
                    message_out = Message(cont_dict=message_out)
            elif isinstance(message_out, Container):
                message_out = Message(container=message_out)

        return message_out

    def backward_impl(self, node, message):
        message_out = Message(cont_dict=message.get_contents())
        return message_out

    # TODO: the loss_propagator method should be moved over to an MCGradProp
    # class.
    def loss_propagator(self):
        """ Returns the default loss propagator that should
        be applied when appending a loss after having called
        manager.forward using the current propagator.

        """
        return LossProp()


class SequenceProp(Propagator):
    """ Base class for sequence based propagators, used to construct them.
    
    """

    def __init_subclass__(cls, propagators=ChainInit(), **kwargs):
        super().__init_subclass__(**kwargs)
        cls.propagators = propagators

    def _split_prop_kwargs(self, kwargs):
        # If a key in kwargs is the name of a propagator in the propagator
        # sequence, the argument corresponding to that key is passed to the
        # specified propagator. Otherwise, the keyword argument is passed
        # to the initialization of the sequence propagator itself.
        prop_kwargs = {}
        for k in self.propagators:
            if k in kwargs:
                prop_kwargs[k] = kwargs.pop(k)
        return prop_kwargs, kwargs

    def __init__(
            self,
            propagators=[],  # A list of already initialized propagators.
            **kwargs):

        # Default initialization keyword arguments.
        default_init_kwargs = copy.copy(self.default_init_kwargs)

        if propagators != []:
            input_kwargs = {**default_init_kwargs, **kwargs}

            super().__init__(**input_kwargs)
            self.propagators = propagators
        else:
            def_prop_kwargs, def_init_kwargs = self._split_prop_kwargs(
                default_init_kwargs)
            prop_kwargs, init_kwargs = self._split_prop_kwargs(kwargs)
            input_kwargs = {**def_init_kwargs, **init_kwargs}

            super().__init__(**input_kwargs)

            input_prop_kwargs = {**def_prop_kwargs, **prop_kwargs}

            self.propagators = self.propagators(**input_prop_kwargs)


class ComboProp(SequenceProp):
    """ Combines propagators, and applies them in a sequence, updating 
    the message and node in-place.

    """

    def forward_impl(self, x, node={}, **kwargs):
        for prop in self.propagators:
            node_out = prop.forward_impl(x, node, **kwargs)
            node.update(node_out)
        return node

    def backward_impl(self, node, message):
        final_message = Message(cont_dict=message.get_contents())
        for prop in reversed(self.propagators):
            message_in = final_message._get_main_message()

            message_out = prop.backward_impl(node, message_in)

            # for backwards compatibility, convert dictionaries
            if isinstance(message_out, dict):
                if 'targets' in message_out:
                    target = message_out.pop('targets')
                    message_out = Message(cont_dict=message_out, target=target)
                else:
                    message_out = Message(cont_dict=message_out)

            final_message.update(message_out)

        return final_message


class BackPropagator(Propagator):
    """ Base propagator that will backprop gradient messages, if they
    are sent into this propagator.

    """

    def backward_impl(self, node, message):

        message_out = bm.backward(node, message)
        return message_out


class BaselineProp(Propagator):
    """ Class for adding a baseline subtraction to the local losses.
    The baseline_func argument may also be a list or tuple, in this
    case the baselines are applied in sequence starting from the
    end of the list.

    """

    def __init__(self, baseline_func=baselines.mean_baseline, **kwargs):
        super().__init__(**kwargs)
        self.baseline_func = baseline_func  # Default baseline function

        # Note: if one wants to change the baseline function for just one
        # forward call compared to the default baseline function in a chain
        # of forward propagations, then they should define a new propagator
        # object for that new forward call. I could also allow giving an
        # additional argument in the forward call to specify a baseline
        # for just that node; however, this would not
        # give a key error if someone accidentally mistypes the key, and
        # may lead to bugs, so I avoid it.

    def backward_impl(self, node, message):
        local_loss = message.pop('local_loss')
        # Need to remove local_loss from previous message, and create
        # a new message to avoid duplicating loss in the ComboProp
        # backward_impl method.
        if isinstance(self.baseline_func, (list, tuple)):
            baselined_loss = copy.copy(local_loss)
            for func in reversed(self.baseline_func):
                baselined_loss = func(baselined_loss, node)
        else:
            baselined_loss = self.baseline_func(local_loss, node)

        message_out = {
            'baselined_loss': baselined_loss,
            'local_loss': local_loss
        }
        return message_out


# A Template for initializing the propagator chains in Monte Carlo
# gradient estimator Propagators.
mcgrad_temp = ChainInitTemplate(
    backprop=Optional(Optional.init(BackPropagator, True), False),
    base=Init,
    baseline=Optional(Optional.init(BaselineProp, True), False))


class PauseBase(Propagator):
    """ A propagator that pauses all incoming gradients, then sends the
    combined gradient backwards. This is useful to prevent errors with
    the computation graph being freed up by undesired gradients 
    propagating through the graph prematurely.
    
    """

    def forward_impl(self, x, node={}, **kwargs):
        node = fm.detached_output(x, **kwargs)
        return node

    def backward_impl(self, node, message):
        message_out = bm.rp_gradient(node, message)
        return message_out


class SkipProp(Propagator):
    """ A propagator that sends all incoming messages backward
    a determined length, skipping the nodes inbetween.

    """

    def __init__(self, skip=1, **kwargs):
        super().__init__(**kwargs)
        self.skip = skip

    def forward_impl(self, x, node={}, **kwargs):
        node['output'] = x
        return node

    def backward_impl(self, node, message):
        out_message = Message(container=copy.copy(message), target=-self.skip)
        return out_message


class PauseProp(ComboProp,
                propagators=mcgrad_temp(backprop=True, base=PauseBase)):
    pass


class SumBase(Propagator):
    """ Propagator that adds a local variable with a different variable in
    messages. This is usually used to accumulate a sum of variables during the
    backward pass, e.g. sum the rewards to obtain the return in 
    reinforcement learning.

    """

    def __init__(self, sum_name, local_variable, **kwargs):
        self.sum_name = sum_name
        self.local_variable = local_variable
        super().__init__(**kwargs)

    def backward_impl(self, node, message):
        current_sum = message.pop(self.sum_name, 0)
        message = {self.sum_name: message[self.local_variable] + current_sum}

        return message


class ChainProp(SequenceProp):
    """ Chains together a set of propagators into a single propagator.
    The implementation is based on creating a new PropagationManager object
    to correctly apply the propagators in sequence without any implementation
    errors. The propagators to chain together should be given as a list
    or tuple of Propagator instances during creation. The propagators
    themselves can also be Chain propagators, which allows for defining
    complex propagation strategies using nested propagation managers.

    """

    def forward_impl(self, x, node={}, chain_kwargs=[], **kwargs):
        manager = pp.PropagationManager(default_propagator=None,
                                        terminal_propagator=None)
        if chain_kwargs:
            for prop, kwarg in zip(self.propagators, chain_kwargs):
                kwarg.update(kwargs)
                x = manager.forward(x, local_propagator=prop, **kwarg)
        else:
            for prop in self.propagators:
                x = manager.forward(x, local_propagator=prop, **kwargs)

        node = {'output': x, 'manager': manager}
        if manager.size() == 0:
            node['register_node'] = False
        return node

    def backward_impl(self, node, message):
        message_out = node['manager'].backward(message=message)
        return message_out

    def loss_propagator(self):
        """ By default, usually the last one in the chain
        contains the correct loss propagator.

        """
        return self.propagators[-1].loss_propagator()


class RPBase(Propagator):
    """ Base class for RP propagator.

    """

    def forward_impl(self, x, node={}, detach=True, **kwargs):
        node = fm.rp_noise(x, detach=detach, **kwargs)
        return node

    def backward_impl(self, node, message):
        message_out = bm.rp_gradient(node, message)
        return message_out


class RPProp(ComboProp, propagators=mcgrad_temp(backprop=True, base=RPBase)):
    """ RP propagator combining the functionality from ComboProp.

    """
    pass


class LossBase(Propagator):
    """ Base class for loss nodes in the computational graph.

    """

    def __init__(self, loss_name='local_loss', **kwargs):
        super().__init__(**kwargs)
        self.loss_name = loss_name

    def forward_impl(self,
                     x,
                     node={},
                     lossgrad=True,
                     lossfunc=None,
                     sum_loss=True,
                     **kwargs):
        if lossfunc:
            if isinstance(x, dict):
                losses = lossfunc(**x)
            else:
                losses = lossfunc(x)
        else:
            losses = x

        node = fm.loss_forward(losses, sum_loss=sum_loss, lossgrad=lossgrad)
        return node

    def backward_impl(self, node, message):
        message_out = bm.loss_backward(node, message, loss_name=self.loss_name)
        return message_out


class LossProp(ComboProp,
               propagators=ChainInit(
                   backprop=Optional(BackPropagator, True),
                   baseline=Optional(BaselineProp,
                                     True,
                                     baseline_func=baselines.mean_baseline),
                   base=LossBase)):
    """ Propagator adding Baseline and ComboProp functionality to Loss nodes.

    """
    pass


class LRBase(Propagator):
    """ Base class for likelihood ratio gradient propagators.

    """

    def forward_impl(self, x, node, **kwargs):
        node = fm.lr_noise(x, **kwargs)
        return node

    def backward_impl(self, node, message):
        message_out = bm.lr_gradient(node, message)
        return message_out


class LRProp(ComboProp,
             propagators=mcgrad_temp(backprop=True, base=LRBase,
                                     baseline=True)):
    """ Class adding ComboProp functionality to LR gradient propagators.

    """

    def loss_propagator(self):
        return LossProp(backprop=False, lossgrad=False)


class TPBase(Propagator):
    """ Base class for total propagation gradient propagators.

    """

    def __init__(self,
                 var_weighting_func=inverse_variance_weighting,
                 **kwargs):
        super().__init__(**kwargs)
        self.var_weighting_func = var_weighting_func

    def forward_impl(self, x, node, **kwargs):
        node = fm.totalprop_noise(x, **kwargs)
        return node

    def backward_impl(self, node, message):
        message_out = bm.totalprop_gradient(
            node, message, var_weighting_func=self.var_weighting_func)
        return message_out


class TotalProp(ComboProp,
                propagators=mcgrad_temp(backprop=True,
                                        base=TPBase,
                                        baseline=True)):
    """ Class adding ComboProp functionality to total propagation 
    gradient propagation nodes.

    """
    pass


class ResampleBase(Propagator):
    """ Base class for a resampling propagator.
    For example, it would fit a Gaussian distribution on the batch
    of x, then resample a new batch from this Gaussian distribution.

    """

    def forward_impl(self, x, node, detach=True, **kwargs):
        node = fm.gauss_resample(x, detach=detach, **kwargs)
        return node

    def backward_impl(self, node, message):
        message_out = bm.resample_back(node, message)
        return message_out


class ResampleProp(ComboProp,
                   propagators=mcgrad_temp(backprop=True, base=ResampleBase)):
    pass


class GRProp(ChainProp, propagators=ChainInit(rp=RPProp, resamp=ResampleProp)):
    """ This implements Gaussian resampling propagation in sequence
    with sampling noise from a Gaussian distribution together with 
    reparameterization gradients. Potentially, the implementation could
    be improved by detaching the RPProp.

    """
    pass


class GaussShapeBase(Propagator):

    def forward_impl(self, x, node, detach=True, **kwargs):
        node = fm.gauss_shape(x, **kwargs)
        return node

    def backward_impl(self, node, message):
        message_out = bm.gs_back(node, message)
        return message_out


class GSProp(ComboProp,
             propagators=ChainInit(backprop=Optional(BackPropagator, True),
                                   sum_base=Init(SumBase,
                                                 sum_name='local_loss',
                                                 local_variable='shaped_stat'),
                                   gs_base=GaussShapeBase)):
    pass


class GSLossBase(ComboProp,
                 propagators=mcgrad_temp(backprop=True, base=LossBase)):
    pass


class GSLoss(ChainProp,
             propagators=ChainInit(gs_prop=Init(GSProp,
                                                shaped_grad=True,
                                                backprop=True),
                                   gs_loss=Init(
                                       GSLossBase,
                                       sum_loss=False,
                                       base={'loss_name': 'shaped_loss'}),
                                   skip=Init(SkipProp, skip=2))):
    pass
