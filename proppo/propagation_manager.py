import torch
import proppo.back_methods as back_methods
import proppo.propagators as propagators
from proppo.containers import Node, Message


def _reversed_enumerate(l):
    count = len(l)
    for value in reversed(l):
        count -= 1
        yield count, value


class PropagationManager:
    """ Propagation Manager class.

    This class enables custom forward and backward propagations for flexibly
    designing new gradient estimation and learning algorithms for computational
    graphs, e.g. neural networks.

    .. code-block:: python

        import proppo

        from proppo.propagators import RPProp

        manager = PropagationManager(default_propagator=RPProp(args))

        model = SomeModel()

        s = torch.rand(32, 256)
        for _ in range(100):
            s = model(s)
            s = manager.forward(s) # register s as a node and inject rp noise

        loss = ((s - 1.0) ** 2)

        manager.backward(loss) # backward from the leaf tensor


    Attributes:
        nodes (list): list of Node objects.
        default_propagator (callable): default propagator
            See `proppo.propagators`.

    """

    def __init__(self,
                 default_propagator=propagators.BackPropagator(),
                 loss_propagator=None,
                 terminal_propagator=propagators.BackPropagator()):
        """ __init__ method.
        
        """
        self.nodes = []
        self.node_pointer = 0  # Pointer for the current position on the tape.
        self.default_propagator = default_propagator
        self.propagators = {}  # dictionary of equipped named propagators.
        if loss_propagator:
            self.loss_propagator = loss_propagator
        elif default_propagator:
            self.loss_propagator = self.default_propagator.loss_propagator()
        # The terminal propagator exists to handle any remaining messages
        # once the backward pass has finished. For example, a common use
        # case is to use BackPropagator() to call backprop once all
        # outputs and gradients have been assembled for the backprop call,
        # if these can be performed in parallel.
        self.terminal_propagator = terminal_propagator
        if self.terminal_propagator != None:
            self.forward(x=None, local_propagator=terminal_propagator)
            self.nodes[0]['clear'] = False

    def add_propagator(self, name, propagator):
        self.propagators[name] = propagator

    def forward(self,
                x,
                force_targets=None,
                local_propagator=None,
                get_node=False,
                **kwargs):
        """ Register input as a node, and returns output through local forward
            function.

        Arguments:
            x: input
            **kwargs (any): any number of arguments for local forward function.

        Returns:
            output: output variable

        """
        # TODO: Currently for each iteration a new Node object is created,
        # but we may want to keep using the same node between different
        # iterations of the algorithm.
        # If the node already exists, e.g., it was not cleared,
        # then creating a new node will not be necessary.
        # Also, information can be stored in the node, between
        # operations, and this seems useful. Reusing the same nodes
        # will speed up operation.

        # Do modification to x, e.g. add noise
        if local_propagator != None:
            if isinstance(local_propagator, str):
                local_propagator = self.propagators[local_propagator]
            node = local_propagator.forward(x, **kwargs)
        else:
            node = self.default_propagator.forward(x, **kwargs)

        if not isinstance(node, Node):
            node = Node.from_container(node)

        # assign the local_propagator to the node so that it knows what
        # backward method to use in the backward pass.
        node.assign_propagator(local_propagator)

        if force_targets != None:
            node['force_targets'] = force_targets

        if 'output' in node:
            output = node['output']
        else:
            output = None

        if node['register_node']:
            if self.node_pointer > (len(self.nodes) - 1):
                self.nodes.append(node)
            else:
                self.nodes[self.node_pointer] = node
            self.node_pointer += 1

        # Return output as well as optionally pointers to the node.
        # The pointer to the node is used to set other
        # propagators to target their messages there.
        if get_node:
            return (output, self.nodes[self.node_pointer])
        else:
            return output

    def backward(self, loss=None, clear_nodes=False, message=None):
        """ Execute backward propagation at the registered nodes one by one.

        Arguments:
            loss: optional loss as the last node in the propagation graph.
            clear_nodes (bool): flag to clear the nodes after propagation.

        """
        # If a loss node is added, append as the last propagation node.
        if loss != None:
            self.append_loss(loss)

        if message != None:  # Send message to last node in the graph.
            self._send_message(
                message=message, target_node=self.nodes[-1]
            )  # TODO: allow the message to be of Message type and
            # include targets. Currently message should be a Container object.

        # Loop through all nodes in reverse order, calling the
        # custom backward method of that node
        for i, node in _reversed_enumerate(self.nodes):
            self.node_pointer = i
            if node.propagator != None:
                messages = node.backward()
            else:
                messages = self.default_propagator.backward(
                    node, node.messages)

            # Clear node content, then send the message;
            # this allows to send a message to the propagators own slot
            # as well (i.e. to keep a history between different
            # manager.backward() calls). Node can optionally be not cleared.
            # (To keep a history in the node the propagator should
            # send a message to itself TODO: this does not work in
            # the current implementation because the message.clear is
            # in the end. I put the clear in the end at the moment
            # because of the Gaussian shaping gradient implementation.
            # There was some issue with the clear clearing out the message
            # to be sent as well.)
            if node['clear']:
                self.nodes[i].clear()

            if messages != None:  # If there are messages, then send them.
                for target, message_container in messages.messages():
                    # choose the priority target, transform the target,
                    # send the message
                    targets = self._target_conflict_resolution(target, node)
                    target_nodes = self._find_nodes(targets)
                    self._send_messages(message_container, target_nodes)

            self.nodes[i].messages.clear()

        # refresh node history
        if clear_nodes:
            if self.terminal_propagator:
                self.nodes = [self.nodes[0]]
            else:
                self.nodes = []

        # TODO: remove this if-statement by instead changing a default value.
        # Or maybe this is OK for now, because there are only 2 choices?
        if self.terminal_propagator:
            self.node_pointer = 1
        else:
            self.node_pointer = 0

        return messages  # The last remaining messages are returned if desired.

    def _send_message(self, message, target_node):
        if target_node != None:
            target_node.receive(message)

    def _send_messages(self, messages, targets):
        # Note that this allows sending a single message to multiple
        # targets by having a list of targets.
        if isinstance(targets, list):
            for target in targets:
                self._send_message(messages, target)
        else:
            self._send_message(messages, targets)

    def _target_conflict_resolution(self, target, node):
        # Chooses the target where to send the message based on the
        # priority. The 'force_targets' in the node has the highest
        # priority, next is the target in the message, then the
        # regular 'targets' in the node, finally if there is no target,
        # then the target is -1
        if 'force_targets' in node:
            targets = node['force_targets']
            return targets
        elif target != None:
            return target
        elif 'targets' in node:
            return node['targets']
        else:
            targets = -1
            return targets

    def _find_node(self, target):
        """ The 'target' is an address for the node. The current
        method will find the node corresponding to the address and
        return a pointer to it.

        """
        if isinstance(target, int):
            node_index = target + self.node_pointer
            if (node_index < 0) or (node_index > (len(self.nodes) + 1)):
                target = None  # If out of bounds, don't send
            else:
                target = self.nodes[node_index]
        return target

    def _find_nodes(self, targets):
        """ If targets is iterable, it will find the node correspoding to
        each target.

        """
        if isinstance(targets, (list, tuple)):
            targets = [self._find_node(t) for t in targets]
            return targets
        else:
            targets = self._find_node(targets)
            return targets

    def append_loss(self, loss_node, loss_propagator=None, **kwargs):
        """ append_loss simply calls the forward method of the loss propagator
        and adds a node into the propagation graph. This is equivalent
        to calling manager.forward(loss_node, local_propagtor=loss_propagator)

        """
        if loss_propagator is None:
            loss_propagator = self.loss_propagator
        out = self.forward(x=loss_node,
                           targets=None,
                           local_propagator=loss_propagator,
                           get_node=False,
                           **kwargs)
        return out

    def size(self):
        """ Returns the number of the registered nodes.

        Returns:
            int: the number of nodes.

        """
        return len(self.nodes)
