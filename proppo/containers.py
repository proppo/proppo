import abc
from typing import List


class Content():

    __slots__ = ('_content', )

    def __init__(self, content):
        if isinstance(content, Content):
            self._content = content.get()
        else:
            self._content = content

    def get(self):
        return self._content

    def set(self, value):
        if isinstance(value, Content):
            v = value.get()
        else:
            v = value

        self._content = v

    def update(self, value):
        if hasattr(self.get(), 'update'):
            if isinstance(value, Content):
                v = value.get()
            else:
                v = value
            self.get().update(v)
        else:
            self.set(value)

    def __repr__(self):
        return 'Content(' + str(self._content) + ')'

    def __str__(self):
        return str(self._content)

    def __add__(self, value):
        if isinstance(value, Content):
            return self.get() + value.get()
        else:
            return self.get() + value

    def __mul__(self, value):
        return self.get() * value.get()

    def __rmul__(self, value):
        return self.get() * value.get()

    def __matmul__(self, value):
        return self.get() @ value.get()


class Summed(Content):

    def update(self, value):
        self.set(self + value)

    def __repr__(self):
        return 'Summed(' + str(self._content) + ')'


class Locked(Content):

    def set(self, value):
        raise RuntimeError(
            'Attempting to call ".set()" on a Locked type'
            ' Content. Locked type Content objects are used'
            ' for contents that are never supposed to be updated.')

    def __repr__(self):
        return 'Locked(' + str(self._content) + ')'


class Listed(Content):

    def __init__(self, *args):
        list_content = []
        for content in args:
            if isinstance(content, Content):
                list_content.append(content.get())
            else:
                list_content.append(content)
        super().__init__(list_content)

    def __repr__(self):
        return 'Listed(' + str(self._content) + ')'

    def update(self, *args):
        if len(args) == 1:
            if not isinstance(args[0], Listed):
                listed_content = Listed(args[0])
            else:
                listed_content = args[0]
        else:
            list_vals = []
            for arg in args:
                if isinstance(arg, Content):
                    if isinstance(arg, Listed):
                        list_vals += arg.get()
                    else:
                        list_vals.append(arg.get())
                else:
                    list_vals.append(arg)
            listed_content = Listed(*list_vals)

        self._content = self + listed_content

    def set(self, *args):
        listed_content = []
        for value in args:
            if isinstance(value, Content):
                v = value.get()
                if not isinstance(value, Listed):
                    v = [v]
            else:
                v = [value]
            listed_content += v

        self._content = listed_content


class Container():

    __slots__ = ('_contents', )

    def __init__(self, cont_dict=None, **kwargs):
        self._contents = {}
        if cont_dict != None:
            kwargs.update(cont_dict)
        for k, v in kwargs.items():
            if not isinstance(v, Content):
                val = Content(v)
            else:
                val = v
            self._contents[k] = val

    def clear(self):
        self._contents.clear()

    def get_contents(self):
        return self._contents

    def set_content(self, key, value):
        if not isinstance(value, Content):
            v = Content(value)
        else:
            v = value

        if key in self._contents:
            self._contents[key].set(v)
        else:
            self._contents[key] = v

    def get(self, key):
        return self[key]

    def item_iter(self):
        for k in self.keys():
            yield (k, self[k])

    def items(self):
        return self.item_iter()

    def keys(self):
        return self._contents.keys()

    def _update_keys(self):
        return self._contents.keys()

    def value_iter(self):
        for k in self.keys():
            yield self[k]

    def values(self):
        return self.value_iter()

    def pop(self, key, default=None):
        try:
            return self._contents.pop(key).get()
        except KeyError as e:
            if default != None:
                return default
            else:
                print(e)

    def __str__(self):
        return 'Contents: ' + str(self._contents)

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, k):
        return self._contents[k].get()

    def __setitem__(self, k, v):
        self.set_content(k, v)

    def __contains__(self, k):
        return k in self._contents

    def update(self, container):
        # Works for both dict or container.
        if isinstance(container, dict):
            c = Container(cont_dict=container)
        else:
            c = container

        for k in c._update_keys():
            if k in self._update_keys():
                self._contents[k].update(c._contents[k])
            else:
                self.set_content(k, c._contents[k])

    def print_all(self):
        # TODO: Remove this
        print(dir(self))


class Node(Container):
    __slots__ = ('_contents', 'messages', 'propagator')

    def __init__(self,
                 cont_dict=None,
                 box_class=Container,
                 propagator=None,
                 **kwargs):
        super().__init__(cont_dict=cont_dict, **kwargs)
        self.messages = box_class()
        self.propagator = propagator

    @classmethod
    def from_container(cls, container):
        kwargs = {}
        if 'box_class' in container:
            kwargs['box_class'] = container.pop('box_class')
        if 'propagator' in container:
            kwargs['propagator'] = container.pop('propagator')
        if isinstance(container, Container):
            cont_dict = container.get_contents()
        elif isinstance(container, dict):
            cont_dict = container
        else:
            raise TypeError('container must be of Container or dict type.')

        return Node(cont_dict=cont_dict, **kwargs)

    def forward(self, x, **kwargs):
        return self.propagator.forward(x, **kwargs)

    def backward(self):
        return self.propagator.backward(self, self.messages)

    def receive(self, message):
        # TODO: add more message box classes
        if isinstance(message, Message):
            for m in message.containers():
                self.messages.update(m)
        else:
            self.messages.update(message)

    def assign_propagator(self, propagator):
        self.propagator = propagator


class Message(Container):
    # TODO: reassign the update method (and other methods) based on the
    # Message multi_message state (get rid of if statements.)
    # TODO: what happens when the message becomes empty via pop()?
    __slots__ = ('_contents', 'multi_message')

    def __init__(self, cont_dict=None, target=-1, container=None, **kwargs):
        if container == None:
            super().__init__(
                cont_dict={target: Container(cont_dict=cont_dict, **kwargs)})
        else:
            super().__init__(cont_dict={target: container})
        self.multi_message = False

    def _switch_multi(self):
        if len(self.targets()) > 1:
            self.multi_message = True
        else:
            self.multi_message = False

    def _get_main_message(self):
        return self._contents[next(iter(self.targets()))].get()

    def get_message(self, target):
        return self._contents[target].get()

    def __str__(self):
        return 'Message: ' + str(self._contents)

    def targets(self):
        return Container.keys(self)

    def iter_containers(self):
        for t in self.targets():
            yield self.get_message(t)

    def containers(self):
        return self.iter_containers()

    def iter_messages(self):
        for t in self.targets():
            yield (t, self.get_message(t))

    def messages(self):
        return self.iter_messages()

    def iter_keys(self):
        for container in self.containers():
            for k in container.keys():
                yield k

    def keys(self):
        if self.multi_message == False:
            return self._get_main_message().keys()
        else:
            return self.iter_keys()

    def iter_items(self):
        for container in self.containers():
            for item in container.items():
                yield item

    def items(self):
        if self.multi_message == False:
            return super().items()
        else:
            return self.iter_items()

    def iter_values(self):
        for container in self.containers():
            for value in container.values():
                yield value

    def values(self):
        if self.multi_message == False:
            return super().values()
        else:
            return self.iter_values()

    def pop(self, k, default=None):
        if default == None:
            args = (k, )
        else:
            args = (k, default)
        if self.multi_message == True:
            out = super().pop(k)
            self._switch_multi()
            return out
        else:
            return self._get_main_message().pop(*args)

    def pop_message(self, k):
        out = super().pop(k)
        self._switch_multi()
        return out

    def update(self, message):
        if isinstance(message, Message):
            super().update(message)
            self._switch_multi(
            )  # check whether a message with a new target was added
        elif isinstance(message, (Container, dict)):
            if self.multi_message == False:
                self._get_main_message().update(message)
            else:
                raise TypeError(
                    'Current Message contains multiple targets. Updating'
                    ' with Container is disabled due to ambiguity'
                    ' in the target. Turn container into Message'
                    ' type with a specified target, then update'
                    ' the current message.')
        else:
            raise TypeError('Message can only be updated with a Message'
                            ' or Container type.')

    def __getitem__(self, k):
        if self.multi_message == False:
            return self._get_main_message()[k]
        else:
            return super().__getitem__(k)

    def __setitem__(self, k, v):
        if self.multi_message == False:
            self._get_main_message().set_content(k, v)
        else:
            super().__setitem__(k, v)

    def __contains__(self, k):
        if self.multi_message == False:
            return k in self._get_main_message()
        return super().__contains(k)
