from collections import OrderedDict
from functools import partial
import copy


class Init():

    def __init__(self, cls, *args, defaults=None, kwdefaults=None, **kwargs):
        self._cls = cls
        # TODO: fix the default arguments so that they are first
        # converted into keyword arguments, and then merged with the
        # keyword arguments. If there is a conflict, e.g., the same
        # keyword exists in both sets, then raise an error.

        if kwdefaults != None:
            self.kwdefaults = {**kwdefaults, **kwargs}
        elif kwargs != {}:
            self.kwdefaults = kwargs
        if defaults != None:
            self.defaults = (*args, *defaults)
        elif args != ():
            self.defaults = args

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'kwdefaults'):
            input_kwargs = {**self.kwdefaults, **kwargs}
        else:
            input_kwargs = kwargs
        # Note that concatenating defaults and args is not allowed
        # due to ambiguity of when to overwrite.
        # TODO: fix the default arguments so that they are first
        # converted into keyword arguments, and then merged with the
        # keyword arguments. If there is a conflict, e.g., the same
        # keyword exists in both sets, then raise an error.
        if hasattr(self, 'defaults') and len(args) == 0:
            input_args = self.defaults
        else:
            input_args = args

        return self._call(*input_args, **input_kwargs)

    def _call(self, *args, **kwargs):
        return self._cls(*args, **kwargs)

    @classmethod
    def init(cls, *args, **kwargs):
        return Init(cls, *args, **kwargs)


class Lock(Init):

    def __init__(self, *args, allow_unused=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_unused = allow_unused

    def __call__(self, *args, allow_unused=False, **kwargs):
        if allow_unused or self.allow_unused:
            return super().__call__()
        elif len(args) == 0 and len(kwargs) == 0:
            return super().__call__()
        else:
            raise RuntimeError('allow_unused permission is False.'
                               ' The Lock initalizer allows only'
                               ' initializing with the default parameters.'
                               ' Remove args and kwargs inputs from'
                               ' initialization.')


class Optional(Init):
    """ Initializer that only initializes when the on flag is
    True. Otherwise, an instance of the class is not created, it returns None.
    """

    def _call(self, on, *args, **kwargs):
        if on:
            return super()._call(*args, **kwargs)
        else:
            return None


class Choice(Init):

    def __init__(self, cls, *args, **kwargs):
        if not isinstance(cls, (tuple, dict)):
            raise TypeError('cls input must be of type tuple or dict.')
        else:
            super().__init__(cls, *args, **kwargs)

    def _call(self, choice, *args, **kwargs):
        return self._cls[choice](*args, **kwargs)


class Empty(Init):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        self._call()

    def _call(self, *args, **kwargs):
        raise RuntimeError('Trying to initialize a ChainInit containing'
                           ' an empty Init. First reconfigure the'
                           ' ChainInit to replace the Empty Init.'
                           ' Empty Inits are used in template'
                           ' ChainInits to indicate what slot has to be'
                           ' changed.')


class ChainInit():

    def __init__(self, regular_dict=False, **kwargs):

        if regular_dict:
            self._chaininit = {}
        else:
            self._chaininit = OrderedDict()

        for k, v in kwargs.items():
            if isinstance(v, Init):
                init = v
            else:
                init = Init(v)
            self._chaininit[k] = init

    def __call__(self, **kwargs):
        return self.init(**kwargs)

    def items(self):
        return self._chaininit.items()

    def values(self):
        return self._chaininit.values()

    def keys(self):
        return self._chaininit.keys()

    def __iter__(self):
        return self._chaininit.__iter__()

    def __contains__(self, key):
        return key in self._chaininit

    def __getitem__(self, key):
        return self._chaininit[key]

    def __len__(self):
        return len(self._chaininit)

    @staticmethod
    def _init_inputs(init, inputs=None):

        if inputs == None:
            obj = init()
        elif isinstance(inputs, tuple):
            arginputs = []
            kwinputs = {}
            for inp in inputs:
                if isinstance(inp, dict):
                    kwinputs.update(inp)
                else:
                    arginputs.append(inp)
            obj = init(*arginputs, **kwinputs)
        elif isinstance(inputs, dict):
            obj = init(**inputs)
        else:
            obj = init(inputs)
        return obj

    def init(self, dictionary=False, **kwargs):

        chainobjs = []
        chainkeys = []

        # Check that all keys exist
        for key in kwargs:
            if key not in self._chaininit:
                raise KeyError(key, 'not included in keys of the ChainInit.')

        for k, init in self._chaininit.items():
            inputs = kwargs.get(k, None)  # kwarg if exists, otherwise None

            obj = self._init_inputs(init, inputs)

            if obj != None:
                chainobjs.append(obj)
                chainkeys.append(k)
        if not dictionary:
            return chainobjs
        else:
            return {k: obj for k, obj in zip(chainkeys, chainobjs)}

    def reconf(self, **kwargs):
        # Create a new ChainInit by reconfiguring the current
        # ChainInit. For example, replace the Empty initializer
        # of the current ChainInit to create a functioning ChainInit.
        # You may also change the default parameters of the Inits.

        chaindict = copy.copy(self._chaininit)
        for k, v in kwargs.items():
            if k not in chaindict:
                raise KeyError('Key does not exist in the ChainInit'
                               ' that you are trying to reconfigure.')
            else:
                if isinstance(v, Init):
                    chaindict[k] = v
                else:
                    initializer = type(chaindict[k])
                    cls = chaindict[k]._cls
                    initializer = partial(initializer, cls)
                    chaindict[k] = self._init_inputs(initializer, v)

        return ChainInit(**chaindict)


class ChainInitTemplate(ChainInit):

    def __call__(self, **kwargs):
        initializers = super().__call__(dictionary=True, **kwargs)
        return ChainInit(**initializers)
