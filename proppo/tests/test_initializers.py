import pytest
from proppo.initializers import (Init, Optional, Empty, ChainInit, Lock,
                                 Choice, ChainInitTemplate)


class Dummy():
    pass


class DummyIn():

    def __init__(self, a, b=2, **kwargs):
        self.a = a
        self.b = b
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_init():
    a = Init(Dummy)
    b = a()

    assert isinstance(b, Dummy)

    a = Init(DummyIn)
    b = a(1)  # First argument to init is 1
    assert b.a == 1
    assert b.b == 2

    # Set default keyword arguments.
    a = Init(DummyIn, kwdefaults={'a': 3, 'b': 5, 'c': 7})
    b = a()
    assert b.a == 3
    assert b.b == 5
    assert b.c == 7

    # Mix of keyword and regular default arguments.
    a = Init(DummyIn, 4, kwdefaults={'b': 5, 'c': 7})
    b = a()
    assert b.a == 4
    assert b.b == 5
    assert b.c == 7

    a = Init(DummyIn, 4, 9, kwdefaults={'c': 7})
    b = a()
    assert b.a == 4
    assert b.b == 9
    assert b.c == 7

    a = Init(DummyIn, 4, 9, kwdefaults={'c': 7})
    b = a(1, 2)  # Overwrite default arguments
    assert b.a == 1
    assert b.b == 2
    assert b.c == 7

    # Use "defaults" keyword.
    a = Init(DummyIn, defaults=(4, 9), kwdefaults={'c': 7})
    b = a(1)
    assert b.a == 1
    assert b.b == 2
    assert b.c == 7

    # Use keyword argument to Init in addition to kwdefaults
    # Both are interchangeable methods to add default kwargs.
    a = Init(DummyIn, kwdefaults={'c': 7}, d=10)
    b = a(1)
    assert b.a == 1
    assert b.b == 2
    assert b.c == 7
    assert b.d == 10


def test_optional():
    a = Optional(Dummy)

    b = a(True)
    assert isinstance(b, Dummy)

    b = a(False)
    assert b == None

    a = Optional(DummyIn, kwdefaults={'a': 3, 'b': 5, 'c': 7})
    with pytest.raises(TypeError):
        b = a()  # This fails because it's missing the on keyword argument.
    b = a(True)
    assert isinstance(b, DummyIn)

    a = Optional(DummyIn, kwdefaults={'on': True, 'a': 3, 'b': 5, 'c': 7})
    b = a()
    assert b.a == 3
    assert b.b == 5
    assert b.c == 7

    a = Optional(DummyIn, kwdefaults={'on': False, 'a': 3, 'b': 5, 'c': 7})
    b = a()
    assert b == None

    # Test mix of kwdefaults and defaults.
    # The "True" will correspond to the "on" argument
    a = Optional(DummyIn, True, kwdefaults={'a': 3, 'b': 5, 'c': 7})
    b = a()
    assert b.a == 3
    assert b.b == 5
    assert b.c == 7

    a = Optional(DummyIn, False, kwdefaults={'a': 3, 'b': 5, 'c': 7})
    b = a()
    assert b == None


def test_empty():
    with pytest.raises(TypeError):
        a = Empty(Dummy)

    a = Empty()

    with pytest.raises(RuntimeError):
        b = a(1)  # Can't initialize an empty Init.


def test_lock():
    a = Lock(DummyIn, a=2, b=4)
    b = a()
    assert b.a == 2
    assert b.b == 4

    with pytest.raises(RuntimeError):
        b = a(1)  # Lock does not allow changing the arguments.

    b = a(1, allow_unused=True)
    assert b.a == 2  # The argument is not changed to 1.
    assert b.b == 4

    # Test "allow_unused" keyword as a default kwarg.
    a = Lock(DummyIn, a=3, b=5, allow_unused=True)
    b = a(1)
    assert b.a == 3
    assert b.b == 5


def test_choice():

    with pytest.raises(TypeError):
        a = Choice(Dummy)  # The input should be a tuple or a dict.

    a = Choice({
        'first': Init(DummyIn, a=3, b=5),
        'second': Init(DummyIn, a=1, b=2, c=10)
    })

    f = a('first')
    assert f.a == 3
    assert f.b == 5

    f = a('second')
    assert f.a == 1
    assert f.b == 2
    assert f.c == 10

    f = a('second', a=2, b=5, d=24)
    assert f.a == 2
    assert f.b == 5
    assert f.c == 10
    assert f.d == 24

    a = Choice(
        {
            'first': Init(DummyIn, a=3, b=5),
            'second': Init(DummyIn, a=1, b=2, c=10)
        },
        a=10)  # Test extra keyword argument

    f = a('first')
    assert f.a == 10

    f = a('first', a=20)  # Test overwriting the default "a"
    assert f.a == 20

    # Test initialization from a tuple
    a = Choice((Init(DummyIn, a=3, b=5), Init(DummyIn, a=1, b=2, c=10)))

    f = a(0)  # Choice by index
    assert f.a == 3
    assert f.b == 5

    f = a(1)
    assert f.a == 1
    assert f.b == 2
    assert f.c == 10


def test_chaininit():
    # Test simple initialization.
    chain1 = ChainInit(a=Dummy, b=Dummy)

    b = chain1()
    for v in b:
        assert isinstance(v, Dummy)

    # Test initialization with default parameters
    chaininit1 = ChainInit(a=Init(DummyIn, 1), b=Init(DummyIn, 2, 4))

    chain = chaininit1()
    el = chain[0]
    assert el.a == 1
    assert el.b == 2

    el = chain[1]
    assert el.a == 2
    assert el.b == 4

    # Test initialization using Empty init
    chaininit2 = ChainInit(a=Init(DummyIn, 1),
                           c=Empty(),
                           b=Init(DummyIn, 2, 4))

    with pytest.raises(RuntimeError):  # Runtime error due to Empty
        chain = chaininit2()

    # Test iteration
    answer = ['a', 'c', 'b']
    for one, two in zip(answer, chaininit2.keys()):
        assert one == two

    for one, two in zip(answer, chaininit2):
        assert one == two

    items = [('a', Init), ('c', Empty), ('b', Init)]
    for one, two in zip(items, chaininit2.items()):
        onek, onev = one
        twok, twov = two
        assert onek == twok
        assert onev == type(twov)

    items = [('a', Init), ('c', Empty), ('b', Init)]
    for one, two in zip(items, chaininit2.values()):
        onek, onev = one
        assert onev == type(two)

    # Test initialization using Optional init
    chaininit3 = ChainInit(a=Init(DummyIn, a=1),
                           c=Optional(DummyIn,
                                      True,
                                      kwdefaults={
                                          'a': 1,
                                          'b': 5,
                                          'd': 11
                                      }),
                           b=Init(DummyIn, a=2, b=4))

    chain = chaininit3()
    el1 = chain[1]
    assert el1.a == 1
    assert el1.b == 5
    assert el1.d == 11

    chain = chaininit3(c=False)
    assert len(chaininit3) == 3
    assert len(chain) == 2
    assert chain[0].a == 1
    assert chain[0].b == 2
    assert chain[1].a == 2
    assert chain[1].b == 4

    chaininit3 = ChainInit(a=Init(DummyIn, a=1),
                           c=Optional(DummyIn,
                                      True,
                                      kwdefaults={
                                          'a': 1,
                                          'b': 5,
                                          'd': 11
                                      }),
                           b=Init(DummyIn, a=2, b=4))

    # Test targeted keyword arguments.
    chain4 = chaininit3(a={'a': -5, 'b': 7, 'c': 23}, c=False, b={'a': 5})

    assert len(chain4) == 2
    assert chain4[0].a == -5
    assert chain4[0].b == 7
    assert chain4[0].c == 23
    assert chain4[1].a == 5
    assert chain4[1].b == 4

    chaininit3 = ChainInit(a=Init(DummyIn, a=1),
                           c=Optional(DummyIn,
                                      False,
                                      kwdefaults={
                                          'a': 1,
                                          'b': 5,
                                          'd': 11
                                      }),
                           b=Init(DummyIn, a=2, b=4))

    chain5 = chaininit3(
        a={
            'a': -5,
            'b': 7,
            'c': 23
        },
        c=(True, {
            'a': 5,
            'b': 8
        }),  # Testing tuple
        b={'a': 5})

    assert len(chain5) == 3
    assert chain5[0].a == -5
    assert chain5[0].b == 7
    assert chain5[0].c == 23
    assert chain5[1].a == 5
    assert chain5[1].b == 8
    assert chain5[2].a == 5
    assert chain5[2].b == 4

    chaininit3 = ChainInit(a=Init(DummyIn, a=1),
                           c=Optional(DummyIn,
                                      False,
                                      1,
                                      kwdefaults={
                                          'b': 5,
                                          'd': 11
                                      }),
                           b=Init(DummyIn, a=2, b=4))

    chain = chaininit3(
        a={
            'a': -5,
            'b': 7,
            'c': 23
        },
        c=(True, 5, {
            'b': 8
        }),  # Testing multiple args
        b={'a': 5})

    assert len(chain) == 3
    assert chain[0].a == -5
    assert chain[0].b == 7
    assert chain[0].c == 23
    assert chain[1].a == 5
    assert chain[1].b == 8
    assert chain[2].a == 5
    assert chain[2].b == 4

    chaininit2 = ChainInit(a=Init(DummyIn, 1),
                           c=Empty(),
                           b=Init(DummyIn, 2, 4))

    with pytest.raises(RuntimeError):  # Runtime error due to Empty
        chain = chaininit2()

    # Testing reconfiguring the chain to fill the Empty initializer slot.
    chaininit = chaininit2.reconf(c=Init(DummyIn, 3))

    assert isinstance(chaininit['c'], Init)
    assert len(chaininit) == 3

    chain = chaininit()
    assert chain[1].a == 3
    assert len(chain) == 3

    with pytest.raises(KeyError):
        chaininit(d=Init(DummyIn, 2))  # "d" is not a member of the chain

    chaininit = chaininit2.reconf(c=Init(DummyIn, 3),
                                  b=(1, 3),
                                  a={
                                      'd': 4,
                                      'a': 2
                                  })
    chain = chaininit()
    assert chain[2].a == 1
    assert chain[2].b == 3
    assert len(chain) == 3
    assert chain[0].a == 2
    assert chain[0].d == 4

    chaininit = chaininit2.reconf(a=(1, {
        'd': 4,
        'b': 3
    }),
                                  c=Init(DummyIn, 5),
                                  b=(1, {
                                      'b': 6
                                  }))
    chain = chaininit()
    assert len(chain) == 3
    assert chain[0].a == 1
    assert chain[0].d == 4
    assert chain[0].b == 3
    assert chain[1].a == 5
    assert chain[1].b == 2
    assert chain[2].a == 1
    assert chain[2].b == 6

    # Testing Chain with a Choice initializer.
    chaininit2 = ChainInit(a=Init(DummyIn, 1),
                           c=Choice({
                               'first': Init(DummyIn, a=3, b=5),
                               'second': Init(DummyIn, a=1, b=2, c=10)
                           }),
                           b=Init(DummyIn, 2, 4))

    chain = chaininit2(c='first')
    assert chain[1].a == 3
    assert chain[1].b == 5

    chain = chaininit2(c='second')
    assert chain[1].a == 1
    assert chain[1].b == 2

    chain = chaininit2(c=('first', {'c': 10, 'b': 20}))
    assert chain[1].a == 3
    assert chain[1].b == 20
    assert chain[1].c == 10

    # Test dictionary output
    chain = chaininit2(dictionary=True, c=('first', {'c': 10, 'b': 20}))

    assert chain['a'].a == 1
    assert chain['a'].b == 2
    assert chain['c'].a == 3
    assert chain['c'].b == 20
    assert chain['c'].c == 10
    assert chain['b'].a == 2
    assert chain['b'].b == 4

    chaininit2 = ChainInit(a=Optional(DummyIn, False, 1),
                           b=Init(DummyIn, 2, 4))

    chain2 = chaininit2(dictionary=True)

    assert len(chain2) == 1
    assert chain2['b'].a == 2
    assert chain2['b'].b == 4

    # Test an empty chain
    emptychain = ChainInit()
    e = emptychain()

    assert e == []


def test_chain_init_template():
    # TODO: Think about this. Maybe an Optional.init(*args, **kwargs) method would be nice?
    # Perhaps there is some way to use this together with
    template = ChainInitTemplate(first=Optional(Optional,
                                                True,
                                                defaults=(DummyIn, True),
                                                a=2,
                                                b=4),
                                 second=Optional(
                                     Init(Optional,
                                          DummyIn,
                                          True,
                                          a=3,
                                          b=5,
                                          c=7), False))
    chaininit = template()
    assert len(chaininit) == 1

    chain = chaininit()

    assert chain[0].a == 2
    assert chain[0].b == 4
    assert len(chain) == 1

    chaininit2 = template(second=True)

    chain = chaininit2()
    assert len(chain) == 2
    assert chain[0].a == 2
    assert chain[0].b == 4
    assert chain[1].a == 3
    assert chain[1].b == 5
    assert chain[1].c == 7

    # Again but using Init.init()
    template = ChainInitTemplate(
        first=Optional(Optional.init(defaults=(DummyIn, True), a=2, b=4),
                       True),
        second=Optional(Optional.init(DummyIn, True, a=3, b=5, c=7), False))
    chaininit = template()
    assert len(chaininit) == 1

    chain = chaininit()

    assert chain[0].a == 2
    assert chain[0].b == 4
    assert len(chain) == 1

    chaininit2 = template(second=True)

    chain = chaininit2()
    assert len(chain) == 2
    assert chain[0].a == 2
    assert chain[0].b == 4
    assert chain[1].a == 3
    assert chain[1].b == 5
    assert chain[1].c == 7

    chain = chaininit2(first=False)
    assert len(chain) == 1
    assert chain[0].a == 3
    assert chain[0].b == 5
    assert chain[0].c == 7

    chain = chaininit2(second=False)
    assert len(chain) == 1
    assert chain[0].a == 2
    assert chain[0].b == 4

    chain = chaininit2(second=(True, {'a': 4, 'b': 6, 'c': 8}))
    assert len(chain) == 2
    assert chain[1].a == 4
    assert chain[1].b == 6
    assert chain[1].c == 8

    chain = chaininit2(first=False, second=(True, {'a': 4, 'b': 6, 'c': 8}))
    assert len(chain) == 1
    assert chain[0].a == 4
    assert chain[0].b == 6
    assert chain[0].c == 8

    template = ChainInitTemplate(first=Init.init(defaults=(DummyIn, ),
                                                 a=2,
                                                 b=4),
                                 second=Init.init(DummyIn, a=3, b=5, c=7))

    chaininit = template(first={'a': 1, 'b': 3}, second={'a': 2, 'b': 4})
    chain = chaininit()

    assert chain[0].a == 1
    assert chain[0].b == 3
    assert chain[1].a == 2
    assert chain[1].b == 4
    assert chain[1].c == 7

    # Test reassigning an empty propagator.
    template = ChainInitTemplate(first=Init.init(defaults=(DummyIn, ),
                                                 a=2,
                                                 b=4),
                                 second=Init.init(DummyIn, a=3, b=5, c=7),
                                 third=Init)

    chaininit = template(third=(DummyIn, {'a': 9, 'b': 11, 'd': 13}))
    chain = chaininit()

    assert chain[2].a == 9
    assert chain[2].b == 11
    assert chain[2].d == 13
