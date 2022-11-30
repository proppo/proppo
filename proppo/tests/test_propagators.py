from proppo.propagators import (SequenceProp, BackPropagator, RPProp,
                                BaselineProp, PauseBase, Propagator, RPBase,
                                LRProp, LRBase, TotalProp, TPBase)
from proppo.initializers import ChainInit, Optional, Choice, Init, Empty, Lock
import pytest


class DummyProp(Propagator):

    def __init__(self, a=1, b=1, c=1, **kwargs):
        self.a = a
        self.b = b
        self.c = c
        super().__init__(**kwargs)


def test_sequence_prop():

    # Test initialization with propagators.
    class TestSequence(SequenceProp,
                       propagators=ChainInit(
                           backprop=Optional(BackPropagator, True),
                           rp=Lock(PauseBase),
                           baseline=Optional(BaselineProp, False))):
        pass

    test_sequence = TestSequence()
    assert isinstance(test_sequence.propagators[0], BackPropagator)
    assert isinstance(test_sequence.propagators[1], PauseBase)
    assert len(test_sequence.propagators) == 2

    test_sequence = TestSequence(backprop=False, baseline=True)
    assert isinstance(test_sequence.propagators[0], PauseBase)
    assert isinstance(test_sequence.propagators[1], BaselineProp)
    assert len(test_sequence.propagators) == 2

    test_sequence = TestSequence(baseline=True)
    assert isinstance(test_sequence.propagators[0], BackPropagator)
    assert isinstance(test_sequence.propagators[1], PauseBase)
    assert isinstance(test_sequence.propagators[2], BaselineProp)
    assert len(test_sequence.propagators) == 3

    # Test initialization with a Dummy class, and default arguments
    class TestSequence(SequenceProp,
                       propagators=ChainInit(first=Optional(DummyProp, True),
                                             second=Init(DummyProp,
                                                         a=2,
                                                         b=3,
                                                         c=4),
                                             third=Optional(DummyProp,
                                                            False,
                                                            a=22,
                                                            b=5,
                                                            c=7))):
        pass

    dummy = TestSequence()
    answer1 = {'a': 1, 'b': 1, 'c': 1}
    for k, v in answer1.items():
        assert getattr(dummy.propagators[0], k) == v
    answer2 = {'a': 2, 'b': 3, 'c': 4}
    for k, v in answer2.items():
        assert getattr(dummy.propagators[1], k) == v
    assert len(dummy.propagators) == 2

    dummy = TestSequence(third=True)
    answer3 = {'a': 22, 'b': 5, 'c': 7}
    for k, v in answer3.items():
        assert getattr(dummy.propagators[2], k) == v

    dummy = TestSequence(third=(True, {'a': 1, 'b': 2, 'c': 3}))
    answer3 = {'a': 1, 'b': 2, 'c': 3}
    for k, v in answer3.items():
        assert getattr(dummy.propagators[2], k) == v

    # Test initialization with propagators.
    class TestSequence(SequenceProp,
                       propagators=ChainInit(
                           backprop=Optional(BackPropagator, True),
                           rp=Lock(RPProp),
                           baseline=Optional(BaselineProp, False))):
        pass

    test_sequence = TestSequence()
    assert isinstance(test_sequence.propagators[0], BackPropagator)
    assert isinstance(test_sequence.propagators[1], RPProp)
    assert len(test_sequence.propagators) == 2

    with pytest.raises(RuntimeError):
        test_sequence = TestSequence(rp={'backprop': False})


def test_rpprop():
    rpprop = RPProp()

    assert isinstance(rpprop.propagators[0], BackPropagator)
    assert isinstance(rpprop.propagators[1], RPBase)

    rpprop = RPProp(backprop=False)

    assert isinstance(rpprop.propagators[0], RPBase)


def test_lrprop():
    lrprop = LRProp()

    assert isinstance(lrprop.propagators[0], BackPropagator)
    assert isinstance(lrprop.propagators[1], LRBase)
    assert isinstance(lrprop.propagators[2], BaselineProp)

    lrprop = LRProp(backprop=False)

    assert isinstance(lrprop.propagators[0], LRBase)
    assert isinstance(lrprop.propagators[1], BaselineProp)


def test_tpprop():
    tpprop = TotalProp()

    assert isinstance(tpprop.propagators[0], BackPropagator)
    assert isinstance(tpprop.propagators[1], TPBase)
    assert isinstance(tpprop.propagators[2], BaselineProp)

    tpprop = TotalProp(backprop=False)

    assert isinstance(tpprop.propagators[0], TPBase)
    assert isinstance(tpprop.propagators[1], BaselineProp)

    tpprop = TotalProp(baseline=False)

    assert isinstance(tpprop.propagators[0], BackPropagator)
    assert isinstance(tpprop.propagators[1], TPBase)
