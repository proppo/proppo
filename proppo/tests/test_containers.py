import pytest

import proppo.containers as c


#@pytest.mark.parametrize("value", [1])
def test_content(value=1):
    mycont = c.Content(value)

    a = mycont.get()
    assert a == value

    mycont.set(2)
    b = mycont.get()
    assert b == 2

    mycont.update(3)
    d = mycont.get()
    assert d == 3

    e = c.Content(5)
    mycont.update(e)
    d = mycont.get()
    assert d == 5

    f = c.Content(mycont)
    g = f.get()
    assert g == 5

    mycont = c.Content({'a': 1, 'b': 2})
    mycont.update({'c': 3, 'a': 7})
    answer = {'a': 7, 'b': 2, 'c': 3}
    assert mycont.get() == answer


def test_summed(value=1):
    a = c.Summed(value)
    a.update(2)
    b = a.get()
    assert b == value + 2

    d = c.Summed(1)
    e = c.Content(2)
    d.update(e)
    f = d.get()

    assert f == 3

    g = c.Content(3)
    d.update(g)
    f = d.get()
    assert f == 6

    d.update(4)
    f = d.get()
    assert f == 10


def test_locked(value=1):
    a = c.Locked(value)

    with pytest.raises(RuntimeError):
        a.update(2)


def test_listed(value=1):
    a = c.Listed(value)
    a.update(5)
    b = [value, 5]

    assert a.get() == b

    a.update(c.Content(7))
    d = [value, 5, 7]
    assert a.get() == d

    e = c.Listed(9, 12)
    a.update(e)
    f = [value, 5, 7, 9, 12]
    a.get() == f

    a.set(1, 2, 3)
    b = [1, 2, 3]
    assert a.get() == b

    a.update(4, 5)
    b = [1, 2, 3, 4, 5]
    assert a.get() == b

    a.update(c.Listed(6, 7))
    b = [1, 2, 3, 4, 5, 6, 7]
    assert a.get() == b

    a.update(c.Listed(8, 9), 10, 11)
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert a.get() == b


def test_container(value=1):
    mycont = c.Container(a=value)

    b = mycont['a']
    d = mycont.get('a')

    assert b == value
    assert d == value

    mycont['a'] = 2
    e = mycont['a']
    assert e == 2
    assert isinstance(mycont._contents['a'], c.Content)

    adict = {'a': 1, 'b': 2, 'c': 3}
    conta = c.Container(cont_dict=adict)

    assert 'a' in conta
    assert 'b' in conta
    assert 'c' in conta

    for one, two in zip(adict.items(), conta.items()):
        assert one == two
    for one, two in zip(adict.keys(), conta.keys()):
        assert one == two
    for one, two in zip(adict.values(), conta.values()):
        assert one == two

    bdict = {'a': 7, 'c': 5, 'd': 4}
    contb = c.Container(**bdict)

    conta.update(contb)
    answer = {'a': 7, 'b': 2, 'c': 5, 'd': 4}
    for k, v in conta.items():
        assert answer[k] == v

    adict = {'a': c.Locked(1), 'b': 2, 'c': 3}
    conta = c.Container(cont_dict=adict)

    with pytest.raises(RuntimeError):
        conta.update(contb)

    adict = {'a': c.Listed(1), 'b': 2, 'c': 3}
    conta = c.Container(cont_dict=adict)

    conta.update(contb)
    assert conta['a'] == [1, 7]

    adict = {'a': c.Summed(1), 'b': 2, 'c': 3}
    conta = c.Container(cont_dict=adict)

    conta.update(contb)
    assert conta['a'] == 8

    adict = {'a': c.Summed(1), 'b': 2, 'c': c.Listed(3)}
    conta = c.Container(cont_dict=adict)

    conta.update(contb)
    answer = {'a': 8, 'b': 2, 'c': [3, 5], 'd': 4}
    for k, v in conta.items():
        assert answer[k] == v

    for k, v in answer.items():
        assert conta[k] == v

    for one, two in zip(answer.items(), conta.items()):
        assert one == two

    conta.update(contb)
    answer = {'a': 15, 'b': 2, 'c': [3, 5, 5], 'd': 4}
    for one, two in zip(answer.items(), conta.items()):
        assert one == two

    popc = conta.pop('c')
    answer = {'a': 15, 'b': 2, 'd': 4}
    answerc = [3, 5, 5]
    assert popc == answerc
    for one, two in zip(answer.items(), conta.items()):
        assert one == two

    conta.clear()
    assert conta._contents == {}


def test_message():
    m = c.Message(target=2, a=1, b=2, c=3)
    test = m['a']
    assert test == 1

    test = m

    # Test iterators.
    # Single message mode.
    answer = {'a': 1, 'b': 2, 'c': 3}
    for one, two in zip(answer.items(), test.items()):
        assert one == two
    for one, two in zip(answer.keys(), test.keys()):
        assert one == two
    for one, two in zip(answer.values(), test.values()):
        assert one == two
    for cont in test.containers():
        assert isinstance(cont, c.Container)
    for target, mess in test.messages():
        assert target == 2
        assert isinstance(mess, c.Container)

    assert 'a' in test
    assert 'b' in test
    assert 'c' in test

    m2 = c.Message(target=1, a=1, b=2, c=4)

    assert m.multi_message == False

    # Multi message mode test iterators.
    m.update(m2)

    answerk = ['a', 'b', 'c', 'a', 'b', 'c']
    answert = [2, 1]
    answerv = [1, 2, 3, 1, 2, 4]
    assert answert == list(m.targets())
    assert m.multi_message == True
    assert answerk == list(m.keys())
    assert answerv == list(m.values())

    for cont in m.containers():
        assert isinstance(cont, c.Container)
    for ans_target, message_tuple in zip(answert, m.messages()):
        target, mess = message_tuple
        assert target == ans_target
        assert isinstance(mess, c.Container)

    # Test selecting a message using the target.
    answer1 = {'a': 1, 'b': 2, 'c': 3}
    for one, two in zip(answer1.items(), m[2].items()):
        assert one == two

    assert m[2]['a'] == 1
    assert m[1]['c'] == 4

    m[2]['a'] = 5
    assert m[2]['a'] == 5

    answer2 = {'a': 1, 'b': 2, 'c': 4}
    for one, two in zip(answer2.items(), m[1].items()):
        assert one == two

    # Test pop.
    m.pop(2)
    assert m.multi_message == False
    m['c'] = 5
    assert m['c'] == 5
    answer = {'a': 1, 'b': 2, 'c': 5}
    for one, two in zip(answer.items(), m.items()):
        assert one == two

    out = m.pop('b')
    assert out == 2
    answer = {'a': 1, 'c': 5}
    for one, two in zip(answer.items(), m.items()):
        assert one == two

    # Test getting a message and setting the content.
    m.get_message(1).set_content('a', 2)
    assert m['a'] == 2

    # Test smart contents and update.
    m = c.Message(target=2, a=c.Listed(1), b=c.Summed(2), c=3)
    m2 = c.Message(target=1, a=1, b=2, c=4, d=5)
    m3 = c.Message(target=2, a=2, b=7, e=8)

    m.update(m2)
    m.update(m3)

    answer1 = {'a': [1, 2], 'b': 9, 'c': 3, 'e': 8}
    for one, two in zip(answer1.items(), m[2].items()):
        assert one == two

    answer2 = {'a': 1, 'b': 2, 'c': 4, 'd': 5}
    for one, two in zip(answer2.items(), m[1].items()):
        assert one == two

    c1 = c.Container(a=1, b=2, c=3)
    with pytest.raises(TypeError):
        m.update(c1)

    with pytest.raises(TypeError):
        m.update(1)

    m = c.Message(target=2, a=c.Listed(1), b=c.Summed(2), c=3)
    m3 = c.Message(target=2, a=2, b=7, e=8)
    m.update(m3)

    d1 = {'a': 3, 'b': 1}
    m.update(d1)
    answer1 = {'a': [1, 2, 3], 'b': 10, 'c': 3, 'e': 8}
    for one, two in zip(answer1.items(), m.items()):
        assert one == two


def test_node():
    # TODO: test receiving messages, and updating the contents
    # with both node and container and dict?
    n = c.Node(a=1, b=2, c=3)
    m = c.Message(b=4, d=5)
    n.receive(m)

    assert n.messages['b'] == 4
    answer = {'a': 1, 'b': 2, 'c': 3}
    assert n['a'] == 1
    for one, two in zip(answer.items(), n.items()):
        assert one == two

    message_contents = {'b': 4, 'd': 5}
    for one, two in zip(message_contents.items(), n.messages.items()):
        assert one == two

    # Test Node.from_container
    cont = c.Container(a=1, b=2, c=3, box_class=dict)
    no1 = c.Node.from_container(cont)

    assert no1.messages == {}
    answer = dict(a=1, b=2, c=3)
    for one, two in zip(answer.items(), no1.items()):
        assert one == two
