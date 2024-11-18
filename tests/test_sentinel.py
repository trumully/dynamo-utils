import pickle

from src.sentinel import Sentinel, sentinel


def test_sentinel_identity():
    s1 = sentinel("TEST_SENTINEL")
    s2 = sentinel("TEST_SENTINEL")

    assert s1 is s2
    assert s1 == s2


def test_sentinel_repr():
    s = sentinel("TEST_SENTINEL", repr="<CustomRepr>")

    assert repr(s) == "<CustomRepr>"


def test_sentinel_truthiness():
    s = sentinel("TEST_SENTINEL")

    assert not bool(s)


def test_sentinel_pickle():
    original = sentinel("PICKLE_TEST")
    pickled = pickle.dumps(original)
    unpickled = pickle.loads(pickled)  # noqa: S301

    assert original is unpickled


def test_sentinel_subclass():
    class MySentinel(Sentinel):
        pass

    s1 = MySentinel("test_module", "TEST")
    s2 = MySentinel("test_module", "TEST")

    assert s1 is s2
    assert s1 == s2
