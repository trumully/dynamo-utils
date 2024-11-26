import pickle  # noqa: S403

from dynamo_utils.sentinel import Sentinel


def test_sentinel_identity() -> None:
    s1 = Sentinel("TEST_SENTINEL")
    s2 = Sentinel("TEST_SENTINEL")

    assert s1 is s2
    assert s1 == s2
    assert isinstance(s1, s1)  # type: ignore[reportArgumentType]
    assert isinstance(s1, s2)  # type: ignore[reportArgumentType]


def test_sentinel_repr() -> None:
    s = Sentinel("WITH_CUSTOM_REPR", repr="<CustomRepr>")

    assert repr(s) == "<CustomRepr>"


def test_sentinel_truthiness() -> None:
    s = Sentinel("TEST_SENTINEL")

    assert not bool(s)

    s = Sentinel("TRUTHY_SENTINEL", truthiness=True)

    assert bool(s)


def test_sentinel_pickle() -> None:
    original = Sentinel("PICKLE_TEST")
    pickled = pickle.dumps(original)
    unpickled = pickle.loads(pickled)  # noqa: S301

    assert original is unpickled


def test_sentinel_subclass() -> None:
    class MySentinel(Sentinel):
        pass

    s1 = MySentinel("TEST", module_name="test_module")
    s2 = MySentinel("TEST", module_name="test_module")

    assert s1 is s2
    assert s1 == s2
