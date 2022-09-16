import pytest


def average(n1, n2):
    return (n1 + n2) / 2


examples = [
    pytest.param(-1, -2, id="neg-neg"),
    pytest.param(2, 3, id="pos-pos"),
    pytest.param(0, 0, id="0-0"),
]


@pytest.mark.parametrize("n1, n2", examples)
def test_is_float(n1, n2):
    assert isinstance(average(n1, n2), float)
