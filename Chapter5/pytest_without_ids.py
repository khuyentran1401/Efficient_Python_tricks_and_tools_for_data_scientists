from pytest import mark


def average(n1, n2):
    return (n1 + n2) / 2

@mark.parametrize(
    "n1, n2",
    [(-1, -2), (2, 3), (0, 0)],
)
def test_is_float(n1, n2):
    assert isinstance(average(n1, n2), float)
