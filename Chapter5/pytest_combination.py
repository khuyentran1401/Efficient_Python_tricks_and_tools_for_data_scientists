from pytest import mark

def average(n1, n2):
    return (n1 + n2) / 2

def perc_difference(n1, n2):
    return (n2 - n1)/n1 * 100

@mark.parametrize("operation", [average, perc_difference])
@mark.parametrize("n1, n2", [(1, 2), (2, 3)])
def test_is_float(operation, n1, n2):
    assert isinstance(operation(n1, n2), float)