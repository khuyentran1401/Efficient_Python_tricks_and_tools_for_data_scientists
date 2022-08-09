import pytest 

def divide_two_nums(num1, num2):
    return num1 / num2

@pytest.mark.xfail(raises=ZeroDivisionError)
def test_divide_by_zero():
    res = divide_two_nums(2, 0)
