from pytest_steps import test_steps


def sum(n1, n2):
    return n1 + n2


def average_2_nums(sum):
    return sum / 2


def sum_test(steps_data):
    res = sum(1, 3)
    assert res == 4
    steps_data.res = res


def average_2_nums_test(steps_data):
    avg = average_2_nums(steps_data.res)
    assert avg == 2


@test_steps(sum_test, average_2_nums_test)
def test_calc_suite(test_step, steps_data):
    if test_step == 'sum_test':
        sum_test(steps_data)
    elif test_step == 'average_2_nums_test':
        average_2_nums_test(steps_data)
