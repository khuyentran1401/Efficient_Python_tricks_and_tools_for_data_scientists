import pytest 

@pytest.fixture(scope="session")
def my_data():
    print("Reading data...")
    return 1

def test_division(my_data):
    print("Test division...")
    assert my_data / 2 == 0.5

def test_modulus(my_data):
    print("Test modulus...")
    assert my_data % 2 == 1