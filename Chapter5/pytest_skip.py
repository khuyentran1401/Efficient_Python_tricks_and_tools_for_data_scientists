import sys
import pytest 

def add_two(num: int):
    return num + 2 

@pytest.mark.skipif(sys.version_info < (3, 9), reason="Eequires Python 3.9 or higher")
def test_add_two(): 
    assert add_two(3) == 5