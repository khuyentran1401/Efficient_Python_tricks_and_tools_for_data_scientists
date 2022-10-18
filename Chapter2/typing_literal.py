from typing import Literal


def get_price(fruit: Literal["apple", "orange"]):
    if fruit == "apple":
        return 1
    else:  # if it is orange
        return 2


get_price("grape")
