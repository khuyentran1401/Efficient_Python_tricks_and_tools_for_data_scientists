from typing import List, Union

def get_name_price(fruits: list) -> Union[list, tuple]:
    return zip(*fruits)

fruits = [('apple', 2), ('orange', 3), ('grape', 2)]
names, prices = get_name_price(fruits)
print(names)  # ('apple', 'orange', 'grape')
print(prices)  # (2, 3, 2)