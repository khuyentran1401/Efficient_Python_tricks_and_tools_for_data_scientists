class Fruit:
    def __init__(self, taste: str) -> None:
        self.taste = taste 

class Orange(Fruit):
    ... 

class Apple(Fruit):
    ... 

def make_fruit(fruit_type: Fruit, taste: str):
    return fruit_type(taste=taste)

orange = make_fruit(Orange, "sour")