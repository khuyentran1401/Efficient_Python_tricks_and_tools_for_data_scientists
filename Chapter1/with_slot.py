from random import randint
from memory_profiler import profile


class Dog:
    __slots__ = ["age"] 
    def __init__(self, age):
        self.age = age


@profile
def main():
    return [Dog(age=randint(0, 30)) for _ in range(100000)]


if __name__ == "__main__":
    main()
