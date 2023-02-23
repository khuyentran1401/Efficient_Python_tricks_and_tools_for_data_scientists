from random import randint
from memory_profiler import profile


class Dog:
    def __init__(self, age):
        self.age = age


@profile
def main():
    return [Dog(age=randint(0, 30)) for _ in range(100000)]


if __name__ == "__main__":
    main()
