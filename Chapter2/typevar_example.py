from typing import TypeVar


T = TypeVar("T")


def last(l: list[T]) -> T:
    return l[-1]


if __name__ == "__main__":
    last(list(range(10)))
    last(list("abc"))
