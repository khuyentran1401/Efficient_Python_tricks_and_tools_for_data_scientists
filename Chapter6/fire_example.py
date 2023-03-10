import fire


def get_mean(numbers: list):
    return sum(numbers) / len(numbers)


def get_modulo(num1: int, num2: int):
    return num1 % num2


if __name__ == "__main__":
    fire.Fire()
