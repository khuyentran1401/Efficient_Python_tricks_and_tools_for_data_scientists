def calculate_statistics(data: list[float]) -> dict[str, float]:
    return {
        "mean": sum(data) / len(data),
        "first": data[0]
    }

numbers = [1, "a", 3]  # mypy error, but code will run
result = calculate_statistics(numbers)  # Fails at runtime during sum()
