def process_data(data: list):
    print("Process data")
    return [num + 1 for num in data]


if __name__ == "__main__":
    process_data([1, 2, 3])
