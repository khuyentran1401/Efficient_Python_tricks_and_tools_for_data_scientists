from watchfiles import run_process


def train():
    print("Detect changes in process_data.py." 
          " Train the model again")


if __name__ == "__main__":
    run_process("process_data.py", target=train)
