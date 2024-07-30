import argparse

def main(message):
    print(f"Message: {message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple CLI with argparse")
    parser.add_argument("message", type=str, help="The message to print")
    args = parser.parse_args()
    main(args.message)
