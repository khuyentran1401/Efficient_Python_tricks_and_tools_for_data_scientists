import typer

def main(message: str):
    print(f"Message: {message}")

if __name__ == "__main__":
    typer.run(main)
