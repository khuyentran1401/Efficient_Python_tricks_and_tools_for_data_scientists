import typer

app = typer.Typer()

@app.command()
def add_numbers(x: float, y: float):
    """Adds two numbers and prints the result."""
    result = x + y
    print(f"The sum of {x} and {y} is {result}.")

if __name__ == "__main__":
    app()
