import click

@click.command()
@click.argument("message")
def main(message):
    print(f"Message: {message}")

if __name__ == "__main__":
    main()
