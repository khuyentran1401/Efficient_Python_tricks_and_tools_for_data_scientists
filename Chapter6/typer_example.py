import typer 

def process_data(data: str, version: int):
    print(f'Processing {data},' 
          f'version {version}')

if __name__ == '__main__':
    typer.run(process_data)