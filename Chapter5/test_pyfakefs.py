from pathlib import Path


def save_result(folder: str, file_name: str, result: str):
    # Create new file inside the folder
    file = Path(folder) / file_name
    file.touch()

    # Write result to the new file
    file.write_text(result)

def test_save_result(fs):
    folder = "new"
    file_name = "my_file.txt"
    result = "The accuracy is 0.9"

    fs.create_dir(folder)

    save_result(folder=folder, file_name=file_name, result=result)
    res = Path(f"{folder}/{file_name}").read_text()
    assert res == result