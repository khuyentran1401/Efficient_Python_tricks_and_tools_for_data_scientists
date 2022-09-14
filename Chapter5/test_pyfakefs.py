from pathlib import Path


def create_file(folder, file_name):
    """Create new file inside the folder"""
    file = Path(folder) / file_name
    file.touch()


def test_create_file(fs):
    """Test if create_file is creating a new file in the fake directory"""

    folder = "new"
    # Create fake directory
    fs.create_dir(folder)
    # Test create_file
    create_file(folder, "my_file.txt")
    assert Path("new/my_file.txt").exists()
