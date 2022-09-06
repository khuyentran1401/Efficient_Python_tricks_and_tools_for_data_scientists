from typing import Annotated


def get_height_in_feet(height: Annotated[float, "meters"]):
    return height * 3.28084


print(get_height_in_feet(height=1.6))
