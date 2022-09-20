from typing import final 

class Dog:
    @final 
    def bark(self) -> None:
        print("Woof")

class Dachshund(Dog):
    def bark(self) -> None:
        print("Ruff")

bim = Dachshund()
bim.bark()