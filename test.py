from typing import List

class Parent:
    def funct(self):
        pass

class Child1(Parent):
    def funct(self):
        print("child1")

class Child2(Parent):
    def funct(self):
        print("child2")

def tst(inp:Parent):
    inp.funct()

lst : List[Parent] = []
lst.append(Child1())
lst.append(Child2())

for l in lst:
    l.funct()