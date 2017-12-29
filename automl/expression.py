

class Expression:
    def __add__(self, other):
        return AddExpression(self, other)
    
    def __sub__(self, other):
        return SubExpression(self, other)

    def __mul__(self, other):
        return MulExpression(self, other)
    
    def __truediv__(self, other):
        return DivExpression(self, other)

class Atom(Expression):
    def __init__(self, index):
        self.index = index

    def eval(self, data):
        return data[:, self.index]

class AddExpression(Expression):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def eval(self, data):
        return self.a.eval(data) + self.b.eval(data)

class SubExpression(Expression):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def eval(self, data):
        return self.a.eval(data) - self.b.eval(data)

class MulExpression(Expression):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def eval(self, data):
        return self.a.eval(data) * self.b.eval(data)

class DivExpression(Expression):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def eval(self, data):
        return self.a.eval(data) / self.b.eval(data)