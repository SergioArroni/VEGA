import math
import random
import operator
import functools
from ..core import Problem, Solution, EPSILON
from ..types import Real, Binary
from abc import ABCMeta


class ZDT(Problem):
    __metaclass__ = ABCMeta

    def __init__(self, nvars):
        super().__init__(nvars, 2)
        self.types[:] = Real(0, 1)


class ZDT1(ZDT):
    def __init__(self):
        super().__init__(30)

    def evaluate(self, solution):
        x = solution.variables[:]
        g = (9.0 / (self.nvars - 1.0)) * sum(x[1:]) + 1.0
        h = 1.0 - math.sqrt(x[0] / g)
        solution.objectives[:] = [x[0], g * h]


class ZDT2(ZDT):
    def __init__(self):
        super().__init__(30)

    def evaluate(self, solution):
        x = solution.variables[:]
        g = (9.0 / (self.nvars - 1.0)) * sum(x[1:]) + 1.0
        h = 1.0 - math.pow(x[0] / g, 2.0)
        solution.objectives[:] = [x[0], g * h]


class ZDT3(ZDT):
    def __init__(self):
        super().__init__(30)

    def evaluate(self, solution):
        x = solution.variables[:]
        g = (9.0 / (self.nvars - 1.0)) * sum(x[1:]) + 1.0
        h = 1.0 - math.sqrt(x[0] / g) - (x[0] / g) * math.sin(10.0 * math.pi * x[0])
        solution.objectives[:] = [x[0], g * h]


class ZDT4(ZDT):
    def __init__(self):
        super().__init__(10)

    def evaluate(self, solution):
        x = solution.variables[:]
        g = (
            1.0
            + 10.0 * (self.nvars - 1)
            + sum(
                [
                    math.pow(x[i], 2.0) - 10.0 * math.cos(4.0 * math.pi * x[i])
                    for i in range(1, self.nvars)
                ]
            )
        )
        h = 1.0 - math.sqrt(x[0] / g)
        solution.objectives[:] = [x[0], g * h]


class ZDT5(ZDT):
    def __init__(self):
        super().__init__(11)
        self.types[0] = Binary(30)
        self.types[1:] = Binary(5)

    def evaluate(self, solution):
        f = 1.0 + sum(solution.variables[0])
        g = sum([2 + sum(v) if sum(v) < 5 else 1 for v in solution.variables[1:]])
        h = 1.0 / f
        solution.objectives[:] = [f, g * h]


class ZDT6(ZDT):
    def __init__(self):
        super().__init__(10)

    def evaluate(self, solution):
        x = solution.variables[:]
        f = 1.0 - math.exp(-4.0 * x[0]) * math.pow(math.sin(6.0 * math.pi * x[0]), 6.0)
        g = 1.0 + 9.0 * math.pow(sum(x[1:]) / (self.nvars - 1.0), 0.25)
        h = 1.0 - math.pow(f / g, 2.0)
        solution.objectives[:] = [f, g * h]
