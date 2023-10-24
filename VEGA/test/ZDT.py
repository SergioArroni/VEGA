import math
import numpy as np


class ZDT:
    def __init__(self, m=30):
        self.m = m

    # Función objetivo 1
    def func1(self, x):
        pass

    # Función objetivo 2
    def func2(self, x):
        pass


class ZDT1(ZDT):
    def __init__(self, m=30):
        super().__init__(m)

    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * (sum(x[1:]) / (len(x) - 1))
        h = 1 - np.sqrt(self.func1(x) / g)
        return g * h

    def __str__(self):
        return "ZDT1"


class ZDT2(ZDT):
    def __init__(self, m=30):
        super().__init__(m)

    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * (sum(x[1:]) / (len(x) - 1))
        h = 1 - math.pow(self.func1(x) / g, 2.0)
        return g * h

    def __str__(self):
        return "ZDT2"


class ZDT3(ZDT):
    def __init__(self, m=30):
        super().__init__(m)

    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * (sum(x[1:]) / (len(x) - 1))
        h = (
            1
            - math.sqrt(self.func1(x) / g)
            - (self.func1(x) / g) * math.sin(10 * math.pi * self.func1(x))
        )
        return g * h

    def __str__(self):
        return "ZDT3"


class ZDT4(ZDT):
    def __init__(self, m=10):
        super().__init__(m)

    # Función objetivo 1
    def func1(self, x):
        return x[0]

    # Función objetivo 2
    def func2(self, x):
        g = (
            1.0
            + 10.0 * (len(x) - 1)
            + sum(
                [
                    math.pow(x[i], 2.0) - 10.0 * math.cos(4.0 * math.pi * x[i])
                    for i in range(1, len(x))
                ]
            )
        )
        h = 1.0 - math.sqrt(x[0] / g)
        return g * h

    def __str__(self):
        return "ZDT4"


class ZDT5(ZDT):
    def __init__(self, m=11):
        super().__init__(m)

    # Función objetivo 1
    def func1(self, x):
        return 1 + bin(int(x[0] * 10**13)).count("1")

    # Función objetivo 2
    def func2(self, x):
        g = sum(
            [
                2 + bin(int(v * 10**13)).count("1")
                if bin(int(v * 10**13)).count("1") < 5
                else 1
                for v in x[1:]
            ]
        )
        h = 1 / self.func1(x)
        return g * h

    def __str__(self):
        return "ZDT5"


class ZDT6(ZDT):
    def __init__(self, m=10):
        super().__init__(m)

    # Función objetivo 1
    def func1(self, x):
        return 1 - math.exp(-4 * x[0]) * math.pow(math.sin(6 * math.pi * x[0]), 6)

    # Función objetivo 2
    def func2(self, x):
        g = 1 + 9 * math.pow(sum(x[1:]) / (len(x) - 1), 0.25)
        h = 1 - math.pow(self.func1(x) / g, 2)
        return g * h

    def __str__(self):
        return "ZDT6"
