from math import gcd
from functools import reduce

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def lcm_range(numbers):
    return reduce(lcm, numbers)

def find_min_N():
    N = 1
    while True:
        valid = True
        for n in range(1, 1000000):  # Проверим на первых 100 n
            hok_1 = lcm_range(range(n, n + 10))
            hok_2 = lcm_range(range(n + 1, n + 11))

            if N * hok_2 < hok_1:
                valid = False
                break

        if valid:
            return N
        N += 1

result = find_min_N()
print("Наименьшее N:", result)
