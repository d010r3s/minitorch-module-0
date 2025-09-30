"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    return x if x >= y else y

def is_close(x: float, y: float) -> float:
    return 1.0 if abs(x - y) <= 1e-2 else 0.0

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def relu(x: float) -> float:
    return x if x > 0.0 else 0.0

def log(x: float) -> float:
    return math.log(x + 1e-6)

def exp(x: float) -> float:
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    return d * (1.0 / x)

def inv(x: float) -> float:
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    return d * (-(1.0 / (x * x)))

def relu_back(x: float, d: float) -> float:
    return d if x > 0.0 else 0.0

# task 3

from typing import Callable, Iterable, List

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def _apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return _apply


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def _apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        a, b = list(ls1), list(ls2)
        if len(a) != len(b):
            raise ValueError("zipWith: lists must have the same length")
        return [fn(x, y) for x, y in zip(a, b)]
    return _apply

def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def _apply(ls: Iterable[float]) -> float:
        acc = start
        for x in ls:
            acc = fn(acc, x)
        return acc

    return _apply

def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)
